//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vcl_tests_common.hpp"

#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/subgraph_builders/matmul_bias.hpp"
#include "common_test_utils/subgraph_builders/split_concat.hpp"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace VCLTest {

class VCLFunctionalTestsCommon : public virtual VCLTestsCommon {
public:
    static IRInfoTestType getDummyIRInfos() {
        return buildDummyIRInfos();
    }

    void SetUp() override {
        const auto ir = GetParam();
        const auto netInfo = std::get<0>(ir);

        const auto network = netInfo.at("network");
        const auto device = netInfo.at("device");
        postProcessNetOptions(device);

        auto model = createDummyModel(network);
        if (model == nullptr) {
            GTEST_SKIP() << "Unknown dummy model: " << network;
        }

        const auto* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
        auto safeName = std::string(testInfo->test_suite_name()) + "_" + testInfo->name();
        // Replacing any non-alphanumeric character with _
        std::replace_if(
                safeName.begin(), safeName.end(),
                [](unsigned char ch) {
                    return !std::isalnum(ch);
                },
                '_');

        const auto tmpDir = std::filesystem::temp_directory_path();
        // Append a per-invocation random suffix to prevent cross-process collisions
        // when the same test runs concurrently in multiple shards or workers.
        const auto uniqueSuffix = std::to_string(std::random_device{}());
        const auto xmlPath = (tmpDir / (safeName + "_" + uniqueSuffix + "_functional.xml")).string();
        const auto binPath = (tmpDir / (safeName + "_" + uniqueSuffix + "_functional.bin")).string();

        vcl_result_t initRes = VCL_RESULT_SUCCESS;
        try {
            ov::pass::Manager passManager;
            passManager.register_pass<ov::pass::Serialize>(xmlPath, binPath);
            passManager.run_passes(std::move(model));
            initRes = initModelData(xmlPath.c_str(), binPath.c_str());
        } catch (const std::exception& e) {
            (void)std::remove(xmlPath.c_str());
            (void)std::remove(binPath.c_str());
            ADD_FAILURE() << "Exception during model serialization or initialization: " << e.what();
            throw std::runtime_error(std::string("Model setup failed: ") + e.what());
        }
        ASSERT_EQ(initRes, VCL_RESULT_SUCCESS);
        (void)std::remove(xmlPath.c_str());
        (void)std::remove(binPath.c_str());
    }

private:
    static IRInfoTestType buildDummyIRInfos() {
        const std::vector<std::string> networks = {
                "split_concat",
                "matmul_bias",
                "conv_pool_relu",
        };

        const std::vector<std::string> devices = [] {
            std::vector<std::string> d{"3720", "4000", "5010", "5020"};
            return d;
        }();

        IRInfoTestType infos;
        infos.reserve(networks.size() * devices.size());

        for (const auto& network : networks) {
            for (const auto& device : devices) {
                infos.push_back({{"network", network}, {"device", device}, {"info", ""}});
            }
        }

        return infos;
    }

    static std::shared_ptr<ov::Model> createDummyModel(const std::string& network) {
        if (network == "split_concat") {
            return ov::test::utils::make_split_concat();
        }

        if (network == "matmul_bias") {
            return ov::test::utils::make_matmul_bias();
        }

        if (network == "conv_pool_relu") {
            return ov::test::utils::make_conv_pool_relu();
        }

        return nullptr;
    }
};

inline auto getVCLFunctionalTestParams() {
    return testing::Combine(testing::ValuesIn(VCLFunctionalTestsCommon::getDummyIRInfos()));
}

}  // namespace VCLTest
