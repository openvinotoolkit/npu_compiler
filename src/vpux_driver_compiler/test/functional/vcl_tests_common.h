//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "npu_driver_compiler.h"

#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <openvino/opsets/opset3.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>

#if defined(_WIN32)
#include "Shlwapi.h"
// These two undefs are to avoid min/max macro interfering introduced by Shlwapi.h.
#undef min
#undef max
#else
#include <sys/stat.h>
#endif

namespace VCLTestsUtils {

using IRInfoTestType = std::vector<std::unordered_map<std::string, std::string>>;
using VCLTestsParams = std::tuple<std::unordered_map<std::string, std::string>>;

/// The file contains the models and its configuration for somke test
const std::string SMOKE_TEST_CONFIG = "/test_smoke.json";
/// The file contains the modesl and its configuration for noraml test
const std::string TEST_CONFIG = "/test.json";

/**
 * @brief Base class to parse config file to get test cases and provide helper functions
 */
class VCLTestsCommon : public testing::WithParamInterface<VCLTestsParams>, public testing::Test {
public:
    VCLTestsCommon(): modelIR(), modelIRSize(0) {
    }
    virtual ~VCLTestsCommon() = default;

    /**
     * @brief Create the test name suffixes with content of test params
     */
    static std::string getTestCaseName(const testing::TestParamInfo<VCLTestsParams>& obj) {
        auto param = obj.param;
        auto netInfo = std::get<0>(param);
        const std::string netName = netInfo.at("network");
        const std::string deviceID = netInfo.at("device");
        return netName + std::string("_VPUX.") + deviceID;
    }

    /**
     * @brief Prepare modelIRData of one test case for compiler
     */
    vcl_result_t initModelData(const char* netName, const char* weightName);

    /**
     * @brief Create a simple model to test compiler
     */
    std::shared_ptr<ov::Model> createSimpleModel();

    /**
     * @brief Get the location of model package defined by POR_PATH
     */
    std::string getTestModelsBasePath();

    /**
     * @brief Get the location of config files for tests
     */
    static std::string getCidToolPath();

    /**
     * @brief Parse config file and detect test cases
     */
    static IRInfoTestType readJson2Vec(std::string fileName);

    /**
     * @brief Add platform info to model build options
     */
    void postProcessNetOptions(const std::string& device);

    /**
     * @brief Return the build options of model
     */
    std::string getNetOptions() {
        return netOptions;
    };

    /**
     * @brief Prepare modelIRData of all tests for compiler
     */
    void SetUp() override;

    std::vector<uint8_t>& getModelIR() {
        return modelIR;
    }

    size_t getModelIRSize() {
        return modelIRSize;
    }

private:
    /// Model build flags
    std::string netOptions;
    /// The data of modelIR to create executable
    std::vector<uint8_t> modelIR;
    size_t modelIRSize;
};
}  // namespace VCLTestsUtils
