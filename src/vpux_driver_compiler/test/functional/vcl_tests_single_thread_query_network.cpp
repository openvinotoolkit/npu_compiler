//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_tests_common.hpp"

#include <iostream>
#include <vector>

namespace VCLTest {

using VCLQueryNetworkSingleThreadTest = VCLSingleThreadTestsCommon;

TEST_P(VCLQueryNetworkSingleThreadTest, queryNetwork) {
    vcl_compiler_handle_t compiler = nullptr;
    vcl_log_handle_t logHandle = nullptr;

    auto ret = createCompiler(compiler, logHandle, VCL_LOG_INFO);
    ASSERT_EQ(ret, VCL_RESULT_SUCCESS);

    vcl_query_handle_t query = nullptr;
    const auto desc = makeQueryDesc(getNetOptions());

    ret = vclQueryNetworkCreate(compiler, desc, &query);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to create query handle! Result: 0x", ret);
        (void)destroyCompiler(compiler);
        ADD_FAILURE() << "vclQueryNetworkCreate failed! Result:0x" << std::hex << uint64_t(ret) << std::dec;
        return;
    }

    uint64_t layerSize = 0;
    ret = vclQueryNetwork(query, nullptr, &layerSize);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to get query result size! Result: 0x", ret);
        const auto destroyRet = vclQueryNetworkDestroy(query);
        if (destroyRet != VCL_RESULT_SUCCESS) {
            printErrorInfo("Failed to destroy query handle during cleanup! Result: 0x", destroyRet);
            ADD_FAILURE() << "vclQueryNetworkDestroy failed during cleanup! Result:0x" << std::hex
                          << uint64_t(destroyRet) << std::dec;
        }
        (void)destroyCompiler(compiler);
        ADD_FAILURE() << "vclQueryNetwork(size) failed! Result:0x" << std::hex << uint64_t(ret) << std::dec;
        return;
    }

    if (layerSize == 0) {
        std::cerr << "Query result size is zero after first vclQueryNetwork call." << std::endl;
        const auto destroyRet = vclQueryNetworkDestroy(query);
        if (destroyRet != VCL_RESULT_SUCCESS) {
            printErrorInfo("Failed to destroy query handle during cleanup! Result: 0x", destroyRet);
            ADD_FAILURE() << "vclQueryNetworkDestroy failed during cleanup! Result:0x" << std::hex
                          << uint64_t(destroyRet) << std::dec;
        }
        (void)destroyCompiler(compiler);
        ADD_FAILURE() << "layerSize is zero after first vclQueryNetwork call";
        return;
    }

    std::vector<uint8_t> layerRawData(layerSize);
    ret = vclQueryNetwork(query, layerRawData.data(), &layerSize);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to get query result! Result: 0x", ret);
        const auto destroyRet = vclQueryNetworkDestroy(query);
        if (destroyRet != VCL_RESULT_SUCCESS) {
            printErrorInfo("Failed to destroy query handle during cleanup! Result: 0x", destroyRet);
            ADD_FAILURE() << "vclQueryNetworkDestroy failed during cleanup! Result:0x" << std::hex
                          << uint64_t(destroyRet) << std::dec;
        }
        (void)destroyCompiler(compiler);
        ADD_FAILURE() << "vclQueryNetwork(data) failed! Result:0x" << std::hex << uint64_t(ret) << std::dec;
        return;
    }

    ret = vclQueryNetworkDestroy(query);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy query handle! Result: 0x", ret);
        (void)destroyCompiler(compiler);
        ADD_FAILURE() << "vclQueryNetworkDestroy failed! Result:0x" << std::hex << uint64_t(ret) << std::dec;
        return;
    }

    EXPECT_EQ(destroyCompiler(compiler), VCL_RESULT_SUCCESS);
}

/// The path of config files for tests
const auto cidTool = VCLQueryNetworkSingleThreadTest::getCidToolPath();
/// Models and configs for smoke test
const auto smokeIRInfos = VCLQueryNetworkSingleThreadTest::readJson2Vec(cidTool + VCLTest::SMOKE_TEST_CONFIG.data());
/// Models and configs for normal test
const auto irInfos = VCLQueryNetworkSingleThreadTest::readJson2Vec(cidTool + VCLTest::TEST_CONFIG.data());
/// Parameters for smoke tests
const auto smokeParams = testing::Combine(testing::ValuesIn(smokeIRInfos));
/// Parameters for normal tests
const auto params = testing::Combine(testing::ValuesIn(irInfos));

INSTANTIATE_TEST_SUITE_P(smoke_SingleThreadQueryNetwork, VCLQueryNetworkSingleThreadTest, smokeParams,
                         VCLQueryNetworkSingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(SingleThreadQueryNetwork, VCLQueryNetworkSingleThreadTest, params,
                         VCLQueryNetworkSingleThreadTest::getTestCaseName);

}  // namespace VCLTest
