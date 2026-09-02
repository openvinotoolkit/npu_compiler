//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common.hpp"

namespace VCLTest {

class VCLQueryNetworkParallelBehaviorTest : public VCLFunctionalTestsCommon {
public:
    vcl_result_t parallelQueryNetwork(const std::string& options);
};

vcl_result_t VCLQueryNetworkParallelBehaviorTest::parallelQueryNetwork(const std::string& options) {
    vcl_compiler_handle_t compiler = nullptr;
    vcl_log_handle_t logHandle = nullptr;
    if (const auto ret = createCompiler(compiler, logHandle, VCL_LOG_INFO); ret != VCL_RESULT_SUCCESS) {
        return ret;
    }

    const auto desc = makeQueryDesc(options);
    std::vector<std::thread> queryThreads;
    std::vector<vcl_result_t> queryResults(numCompilationThreads, VCL_RESULT_SUCCESS);

    for (int i = 0; i < numCompilationThreads; ++i) {
        queryThreads.emplace_back([&, i] {
            vcl_query_handle_t queryHandle = nullptr;
            queryResults[i] = vclQueryNetworkCreate(compiler, desc, &queryHandle);
            if (queryResults[i] != VCL_RESULT_SUCCESS) {
                return;
            }

            uint64_t layerSize = 0;
            queryResults[i] = vclQueryNetwork(queryHandle, nullptr, &layerSize);
            if (queryResults[i] != VCL_RESULT_SUCCESS) {
                (void)vclQueryNetworkDestroy(queryHandle);
                return;
            }
            if (layerSize == 0) {
                std::cerr << "Query result size is zero after first vclQueryNetwork call in thread " << i << "."
                          << std::endl;
                queryResults[i] = VCL_RESULT_ERROR_INVALID_ARGUMENT;
                (void)vclQueryNetworkDestroy(queryHandle);
                return;
            }

            std::vector<uint8_t> layerRawData;
            layerRawData.resize(layerSize);
            queryResults[i] = vclQueryNetwork(queryHandle, layerRawData.data(), &layerSize);
            if (queryResults[i] != VCL_RESULT_SUCCESS) {
                (void)vclQueryNetworkDestroy(queryHandle);
                return;
            }

            queryResults[i] = vclQueryNetworkDestroy(queryHandle);
        });
    }

    for (auto& thread : queryThreads) {
        thread.join();
    }

    for (size_t i = 0; i < queryResults.size(); ++i) {
        if (queryResults[i] != VCL_RESULT_SUCCESS) {
            std::string printStr = "Failed query-network API flow with " + std::to_string(i) +
                                   " thread!\n"
                                   "Result:0x";
            printErrorInfo(printStr, queryResults[i]);
            (void)destroyCompiler(compiler);
            return queryResults[i];
        }
    }

    return destroyCompiler(compiler);
}

TEST_P(VCLQueryNetworkParallelBehaviorTest, ParallelQueryNetwork) {
    setThreadCount(8);
    const auto ret = parallelQueryNetwork(getNetOptions());
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run parallel query-network test! Result:0x" << std::hex
                                       << uint64_t(ret) << std::dec << std::endl;
}
INSTANTIATE_TEST_SUITE_P(ParallelQueryNetworkTest, VCLQueryNetworkParallelBehaviorTest, getVCLFunctionalTestParams(),
                         VCLQueryNetworkParallelBehaviorTest::getTestCaseName);
}  // namespace VCLTest
