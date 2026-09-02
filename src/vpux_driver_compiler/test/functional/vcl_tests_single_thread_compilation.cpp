//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_tests_common.hpp"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string_view>
#include <type_traits>

namespace VCLTest {
using VCLSingleThreadTest = VCLSingleThreadTestsCommon;

TEST_P(VCLSingleThreadTest, compileModel) {
    vcl_compiler_handle_t compiler = nullptr;
    vcl_log_handle_t logHandle = nullptr;

    auto ret = createCompiler(compiler, logHandle, VCL_LOG_INFO);
    ASSERT_EQ(ret, VCL_RESULT_SUCCESS);

    vcl_executable_handle_t executable = nullptr;
    const auto options = getNetOptions();
    auto exeDesc = makeExeDesc(options);

    ret = vclExecutableCreate(compiler, exeDesc, &executable);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to create executable handle! Result: 0x", ret);
        (void)destroyCompiler(compiler);
    }
    ASSERT_EQ(ret, VCL_RESULT_SUCCESS);

    uint64_t blobSize = 0;
    ret = vclExecutableGetSerializableBlob(executable, nullptr, &blobSize);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to get blob size! Result: 0x", ret);
        vclExecutableDestroy(executable);
        (void)destroyCompiler(compiler);
    }
    ASSERT_EQ(ret, VCL_RESULT_SUCCESS);

    if (blobSize == 0) {
        std::cerr << "Blob size is zero after first vclExecutableGetSerializableBlob call." << std::endl;
        vclExecutableDestroy(executable);
        (void)destroyCompiler(compiler);
    }
    ASSERT_NE(blobSize, 0U);

    uint8_t* blob = static_cast<uint8_t*>(malloc(blobSize));
    if (!blob) {
        std::cerr << "Failed to alloc memory for blob!\n";
        vclExecutableDestroy(executable);
        (void)destroyCompiler(compiler);
    }
    ASSERT_NE(blob, nullptr);

    ret = vclExecutableGetSerializableBlob(executable, blob, &blobSize);
    free(blob);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to get blob! Result: 0x", ret);
        vclExecutableDestroy(executable);
        (void)destroyCompiler(compiler);
    }
    ASSERT_EQ(ret, VCL_RESULT_SUCCESS);

    ret = vclExecutableDestroy(executable);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy executable! Result: 0x", ret);
        (void)destroyCompiler(compiler);
    }
    ASSERT_EQ(ret, VCL_RESULT_SUCCESS);

    EXPECT_EQ(destroyCompiler(compiler), VCL_RESULT_SUCCESS);
}

/// The path of config files for tests
const auto cidTool = VCLSingleThreadTest::getCidToolPath();
/// Models and configs for smoke test
const auto smokeIRInfos = VCLSingleThreadTest::readJson2Vec(cidTool + VCLTest::SMOKE_TEST_CONFIG.data());
/// Models and configs for normal test
const auto irInfos = VCLSingleThreadTest::readJson2Vec(cidTool + VCLTest::TEST_CONFIG.data());
/// Parameters for smoke tests
const auto smokeParams = testing::Combine(testing::ValuesIn(smokeIRInfos));
/// Parameters for normal tests
const auto params = testing::Combine(testing::ValuesIn(irInfos));

INSTANTIATE_TEST_SUITE_P(smoke_SingleThreadCompilation, VCLSingleThreadTest, smokeParams,
                         VCLSingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(SingleThreadCompilation, VCLSingleThreadTest, params, VCLSingleThreadTest::getTestCaseName);

}  // namespace VCLTest
