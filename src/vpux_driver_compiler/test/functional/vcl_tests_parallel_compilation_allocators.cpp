//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_tests_common.hpp"

#include <stdint.h>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace VCLTest {

/// Tests parallel compilation using the vclAllocatedExecutableCreateX APIs
/// with 8 concurrent threads.
class VCLParallelCompilationAllocatorsTest : public VCLAllocatedExecutableTestsBase {};

TEST_P(VCLParallelCompilationAllocatorsTest, ParallelCompilationWithAllocator2) {
    setThreadCount(8);
    const auto ret = runParallelWithAllocator2(
            getNetOptions(),
            [](vcl_compiler_handle_t compiler, vcl_executable_desc_t exeDesc, vcl_allocator2_t& allocator,
               int /*i*/) -> vcl_result_t {
                uint8_t* blob = nullptr;
                uint64_t blobSize = 0;

                const auto result = vclAllocatedExecutableCreate2(compiler, exeDesc, &allocator, &blob, &blobSize);
                if (result != VCL_RESULT_SUCCESS || blob == nullptr || blobSize == 0) {
                    if (blob != nullptr) {
                        allocator.deallocate(&allocator, blob);
                    }
                    return result != VCL_RESULT_SUCCESS ? result : VCL_RESULT_ERROR_UNKNOWN;
                }

                allocator.deallocate(&allocator, blob);
                return VCL_RESULT_SUCCESS;
            });
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run parallel vclAllocatedExecutableCreate2 test! Result:0x"
                                       << std::hex << uint64_t(ret) << std::dec << std::endl;
}

TEST_P(VCLParallelCompilationAllocatorsTest, ParallelCompilationWithAllocator4) {
    setThreadCount(8);
    const auto ret = runParallelWithAllocator2(
            getNetOptions(),
            [](vcl_compiler_handle_t compiler, vcl_executable_desc_t exeDesc, vcl_allocator2_t& allocator,
               int /*i*/) -> vcl_result_t {
                uint8_t* blob = nullptr;
                uint64_t blobSize = 0;
                vcl_executable_handle_t executable = nullptr;

                auto result =
                        vclAllocatedExecutableCreate4(compiler, exeDesc, &allocator, &blob, &blobSize, &executable);
                if (result != VCL_RESULT_SUCCESS || blob == nullptr || blobSize == 0 || executable == nullptr) {
                    if (blob != nullptr) {
                        allocator.deallocate(&allocator, blob);
                    }
                    if (executable != nullptr) {
                        (void)vclExecutableDestroy(executable);
                    }
                    return result != VCL_RESULT_SUCCESS ? result : VCL_RESULT_ERROR_UNKNOWN;
                }

                result = VCLAllocatedExecutableTestsBase::getAndValidateCompatibilityString(executable);

                allocator.deallocate(&allocator, blob);

                const auto destroyExecutableRes = vclExecutableDestroy(executable);
                if (result == VCL_RESULT_SUCCESS && destroyExecutableRes != VCL_RESULT_SUCCESS) {
                    result = destroyExecutableRes;
                }

                return result;
            });
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run parallel vclAllocatedExecutableCreate4 test! Result:0x"
                                       << std::hex << uint64_t(ret) << std::dec << std::endl;
}

TEST_P(VCLParallelCompilationAllocatorsTest, AllocatedExecutableCreateWSOneShot2) {
    const auto netInfo = std::get<0>(GetParam());
    const auto device = netInfo.at("device");
    if (!VCLParallelCompilationAllocatorsTest::isWSOneShotSupported(device)) {
        GTEST_SKIP() << "vclAllocatedExecutableCreateWSOneShot2 is not supported for NPU" << device;
    }

    setThreadCount(8);
    const auto ret = runParallelWithAllocator2(
            getNetOptions(),
            [](vcl_compiler_handle_t compiler, vcl_executable_desc_t exeDesc, vcl_allocator2_t& allocator,
               int /*i*/) -> vcl_result_t {
                vcl_executable_handle_t executable = nullptr;
                auto result = vclAllocatedExecutableCreateWSOneShot2(compiler, exeDesc, &allocator, &executable);
                if (result != VCL_RESULT_SUCCESS || executable == nullptr) {
                    return result != VCL_RESULT_SUCCESS ? result : VCL_RESULT_ERROR_UNKNOWN;
                }

                result = VCLAllocatedExecutableTestsBase::getAndValidateCompatibilityString(executable);

                const auto destroyResult = vclExecutableDestroy(executable);
                return result == VCL_RESULT_SUCCESS ? destroyResult : result;
            });
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS)
            << "Failed to run parallel vclAllocatedExecutableCreateWSOneShot2 test! Result:0x" << std::hex
            << uint64_t(ret) << std::dec << std::endl;
}

TEST_P(VCLParallelCompilationAllocatorsTest, CompatibilityStringMatchesWSOneShot2) {
    const auto netInfo = std::get<0>(GetParam());
    const auto device = netInfo.at("device");
    if (!VCLParallelCompilationAllocatorsTest::isWSOneShotSupported(device)) {
        GTEST_SKIP() << "vclAllocatedExecutableCreateWSOneShot2 is not supported for NPU" << device;
    }

    vcl_compiler_handle_t compiler = nullptr;
    vcl_log_handle_t logHandle = nullptr;
    auto result = createCompiler(compiler, logHandle, VCL_LOG_INFO);
    ASSERT_EQ(result, VCL_RESULT_SUCCESS);

    vcl_allocator2_t allocator;
    allocator.allocate = VCLTest::allocateBlob2;
    allocator.deallocate = VCLTest::deallocateBlob2;
    const auto exeDesc = makeExeDesc(getNetOptions());

    auto getCompatibilityString = [&](vcl_executable_handle_t executable) {
        uint64_t size = 0;
        auto ret = vclExecutableGetCompatibilityString(executable, nullptr, &size);
        if (ret != VCL_RESULT_SUCCESS || size <= 1) {
            return std::pair<vcl_result_t, std::string>{ret != VCL_RESULT_SUCCESS ? ret : VCL_RESULT_ERROR_UNKNOWN, {}};
        }

        std::string compatibilityString(size, '\0');
        ret = vclExecutableGetCompatibilityString(executable, compatibilityString.data(), &size);
        if (ret != VCL_RESULT_SUCCESS || compatibilityString.back() != '\0') {
            return std::pair<vcl_result_t, std::string>{
                    ret != VCL_RESULT_SUCCESS ? ret : VCL_RESULT_ERROR_INVALID_ARGUMENT,
                    {}};
        }
        compatibilityString.pop_back();
        return std::pair<vcl_result_t, std::string>{VCL_RESULT_SUCCESS, std::move(compatibilityString)};
    };

    uint8_t* blob = nullptr;
    uint64_t blobSize = 0;
    vcl_executable_handle_t allocator4Executable = nullptr;
    result = vclAllocatedExecutableCreate4(compiler, exeDesc, &allocator, &blob, &blobSize, &allocator4Executable);
    ASSERT_EQ(result, VCL_RESULT_SUCCESS);
    ASSERT_NE(allocator4Executable, nullptr);
    ASSERT_NE(blob, nullptr);
    ASSERT_GT(blobSize, 0);

    const auto allocator4Compatibility = getCompatibilityString(allocator4Executable);
    ASSERT_EQ(allocator4Compatibility.first, VCL_RESULT_SUCCESS);
    ASSERT_EQ(VCLAllocatedExecutableTestsBase::validateCompatibilityString(allocator4Compatibility.second),
              VCL_RESULT_SUCCESS);

    vcl_executable_handle_t wsOneShotExecutable = nullptr;
    result = vclAllocatedExecutableCreateWSOneShot2(compiler, exeDesc, &allocator, &wsOneShotExecutable);
    ASSERT_EQ(result, VCL_RESULT_SUCCESS);
    ASSERT_NE(wsOneShotExecutable, nullptr);

    const auto wsOneShotCompatibility = getCompatibilityString(wsOneShotExecutable);
    ASSERT_EQ(wsOneShotCompatibility.first, VCL_RESULT_SUCCESS);
    ASSERT_EQ(VCLAllocatedExecutableTestsBase::validateCompatibilityString(wsOneShotCompatibility.second),
              VCL_RESULT_SUCCESS);
    EXPECT_EQ(allocator4Compatibility.second, wsOneShotCompatibility.second);

    const auto wsOneShotDestroyResult = vclExecutableDestroy(wsOneShotExecutable);
    const auto allocator4DestroyResult = vclExecutableDestroy(allocator4Executable);
    allocator.deallocate(&allocator, blob);
    const auto destroyCompilerResult = destroyCompiler(compiler);

    EXPECT_EQ(wsOneShotDestroyResult, VCL_RESULT_SUCCESS);
    EXPECT_EQ(allocator4DestroyResult, VCL_RESULT_SUCCESS);
    EXPECT_EQ(destroyCompilerResult, VCL_RESULT_SUCCESS);
}

/// The path of config files for tests
const auto cidTool = VCLParallelCompilationAllocatorsTest::getCidToolPath();
/// Models and configs for smoke test
const auto smokeIrInfos =
        VCLParallelCompilationAllocatorsTest::readJson2Vec(cidTool + VCLTest::SMOKE_TEST_CONFIG.data());
/// Models and configs for normal test
const auto irInfos = VCLParallelCompilationAllocatorsTest::readJson2Vec(cidTool + VCLTest::TEST_CONFIG.data());
/// Parameters for smoke tests
const auto smokeParams = testing::Combine(testing::ValuesIn(smokeIrInfos));
/// Parameters for normal tests
const auto params = testing::Combine(testing::ValuesIn(irInfos));

INSTANTIATE_TEST_SUITE_P(smoke_ParallelCompilationTest, VCLParallelCompilationAllocatorsTest, smokeParams,
                         VCLParallelCompilationAllocatorsTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ParallelCompilationTest, VCLParallelCompilationAllocatorsTest, params,
                         VCLParallelCompilationAllocatorsTest::getTestCaseName);

}  // namespace VCLTest
