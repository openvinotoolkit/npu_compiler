//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common.hpp"

#include <mutex>
#include <string_view>

namespace VCLTest {

/// Inherits dummy-model SetUp from VCLFunctionalTestsCommon and the allocator parallel
/// compilation helper from VCLAllocatedExecutableTestsBase.
/// Both bases share a single VCLTestsCommon subobject via virtual inheritance.
class VCLAllocatedExecutableParallelCompilationBehaviorTest :
        public VCLFunctionalTestsCommon,
        public VCLAllocatedExecutableTestsBase {
protected:
    static constexpr int kDefaultParallelThreads = 8;
    static constexpr unsigned int kMultiCompilerThreads = 5;
    static constexpr unsigned int kMultiCompilerIterations = 10;
};

TEST_P(VCLAllocatedExecutableParallelCompilationBehaviorTest, ParallelCompilationWithAllocator2) {
    setThreadCount(kDefaultParallelThreads);
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
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run parallel vclAllocatedExecutableCreate2 test " << "! Result:0x"
                                       << std::hex << uint64_t(ret) << std::dec << std::endl;
}

TEST_P(VCLAllocatedExecutableParallelCompilationBehaviorTest, ParallelCompilationWithAllocator4) {
    setThreadCount(kDefaultParallelThreads);
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
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run parallel vclAllocatedExecutableCreate4 test " << "! Result:0x"
                                       << std::hex << uint64_t(ret) << std::dec << std::endl;
}

// Mimics CoreThreadingTestsWithIter::smoke_CompileModel_MultipleCores by running
// compile in multiple threads and multiple iterations, where each iteration
// uses its own compiler instance.
TEST_P(VCLAllocatedExecutableParallelCompilationBehaviorTest, ParallelCompilationWithAllocator4MultipleCompilers) {
    std::mutex firstFailureMutex;
    vcl_result_t firstFailure = VCL_RESULT_SUCCESS;
    const auto options = getNetOptions();

    auto hasFailure = [&]() -> bool {
        std::lock_guard<std::mutex> lock(firstFailureMutex);
        return firstFailure != VCL_RESULT_SUCCESS;
    };

    auto recordFailure = [&](vcl_result_t ret) {
        std::lock_guard<std::mutex> lock(firstFailureMutex);
        if (firstFailure == VCL_RESULT_SUCCESS) {
            firstFailure = ret;
        }
    };

    auto runOneCompilation = [&]() -> vcl_result_t {
        vcl_compiler_handle_t compiler = nullptr;
        vcl_log_handle_t logHandle = nullptr;
        auto result = createCompiler(compiler, logHandle, VCL_LOG_INFO);
        if (result != VCL_RESULT_SUCCESS) {
            return result;
        }

        vcl_allocator2_t allocator;
        allocator.allocate = VCLTest::allocateBlob2;
        allocator.deallocate = VCLTest::deallocateBlob2;

        const auto exeDesc = makeExeDesc(options);

        uint8_t* blob = nullptr;
        uint64_t blobSize = 0;
        vcl_executable_handle_t executable = nullptr;

        result = vclAllocatedExecutableCreate4(compiler, exeDesc, &allocator, &blob, &blobSize, &executable);
        if (result != VCL_RESULT_SUCCESS || blob == nullptr || blobSize == 0 || executable == nullptr) {
            if (blob != nullptr) {
                allocator.deallocate(&allocator, blob);
            }
            if (executable != nullptr) {
                (void)vclExecutableDestroy(executable);
            }
            (void)destroyCompiler(compiler);
            return result != VCL_RESULT_SUCCESS ? result : VCL_RESULT_ERROR_UNKNOWN;
        }

        result = VCLAllocatedExecutableTestsBase::getAndValidateCompatibilityString(executable);

        allocator.deallocate(&allocator, blob);

        const auto destroyExecutableRes = vclExecutableDestroy(executable);
        if (result == VCL_RESULT_SUCCESS && destroyExecutableRes != VCL_RESULT_SUCCESS) {
            result = destroyExecutableRes;
        }

        const auto destroyCompilerRes = destroyCompiler(compiler);
        if (result == VCL_RESULT_SUCCESS && destroyCompilerRes != VCL_RESULT_SUCCESS) {
            result = destroyCompilerRes;
        }

        return result;
    };

    std::vector<std::thread> threads;
    threads.reserve(kMultiCompilerThreads);
    for (unsigned int t = 0; t < kMultiCompilerThreads; ++t) {
        threads.emplace_back([&] {
            for (unsigned int i = 0; i < kMultiCompilerIterations; ++i) {
                if (hasFailure()) {
                    return;
                }
                const auto ret = runOneCompilation();
                if (ret != VCL_RESULT_SUCCESS) {
                    recordFailure(ret);
                    return;
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    const auto ret = [&]() {
        std::lock_guard<std::mutex> lock(firstFailureMutex);
        return firstFailure;
    }();
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS)
            << "Failed to run parallel vclAllocatedExecutableCreate4 multi-compiler test! Result:0x" << std::hex
            << uint64_t(ret) << std::dec << std::endl;
}

TEST_P(VCLAllocatedExecutableParallelCompilationBehaviorTest, AllocatedExecutableCreateWSOneShot) {
    const auto netInfo = std::get<0>(GetParam());
    const auto device = netInfo.at("device");
    if (!VCLAllocatedExecutableParallelCompilationBehaviorTest::isWSOneShotSupported(device)) {
        GTEST_SKIP() << "vclAllocatedExecutableCreateWSOneShot is not supported for NPU" << device;
    }

    setThreadCount(kDefaultParallelThreads);
    const auto ret = runParallelWithAllocator2(getNetOptions(),
                                               [](vcl_compiler_handle_t compiler, vcl_executable_desc_t exeDesc,
                                                  vcl_allocator2_t& allocator, int /*i*/) -> vcl_result_t {
                                                   try {
                                                       return vclAllocatedExecutableCreateWSOneShot(compiler, exeDesc,
                                                                                                    &allocator);
                                                   } catch (const std::exception&) {
                                                       return VCL_RESULT_ERROR_UNSUPPORTED_FEATURE;
                                                   }
                                               });

    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run parallel vclAllocatedExecutableCreateWSOneShot test! Result:0x"
                                       << std::hex << uint64_t(ret) << std::dec << std::endl;
}

TEST_P(VCLAllocatedExecutableParallelCompilationBehaviorTest, AllocatedExecutableCreateWSOneShot2) {
    const auto netInfo = std::get<0>(GetParam());
    const auto device = netInfo.at("device");
    if (!VCLAllocatedExecutableParallelCompilationBehaviorTest::isWSOneShotSupported(device)) {
        GTEST_SKIP() << "vclAllocatedExecutableCreateWSOneShot2 is not supported for NPU" << device;
    }

    setThreadCount(kDefaultParallelThreads);
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

TEST_P(VCLAllocatedExecutableParallelCompilationBehaviorTest, CompatibilityStringMatchesWSOneShot2) {
    const auto netInfo = std::get<0>(GetParam());
    const auto device = netInfo.at("device");
    if (!VCLAllocatedExecutableParallelCompilationBehaviorTest::isWSOneShotSupported(device)) {
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

INSTANTIATE_TEST_SUITE_P(ParallelCompilationTest, VCLAllocatedExecutableParallelCompilationBehaviorTest,
                         getVCLFunctionalTestParams(),
                         VCLAllocatedExecutableParallelCompilationBehaviorTest::getTestCaseName);

}  // namespace VCLTest
