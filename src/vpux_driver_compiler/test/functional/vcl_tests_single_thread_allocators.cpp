//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_tests_common.hpp"

#include <stdint.h>
#include <iostream>
#include <string_view>
#include <vector>

namespace VCLTest {

using VCLAllocator2SingleThreadTest = VCLSingleThreadTestsCommon;

TEST_P(VCLAllocator2SingleThreadTest, compileModel) {
    vcl_compiler_handle_t compiler = nullptr;
    vcl_log_handle_t logHandle = nullptr;

    auto ret = createCompiler(compiler, logHandle, VCL_LOG_INFO);
    ASSERT_EQ(ret, VCL_RESULT_SUCCESS);

    uint8_t* blob = nullptr;
    uint64_t size = 0;

    const auto options = getNetOptions();
    auto exeDesc = makeExeDesc(options);
    vcl_allocator2_t allocator;
    allocator.allocate = VCLTest::allocateBlob2;
    allocator.deallocate = VCLTest::deallocateBlob2;

    ret = vclAllocatedExecutableCreate2(compiler, exeDesc, &allocator, &blob, &size);
    if (ret != VCL_RESULT_SUCCESS || blob == nullptr || size == 0) {
        printErrorInfo("Failed to create executable handle! Result: 0x", ret);
        if (blob != nullptr) {
            allocator.deallocate(&allocator, blob);
        }
        (void)destroyCompiler(compiler);
        ADD_FAILURE() << "vclAllocatedExecutableCreate2 failed or returned invalid output. Result:0x" << std::hex
                      << uint64_t(ret) << std::dec;
        return;
    }

    allocator.deallocate(&allocator, blob);

    EXPECT_EQ(destroyCompiler(compiler), VCL_RESULT_SUCCESS);
}

using VCLAllocator4SingleThreadTest = VCLSingleThreadTestsCommon;

namespace {
// RAII guards to ensure resources are cleaned up on any exit path.
struct BlobGuard {
    uint8_t*& blob;
    vcl_allocator2_t* allocator;

    BlobGuard(uint8_t*& blobRef, vcl_allocator2_t* allocatorPtr): blob(blobRef), allocator(allocatorPtr) {
    }

    BlobGuard(const BlobGuard&) = delete;
    BlobGuard& operator=(const BlobGuard&) = delete;
    BlobGuard(BlobGuard&&) = delete;
    BlobGuard& operator=(BlobGuard&&) = delete;

    ~BlobGuard() {
        if (blob != nullptr && allocator != nullptr) {
            allocator->deallocate(allocator, blob);
            blob = nullptr;
        }
    }
};

struct ExecutableGuard {
    vcl_executable_handle_t& executable;

    explicit ExecutableGuard(vcl_executable_handle_t& executableRef): executable(executableRef) {
    }

    ExecutableGuard(const ExecutableGuard&) = delete;
    ExecutableGuard& operator=(const ExecutableGuard&) = delete;
    ExecutableGuard(ExecutableGuard&&) = delete;
    ExecutableGuard& operator=(ExecutableGuard&&) = delete;

    ~ExecutableGuard() {
        if (executable != nullptr) {
            (void)vclExecutableDestroy(executable);
            executable = nullptr;
        }
    }
};

struct CompilerGuard {
    vcl_compiler_handle_t& compiler;
    VCLTestsCommon* test_instance;

    CompilerGuard(vcl_compiler_handle_t& compilerRef, VCLTestsCommon* test)
            : compiler(compilerRef), test_instance(test) {
    }

    CompilerGuard(const CompilerGuard&) = delete;
    CompilerGuard& operator=(const CompilerGuard&) = delete;
    CompilerGuard(CompilerGuard&&) = delete;
    CompilerGuard& operator=(CompilerGuard&&) = delete;

    ~CompilerGuard() {
        if (compiler != nullptr) {
            (void)test_instance->destroyCompiler(compiler);
            compiler = nullptr;
        }
    }
};
}  // anonymous namespace

TEST_P(VCLAllocator4SingleThreadTest, compileModelWithCompatibilityString) {
    vcl_compiler_handle_t compiler = nullptr;
    vcl_log_handle_t logHandle = nullptr;

    auto ret = createCompiler(compiler, logHandle, VCL_LOG_INFO);
    ASSERT_EQ(ret, VCL_RESULT_SUCCESS);

    uint8_t* blob = nullptr;
    uint64_t blobSize = 0;
    vcl_executable_handle_t executable = nullptr;

    const auto options = getNetOptions();
    auto exeDesc = makeExeDesc(options);
    vcl_allocator2_t allocator;
    allocator.allocate = VCLTest::allocateBlob2;
    allocator.deallocate = VCLTest::deallocateBlob2;

    // RAII guards ensure resources are cleaned up automatically on any exit path.
    CompilerGuard compiler_guard{compiler, this};
    ExecutableGuard executable_guard{executable};
    BlobGuard blob_guard{blob, &allocator};

    ret = vclAllocatedExecutableCreate4(compiler, exeDesc, &allocator, &blob, &blobSize, &executable);
    if (ret != VCL_RESULT_SUCCESS || blob == nullptr || blobSize == 0 || executable == nullptr) {
        printErrorInfo("Failed to create executable via vclAllocatedExecutableCreate4! Result: 0x", ret);
        ADD_FAILURE() << "vclAllocatedExecutableCreate4 failed or returned invalid output. Result:0x" << std::hex
                      << uint64_t(ret) << std::dec;
        return;
    }

    uint64_t compatibilityStringSize = 0;
    ret = vclExecutableGetCompatibilityString(executable, nullptr, &compatibilityStringSize);
    if (ret != VCL_RESULT_SUCCESS || compatibilityStringSize == 0) {
        printErrorInfo("Failed to get compatibility string size! Result: 0x", ret);
        ADD_FAILURE() << "vclExecutableGetCompatibilityString(size) failed or returned empty output. Result:0x"
                      << std::hex << uint64_t(ret) << std::dec;
        return;
    }

    std::vector<char> compatibilityString(compatibilityStringSize);
    ret = vclExecutableGetCompatibilityString(executable, compatibilityString.data(), &compatibilityStringSize);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to get compatibility string! Result: 0x", ret);
        ADD_FAILURE() << "vclExecutableGetCompatibilityString(data) failed. Result:0x" << std::hex << uint64_t(ret)
                      << std::dec;
        return;
    }

    std::cout << "Compatibility string: " << compatibilityString.data() << "\n";
    EXPECT_TRUE(std::string_view(compatibilityString.data(), compatibilityString.size())
                        .substr(0, sizeof("compiler=") - 1) == "compiler=");

    ret = vclExecutableDestroy(executable);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy executable! Result: 0x", ret);
        ADD_FAILURE() << "vclExecutableDestroy failed. Result:0x" << std::hex << uint64_t(ret) << std::dec;
        executable = nullptr;
        return;
    }
    // Suppress guard's destructor since we successfully destroyed manually.
    executable = nullptr;

    EXPECT_EQ(destroyCompiler(compiler), VCL_RESULT_SUCCESS);
    // Suppress guard's destructor since we successfully destroyed manually.
    compiler = nullptr;
}

/// The path of config files for tests
const auto cidTool = VCLAllocator2SingleThreadTest::getCidToolPath();
/// Models and configs for smoke test
const auto smokeIRInfos = VCLAllocator2SingleThreadTest::readJson2Vec(cidTool + VCLTest::SMOKE_TEST_CONFIG.data());
/// Models and configs for normal test
const auto irInfos = VCLAllocator2SingleThreadTest::readJson2Vec(cidTool + VCLTest::TEST_CONFIG.data());
/// Parameters for smoke tests
const auto smokeParams = testing::Combine(testing::ValuesIn(smokeIRInfos));
/// Parameters for normal tests
const auto params = testing::Combine(testing::ValuesIn(irInfos));

INSTANTIATE_TEST_SUITE_P(smoke_SingleThreadCompilation, VCLAllocator2SingleThreadTest, smokeParams,
                         VCLAllocator2SingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(SingleThreadCompilation, VCLAllocator2SingleThreadTest, params,
                         VCLAllocator2SingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SingleThreadCompilation, VCLAllocator4SingleThreadTest, smokeParams,
                         VCLAllocator4SingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(SingleThreadCompilation, VCLAllocator4SingleThreadTest, params,
                         VCLAllocator4SingleThreadTest::getTestCaseName);

}  // namespace VCLTest
