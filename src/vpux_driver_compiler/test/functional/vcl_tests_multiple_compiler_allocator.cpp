//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_tests_common.hpp"

#include <stdint.h>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace VCLTest {

class VCLAllocatorMultipleCompilerTest : public VCLTestsCommon {
public:
    /**
     * @brief Use compiler to compile one model using vclAllocatedExecutableCreate2
     *
     * @param options  Build flags of a model
     */
    vcl_result_t singleCompilationWithAllocator2(const std::string& options);

    /**
     * @brief Use compiler to compile one model using vclAllocatedExecutableCreate4
     *        and verify the compatibility string via vclExecutableGetCompatibilityString
     *
     * @param options  Build flags of a model
     */
    vcl_result_t singleCompilationWithAllocator4(const std::string& options);

    /**
     * @brief Check if all compilations have created blob
     */
    bool check() const;

    size_t getOutputSize() const {
        return outputs.size();
    }

private:
    std::vector<std::string> outputs;
    std::mutex lock;
};

vcl_result_t VCLAllocatorMultipleCompilerTest::singleCompilationWithAllocator2(const std::string& options) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    vcl_compiler_handle_t compiler = nullptr;
    vcl_log_handle_t logHandle = nullptr;
    ret = createCompiler(compiler, logHandle, VCL_LOG_INFO);

    if (ret != VCL_RESULT_SUCCESS) {
        return ret;
    }

    vcl_allocator2_t allocator;
    allocator.allocate = VCLTest::allocateBlob2;
    allocator.deallocate = VCLTest::deallocateBlob2;
    uint8_t* blob = nullptr;
    uint64_t size = 0;

    auto exeDesc = makeExeDesc(options);
    ret = vclAllocatedExecutableCreate2(compiler, exeDesc, &allocator, &blob, &size);

    if (ret != VCL_RESULT_SUCCESS || blob == nullptr || size == 0) {
        printErrorInfo("Failed to create executable! Result: 0x", ret);
        (void)destroyCompiler(compiler);
        return ret != VCL_RESULT_SUCCESS ? ret : VCL_RESULT_ERROR_UNKNOWN;
    }

    std::string output(reinterpret_cast<char*>(blob), size);
    {
        std::lock_guard<std::mutex> guard(lock);
        outputs.push_back(std::move(output));
    }

    allocator.deallocate(&allocator, blob);

    return destroyCompiler(compiler);
}

vcl_result_t VCLAllocatorMultipleCompilerTest::singleCompilationWithAllocator4(const std::string& options) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    vcl_compiler_handle_t compiler = nullptr;
    vcl_log_handle_t logHandle = nullptr;
    ret = createCompiler(compiler, logHandle, VCL_LOG_INFO);
    if (ret != VCL_RESULT_SUCCESS) {
        return ret;
    }

    vcl_allocator2_t allocator;
    allocator.allocate = VCLTest::allocateBlob2;
    allocator.deallocate = VCLTest::deallocateBlob2;
    uint8_t* blob = nullptr;
    uint64_t blobSize = 0;
    vcl_executable_handle_t executable = nullptr;

    auto exeDesc = makeExeDesc(options);
    ret = vclAllocatedExecutableCreate4(compiler, exeDesc, &allocator, &blob, &blobSize, &executable);

    if (ret != VCL_RESULT_SUCCESS || blob == nullptr || blobSize == 0 || executable == nullptr) {
        printErrorInfo("Failed to create executable! Result: 0x", ret);
        if (blob != nullptr) {
            allocator.deallocate(&allocator, blob);
        }
        if (executable != nullptr) {
            (void)vclExecutableDestroy(executable);
        }
        (void)destroyCompiler(compiler);
        return ret != VCL_RESULT_SUCCESS ? ret : VCL_RESULT_ERROR_UNKNOWN;
    }

    uint64_t compatibilityStringSize = 0;
    ret = vclExecutableGetCompatibilityString(executable, nullptr, &compatibilityStringSize);
    if (ret == VCL_RESULT_SUCCESS && compatibilityStringSize != 0) {
        std::vector<char> compatibilityString(compatibilityStringSize);
        ret = vclExecutableGetCompatibilityString(executable, compatibilityString.data(), &compatibilityStringSize);
        if (ret == VCL_RESULT_SUCCESS) {
            constexpr size_t prefixLen = sizeof("compiler=") - 1;
            if (compatibilityStringSize <= prefixLen ||
                strncmp(compatibilityString.data(), "compiler=", prefixLen) != 0) {
                ret = VCL_RESULT_ERROR_INVALID_ARGUMENT;
            }
        }
    } else {
        ret = ret != VCL_RESULT_SUCCESS ? ret : VCL_RESULT_ERROR_UNKNOWN;
    }

    if (ret == VCL_RESULT_SUCCESS) {
        std::string output(reinterpret_cast<char*>(blob), blobSize);
        std::lock_guard<std::mutex> guard(lock);
        outputs.push_back(std::move(output));
    }

    allocator.deallocate(&allocator, blob);

    const auto destroyRet = vclExecutableDestroy(executable);
    if (ret == VCL_RESULT_SUCCESS && destroyRet != VCL_RESULT_SUCCESS) {
        ret = destroyRet;
    }

    const auto compilerRet = destroyCompiler(compiler);
    if (ret == VCL_RESULT_SUCCESS) {
        ret = compilerRet;
    }
    return ret;
}

bool VCLAllocatorMultipleCompilerTest::check() const {
    const size_t count = outputs.size();
    if (count == 0) {
        std::cerr << "No outputs!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < count; i++) {
        if (outputs[i].size() == 0) {
            std::cerr << "blob " << i << "'s size is zero." << std::endl;
            return false;
        }
    }
    return true;
}

TEST_P(VCLAllocatorMultipleCompilerTest, CompilerInstanceWithAllocator2) {
    std::vector<vcl_result_t> res(5, VCL_RESULT_SUCCESS);
    std::vector<std::thread> threads;
    // Get the ref output from single thread env;
    threads.emplace_back([&res, this] {
        res[0] = singleCompilationWithAllocator2(this->getNetOptions());
    });

    // Get the outputs from multiple threads env;
    for (int i = 1; i < 5; ++i) {
        threads.emplace_back([&res, this, i] {
            res[i] = singleCompilationWithAllocator2(this->getNetOptions());
        });
    }

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

    for (auto i = res.begin(); i != res.end(); ++i) {
        EXPECT_EQ(*i, VCL_RESULT_SUCCESS) << "Thread " << std::distance(res.begin(), i) << " failed! Result:0x"
                                          << std::hex << uint64_t(*i) << std::dec;
    }

    EXPECT_EQ(getOutputSize(), 5) << "Did not get all outputs successfully!" << std::endl;
    EXPECT_EQ(check(), true);
}

TEST_P(VCLAllocatorMultipleCompilerTest, CompilerInstanceWithAllocator4) {
    std::vector<vcl_result_t> res(5, VCL_RESULT_SUCCESS);
    std::vector<std::thread> threads;
    threads.emplace_back([&res, this] {
        res[0] = singleCompilationWithAllocator4(this->getNetOptions());
    });

    for (int i = 1; i < 5; ++i) {
        threads.emplace_back([&res, this, i] {
            res[i] = singleCompilationWithAllocator4(this->getNetOptions());
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (auto i = res.begin(); i != res.end(); ++i) {
        EXPECT_EQ(*i, VCL_RESULT_SUCCESS) << "Thread " << std::distance(res.begin(), i) << " failed! Result:0x"
                                          << std::hex << uint64_t(*i) << std::dec;
    }

    EXPECT_EQ(getOutputSize(), 5) << "Did not get all outputs successfully!" << std::endl;
    EXPECT_EQ(check(), true);
}

/// The path of config files for tests
const auto cidTool = VCLAllocatorMultipleCompilerTest::getCidToolPath();
/// Models and configs for smoke test
const auto smokeIRInfos = VCLAllocatorMultipleCompilerTest::readJson2Vec(cidTool + VCLTest::SMOKE_TEST_CONFIG.data());
/// Models and configs for normal test
const auto irInfos = VCLAllocatorMultipleCompilerTest::readJson2Vec(cidTool + VCLTest::TEST_CONFIG.data());
/// Parameters for smoke tests
const auto smokeParams = testing::Combine(testing::ValuesIn(smokeIRInfos));
/// Parameters for normal tests
const auto params = testing::Combine(testing::ValuesIn(irInfos));

INSTANTIATE_TEST_SUITE_P(smoke_MultipleCompilerInstanceTest, VCLAllocatorMultipleCompilerTest, smokeParams,
                         VCLAllocatorMultipleCompilerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MultipleCompilerInstanceTest, VCLAllocatorMultipleCompilerTest, params,
                         VCLAllocatorMultipleCompilerTest::getTestCaseName);

}  // namespace VCLTest
