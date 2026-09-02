//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common.hpp"

namespace VCLTest {
class VCLParallelCompilationBehaviorTest : public VCLFunctionalTestsCommon {
public:
    VCLParallelCompilationBehaviorTest(): numGetBlobThreads(0) {
        outputs.clear();
    }

    /**
     * @brief Set the number of threads during tests
     *
     * @param compilationThreads The number of threads to do compilation
     * @param getBlobThreads The number of threads to test executable
     */
    void setThreadCount(int compilationThreads, int getBlobThreads) {
        numCompilationThreads = compilationThreads;
        numGetBlobThreads = getBlobThreads;
    }

    size_t getOutputSize() const {
        return outputs.size();
    }

    bool check() const {
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

    /**
     * @brief Multiple threads to call API of one compiler
     *
     * @param options Build flags of a model
     */
    vcl_result_t parallelCompilation(const std::string& options);

protected:
    std::vector<std::string> outputs;

private:
    int numGetBlobThreads;
};

vcl_result_t VCLParallelCompilationBehaviorTest::parallelCompilation(const std::string& options) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    vcl_compiler_handle_t compiler = nullptr;
    vcl_log_handle_t logHandle = nullptr;
    ret = createCompiler(compiler, logHandle, VCL_LOG_INFO);

    if (ret != VCL_RESULT_SUCCESS) {
        return ret;
    }

    auto exeDesc = makeExeDesc(options);

    /// Create multiple thread to do compilation
    std::vector<std::thread> compilationThreads;
    /// The compilation results of each thread
    std::vector<std::pair<vcl_executable_handle_t, uint64_t>> exeHandles(numCompilationThreads, {nullptr, 0});
    /// The execution result of compilation of each thread
    std::vector<vcl_result_t> resCreate(numCompilationThreads, VCL_RESULT_SUCCESS);
    /// Blob data and sizes collected
    std::vector<std::pair<uint8_t*, uint64_t>> blobs;

    /// Pass freeBlobs=true once the blobs vector may be non-empty.
    auto cleanup = [&](bool freeBlobs) {
        if (freeBlobs) {
            for (auto& p : blobs) {
                free(p.first);
            }
        }
        for (auto& pair : exeHandles) {
            if (pair.first != nullptr) {
                (void)vclExecutableDestroy(pair.first);
                pair.first = nullptr;
            }
        }
        (void)destroyCompiler(compiler);
    };

    /// Create multiple threads to do compilation with one compiler
    for (int i = 0; i < numCompilationThreads; i++) {
        vcl_executable_handle_t* exeHandle = &exeHandles[i].first;
        std::thread thread{[&resCreate, &compiler, &exeDesc, exeHandle, i] {
            resCreate[i] = vclExecutableCreate(compiler, exeDesc, exeHandle);
        }};
        compilationThreads.push_back(std::move(thread));
    }

    /// Wait for all threads to finish
    for (auto& compilationThread : compilationThreads) {
        compilationThread.join();
    }

    /// Check all execution results
    for (size_t t = 0; t < resCreate.size(); ++t) {
        if (resCreate[t] != VCL_RESULT_SUCCESS) {
            std::string printStr = "Failed to vclExecutableCreate with " + std::to_string(t) +
                                   " thread!\n"
                                   "Result:0x";
            printErrorInfo(printStr, resCreate[t]);
            cleanup(false);
            return resCreate[t];
        }
    }

    /// Multiple threads to get blob size from VCLExecutable
    std::vector<std::thread> getBlobSizeThreads;
    std::vector<vcl_result_t> resGetBlobInit(numCompilationThreads, VCL_RESULT_SUCCESS);
    unsigned idx = 0;
    for (auto& pair : exeHandles) {
        vcl_executable_handle_t exeHandle = pair.first;
        uint64_t& blobSize = pair.second;
        uint8_t* blob = nullptr;
        std::thread thread{[&blobSize, idx, &resGetBlobInit, exeHandle, blob] {
            resGetBlobInit[idx] = vclExecutableGetSerializableBlob(exeHandle, blob, &blobSize);
        }};
        getBlobSizeThreads.push_back(std::move(thread));
        idx++;
    }
    for (auto& getBlobSizeThread : getBlobSizeThreads) {
        getBlobSizeThread.join();
    }

    for (size_t i = 0; i < resGetBlobInit.size(); i++) {
        const auto result = resGetBlobInit[i];
        if (result != VCL_RESULT_SUCCESS || exeHandles[i].second == 0) {
            std::cerr << "Failed to vclExecutableGetSerializableBlob initially with " << i << " thread!" << std::endl;
            printErrorInfo("Result:0x", result != VCL_RESULT_SUCCESS ? result : VCL_RESULT_ERROR_UNKNOWN);
            cleanup(false);
            return result != VCL_RESULT_SUCCESS ? result : VCL_RESULT_ERROR_UNKNOWN;
        }
    }

    /// Multiple threads to get blob data from vclExecutableGetSerializableBlob
    std::vector<std::thread> getBlobThreads;
    std::vector<vcl_result_t> resGetBlob(numGetBlobThreads, VCL_RESULT_SUCCESS);
    std::vector<uint64_t> returnedBlobSizes(numGetBlobThreads, 0);
    bool mallocFailed = false;

    for (int i = 0; i < numGetBlobThreads; i++) {
        const auto getBlobIdx = static_cast<size_t>(i % numCompilationThreads);

        vcl_executable_handle_t exe = exeHandles[getBlobIdx].first;
        uint64_t blobSize = exeHandles[getBlobIdx].second;
        uint8_t* blob = (uint8_t*)malloc(blobSize);
        if (blob == nullptr) {
            std::cerr << "Failed to malloc memory to store blob!" << std::endl;
            mallocFailed = true;
            break;
        }
        std::thread thread{[&resGetBlob, &returnedBlobSizes, exe, i, blob, blobSize] {
            uint64_t size = blobSize;
            resGetBlob[i] = vclExecutableGetSerializableBlob(exe, blob, &size);
            returnedBlobSizes[i] = size;
        }};
        blobs.push_back(std::make_pair(blob, blobSize));

        getBlobThreads.push_back(std::move(thread));
    }
    for (auto& getBlobThread : getBlobThreads) {
        getBlobThread.join();
    }

    if (mallocFailed) {
        cleanup(true);
        return VCL_RESULT_ERROR_OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < blobs.size(); i++) {
        const auto result = resGetBlob[i];
        if (result != VCL_RESULT_SUCCESS) {
            std::string printStr =
                    "Failed to vclExecutableGetSerializableBlob with " + std::to_string(i) + " thread!\n" + "Result:0x";
            printErrorInfo(printStr, result);
            cleanup(true);
            return result;
        }
    }

    /// Save all result blobs
    for (size_t i = 0; i < blobs.size(); ++i) {
        auto blob = blobs[i].first;
        const auto blobSize = returnedBlobSizes[i];
        std::string output(reinterpret_cast<char*>(blob), blobSize);
        outputs.push_back(std::move(output));
        free(blob);
    }
    vcl_result_t destroyExeRes = VCL_RESULT_SUCCESS;
    for (auto& pair : exeHandles) {
        const auto cur = vclExecutableDestroy(pair.first);
        pair.first = nullptr;
        if (destroyExeRes == VCL_RESULT_SUCCESS && cur != VCL_RESULT_SUCCESS) {
            destroyExeRes = cur;
        }
    }
    const auto destroyCompilerRes = destroyCompiler(compiler);
    if (destroyExeRes != VCL_RESULT_SUCCESS) {
        return destroyExeRes;
    }
    return destroyCompilerRes;
}

TEST_P(VCLParallelCompilationBehaviorTest, ParallelCompilation) {
    setThreadCount(1, 1);
    vcl_result_t ret = parallelCompilation(getNetOptions());
    ASSERT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run test to create ref! Result:0x" << std::hex << uint64_t(ret)
                                       << std::dec << std::endl;

    // Get the outputs from multiple threads env;
    int compilationThreads = 5;
    int getBlobThreads = 17;
    setThreadCount(compilationThreads, getBlobThreads);
    ret = parallelCompilation(getNetOptions());
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run thread test! Result:0x" << std::hex << uint64_t(ret)
                                       << std::dec << std::endl;
    EXPECT_EQ(getOutputSize(), getBlobThreads + 1) << "Not get all outputs successfully!" << std::endl;
    EXPECT_EQ(check(), true);
}

INSTANTIATE_TEST_SUITE_P(ParallelCompilationTest, VCLParallelCompilationBehaviorTest, getVCLFunctionalTestParams(),
                         VCLParallelCompilationBehaviorTest::getTestCaseName);

}  // namespace VCLTest
