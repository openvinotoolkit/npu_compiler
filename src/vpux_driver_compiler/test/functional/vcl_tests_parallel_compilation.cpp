//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vcl_tests_common.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

class VCLParallelCompilationTest : public VCLTestsUtils::VCLTestsCommon {
public:
    VCLParallelCompilationTest(): numCompilationThreads(0), numGetBlobThreads(0) {
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

    /**
     * @brief Multiple threads to call API of one compiler
     *
     * @param options Build flags of a model
     */
    vcl_result_t parallelCompilation(const std::string& options);

    /**
     * @brief Check if all compilations have created blob
     */
    bool check() const;

    /**
     * @brief Test parallel excution of one compiler and check results
     */
    void run();

private:
    int numCompilationThreads;
    int numGetBlobThreads;
    std::vector<std::string> outputs;
    std::mutex lock;
};

vcl_result_t VCLParallelCompilationTest::parallelCompilation(const std::string& options) {
    static int count = 0;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto id = std::this_thread::get_id();
    std::stringstream ss;
    ss << id;
    std::string threadName = ss.str();
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    /// Default device is 4000, can be updated by test config
    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = VCL_LOG_ERROR;
    vcl_device_desc_t deviceDesc = {sizeof(vcl_device_desc_t), 0x643e, 3, 5};
    vcl_compiler_handle_t compiler = nullptr;
    ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, nullptr);
    if (ret) {
        std::cerr << "Failed to create compiler! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret) {
        std::cerr << "Failed to query compiler props! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        std::cout << "############################################" << std::endl;
        std::cout << threadName.c_str() << " Current compiler info:" << std::endl;
        std::cout << threadName.c_str() << " ID: " << compilerProp.id << std::endl;
        std::cout << threadName.c_str() << " Version:" << compilerProp.version.major << "."
                  << compilerProp.version.minor << std::endl;
        std::cout << threadName.c_str() << "\tSupported opsets:" << compilerProp.supportedOpsets << std::endl;
        std::cout << "############################################" << std::endl;
    }
    vcl_executable_desc_t exeDesc = {getModelIR().data(), getModelIRSize(), options.c_str(), options.size() + 1};

    /// Create multiple thread to do compilation
    std::vector<std::thread> compilationThreads;
    /// The compilation results of each thread
    std::vector<std::pair<vcl_executable_handle_t*, uint64_t>> exeHandles;
    /// The execution result of compilation of each thread
    std::vector<vcl_result_t> resCreate(numCompilationThreads, VCL_RESULT_SUCCESS);

    /// Create multiple threads to do compilation with one compiler
    for (int i = 0; i < numCompilationThreads; i++) {
        vcl_executable_handle_t* exeHandle = new vcl_executable_handle_t();
        uint64_t blobSize = 0;
        std::thread thread{[&resCreate, &compiler, &exeDesc, exeHandle, i] {
            resCreate[i] = vclExecutableCreate(compiler, exeDesc, exeHandle);
        }};
        exeHandles.push_back(std::make_pair(exeHandle, blobSize));

        compilationThreads.push_back(move(thread));
    }

    /// Wait for all threads to finish
    for (auto& compilationThread : compilationThreads) {
        compilationThread.join();
    }

    /// Check all execution results
    for (auto i = resCreate.begin(); i != resCreate.end(); ++i) {
        if (*i != VCL_RESULT_SUCCESS) {
            std::cerr << "Failed to vclExecutableCreate with " << std::distance(resCreate.begin(), i) << " thread!"
                      << std::endl;
            std::cerr << "Result:0x" << std::hex << uint64_t(*i) << std::dec << std::endl;
            vclCompilerDestroy(compiler);
            return *i;
        }
    }

    /// Multiple threads to get blob size from VCLExecutable
    std::vector<std::thread> getBlobSizeThreads;
    std::vector<vcl_result_t> resGetBlobInit(numCompilationThreads, VCL_RESULT_SUCCESS);
    unsigned idx = 0;
    for (auto& pair : exeHandles) {
        vcl_executable_handle_t exeHandle = *(pair.first);
        uint64_t& blobSize = pair.second;
        uint8_t* blob = nullptr;
        std::thread thread{[&blobSize, idx, &resGetBlobInit, exeHandle, blob] {
            resGetBlobInit[idx] = vclExecutableGetSerializableBlob(exeHandle, blob, &blobSize);
        }};
        getBlobSizeThreads.push_back(move(thread));
        idx++;
    }
    for (auto& getBlobSizeThread : getBlobSizeThreads) {
        getBlobSizeThread.join();
    }

    for (size_t i = 0; i < resGetBlobInit.size(); i++) {
        const auto result = resGetBlobInit[i];
        if (result != VCL_RESULT_SUCCESS) {
            std::cerr << "Failed to vclExecutableGetSerializableBlob initially with " << i << " thread!" << std::endl;
            std::cerr << "Result:0x" << std::hex << result << std::dec << std::endl;
            vclCompilerDestroy(compiler);
            return result;
        }
    }

    /// Multiple threads to get blob data from VCLExetuable
    std::vector<std::thread> getBlobThreads;
    std::vector<std::pair<uint8_t*, uint64_t>> blobs;
    std::vector<vcl_result_t> resGetBlob(numGetBlobThreads, VCL_RESULT_SUCCESS);

    for (int i = 0; i < numGetBlobThreads; i++) {
        int idx = i % numCompilationThreads;

        vcl_executable_handle_t exe = *(exeHandles[idx].first);
        uint64_t blobSize = exeHandles[idx].second;
        uint8_t* blob = (uint8_t*)malloc(blobSize);
        if (blob == nullptr) {
            std::cerr << "Failed to malloc memory to store blob!" << std::endl;
            break;
        }
        std::thread thread{[&resGetBlob, &exeHandles, exe, i, blob, idx] {
            resGetBlob[i] = vclExecutableGetSerializableBlob(exe, blob, &(exeHandles[idx].second));
        }};
        blobs.push_back(std::make_pair(blob, exeHandles[idx].second));

        getBlobThreads.push_back(move(thread));
    }
    for (auto& getBlobThread : getBlobThreads) {
        getBlobThread.join();
    }

    for (size_t i = 0; i < resGetBlob.size(); i++) {
        const auto result = resGetBlob[i];
        if (result != VCL_RESULT_SUCCESS) {
            std::cerr << "Failed to vclExecutableGetSerializableBlob with " << i << " thread!" << std::endl;
            std::cerr << "Result:0x" << std::hex << result << std::dec << std::endl;
            vclCompilerDestroy(compiler);
            return result;
        }
    }

    /// Save all result blobs
    for (auto pair : blobs) {
        auto blob = pair.first;
        auto blobSize = pair.second;
#ifdef BLOB_DUMP
        std::string blobName = "ct2_" + std::to_string(count) + "_" + threadName + ".net";
        std::ofstream bfos(blobName, std::ios::binary);
        if (!bfos.is_open()) {
            std::cerr << "Failed to open " << blobName << ", skip dump!" << std::endl;
        } else {
            bfos.write(reinterpret_cast<char*>(blob), blobSize);
            if (bfos.fail()) {
                std::cerr << "Short write to " << blobName << ", the file is invalid!" << std::endl;
            }
        }
        bfos.close();
#endif  // BLOB_DUMP
        std::string output(reinterpret_cast<char*>(blob), blobSize);
        outputs.push_back(std::move(output));
        free(blob);
        count++;
    }
    for (auto& pair : exeHandles) {
        ret = vclExecutableDestroy(*(pair.first));
        if (ret != VCL_RESULT_SUCCESS) {
            std::cerr << "Failed to destroy executable! Result:0x" << std::hex << uint64_t(ret) << std::dec
                      << std::endl;
            vclCompilerDestroy(compiler);
            return ret;
        }
    }
    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        std::cerr << "Failed to destroy compiler! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        return ret;
    }
    return ret;
}

bool VCLParallelCompilationTest::check() const {
    const size_t count = outputs.size();
    if (count == 0) {
        std::cerr << "No outputs!" << std::endl;
        return false;
    }
    for (size_t i = 1; i < count; i++) {
        if (outputs[i].size() == 0) {
            std::cerr << "blob " << i << "'s size is zero." << std::endl;
            return false;
        }
    }
    return true;
}

void VCLParallelCompilationTest::run() {
    setThreadCount(1, 1);
    vcl_result_t ret = parallelCompilation(getNetOptions());
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run test to create ref! Result:0x" << std::hex << uint64_t(ret)
                                       << std::dec << std::endl;

    // Get the outputs from multiple threads env;
    int numCompilationThreads = 5;
    int numGetBlobThreads = 17;
    setThreadCount(numCompilationThreads, numGetBlobThreads);
    ret = parallelCompilation(getNetOptions());
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run thread test! Result:0x" << std::hex << uint64_t(ret)
                                       << std::dec << std::endl;
    EXPECT_EQ(getOutputSize(), 18) << "Not get all outputs successfully!" << std::endl;
    EXPECT_EQ(check(), true);
}

TEST_P(VCLParallelCompilationTest, ParallelCompilation) {
    run();
}

/// The path of config files for tests
const auto cidTool = VCLParallelCompilationTest::getCidToolPath();
/// Models and configs for smoke test
const auto smokeIRInfos = VCLParallelCompilationTest::readJson2Vec(cidTool + VCLTestsUtils::SMOKE_TEST_CONFIG);
/// Models and configs for normal test
const auto irInfos = VCLParallelCompilationTest::readJson2Vec(cidTool + VCLTestsUtils::TEST_CONFIG);
/// Params for somke tests
const auto smokeParams = testing::Combine(testing::ValuesIn(smokeIRInfos));
/// Params for normal tests
const auto params = testing::Combine(testing::ValuesIn(irInfos));

INSTANTIATE_TEST_SUITE_P(smoke_ParallelCompilationTest, VCLParallelCompilationTest, smokeParams,
                         VCLParallelCompilationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ParallelCompilationTest, VCLParallelCompilationTest, params,
                         VCLParallelCompilationTest::getTestCaseName);
