//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_tests_common.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

namespace VCLTest {
class VCLParallelCompilationTest : public VCLTestsCommon {
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
     * @brief Test parallel execution of a compiler and check the results
     */
    void run();

private:
    int numCompilationThreads;
    int numGetBlobThreads;
    std::vector<std::string> outputs;
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
        printErrorInfo("Failed to create compiler! Result:0x", ret);
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret) {
        printErrorInfo("Failed to query compiler props! Result:0x", ret);
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
        compilationThreads.push_back(std::move(thread));
    }

    /// Wait for all threads to finish
    for (auto& compilationThread : compilationThreads) {
        compilationThread.join();
    }

    /// Check all execution results
    for (auto i = resCreate.begin(); i != resCreate.end(); ++i) {
        if (*i != VCL_RESULT_SUCCESS) {
            std::string printStr = "Failed to vclExecutableCreate with " +
                                   std::to_string(std::distance(resCreate.begin(), i)) + " thread!\n" + "Result:0x";
            printErrorInfo(printStr, *i);
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
        getBlobSizeThreads.push_back(std::move(thread));
        idx++;
    }
    for (auto& getBlobSizeThread : getBlobSizeThreads) {
        getBlobSizeThread.join();
    }

    for (size_t i = 0; i < resGetBlobInit.size(); i++) {
        const auto result = resGetBlobInit[i];
        if (result != VCL_RESULT_SUCCESS) {
            std::cerr << "Failed to vclExecutableGetSerializableBlob initially with " << i << " thread!" << std::endl;
            printErrorInfo("Result:0x", result);
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

        getBlobThreads.push_back(std::move(thread));
    }
    for (auto& getBlobThread : getBlobThreads) {
        getBlobThread.join();
    }

    for (size_t i = 0; i < resGetBlob.size(); i++) {
        const auto result = resGetBlob[i];
        if (result != VCL_RESULT_SUCCESS) {
            std::string printStr =
                    "Failed to vclExecutableGetSerializableBlob with " + std::to_string(i) + " thread!\n" + "Result:0x";
            printErrorInfo(printStr, result);
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
            printErrorInfo("Failed to destroy executable! Result:0x", ret);
            vclCompilerDestroy(compiler);
            return ret;
        }
    }
    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy compiler! Result:0x", ret);
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
const auto smokeIRInfos = VCLParallelCompilationTest::readJson2Vec(cidTool + VCLTest::SMOKE_TEST_CONFIG);
/// Models and configs for normal test
const auto irInfos = VCLParallelCompilationTest::readJson2Vec(cidTool + VCLTest::TEST_CONFIG);
/// Parameters for smoke tests
const auto smokeParams = testing::Combine(testing::ValuesIn(smokeIRInfos));
/// Parameters for normal tests
const auto params = testing::Combine(testing::ValuesIn(irInfos));

INSTANTIATE_TEST_SUITE_P(smoke_ParallelCompilationTest, VCLParallelCompilationTest, smokeParams,
                         VCLParallelCompilationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ParallelCompilationTest, VCLParallelCompilationTest, params,
                         VCLParallelCompilationTest::getTestCaseName);

class VCLAllocatorParallelCompilationTest : public VCLTest::VCLTestsCommon {
public:
    VCLAllocatorParallelCompilationTest(): numCompilationThreads(0) {
        outputs.clear();
    }
    /**
     * @brief Set the number of threads during tests
     *
     * @param compilationThreads The number of threads to do compilation
     */
    void setThreadCount(int compilationThreads) {
        numCompilationThreads = compilationThreads;
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
     * @brief Test parallel execution of one compiler and check results
     */
    void run();
    struct CreateThreadParams {
        vcl_compiler_handle_t compiler;
        vcl_executable_desc_t& exeDesc;
        vcl_allocator_t& allocator;
        std::vector<vcl_result_t>& resCreate;
        std::vector<std::pair<uint8_t*, uint64_t>>& blobs;
        std::mutex& lock;
    };

    class CreateThread {
    public:
        CreateThread(const CreateThreadParams& params)
                : m_compiler(params.compiler),
                  m_exeDesc(params.exeDesc),
                  m_allocator(params.allocator),
                  m_resCreate(params.resCreate),
                  m_blobs(params.blobs),
                  m_lock(params.lock) {
        }

        void createThreadVector(int i) {
            uint8_t* blob;
            uint64_t size = 0;
            m_resCreate[i] = vclAllocatedExecutableCreate(m_compiler, m_exeDesc, &m_allocator, &blob, &size);
            EXPECT_TRUE(blob != nullptr) << "blob is empty!" << std::endl;
            {
                std::lock_guard<std::mutex> guard(m_lock);
                m_blobs.push_back(std::make_pair(blob, size));
            }
        }

    private:
        vcl_compiler_handle_t m_compiler;
        vcl_executable_desc_t& m_exeDesc;
        vcl_allocator_t& m_allocator;
        std::vector<vcl_result_t>& m_resCreate;
        std::vector<std::pair<uint8_t*, uint64_t>>& m_blobs;
        std::mutex& m_lock;
    };

private:
    int numCompilationThreads;
    std::vector<std::string> outputs;
};

vcl_result_t VCLAllocatorParallelCompilationTest::parallelCompilation(const std::string& options) {
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
    compilerDesc.debugLevel = VCL_LOG_INFO;
    vcl_device_desc_t deviceDesc = {sizeof(vcl_device_desc_t), 0x643e, 3, 5};
    vcl_compiler_handle_t compiler = nullptr;
    ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, nullptr);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to create compiler! Result: 0x", ret);
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to query compiler props! Result: 0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }
    std::cout << "############################################" << std::endl;
    std::cout << threadName.c_str() << " Current compiler info:" << std::endl;
    std::cout << threadName.c_str() << " ID: " << compilerProp.id << std::endl;
    std::cout << threadName.c_str() << " Version:" << compilerProp.version.major << "." << compilerProp.version.minor
              << std::endl;
    std::cout << threadName.c_str() << "\tSupported opsets:" << compilerProp.supportedOpsets << std::endl;
    std::cout << "############################################" << std::endl;

    vcl_allocator_t allocator;
    allocator.allocate = VCLTest::allocateBlob;
    allocator.deallocate = VCLTest::deallocateBlob;

    vcl_executable_desc_t exeDesc = {getModelIR().data(), getModelIRSize(), options.c_str(), options.size() + 1};

    /// Create multiple thread to do compilation
    std::vector<std::thread> compilationThreads;
    compilationThreads.reserve(numCompilationThreads);
    /// The execution result of compilation of each thread
    std::vector<vcl_result_t> resCreate(numCompilationThreads, VCL_RESULT_SUCCESS);
    std::vector<std::pair<uint8_t*, uint64_t>> blobs;
    std::mutex lock;

    /// Create multiple threads to do compilation with one compiler
    CreateThreadParams params = {compiler, exeDesc, allocator, resCreate, blobs, lock};
    CreateThread createThread(params);
    for (int i = 0; i < numCompilationThreads; i++) {
        compilationThreads.emplace_back([&createThread, i] {
            createThread.createThreadVector(i);
        });
    }

    /// Wait for all threads to finish
    for (auto& compilationThread : compilationThreads) {
        compilationThread.join();
    }

    /// Check all execution results
    for (auto i = resCreate.begin(); i != resCreate.end(); ++i) {
        if (*i != VCL_RESULT_SUCCESS) {
            std::string printStr = "Failed to vclExecutableCreate with " +
                                   std::to_string(std::distance(resCreate.begin(), i)) + " thread!\n" + "Result:0x";
            printErrorInfo(printStr, *i);
            vclCompilerDestroy(compiler);
            return *i;
        }
    }

    /// Save all result blobs
    for (auto pair : blobs) {
        auto blob = pair.first;
        auto blobSize = pair.second;
#ifdef BLOB_DUMP
        auto ir = GetParam();
        auto netInfo = std::get<0>(ir);
        const std::string networkName = netInfo.at("network");
        std::string blobName = "ct2_" + std::to_string(count) + "_" + threadName + networkName + ".net.allocator";
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
        allocator.deallocate(blob);
        count++;
    }

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy compiler! Result: 0x", ret);
        return ret;
    }
    return ret;
}

bool VCLAllocatorParallelCompilationTest::check() const {
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

void VCLAllocatorParallelCompilationTest::run() {
    setThreadCount(1);
    vcl_result_t ret = parallelCompilation(getNetOptions());
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run test to create ref! Result: 0x" << std::hex << ret
                                       << std::endl;

    // Get the outputs from multiple threads env;
    int numCompilationThreads = 5;
    setThreadCount(numCompilationThreads);
    ret = parallelCompilation(getNetOptions());
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run thread test! Result: 0x" << std::hex << ret << std::endl;
    EXPECT_EQ(getOutputSize(), 6) << "Not get all outputs successfully!" << std::endl;
    EXPECT_EQ(check(), true);
}

TEST_P(VCLAllocatorParallelCompilationTest, ParallelCompilation) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_ParallelCompilationTest, VCLAllocatorParallelCompilationTest, smokeParams,
                         VCLAllocatorParallelCompilationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ParallelCompilationTest, VCLAllocatorParallelCompilationTest, params,
                         VCLAllocatorParallelCompilationTest::getTestCaseName);

class VCLQueryNetworkParallelTest : public VCLTestsCommon {
public:
    VCLQueryNetworkParallelTest(): numQueryThreads(0) {
    }

    void setThreadCount(int queryThreads) {
        numQueryThreads = queryThreads;
    }

    vcl_result_t parallelQueryNetwork(const std::string& options);
    void run();

private:
    int numQueryThreads;
};

vcl_result_t VCLQueryNetworkParallelTest::parallelQueryNetwork(const std::string& options) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = VCL_LOG_ERROR;
    vcl_device_desc_t deviceDesc = {sizeof(vcl_device_desc_t), 0x643e, 3, 5};

    vcl_compiler_handle_t compiler = nullptr;
    ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, nullptr);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to create compiler! Result:0x", ret);
        return ret;
    }

    vcl_query_desc_t desc = {getModelIR().data(), getModelIRSize(), options.c_str(), options.size() + 1};
    std::vector<std::thread> queryThreads;
    std::vector<vcl_result_t> queryResults(numQueryThreads, VCL_RESULT_SUCCESS);

    for (int i = 0; i < numQueryThreads; ++i) {
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
            (void)vclCompilerDestroy(compiler);
            return queryResults[i];
        }
    }

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy compiler! Result:0x", ret);
        return ret;
    }

    return VCL_RESULT_SUCCESS;
}

void VCLQueryNetworkParallelTest::run() {
    setThreadCount(8);
    const auto ret = parallelQueryNetwork(getNetOptions());
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run parallel query-network test! Result:0x" << std::hex
                                       << uint64_t(ret) << std::dec << std::endl;
}

TEST_P(VCLQueryNetworkParallelTest, ParallelQueryNetwork) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_ParallelQueryNetworkTest, VCLQueryNetworkParallelTest, smokeParams,
                         VCLQueryNetworkParallelTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ParallelQueryNetworkTest, VCLQueryNetworkParallelTest, params,
                         VCLQueryNetworkParallelTest::getTestCaseName);
}  // namespace VCLTest
