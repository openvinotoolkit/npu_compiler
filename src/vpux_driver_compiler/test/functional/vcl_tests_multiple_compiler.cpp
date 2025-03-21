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

class VCLMultipleCompilerTest : public VCLTestsUtils::VCLTestsCommon {
public:
    /**
     * @brief Use compiler to compile one model
     *
     * @param options  Build flags of a model
     */
    vcl_result_t singleCompilation(const std::string& options);

    /**
     * @brief Check if all compilations have created blob
     */
    bool check() const;

    size_t getOutputSize() const {
        return outputs.size();
    }

    /**
     * @brief Create multiple compilers and do compilation at same time
     */
    void run();

private:
    std::vector<std::string> outputs;
    std::mutex lock;
};

vcl_result_t VCLMultipleCompilerTest::singleCompilation(const std::string& options) {
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

    vcl_executable_handle_t executable = nullptr;
    vcl_executable_desc_t exeDesc = {getModelIR().data(), getModelIRSize(), options.c_str(), options.size() + 1};

    ret = vclExecutableCreate(compiler, exeDesc, &executable);
    if (ret != VCL_RESULT_SUCCESS) {
        std::cerr << "Failed to create executable handle! Result:0x" << std::hex << uint64_t(ret) << std::dec
                  << std::endl;
        vclCompilerDestroy(compiler);
        return ret;
    }
    uint64_t blobSize = 0;
    ret = vclExecutableGetSerializableBlob(executable, nullptr, &blobSize);
    if (ret != VCL_RESULT_SUCCESS || blobSize == 0) {
        std::cerr << "Failed to get blob size! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        vclExecutableDestroy(executable);
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        uint8_t* blob = (uint8_t*)malloc(blobSize);
        if (blob == nullptr) {
            std::cerr << "Failed to malloc memory to store blob!" << std::endl;
            vclExecutableDestroy(executable);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_OUT_OF_MEMORY;
        }
        ret = vclExecutableGetSerializableBlob(executable, blob, &blobSize);
        if (ret == VCL_RESULT_SUCCESS) {
#ifdef BLOB_DUMP
            std::string blobName = "ct1_" + threadName + ".net";
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
            lock.lock();
            outputs.push_back(std::move(output));
            lock.unlock();
        }
        free(blob);
    }

    ret = vclExecutableDestroy(executable);
    if (ret != VCL_RESULT_SUCCESS) {
        std::cerr << "Failed to destroy executable! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        vclCompilerDestroy(compiler);
        return ret;
    }
    executable = nullptr;

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        std::cerr << "Failed to destroy compiler! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        return ret;
    }
    return ret;
}

bool VCLMultipleCompilerTest::check() const {
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

void VCLMultipleCompilerTest::run() {
    std::vector<vcl_result_t> res(5, VCL_RESULT_SUCCESS);
    /// Get the ref output from single thread env;
    std::thread t0{[&res, this] {
        res[0] = singleCompilation(this->getNetOptions());
    }};

    t0.join();
    /// Get the outputs from multiple threads env;
    std::thread t1{[&res, this] {
        res[1] = singleCompilation(this->getNetOptions());
    }};
    std::thread t2{[&res, this] {
        res[2] = singleCompilation(this->getNetOptions());
    }};
    std::thread t3{[&res, this] {
        res[3] = singleCompilation(this->getNetOptions());
    }};
    std::thread t4{[&res, this] {
        res[4] = singleCompilation(this->getNetOptions());
    }};

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    for (auto i = res.begin(); i != res.end(); ++i) {
        if (*i != VCL_RESULT_SUCCESS) {
            std::cerr << "Failed to run " << std::distance(res.begin(), i) << " thread!" << std::endl;
            std::cerr << "Result:0x" << std::hex << uint64_t(*i) << std::dec << std::endl;
        }
    }

    EXPECT_EQ(getOutputSize(), 5) << "Not get all outputs successfully!" << std::endl;
    EXPECT_EQ(check(), true);
}

TEST_P(VCLMultipleCompilerTest, CompilerInstance) {
    run();
}

/// The path of config files for tests
const auto cidTool = VCLMultipleCompilerTest::getCidToolPath();
/// Models and configs for smoke test
const auto smokeIRInfos = VCLMultipleCompilerTest::readJson2Vec(cidTool + VCLTestsUtils::SMOKE_TEST_CONFIG);
/// Models and configs for normal test
const auto irInfos = VCLMultipleCompilerTest::readJson2Vec(cidTool + VCLTestsUtils::TEST_CONFIG);
/// Params for somke tests
const auto smokeParams = testing::Combine(testing::ValuesIn(smokeIRInfos));
/// Params for normal tests
const auto params = testing::Combine(testing::ValuesIn(irInfos));

INSTANTIATE_TEST_SUITE_P(smoke_MultipleCompilerInstanceTest, VCLMultipleCompilerTest, smokeParams,
                         VCLMultipleCompilerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MultipleCompilerInstanceTest, VCLMultipleCompilerTest, params,
                         VCLMultipleCompilerTest::getTestCaseName);
