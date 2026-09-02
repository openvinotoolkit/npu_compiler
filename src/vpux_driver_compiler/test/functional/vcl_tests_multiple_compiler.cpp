//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_tests_common.hpp"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

namespace VCLTest {
class VCLMultipleCompilerTest : public VCLTestsCommon {
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

private:
    std::vector<std::string> outputs;
    std::mutex lock;
};

vcl_result_t VCLMultipleCompilerTest::singleCompilation(const std::string& options) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    vcl_compiler_handle_t compiler = nullptr;
    vcl_log_handle_t logHandle = nullptr;
    ret = createCompiler(compiler, logHandle, VCL_LOG_INFO);
    if (ret != VCL_RESULT_SUCCESS) {
        return ret;
    }

    vcl_executable_handle_t executable = nullptr;
    auto exeDesc = makeExeDesc(options);

    ret = vclExecutableCreate(compiler, exeDesc, &executable);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to create executable handle! Result:0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }
    uint64_t blobSize = 0;
    ret = vclExecutableGetSerializableBlob(executable, nullptr, &blobSize);
    if (ret != VCL_RESULT_SUCCESS || blobSize == 0) {
        printErrorInfo("Failed to get blob size! Result:0x", ret);
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
            std::string output(reinterpret_cast<char*>(blob), blobSize);
            lock.lock();
            outputs.push_back(std::move(output));
            lock.unlock();
        }
        free(blob);
    }

    ret = vclExecutableDestroy(executable);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy executable! Result:0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }
    executable = nullptr;

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy compiler! Result:0x", ret);
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

TEST_P(VCLMultipleCompilerTest, CompilerInstance) {
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
            std::string printStr =
                    "Failed to run " + std::to_string(std::distance(res.begin(), i)) + " thread!\n" + "Result:0x";
            printErrorInfo(printStr, *i);
        }
    }

    EXPECT_EQ(getOutputSize(), 5) << "Not get all outputs successfully!" << std::endl;
    EXPECT_EQ(check(), true);
}

/// The path of config files for tests
const auto cidTool = VCLMultipleCompilerTest::getCidToolPath();
/// Models and configs for smoke test
const auto smokeIRInfos = VCLMultipleCompilerTest::readJson2Vec(cidTool + VCLTest::SMOKE_TEST_CONFIG.data());
/// Models and configs for normal test
const auto irInfos = VCLMultipleCompilerTest::readJson2Vec(cidTool + VCLTest::TEST_CONFIG.data());
/// Parameters for smoke tests
const auto smokeParams = testing::Combine(testing::ValuesIn(smokeIRInfos));
/// Parameters for normal tests
const auto params = testing::Combine(testing::ValuesIn(irInfos));

INSTANTIATE_TEST_SUITE_P(smoke_MultipleCompilerInstanceTest, VCLMultipleCompilerTest, smokeParams,
                         VCLMultipleCompilerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MultipleCompilerInstanceTest, VCLMultipleCompilerTest, params,
                         VCLMultipleCompilerTest::getTestCaseName);

}  // namespace VCLTest
