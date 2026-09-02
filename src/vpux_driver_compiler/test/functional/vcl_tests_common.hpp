//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vcl_api.hpp"

#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <openvino/opsets/opset3.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>

#if defined(_WIN32)
#include "Shlwapi.h"
// These two undefs are to avoid min/max macro interfering introduced by Shlwapi.h.
#undef min
#undef max
#else
#include <sys/stat.h>
#endif

namespace VCLTest {

using IRInfoTestType = std::vector<std::unordered_map<std::string, std::string>>;
using VCLTestsParams = std::tuple<std::unordered_map<std::string, std::string>>;

/// The file contains the models and their configuration for smoke tests
constexpr std::string_view SMOKE_TEST_CONFIG = "/test_smoke.json";
/// The file contains the models and their configuration for normal tests
constexpr std::string_view TEST_CONFIG = "/test.json";

// The two functions are prepared to call the vclAllocatedExecutableCreate.
uint8_t* allocateBlob(uint64_t size);
void deallocateBlob(uint8_t* ptr);
// Used with vclAllocatedExecutableCreate2
uint8_t* allocateBlob2(vcl_allocator2_t*, uint64_t size);
void deallocateBlob2(vcl_allocator2_t*, uint8_t* ptr);

/**
 * @brief Base class to parse config file to get test cases and provide helper functions
 */
class VCLTestsCommon : public testing::WithParamInterface<VCLTestsParams>, public testing::Test {
public:
    VCLTestsCommon(): modelIR(), modelIRSize(0) {
        (void)VCLApi::getInstance();
    }
    virtual ~VCLTestsCommon() = default;

    /**
     * @brief Create the test name suffixes with content of test params
     */
    static std::string getTestCaseName(const testing::TestParamInfo<VCLTestsParams>& obj) {
        auto param = obj.param;
        auto netInfo = std::get<0>(param);
        const std::string netName = netInfo.at("network");
        const std::string deviceID = netInfo.at("device");
        return netName + std::string("_NPU") + deviceID;
    }

    /**
     * @brief Prepare modelIRData of one test case for compiler
     */
    vcl_result_t initModelData(const char* netName, const char* weightName);

    /**
     * @brief Create a simple model to test compiler
     */
    std::shared_ptr<ov::Model> createSimpleModel();

    /**
     * @brief Get the location of model package defined by POR_PATH
     */
    std::string getTestModelsBasePath();

    /**
     * @brief Get the location of config files for tests
     */
    static std::string getCidToolPath();

    /**
     * @brief Parse config file and detect test cases
     */
    static IRInfoTestType readJson2Vec(const std::string& fileName);

    /**
     * @brief Add platform info to model build options
     */
    void postProcessNetOptions(const std::string& device);

    /**
     * @brief Build a vcl_device_desc_t from the current test parameter's device string
     */
    vcl_device_desc_t getDeviceDesc() const;

    /**
     * @brief Return the build options of model
     */
    std::string getNetOptions() const {
        return netOptions;
    }

    /**
     * @brief Prepare modelIRData of all tests for compiler
     */
    void SetUp() override;

    std::vector<uint8_t>& getModelIR() {
        return modelIR;
    }

    const std::vector<uint8_t>& getModelIR() const {
        return modelIR;
    }

    size_t getModelIRSize() const {
        return modelIRSize;
    }

    /**
     * @brief Print error info to pass coverity scanner
     */
    void printErrorInfo(const std::string& errorStr, vcl_result_t ret) {
        std::ios::fmtflags originalFormat = std::cerr.flags();
        std::cerr << errorStr << std::hex << ret << std::endl;
        std::cerr.flags(originalFormat);
    }

    /**
     * @brief Set the number of threads for the parallel test
     */
    void setThreadCount(int n) {
        numCompilationThreads = n;
    }

    /**
     * @brief Create a compiler with standard settings
     *
     * @param compiler Output compiler handle
     * @param debugLevel VCL debug log level
     * @return VCL_RESULT_SUCCESS on success
     */
    vcl_result_t createCompiler(vcl_compiler_handle_t& compiler, vcl_log_handle_t& logHandle,
                                vcl_log_level_t debugLevel = VCL_LOG_ERROR);

    /**
     * @brief Destroy a compiler handle
     *
     * @param compiler Compiler handle to destroy
     * @return VCL_RESULT_SUCCESS on success
     */
    vcl_result_t destroyCompiler(vcl_compiler_handle_t compiler);

    /**
     * @brief Build an executable descriptor from the current model IR and options
     *
     * @param options Compilation options string
     * @return Populated vcl_executable_desc_t
     */
    vcl_executable_desc_t makeExeDesc(const std::string& options) const {
        return {getModelIR().data(), getModelIRSize(), options.c_str(), options.size() + 1};
    }

    /**
     * @brief Build a query descriptor from the current model IR and options
     *
     * @param options Compilation options string
     * @return Populated vcl_query_desc_t
     */
    vcl_query_desc_t makeQueryDesc(const std::string& options) const {
        return {getModelIR().data(), getModelIRSize(), options.c_str(), options.size() + 1};
    }

    /**
     * @brief Query compiler properties
     *
     * @param compiler Compiler handle to query
     * @param props Output compiler properties
     * @return VCL_RESULT_SUCCESS on success
     */
    vcl_result_t queryCompilerProperties(vcl_compiler_handle_t compiler, vcl_compiler_properties_t& props);

protected:
    int numCompilationThreads = 0;

private:
    /// Model build flags
    std::string netOptions;
    /// The data of modelIR to create executable
    std::vector<uint8_t> modelIR;
    size_t modelIRSize;
};

/// Shared fixture for single-thread functional tests.
class VCLSingleThreadTestsCommon : public VCLTestsCommon {
public:
};

/// Extends VCLTestsCommon with parallel compilation helpers that use vcl_allocator2_t.
/// Fixture classes for vclAllocatedExecutableCreateX tests inherit from this class.
/// Virtual inheritance on VCLTestsCommon supports diamond derivation when a fixture also
/// inherits a SetUp-providing base (e.g. VCLFunctionalTestsCommon).
class VCLAllocatedExecutableTestsBase : public virtual VCLTestsCommon {
public:
    static bool isWSOneShotSupported([[maybe_unused]] std::string_view device) {
        return true;
    }

    static vcl_result_t validateCompatibilityString(std::string_view compatibilityString) {
        constexpr auto compatibilityStringPattern =
                R"(compiler=[0-9]+\.[0-9]+;npu=[0-9]+;t=[0-9]+;elf=[0-9]+\.[0-9]+\.[0-9]+;mi=[0-9]+\.[0-9]+\.[0-9]+)";
        return std::regex_match(compatibilityString.begin(), compatibilityString.end(),
                                std::regex(compatibilityStringPattern))
                       ? VCL_RESULT_SUCCESS
                       : VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    static vcl_result_t getAndValidateCompatibilityString(vcl_executable_handle_t executable) {
        uint64_t compatibilityStringSize = 0;
        auto result = vclExecutableGetCompatibilityString(executable, nullptr, &compatibilityStringSize);
        if (result == VCL_RESULT_SUCCESS && compatibilityStringSize > 1) {
            std::vector<char> compatibilityString(compatibilityStringSize);
            result = vclExecutableGetCompatibilityString(executable, compatibilityString.data(),
                                                         &compatibilityStringSize);
            if (result == VCL_RESULT_SUCCESS) {
                const std::string_view sv(compatibilityString.data(), compatibilityStringSize - 1);
                if (compatibilityString[compatibilityStringSize - 1] != '\0' ||
                    validateCompatibilityString(sv) != VCL_RESULT_SUCCESS) {
                    result = VCL_RESULT_ERROR_INVALID_ARGUMENT;
                }
            }
        } else {
            result = result != VCL_RESULT_SUCCESS ? result : VCL_RESULT_ERROR_UNKNOWN;
        }

        return result;
    }

    /// Sets up a compiler and a vcl_allocator2_t, then runs @p perThreadWork in @p numCompilationThreads
    /// parallel threads with signature (compiler, exeDesc, allocator, threadIndex) -> vcl_result_t.
    /// Returns the first non-success thread result, or the result of destroyCompiler on success.
    template <typename ThreadFunc>
    vcl_result_t runParallelWithAllocator2(const std::string& options, ThreadFunc&& perThreadWork) {
        if (numCompilationThreads <= 0) {
            ADD_FAILURE() << "numCompilationThreads must be > 0. Call setThreadCount(n) before invoking this helper.";
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        }
        vcl_compiler_handle_t compiler = nullptr;
        vcl_log_handle_t logHandle = nullptr;
        if (const auto ret = createCompiler(compiler, logHandle, VCL_LOG_INFO); ret != VCL_RESULT_SUCCESS) {
            return ret;
        }

        vcl_allocator2_t allocator;
        allocator.allocate = allocateBlob2;
        allocator.deallocate = deallocateBlob2;

        // exeDesc holds a raw pointer into modelIR. The vector must not be resized or
        // modified while threads are running; SetUp() populates it once before any test
        // method executes, so the pointer remains valid for the lifetime of this call.
        const auto exeDesc = makeExeDesc(options);

        // compiler is shared across all threads. vclAllocatedExecutableCreate2/4 is
        // assumed to be thread-safe for concurrent calls on the same compiler handle;
        // this test intentionally exercises that guarantee.
        std::vector<std::thread> threads;
        threads.reserve(numCompilationThreads);
        std::vector<vcl_result_t> results(numCompilationThreads, VCL_RESULT_SUCCESS);

        for (int i = 0; i < numCompilationThreads; i++) {
            threads.emplace_back([&results, &perThreadWork, &compiler, &exeDesc, &allocator, i] {
                try {
                    results[i] = perThreadWork(compiler, exeDesc, allocator, i);
                } catch (const std::exception& e) {
                    std::cerr << "Thread " << i << " threw: " << e.what() << std::endl;
                    results[i] = VCL_RESULT_ERROR_UNKNOWN;
                } catch (...) {
                    std::cerr << "Thread " << i << " threw an unknown exception" << std::endl;
                    results[i] = VCL_RESULT_ERROR_UNKNOWN;
                }
            });
        }
        for (auto& t : threads) {
            t.join();
        }

        vcl_result_t firstFailure = VCL_RESULT_SUCCESS;
        for (size_t i = 0; i < results.size(); i++) {
            if (results[i] != VCL_RESULT_SUCCESS) {
                printErrorInfo("Thread " + std::to_string(i) + " failed! Result:0x", results[i]);
                if (firstFailure == VCL_RESULT_SUCCESS) {
                    firstFailure = results[i];
                }
            }
        }
        const auto destroyRet = destroyCompiler(compiler);
        if (firstFailure != VCL_RESULT_SUCCESS) {
            return firstFailure;
        }
        return destroyRet;
    }
};

}  // namespace VCLTest
