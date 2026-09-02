//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common.hpp"

#include <string_view>

namespace VCLTest {

using VCLApiBehaviorTest = VCLFunctionalTestsCommon;

// RAII guard to ensure compiler handle is destroyed exactly once.
// Stores a pointer to the test instance to call destroyCompiler().
struct CompilerGuard {
    vcl_compiler_handle_t* compiler;
    VCLTestsCommon* test_instance;

    CompilerGuard(vcl_compiler_handle_t* c, VCLTestsCommon* t): compiler(c), test_instance(t) {
    }

    ~CompilerGuard() {
        if (compiler && *compiler != nullptr) {
            (void)test_instance->destroyCompiler(*compiler);
            *compiler = nullptr;
        }
    }
};

TEST_P(VCLApiBehaviorTest, VclGetVersionReturnsRuntimeVersion) {
    vcl_version_info_t compilerVersion{};
    vcl_version_info_t profilingVersion{};

    const auto ret = vclGetVersion(&compilerVersion, &profilingVersion);
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "vclGetVersion failed! Result:0x" << std::hex << uint64_t(ret) << std::dec
                                       << std::endl;
    EXPECT_NE(compilerVersion.major, 0);
    EXPECT_NE(profilingVersion.major, 0);
}

TEST_P(VCLApiBehaviorTest, CompilerSupportedOptionsApi) {
    vcl_compiler_handle_t compiler = nullptr;
    vcl_log_handle_t logHandle = nullptr;
    auto ret = createCompiler(compiler, logHandle, VCL_LOG_INFO);
    ASSERT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to create compiler! Result:0x" << std::hex << uint64_t(ret)
                                       << std::dec << std::endl;

    // RAII guard ensures compiler is destroyed exactly once, regardless of assertion outcomes or early returns.
    CompilerGuard compiler_guard{&compiler, this};

    uint64_t optionsSize = 0;

    ret = vclGetCompilerSupportedOptions(compiler, nullptr, &optionsSize);
    ASSERT_EQ(ret, VCL_RESULT_SUCCESS) << "vclGetCompilerSupportedOptions(size) failed! Result:0x" << std::hex
                                       << uint64_t(ret) << std::dec << std::endl;
    ASSERT_NE(optionsSize, 0U);

    std::vector<char> options(optionsSize, '\0');
    ret = vclGetCompilerSupportedOptions(compiler, options.data(), &optionsSize);
    ASSERT_EQ(ret, VCL_RESULT_SUCCESS) << "vclGetCompilerSupportedOptions(data) failed! Result:0x" << std::hex
                                       << uint64_t(ret) << std::dec << std::endl;

    EXPECT_EQ(options.back(), '\0') << "Supported options list is expected to be null-terminated";
    const std::string_view optionsList(options.data());
    EXPECT_FALSE(optionsList.empty()) << "Supported options list should not be empty";
}

INSTANTIATE_TEST_SUITE_P(ApiBehaviorTest, VCLApiBehaviorTest, getVCLFunctionalTestParams(),
                         VCLApiBehaviorTest::getTestCaseName);

}  // namespace VCLTest
