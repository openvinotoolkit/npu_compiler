//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_common.hpp>
#include "vpux/compiler/compiler.hpp"
#include "vpux/utils/ov/config.hpp"
#include "vpux/utils/ov/options.hpp"

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1_decl.hpp>
#include "openvino/op/add.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace vpux;

class CompilerProfilingTest : public testing::Test {
public:
    CompilerProfilingTest()
            : _options{std::make_shared<vpux::OV::OptionsDesc>()},
              _config{_options},
              _compiler{std::make_shared<CompilerImpl>()} {
        _options->add<vpux::OV::PLATFORM>();
        _options->add<vpux::OV::COMPILER_TYPE>();
        _options->add<vpux::OV::COMPILATION_MODE_PARAMS>();

        _config.update({{vpux::OV::COMPILER_TYPE::key().data(), "PLUGIN"}});
        // Note: for this test, it does not matter which platform is used -
        // select any public one that is still "actual"
        _config.update({{vpux::OV::PLATFORM::key().data(), "VPU5010"}});
    }

    void setProfilingTool(const std::string& toolStr) {
        _config.update({{vpux::OV::COMPILATION_MODE_PARAMS::key().data(), "compiler-profiler-tool=" + toolStr}});
    }

    void SetUp() override {
        const std::filesystem::path tempDir = std::filesystem::temp_directory_path();
        _tmpFilePath = tempDir / ("compiler_profiling_dump.json");
    }

    void TearDown() override {
        std::remove(_tmpFilePath.string().c_str());
    }

protected:
    std::shared_ptr<vpux::OV::OptionsDesc> _options;
    vpux::OV::Config _config;
    std::shared_ptr<ICompiler> _compiler;
    std::filesystem::path _tmpFilePath;

    std::shared_ptr<ov::Model> createSomeModel() const {
        auto elementType = ov::element::f32;
        auto shape = ov::Shape{1, 2, 5, 5};
        auto input1 = std::make_shared<ov::op::v0::Parameter>(elementType, shape);
        input1->set_friendly_name("input1");
        input1->output(0).get_tensor().set_names({"input1"});
        auto input2 = std::make_shared<ov::op::v0::Parameter>(elementType, shape);
        input2->set_friendly_name("input2");
        input2->output(0).get_tensor().set_names({"input2"});

        auto add = std::make_shared<ov::op::v1::Add>(input1, input2);
        add->set_friendly_name("add");

        auto output = std::make_shared<ov::op::v0::Result>(add);
        output->set_friendly_name("output");
        output->output(0).get_tensor().set_names({"output"});

        ov::ParameterVector parameters({input1, input2});
        ov::ResultVector results({output});
        auto model = std::make_shared<ov::Model>(results, parameters);
        model->set_friendly_name("some_model");

        return model;
    }
};

TEST_F(CompilerProfilingTest, EmptyProfiler) {
    const auto model = createSomeModel();
    auto result = _compiler->compile(model, _config);
    ASSERT_FALSE(result.compiledNetwork.empty()) << "Compilation must succeed if the profiler is not set";
}

TEST_F(CompilerProfilingTest, NoneProfiler) {
    setProfilingTool("none");
    const auto model = createSomeModel();
    auto result = _compiler->compile(model, _config);
    ASSERT_FALSE(result.compiledNetwork.empty()) << "Compilation must succeed if the profiler is not set";
}

TEST_F(CompilerProfilingTest, UnknownProfiler) {
    setProfilingTool("unknown");
    const auto model = createSomeModel();

    try {
        auto result = _compiler->compile(model, _config);
        GTEST_FAIL() << "Compilation must fail since profiler tool is unknown";
    } catch (const std::exception& e) {
        const auto pos = std::string(e.what()).find("Unknown compiler profiler tool");
        ASSERT_NE(pos, std::string::npos)
                << "Compilation must fail since profiler tool is unknown. Error: " << e.what();
    }
}

TEST_F(CompilerProfilingTest, MlirBuiltinProfiler) {
    setProfilingTool("mlir-profiler=path=" + _tmpFilePath.string());

    auto result = _compiler->compile(createSomeModel(), _config);
    ASSERT_FALSE(result.compiledNetwork.empty()) << "Compilation must succeed if the profiler is valid";

    std::fstream dumpFile(_tmpFilePath, std::ios::in);
    ASSERT_TRUE(dumpFile.is_open());
    std::ostringstream ss;
    ss << dumpFile.rdbuf();
    const auto dumpContent = ss.str();

    // probe the dump for different passes that are known to be always present

    const auto initResourcesLine = dumpContent.find("`pass-execution` running `InitResources`");
    ASSERT_NE(initResourcesLine, std::string::npos)
            << "InitResources is the first pass in compiler. Dump content: " << dumpContent;

    const auto convertShapeTo4DLine = dumpContent.find("`pass-execution` running `ConvertShapeTo4D`");
    ASSERT_NE(convertShapeTo4DLine, std::string::npos)
            << "ConvertShapeTo4D pass is the most common legalization pass from IE";

    const auto bufferizationLine = dumpContent.find("`pass-execution` running `OneShotBufferizeVPU2VPUIP`");
    ASSERT_NE(bufferizationLine, std::string::npos)
            << "OneShotBufferizeVPU2VPUIP pass is the required pass to get buffers in MLIR";

    const auto feasibleAllocationLine = dumpContent.find("`pass-execution` running `FeasibleAllocation`");
    ASSERT_NE(feasibleAllocationLine, std::string::npos) << "FeasibleAllocation pass is the required pass from VPUIP";
}

constexpr bool isValgrindPresent() {
#ifdef NPU_TESTS_VALGRIND_FOUND
    return true;
#else
    return false;
#endif
}

TEST_F(CompilerProfilingTest, CallgrindProfiler) {
    setProfilingTool("callgrind");
    const auto model = createSomeModel();

    // Note: This test is a bit "dumb" since the case where valgrind is present
    // does not really test much. yet, it is still nice to have to e.g. test
    // locally. In CI, expect valgrind to not be found, so the compilation must
    // fail with an exception, which is what we want: it shows that the
    // compilation option was parsed successfully and the profiler constructor
    // was reached eventually, and that failed as it should be.
    if constexpr (isValgrindPresent()) {
        auto result = _compiler->compile(model, _config);
        ASSERT_FALSE(result.compiledNetwork.empty()) << "Compilation must succeed if the profiler is found";
    } else {
        try {
            _compiler->compile(model, _config);
            GTEST_FAIL();
        } catch (const std::exception& e) {
            const auto pos = std::string(e.what()).find("Callgrind profiler is requested but valgrind is not found");
            ASSERT_NE(pos, std::string::npos)
                    << "Compilation must fail since valgrind is not found in the system. Got: " << e.what();
        }
    }
}

TEST_F(CompilerProfilingTest, CallgrindProfilerWithRegex) {
    setProfilingTool("callgrind=regex=run-ngraph-passes|convert-shape-to-4d");
    const auto model = createSomeModel();

    if constexpr (isValgrindPresent()) {
        auto result = _compiler->compile(model, _config);
        ASSERT_FALSE(result.compiledNetwork.empty()) << "Compilation must succeed if the profiler is found";
    } else {
        try {
            _compiler->compile(model, _config);
            GTEST_FAIL();
        } catch (const std::exception& e) {
            const auto pos = std::string(e.what()).find("Callgrind profiler is requested but valgrind is not found");
            ASSERT_NE(pos, std::string::npos)
                    << "Compilation must fail since valgrind is not found in the system. Got: " << e.what();
        }
    }
}

TEST_F(CompilerProfilingTest, CallgrindProfilerWithRegexAndMultipleFiles) {
    setProfilingTool("callgrind='regex=convert-shape-to-4d separate-dumps=1'");
    const auto model = createSomeModel();

    if constexpr (isValgrindPresent()) {
        auto result = _compiler->compile(model, _config);
        ASSERT_FALSE(result.compiledNetwork.empty()) << "Compilation must succeed if the profiler is found";
    } else {
        try {
            _compiler->compile(model, _config);
            GTEST_FAIL();
        } catch (const std::exception& e) {
            const auto pos = std::string(e.what()).find("Callgrind profiler is requested but valgrind is not found");
            ASSERT_NE(pos, std::string::npos)
                    << "Compilation must fail since valgrind is not found in the system. Got: " << e.what();
        }
    }
}

TEST_F(CompilerProfilingTest, VTuneProfiler) {
    setProfilingTool("vtune=regex=.*");

    auto result = _compiler->compile(createSomeModel(), _config);
    ASSERT_FALSE(result.compiledNetwork.empty()) << "Compilation must succeed for the VTune profiler";
}

TEST_F(CompilerProfilingTest, VTuneProfilerWithEmptyRegex) {
    setProfilingTool("vtune=regex=");

    auto result = _compiler->compile(createSomeModel(), _config);
    ASSERT_FALSE(result.compiledNetwork.empty()) << "Compilation must succeed if the VTune profiler regex is empty";
}

TEST_F(CompilerProfilingTest, VTuneProfilerWithRegex) {
    setProfilingTool("vtune=regex=run-ngraph-passes|convert-shape-to-4d");

    auto result = _compiler->compile(createSomeModel(), _config);
    ASSERT_FALSE(result.compiledNetwork.empty()) << "Compilation must succeed if the VTune profiler regex is set";
}

TEST_F(CompilerProfilingTest, VTuneProfilerWithRegexInQuotes) {
    setProfilingTool("vtune=regex='run-ngraph-passes|convert-shape-to-4d'");

    auto result = _compiler->compile(createSomeModel(), _config);
    ASSERT_FALSE(result.compiledNetwork.empty()) << "Compilation must succeed if the VTune profiler regex is set";
}
