//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_common.hpp>
#include "intel_npu/config/options.hpp"
#include "vpux/compiler/compiler.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/logger/logger.hpp"

#include "llvm/Support/SHA256.h"

#include <gtest/gtest.h>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1_decl.hpp>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/softmax.hpp"

#include <array>
#include <fstream>
#include <future>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using namespace vpux;
using namespace intel_npu;

using CompilationParamsIR = std::tuple<std::vector<std::string>,  // model paths
                                       size_t,                    // number of threads per model
                                       size_t                     // number of compilation iterations per model thread
                                       >;

using CompilationParamsModel = std::tuple<std::vector<std::string>,  // model names
                                          size_t,                    // number of threads per model
                                          size_t  // number of compilation iterations per model thread
                                          >;

// The compilation status for each iteration of each thread, for every model
// Value true represents a successful compilation, while the string represents the failure message if it is present
using Status = std::tuple<bool, std::string>;
using CompilationStatus = std::vector<std::vector<std::vector<Status>>>;

using Checksum = std::array<uint8_t, 32>;
using CompilationChecksums = std::vector<std::vector<std::vector<Checksum>>>;

namespace {

class ChecksumException : public std::exception {
public:
    ChecksumException(const std::string& message): _message(message) {
    }
    const char* what() const noexcept override {
        return _message.c_str();
    }

private:
    std::string _message;
};

std::string stringifyChecksum(const Checksum& checksum) {
    std::string checksumStr;
    checksumStr.reserve(checksum.size() * 2);
    for (const auto byte : checksum) {
        char byteStr[3];
        sprintf(byteStr, "%02x", byte);
        checksumStr += byteStr;
    }
    return checksumStr;
};
}  // namespace

class CompilationTestBase : virtual public ov::test::TestsCommon {
public:
    CompilationTestBase()
            : _options{std::make_shared<OptionsDesc>()},
              _config{_options},
              _compiler{nullptr},
              _numThreads{},
              _numIterations{},
              _log(vpux::Logger::global()) {
    }

protected:
    void SetPlatform(const std::string& platform) {
        _config.update({{PLATFORM::key().data(), platform}});
    }

    void startCompilationThreads(const std::map<std::string, std::vector<std::shared_ptr<ov::Model>>>& threadModels,
                                 std::vector<std::future<Checksum>>& futures,
                                 std::unordered_map<size_t, size_t>& modelForFutures, size_t numThreads,
                                 size_t numIterations, const vpux::Logger& log) const {
        const auto compileNetwork = [](const std::shared_ptr<ICompiler>& compiler,
                                       const std::shared_ptr<const ov::Model>& model,
                                       const Config& config) -> Checksum {
            const auto netDesc = compiler->compile(model, config);
            const auto& blob = netDesc.compiledNetwork;
            return llvm::SHA256::hash(blob);
        };

        const auto threadFunction = [compileNetwork](const std::shared_ptr<ICompiler>& compiler,
                                                     const std::shared_ptr<const ov::Model>& model,
                                                     const Config& config, const size_t iterationsCount) -> Checksum {
            const auto previousHash = compileNetwork(compiler, model, config);
            for (auto i : irange(iterationsCount - 1)) {
                const auto currentHash = compileNetwork(compiler, model, config);
                if (previousHash != currentHash) {
                    std::stringstream ss;
                    ss << "Checksum for iteration " << i + 1 << " (" << stringifyChecksum(currentHash)
                       << ") different than previous hash (" << stringifyChecksum(previousHash) << ")";
                    throw ChecksumException(ss.str());
                }
            }
            return previousHash;
        };

        ov::Core core;
        size_t futureIdx = 0;
        for (auto p : threadModels | indexed) {
            const auto modelIdx = p.index();
            const auto& modelName = p.value().first;

            log.trace("Model {0} with name '{1}'", modelIdx + 1, modelName);
            auto& models = p.value().second;

            for (const size_t tIt : irange(numThreads)) {
                modelForFutures[futureIdx++] = modelIdx;
                const auto& model = models.at(tIt);

                log.nest().trace("Starting thread {0} / {1} with {2} iteration(s)", tIt + 1, numThreads, numIterations);
                futures.push_back(
                        std::async(std::launch::async, threadFunction, _compiler, model, _config, numIterations));
            }
        }
    }

    void compareResults(std::vector<std::future<Checksum>>& futures,
                        std::unordered_map<size_t, size_t>& modelForFutures) const {
        bool anyFailure = false;
        std::unordered_map<size_t, Checksum> modelChecksums;
        for (const auto& future : futures | indexed) {
            try {
                const auto threadChecksum = future.value().get();

                const auto modelIdx = modelForFutures.at(future.index());
                if (modelChecksums.find(modelIdx) == modelChecksums.end()) {
                    modelChecksums[modelIdx] = threadChecksum;
                    continue;
                }
                const auto& modelChecksum = modelChecksums.at(modelIdx);
                if (threadChecksum != modelChecksum) {
                    std::stringstream ss;
                    ss << "Checksum " << stringifyChecksum(threadChecksum)
                       << " different than hash of other threads: " << stringifyChecksum(modelChecksum);
                    throw ChecksumException(ss.str());
                }

                continue;
            } catch (const ChecksumException& hashException) {
                std::cout << "Checksum error for thread " << future.index() << ":" << std::endl;
                std::cout << "    " << hashException.what() << std::endl;
            } catch (const std::exception& compileException) {
                std::cout << "Compilation error for thread " << future.index() << ":" << std::endl;
                std::cout << "    " << compileException.what() << std::endl;
            } catch (...) {
                std::cout << "General error for thread " << future.index() << std::endl;
            }
            anyFailure = true;
        }

        ASSERT_EQ(anyFailure, false);
    }

protected:
    std::shared_ptr<OptionsDesc> _options;
    Config _config;
    std::shared_ptr<ICompiler> _compiler;
    size_t _numThreads;
    size_t _numIterations;

    vpux::Logger _log;
};

class CompilationTestModel :
        public testing::WithParamInterface<CompilationParamsModel>,
        virtual public CompilationTestBase {
public:
    CompilationTestModel(): CompilationTestBase(), _modelNames{} {
        _log.setName("CompilationTestModel");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParamsModel>& obj) {
        std::vector<std::string> modelNames;
        size_t numThreads;
        size_t numIterations;
        std::tie(modelNames, numThreads, numIterations) = obj.param;
        const auto numModels = modelNames.size();

        std::ostringstream result;
        result << "modelNames=[";
        for (size_t i = 0; i < numModels - 1; ++i) {
            result << modelNames[i] << ",";
        }
        result << modelNames[numModels - 1] << "]_";
        result << "threads=" << numThreads << "_";
        result << "iterations=" << numIterations;
        return result.str();
    }

protected:
    void SetUp() override {
        _modelNames = std::get<0>(GetParam());
        _numThreads = std::get<1>(GetParam());
        _numIterations = std::get<2>(GetParam());

        _options->add<intel_npu::PERFORMANCE_HINT>();
        _options->add<intel_npu::PERFORMANCE_HINT_NUM_REQUESTS>();
        _options->add<intel_npu::INFERENCE_PRECISION_HINT>();
        _options->add<intel_npu::PERF_COUNT>();
        _options->add<intel_npu::LOG_LEVEL>();
        _options->add<intel_npu::PLATFORM>();
        _options->add<intel_npu::DEVICE_ID>();
        _options->add<intel_npu::CACHE_DIR>();
        _options->add<intel_npu::LOADED_FROM_CACHE>();
        _options->add<intel_npu::BATCH_MODE>();
        _options->add<intel_npu::COMPILER_TYPE>();
        _options->add<intel_npu::COMPILATION_MODE>();
        _options->add<intel_npu::COMPILATION_MODE_PARAMS>();
        _options->add<intel_npu::BACKEND_COMPILATION_PARAMS>();
        _options->add<intel_npu::COMPILATION_NUM_THREADS>();
        _options->add<intel_npu::TILES>();
        _options->add<intel_npu::STEPPING>();
        _options->add<intel_npu::MAX_TILES>();
        _options->add<intel_npu::DMA_ENGINES>();
        _options->add<intel_npu::DYNAMIC_SHAPE_TO_STATIC>();
        _options->add<intel_npu::EXECUTION_MODE_HINT>();
        _options->add<intel_npu::COMPILER_DYNAMIC_QUANTIZATION>();
        _options->add<intel_npu::QDQ_OPTIMIZATION>();
        _options->add<intel_npu::QDQ_OPTIMIZATION_AGGRESSIVE>();

        _compiler = std::make_shared<CompilerImpl>();
    }

    void Run() const {
        std::map<std::string, std::vector<std::shared_ptr<ov::Model>>> threadModels;
        for (auto p : _modelNames | indexed) {
            const auto modelIdx = p.index();
            const auto& modelName = p.value();

            _log.trace("Model {0} with name '{1}'", modelIdx + 1, modelName);

            for (const size_t tIt : irange(_numThreads)) {
                _log.nest().trace("Preparing model with name '{0}' for thread {1} / {2}", modelName, tIt + 1,
                                  _numThreads);
                if (modelName == "A") {
                    threadModels[modelName].push_back(createModelA());
                } else if (modelName == "B") {
                    threadModels[modelName].push_back(createModelB());
                } else {
                    FAIL() << "Unknown model";
                }
            }
        }

        std::vector<std::future<Checksum>> futures;
        std::unordered_map<size_t, size_t> modelForFutures;
        startCompilationThreads(threadModels, futures, modelForFutures, _numThreads, _numIterations, _log);

        compareResults(futures, modelForFutures);
    }

private:
    std::shared_ptr<ov::Model> createModelA() const {
        auto elementType = ov::element::f32;
        auto shape = ov::Shape{1, 3, 32, 32};
        auto layout = "NCHW";
        auto input = std::make_shared<ov::op::v0::Parameter>(elementType, shape);
        input->set_layout(layout);
        input->set_friendly_name("input");
        input->output(0).get_tensor().set_names({"input"});

        const auto conv_weights =
                ov::op::v0::Constant::create(ov::element::f32, {16, 3, 1, 1}, std::vector<float>{1.f});
        const auto conv = std::make_shared<ov::op::v1::Convolution>(
                input, conv_weights, /*strides=*/ov::Strides{1, 1}, /*padsBegin=*/ov::CoordinateDiff{0, 0},
                /*padsEnd=*/ov::CoordinateDiff{0, 0}, /*dilations=*/ov::Strides{1, 1});
        conv_weights->set_friendly_name("conv_weights");
        conv->set_friendly_name("conv");

        auto add_constant = ov::op::v0::Constant::create(elementType, {1}, {1});
        auto add = std::make_shared<ov::op::v1::Add>(conv, add_constant);
        add_constant->set_friendly_name("add_constant");
        add->set_friendly_name("add");

        auto output = std::make_shared<ov::op::v0::Result>(add);
        output->set_friendly_name("output");
        output->output(0).get_tensor().set_names({"output"});

        ov::ParameterVector parameters({input});
        ov::ResultVector results({output});
        auto model = std::make_shared<ov::Model>(results, parameters);
        model->set_friendly_name("model_a");

        return model;
    }

    std::shared_ptr<ov::Model> createModelB() const {
        auto elementType = ov::element::f16;
        auto shape = ov::Shape{1, 30, 64, 64};
        auto layout = "NCHW";
        auto input = std::make_shared<ov::op::v0::Parameter>(elementType, shape);
        input->set_layout(layout);
        input->set_friendly_name("input");
        input->output(0).get_tensor().set_names({"input"});

        const auto softmax = std::make_shared<ov::op::v1::Softmax>(input, /*axis=*/1);
        softmax->set_friendly_name("softmax");

        auto reshape_constant = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, 30, 128, 32});
        const auto reshape = std::make_shared<ov::opset1::Reshape>(softmax, reshape_constant, false);
        reshape_constant->set_friendly_name("reshape_constant");
        reshape->set_friendly_name("reshape");

        auto output = std::make_shared<ov::op::v0::Result>(reshape);
        output->set_friendly_name("output");
        output->output(0).get_tensor().set_names({"output"});

        ov::ParameterVector parameters({input});
        ov::ResultVector results({output});
        auto model = std::make_shared<ov::Model>(results, parameters);
        model->set_friendly_name("model_b");

        return model;
    }

private:
    std::vector<std::string> _modelNames;
};

class CompilationTestIR : public testing::WithParamInterface<CompilationParamsIR>, virtual public CompilationTestBase {
public:
    CompilationTestIR(): CompilationTestBase(), _modelPaths{} {
        _log.setName("CompilationTestIR");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParamsIR>& obj) {
        std::vector<std::string> modelPaths;
        size_t numThreads;
        size_t numIterations;
        std::tie(modelPaths, numThreads, numIterations) = obj.param;
        const auto numModels = modelPaths.size();

        std::ostringstream result;
        result << "modelPaths=[";
        for (size_t i = 0; i < numModels - 1; ++i) {
            result << modelPaths[i] << ",";
        }
        result << modelPaths[numModels - 1] << "]_";
        result << "threads=" << numThreads << "_";
        result << "iterations=" << numIterations;
        return result.str();
    }

protected:
    void SetUp() override {
        _modelPaths = std::get<0>(GetParam());
        _numThreads = std::get<1>(GetParam());
        _numIterations = std::get<2>(GetParam());

        validateAndExpandModelPaths(_modelPaths);

        _options->add<intel_npu::PERFORMANCE_HINT>();
        _options->add<intel_npu::PERFORMANCE_HINT_NUM_REQUESTS>();
        _options->add<intel_npu::INFERENCE_PRECISION_HINT>();
        _options->add<intel_npu::PERF_COUNT>();
        _options->add<intel_npu::LOG_LEVEL>();
        _options->add<intel_npu::PLATFORM>();
        _options->add<intel_npu::DEVICE_ID>();
        _options->add<intel_npu::CACHE_DIR>();
        _options->add<intel_npu::LOADED_FROM_CACHE>();
        _options->add<intel_npu::BATCH_MODE>();
        _options->add<intel_npu::COMPILER_TYPE>();
        _options->add<intel_npu::COMPILATION_MODE>();
        _options->add<intel_npu::COMPILATION_MODE_PARAMS>();
        _options->add<intel_npu::BACKEND_COMPILATION_PARAMS>();
        _options->add<intel_npu::COMPILATION_NUM_THREADS>();
        _options->add<intel_npu::TILES>();
        _options->add<intel_npu::STEPPING>();
        _options->add<intel_npu::MAX_TILES>();
        _options->add<intel_npu::DMA_ENGINES>();
        _options->add<intel_npu::DYNAMIC_SHAPE_TO_STATIC>();
        _options->add<intel_npu::EXECUTION_MODE_HINT>();
        _options->add<intel_npu::COMPILER_DYNAMIC_QUANTIZATION>();
        _options->add<intel_npu::QDQ_OPTIMIZATION>();
        _options->add<intel_npu::QDQ_OPTIMIZATION_AGGRESSIVE>();

        _compiler = std::make_shared<CompilerImpl>();
    }

    void Run() const {
        ov::Core core;
        std::map<std::string, std::vector<std::shared_ptr<ov::Model>>> threadModels;
        for (auto p : _modelPaths | indexed) {
            const auto modelIdx = p.index();
            const auto& modelPath = p.value();

            _log.trace("Model {0} with path '{1}'", modelIdx + 1, modelPath);

            for (const size_t tIt : irange(_numThreads)) {
                const auto model = core.read_model(modelPath);
                _log.nest().trace("Read model with name '{0}' for thread {1} / {2}", model->get_friendly_name(),
                                  tIt + 1, _numThreads);
                threadModels[modelPath].push_back(model);
            }
        }

        std::vector<std::future<Checksum>> futures;
        std::unordered_map<size_t, size_t> modelForFutures;
        startCompilationThreads(threadModels, futures, modelForFutures, _numThreads, _numIterations, _log);

        compareResults(futures, modelForFutures);
    }

private:
    // Finds instances of environmental variables (e.g. ${MODELS_PATH}) and expands them into the path string
    // In case any path does not point to a valid .xml file (and associated .bin file),
    // the function returns false
    void validateAndExpandModelPaths(std::vector<std::string>& modelPaths) {
        for (auto& path : modelPaths) {
            std::vector<std::tuple<size_t, size_t, std::string>> vars;

            size_t index = 0;
            while ((index = path.find("$", index)) != std::string::npos) {
                const auto indexLPar = index + 1;
                if (indexLPar >= path.size() || path[indexLPar] != '{') {
                    ++index;
                }
                size_t indexRPar = path.find("}", indexLPar);
                if (indexRPar >= path.size()) {
                    ++index;
                    continue;
                }
                const auto envVar = path.substr(indexLPar + 1, indexRPar - indexLPar - 1);
                if (const auto env = std::getenv(envVar.c_str())) {
                    vars.push_back(std::make_tuple(index, indexRPar, env));
                }
                index += envVar.length();
            }

            for (const auto& var : vars) {
                const auto indexStart = std::get<0>(var);
                const auto indexEnd = std::get<1>(var);
                const auto envVarValue = std::get<2>(var);
                path.replace(indexStart, indexEnd - indexStart + 1, envVarValue.c_str());
            }
        }

        for (const auto& path : modelPaths) {
            std::ifstream xmlFile(path.c_str());
            ASSERT_TRUE(xmlFile.good()) << "Invalid model xml path: " << path;

            auto binFilePath = path;
            const std::string binExt(".bin");
            binFilePath.replace(binFilePath.size() - binExt.size(), binExt.size(), binExt);
            std::ifstream binFile(path.c_str());
            ASSERT_TRUE(binFile.good()) << "Invalid model bin path: " << binFilePath;
        }
    }

private:
    std::vector<std::string> _modelPaths;
};

TEST_P(CompilationTestModel, NPU3720) {
    SetPlatform("VPU3720");
    Run();
}

TEST_P(CompilationTestModel, NPU4000) {
    SetPlatform("VPU4000");
    Run();
}
INSTANTIATE_TEST_SUITE_P(single_thread, CompilationTestModel,
                         ::testing::Combine(::testing::Values(std::vector<std::string>{"A"}),
                                            ::testing::ValuesIn(std::vector<size_t>{1}),  // num threads per model
                                            ::testing::Values(3)                          // num iterations per thread
                                            ),
                         CompilationTestModel::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(precommit_multithreaded, CompilationTestModel,
                         ::testing::Combine(::testing::Values(std::vector<std::string>{"A"}),
                                            ::testing::ValuesIn(std::vector<size_t>{4}),  // num threads per model
                                            ::testing::Values(1)                          // num iterations per thread
                                            ),
                         CompilationTestModel::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(precommit_multithreaded_two_models, CompilationTestModel,
                         ::testing::Combine(::testing::Values(std::vector<std::string>{
                                                    "A",
                                                    "B",
                                            }),
                                            ::testing::ValuesIn(std::vector<size_t>{1}),  // num threads per model
                                            ::testing::Values(2)                          // num iterations per thread
                                            ),
                         CompilationTestModel::getTestCaseName);
