//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_compiler.hpp"
#include "vcl_executable.hpp"
#include "vcl_query_network.hpp"

#include <openvino/openvino.hpp>
#include <openvino/util/file_util.hpp>
#include <transformations/utils/utils.hpp>

#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "vpux/compiler/compiler.hpp"

#include <future>
#include <thread>
#include <type_traits>
#include <unordered_set>

#define xstr(s) str(s)
#define str(s) #s

using namespace vpux;

namespace {

constexpr int64_t OLDEST_IR_VERSION_SUPPORTED = 10;

std::string rankToLegacyLayoutString(const size_t rank) {
    switch (rank) {
    case 0:
        return "SCALAR";
    case 1:
        return "C";
    case 2:
        return "NC";
    case 3:
        return "CHW";
    case 4:
        return "NCHW";
    case 5:
        return "NCDHW";
    default:
        return "BLOCKED";
    }
}

// Cannot use std::async as MSVC recycles the underlying thread
template <typename F>
auto run_in_worker_thread_sync(F&& f) -> typename std::invoke_result<F>::type {
    std::packaged_task<typename std::invoke_result<F>::type()> task(std::forward<F>(f));
    auto result = task.get_future();
    std::thread(std::move(task)).join();
    return result.get();
}

/**
 * @brief Adds precision conversion and transposition layers to the model in order to comply with the given precision
 * and layout values.
 * @details In the legacy scenarios when either the older API or the IR version 10 is being used, the "ov::Model"
 * object may not hold the correct I/O metadata values (either a wrong precision or a transposed shape may be used). The
 * objective of the current function is to correct this misalignment by introducing additional precision conversion or
 * transposition layers.
 *
 * Note that the correct precision/layout values are given by the driver. Depending on the plugin version, the origin of
 * these values may be either the metadata stored by the user application in a legacy "InferenceEngine::CNNNetwork"
 * object, or the values found within the "ov::Model" one, which could have been altered as a result of the
 * serialization process.
 *
 * @param model The model representation corresponding to the 2.0 API, this is the target object.
 * @param inputPrecisions The reference input precision values.
 * @param outputPrecisions The reference output precision values.
 * @param inputLayouts The reference input layout values.
 * @param outputLayouts The reference output layout values.
 * @param logger The logger to be used for logging messages.
 * @returns The model altered as to comply with the given precisions/layouts, or the original model if preprocessing
 * is skipped.
 */
std::shared_ptr<ov::Model> preprocessModel(const std::shared_ptr<ov::Model>& model,
                                           const std::unordered_map<std::string, ov::element::Type_t>& inputPrecisions,
                                           const std::unordered_map<std::string, ov::element::Type_t>& outputPrecisions,
                                           const std::unordered_map<std::string, std::string>& inputLayouts,
                                           const std::unordered_map<std::string, std::string>& outputLayouts,
                                           const VPUXDriverCompiler::VCLLogger* logger) {
    auto& runtimeInfoMap = model->get_rt_info();

    int64_t irVersion = OLDEST_IR_VERSION_SUPPORTED;
    if (const auto irVersionMatch = runtimeInfoMap.find("version"); irVersionMatch != runtimeInfoMap.end()) {
        irVersion = irVersionMatch->second.as<int64_t>();
    }

    bool useIndices = false;
    if (const auto useIndicesMatch = runtimeInfoMap.find("use_indices_for_io_metadata");
        useIndicesMatch != runtimeInfoMap.end()) {
        useIndices = useIndicesMatch->second.as<bool>();
    }

    logger->debug("useIndices is {0}, using {1} for parameter/result node identification", useIndices,
                  useIndices ? "indices" : "names");

    // Compiler needs to maintain compatiblity with older OpenVINO (plugin) versions.
    // Compiler needs to support applications that are still using OV 1.0 API.
    //     | OV API Version | IR version | Needs preprocessing? |
    //     | 1.0            | v10        | Yes                  |
    //     | 2.0            | v10        | Yes                  |
    //     | 1.0            | v11        | Invalid usecase      |
    //     | 2.0            | v11        | NO                   |
    // OpenVINO releases >= 23.2 are passing a runtime attribute "is_new_api" to inform
    // the compiler if API2.0 is being used. However, given the compatibility matrix
    // above, compiler will ignore "is_new_api" to maintain compatibility with even
    // older applications that use OV versions < 23.2 and IRv11
    if (irVersion >= 11) {
        logger->info("IR version >= 11. Preprocessing will be skipped.");
        return model;
    }
    logger->info("IR version < 11. Preprocessing will be performed.");

    bool hasRequiredIOInfo =
            !inputPrecisions.empty() && !outputPrecisions.empty() && !inputLayouts.empty() && !outputLayouts.empty();
    if (!hasRequiredIOInfo) {
        logger->warning("The ioInfo options are missing for IR version < 11! Preprocessing will be skipped.");
        return model;
    }
    logger->info("The ioInfo options are provided. Preprocessing will be performed.");

    auto preprocessor = ov::preprocess::PrePostProcessor(model);
    const ov::ParameterVector& parameters = model->get_parameters();
    const ov::ResultVector& results = model->get_results();

    logger->trace("Configuring {0} parameter nodes for preprocessing...", parameters.size());
    for (size_t parameterIndex = 0; parameterIndex < parameters.size(); ++parameterIndex) {
        const std::shared_ptr<ov::op::v0::Parameter>& parameter = parameters[parameterIndex];
        const std::string inputID = useIndices ? std::to_string(parameterIndex) : parameter->get_friendly_name();

        const ov::Layout tensorLayout(inputLayouts.at(inputID));
        const size_t rank = parameter->get_shape().size();
        const ov::Layout modelLayout(rankToLegacyLayoutString(rank));

        ov::preprocess::InputInfo& inputInfo = preprocessor.input(parameterIndex);
        inputInfo.tensor().set_layout(tensorLayout);
        inputInfo.model().set_layout(modelLayout);
        inputInfo.tensor().set_element_type(inputPrecisions.at(inputID));
    }
    logger->trace("Completed the configuration of all parameter nodes!");

    logger->trace("Configuring {0} result nodes for preprocessing...", results.size());
    for (size_t resultIndex = 0; resultIndex < results.size(); ++resultIndex) {
        const std::shared_ptr<ov::op::v0::Result>& result = results[resultIndex];

        std::string outputID;

        if (useIndices) {
            outputID = std::to_string(resultIndex);
        } else {
            // Otherwise, the legacy name of the result node (refers to the name of its parent node) will be used
            outputID = result->get_input_node_ptr(0)->get_friendly_name();
            if (result->get_input_node_ptr(0)->get_output_size() != 1) {
                // If the parent node does not have exactly 1 output port
                if (!outputLayouts.count(outputID)) {
                    // If the legacy name is not found, append the index of the parent node's output port linked to this
                    // result node. Otherwise, do not append anything.
                    outputID += "." + std::to_string(result->input_value(0).get_index());
                }
            }
        }

        ov::Layout tensorLayout;

        try {
            tensorLayout = ov::Layout(outputLayouts.at(outputID));
        } catch (const std::out_of_range& e) {
            // Throw an error and print the list of available indices/names
            std::string availableIDs;
            for (const auto& allIDs : outputLayouts) {
                availableIDs += "\n" + allIDs.first;
            }

            throw std::runtime_error(std::string(e.what()) + "\nFailed to resolve obtained " +
                                     std::string(useIndices ? "index" : "name") + " '" + outputID + "' at the " +
                                     std::to_string(resultIndex) +
                                     "th result node (node name: " + result->get_friendly_name() + "). Available " +
                                     std::string(useIndices ? "indices" : "names") + ": " + availableIDs);
        }

        const size_t rank = result->get_shape().size();
        const ov::Layout modelLayout(rankToLegacyLayoutString(rank));

        ov::preprocess::OutputInfo& outputInfo = preprocessor.output(resultIndex);
        outputInfo.tensor().set_layout(tensorLayout);
        outputInfo.model().set_layout(modelLayout);
        outputInfo.tensor().set_element_type(outputPrecisions.at(outputID));
    }
    logger->trace("Completed the configuration of all result nodes!");

    return preprocessor.build();
}

std::unique_ptr<CompilerImpl> createNPUCompiler() {
    return std::make_unique<CompilerImpl>();
}

}  // namespace

/// Compiler version contains the info of code commit, compiler API version
static const char* COMPILER_VERSION =
        xstr(DRIVER_COMPILER_ID) "." xstr(VCL_COMPILER_VERSION_MAJOR) "." xstr(VCL_COMPILER_VERSION_MINOR);

namespace VPUXDriverCompiler {

VPUXCompilerL0::VPUXCompilerL0(vcl_compiler_desc_t* compilerDesc, vcl_device_desc_t* deviceDesc, VCLLogger* vclLogger)
        : _options(std::make_shared<intel_npu::OptionsDesc>()), _compilerDesc(*compilerDesc), _logger(vclLogger) {
    // Initialize compiler description if it is not empty
    if (deviceDesc) {
        _deviceDesc = *deviceDesc;
        _isDeviceDescEmpty = false;
    } else {
        _deviceDesc = {};
        _isDeviceDescEmpty = true;
        _logger->debug("DeviceDesc is empty! Just use user config value for offline compilation.");
    }

    // Register compiler configuration options
    _options->add<intel_npu::PERFORMANCE_HINT>();
    _options->add<intel_npu::PERFORMANCE_HINT_NUM_REQUESTS>();
    _options->add<intel_npu::INFERENCE_PRECISION_HINT>();
    _options->add<intel_npu::PERF_COUNT>();
    _options->add<intel_npu::LOG_LEVEL>();
    _options->add<intel_npu::PLATFORM>();
    _options->add<intel_npu::COMPILER_TYPE>();
    _options->add<intel_npu::DEVICE_ID>();
    _options->add<intel_npu::BATCH_MODE>();
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
    _options->add<intel_npu::BATCH_COMPILER_MODE_SETTINGS>();
    _options->add<intel_npu::QDQ_OPTIMIZATION_AGGRESSIVE>();
    _options->add<intel_npu::QDQ_OPTIMIZATION>();
    _options->add<intel_npu::TURBO>();

#ifdef VPUX_DEVELOPER_BUILD
    // E#103359: WS is only available in developer builds
    _options->add<intel_npu::WEIGHTLESS_BLOB>();
    _options->add<intel_npu::SEPARATE_WEIGHTS_VERSION>();
    _options->add<intel_npu::WS_COMPILE_CALL_NUMBER>();
    _options->add<intel_npu::CACHE_MODE>();
#endif  // VPUX_DEVELOPER_BUILD

    // Create compiler instance with the default config
    // COMPILER_TYPE DRIVER is assumed
    _compiler = createNPUCompiler();

    // Update the compiler properties
    uint32_t compiler_version = _compiler->get_version();
    _compilerProp.id = COMPILER_VERSION;
    _compilerProp.version.major = compiler_version >> 16;     /// 16Bit msb = major version
    _compilerProp.version.minor = compiler_version & 0xFFFF;  /// 16Bit lsb = minor version

    _compilerProp.supportedOpsets = _compiler->getSupportedOpsetVersion();
}

std::pair<VPUXExecutableL0*, vcl_result_t> VPUXCompilerL0::importNetwork(BuildInfo& buildInfo) {
    StopWatch stopWatch;
    if (buildInfo.enableProfiling) {
        // Output time cost on vcl level
        stopWatch.start();
    }

    auto scoped = Scoped{[&stopWatch, &buildInfo, this]() {
        if (buildInfo.enableProfiling) {
            stopWatch.stop();
            _logger->info("Compile net time: {0} ms", stopWatch.delta_ms());
        }
    }};

    VPUXExecutableL0* exe = nullptr;
    std::shared_ptr<ov::Model> model;

    try {
        model = preprocessModel(buildInfo.model, buildInfo.inputPrecisions, buildInfo.outputPrecisions,
                                buildInfo.inputLayouts, buildInfo.outputLayouts, _logger);
    } catch (const std::exception& error) {
        _logger->outputError(formatv("Failed to process model:\n{0}", error.what()));
        return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
    } catch (...) {
        _logger->outputError("Internal exception! Can not process model!");
        return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
    }

    try {
        // Call compiler to compile the model and create blob
        // Create executable with the result NetworkDescription, profiling option and logger
        // Note we rely on implicit move semantics thanks to compile result being an rvalue,
        // failure to move here would lead to a blob copy!

        // Isolate the MLIR thread to safely destroy MLIR thread_local objects before CiD unload
        auto network = std::make_shared<const NetworkDescription>(run_in_worker_thread_sync([&] {
            if (buildInfo.parsedConfig.get<intel_npu::WEIGHTLESS_BLOB>()) {
                return _compiler->compileWsIterative(model, buildInfo.parsedConfig,
                                                     buildInfo.parsedConfig.get<intel_npu::WS_COMPILE_CALL_NUMBER>());
            }

            return _compiler->compile(model, buildInfo.parsedConfig);
        }));

        exe = new VPUXExecutableL0(network, buildInfo.enableProfiling, _logger);
    } catch (const std::exception& error) {
        _logger->outputError(formatv("Compiler returned msg:\n{0}", error.what()));
        return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
    } catch (...) {
        _logger->outputError("Internal exception! Can not compile!");
        return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
    }

    return std::pair<VPUXExecutableL0*, vcl_result_t>(exe, VCL_RESULT_SUCCESS);
}  // namespace VPUXDriverCompiler

NetworkDescriptionView VPUXCompilerL0::importNetwork(BuildInfo& buildInfo, BlobAllocator& allocator) {
    StopWatch stopWatch;
    if (buildInfo.enableProfiling) {
        // Output time cost on vcl level
        stopWatch.start();
    }

    auto scoped = Scoped{[&stopWatch, &buildInfo, this]() {
        if (buildInfo.enableProfiling) {
            stopWatch.stop();
            _logger->info("Compile net time: {0} ms", stopWatch.delta_ms());
        }
    }};

    std::shared_ptr<ov::Model> model;

    model = preprocessModel(buildInfo.model, buildInfo.inputPrecisions, buildInfo.outputPrecisions,
                            buildInfo.inputLayouts, buildInfo.outputLayouts, _logger);

    // Isolate the MLIR thread to safely destroy MLIR thread_local objects before CiD unload
    return run_in_worker_thread_sync([&] {
        if (buildInfo.parsedConfig.get<intel_npu::WEIGHTLESS_BLOB>()) {
            return _compiler->compileWsIterative(model, buildInfo.parsedConfig,
                                                 buildInfo.parsedConfig.get<intel_npu::WS_COMPILE_CALL_NUMBER>(),
                                                 allocator);
        }

        return _compiler->compile(model, buildInfo.parsedConfig, allocator);
    });
}

vcl_result_t VPUXCompilerL0::queryNetwork(const BuildInfo& buildInfo, VPUXQueryNetworkL0* pQueryNetwork) {
    _logger->info("Start to call query function from compiler to get supported layers!");
    ov::SupportedOpsMap queryNetworkResult;
    try {
        queryNetworkResult = _compiler->query(buildInfo.model, buildInfo.parsedConfig);
    } catch (const std::exception& error) {
        _logger->outputError(formatv("Compiler returned msg:\n{0}", error.what()));
        return VCL_RESULT_ERROR_UNKNOWN;
    } catch (...) {
        _logger->outputError("Failed to call query from compiler!");
        return VCL_RESULT_ERROR_UNKNOWN;
    }
    _logger->info("Successfully query supported layers!");

    // Serialize the result to predefined format
    auto ret = pQueryNetwork->setQueryResult(queryNetworkResult);
    return ret;
}

vcl_result_t VPUXCompilerL0::getSupportedOptions(char* buffer, uint64_t size) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    // get the registered options list, excluding private options (false param)
    std::string optListStr = _options->getSupportedAsString(false);
    // check if it fits
    uint64_t stringsize = optListStr.size() + 1;
    if (stringsize > size) {
        _logger->outputError("Compiler supported options list does not fit into the provided buffer!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    // serialize
    std::memcpy(buffer, optListStr.c_str(), stringsize);

    return ret;
}

vcl_result_t VPUXCompilerL0::getSupportedOptionsSize(uint64_t* stringSize) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    // get the registered options list, excluding private options (false param)
    std::string optionsList = _options->getSupportedAsString(false);
    // get string size +1 for null-termination
    *stringSize = optionsList.size() + 1;
    return ret;
}

bool VPUXCompilerL0::isOptionValueSupported(const char* option, const char* value) {
    // From OV 25.2, the plugin can check private options
    // Return true if the option is supported by transformation in vcl_common
    static std::unordered_set<std::string> compatibilityConfig = {"NPU_DPU_GROUPS"};

    std::string optName(option);
    if (compatibilityConfig.count(optName) > 0) {
        _logger->debug("Option {0} is a compatibility option, returning true", optName);
        return true;
    }

    bool ret = false;
    std::vector<std::string> optList = _options->getSupported(true);  // include private
    if (std::find(optList.begin(), optList.end(), optName) != optList.end()) {
        ret = true;  // found
    } else {
        return false;  // not found
    };
    // see if we need to check for supported value too
    if (value != nullptr) {
        // value is to be checked too
        /*TO BE IMPLEMENTED*/
        return false;
    }

    return ret;
}

}  // namespace VPUXDriverCompiler
