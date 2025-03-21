//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vcl_compiler.hpp"
#include "vcl_executable.hpp"
#include "vcl_query_network.hpp"

#include <openvino/openvino.hpp>
#include <openvino/util/file_util.hpp>
#include <transformations/utils/utils.hpp>

#include "intel_npu/config/common.hpp"
#include "intel_npu/config/compiler.hpp"
#include "intel_npu/config/runtime.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "vpux/compiler/compiler.hpp"

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
 * @param useIndices Indicates whether the rest of the map arguments use indices (converted to strings) or names as
 * keys.
 * @returns The model altered as to comply with the given precisions/layouts.
 */
std::shared_ptr<ov::Model> preprocessModel(const std::shared_ptr<ov::Model>& model,
                                           const std::unordered_map<std::string, ov::element::Type_t>& inputPrecisions,
                                           const std::unordered_map<std::string, ov::element::Type_t>& outputPrecisions,
                                           const std::unordered_map<std::string, std::string>& inputLayouts,
                                           const std::unordered_map<std::string, std::string>& outputLayouts,
                                           const bool useIndices) {
    auto preprocessor = ov::preprocess::PrePostProcessor(model);
    const ov::ParameterVector& parameters = model->get_parameters();
    const ov::ResultVector& results = model->get_results();

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

    for (size_t resultIndex = 0; resultIndex < results.size(); ++resultIndex) {
        const std::shared_ptr<ov::op::v0::Result>& result = results[resultIndex];
        const std::string outputID =
                useIndices ? std::to_string(resultIndex) : ov::op::util::get_ie_output_name(result->input_value(0));

        const ov::Layout tensorLayout(outputLayouts.at(outputID));
        const size_t rank = result->get_shape().size();
        const ov::Layout modelLayout(rankToLegacyLayoutString(rank));

        ov::preprocess::OutputInfo& outputInfo = preprocessor.output(resultIndex);
        outputInfo.tensor().set_layout(tensorLayout);
        outputInfo.model().set_layout(modelLayout);
        outputInfo.tensor().set_element_type(outputPrecisions.at(outputID));
    }

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
        : _options(std::make_shared<intel_npu::OptionsDesc>()),
          _compilerDesc(*compilerDesc),
          _deviceDesc(*deviceDesc),
          _logger(vclLogger) {
    // Prepare default compilation configs
    registerCommonOptions(*_options);
    registerCompilerOptions(*_options);
    registerRunTimeOptions(*_options);

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
    std::shared_ptr<ov::Model> model = buildInfo.model;
    VPUXExecutableL0* exe = nullptr;
    StopWatch stopWatch;
    if (buildInfo.enableProfiling) {
        // Output time cost on vcl level
        stopWatch.start();
    }
    try {
        bool isNewAPI = false;
        int64_t irVersion = OLDEST_IR_VERSION_SUPPORTED;
        bool useIndices = false;

        ov::RTMap& runtimeInfoMap = model->get_rt_info();
        const auto& isNewAPIMatch = runtimeInfoMap.find("is_new_api");
        if (isNewAPIMatch != runtimeInfoMap.end()) {
            isNewAPI = isNewAPIMatch->second.as<bool>();
        }

        const auto& irVersionMatch = runtimeInfoMap.find("version");
        if (irVersionMatch != runtimeInfoMap.end()) {
            irVersion = irVersionMatch->second.as<int64_t>();
        }

        const auto& useIndicesMatch = runtimeInfoMap.find("use_indices_for_io_metadata");
        if (useIndicesMatch != runtimeInfoMap.end()) {
            useIndices = useIndicesMatch->second.as<bool>();
        }

        if (!isNewAPI || irVersion < 11) {
            model = preprocessModel(model, buildInfo.inputPrecisions, buildInfo.outputPrecisions,
                                    buildInfo.inputLayouts, buildInfo.outputLayouts, useIndices);
        }

        // Call compiler to compile the model and create blob
        // Create executable with the result NetworkDescription, profiling option and logger
        // Note we rely on implicit move semantics thanks to compile result being an rvalue,
        // failure to move here would lead to a blob copy!
        auto network = std::make_shared<const NetworkDescription>(_compiler->compile(model, buildInfo.parsedConfig));

        exe = new VPUXExecutableL0(network, buildInfo.enableProfiling, _logger);
    } catch (const std::exception& error) {
        _logger->outputError(formatv("{0}", error.what()));
        return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
    } catch (...) {
        _logger->outputError("Internal exception! Can not compile!");
        return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
    }

    if (buildInfo.enableProfiling) {
        stopWatch.stop();
        _logger->info("Compile net time: {0} ms", stopWatch.delta_ms());
    }

    return std::pair<VPUXExecutableL0*, vcl_result_t>(exe, VCL_RESULT_SUCCESS);
}

namespace {

class VCLBlobAllocator : public BlobAllocator {
public:
    explicit VCLBlobAllocator(const vcl_allocator_t* driverAllocator): allocator(driverAllocator) {
    }
    uint8_t* allocate(Byte size) override {
        return allocator->allocate(static_cast<uint64_t>(size.count()));
    }
    void deallocate(uint8_t* ptr) override {
        allocator->deallocate(ptr);
    }

private:
    const vcl_allocator_t* allocator;
};

}  // namespace

NetworkDescriptionView VPUXCompilerL0::importNetwork(BuildInfo& buildInfo, const vcl_allocator_t* allocator) {
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

    auto model = buildInfo.model;
    auto& runtimeInfoMap = model->get_rt_info();

    bool isNewAPI = false;
    if (const auto isNewAPIMatch = runtimeInfoMap.find("is_new_api"); isNewAPIMatch != runtimeInfoMap.end()) {
        isNewAPI = isNewAPIMatch->second.as<bool>();
    }

    int64_t irVersion = OLDEST_IR_VERSION_SUPPORTED;
    if (const auto irVersionMatch = runtimeInfoMap.find("version"); irVersionMatch != runtimeInfoMap.end()) {
        irVersion = irVersionMatch->second.as<int64_t>();
    }

    bool useIndices = false;
    if (const auto useIndicesMatch = runtimeInfoMap.find("use_indices_for_io_metadata");
        useIndicesMatch != runtimeInfoMap.end()) {
        useIndices = useIndicesMatch->second.as<bool>();
    }

    if (!isNewAPI || irVersion < 11) {
        model = preprocessModel(model, buildInfo.inputPrecisions, buildInfo.outputPrecisions, buildInfo.inputLayouts,
                                buildInfo.outputLayouts, useIndices);
    }

    auto vclAllocator = VCLBlobAllocator{allocator};
    return _compiler->compile(model, buildInfo.parsedConfig, vclAllocator);
}

vcl_result_t VPUXCompilerL0::queryNetwork(const BuildInfo& buildInfo, VPUXQueryNetworkL0* pQueryNetwork) {
    _logger->info("Start to call query function from compiler to get supported layers!");
    ov::SupportedOpsMap queryNetworkResult;
    try {
        queryNetworkResult = _compiler->query(buildInfo.model, buildInfo.parsedConfig);
    } catch (const std::exception& error) {
        _logger->outputError(error.what());
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

}  // namespace VPUXDriverCompiler
