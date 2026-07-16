//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/compiler.hpp"
#include <optional>
#include <utility>

#include "intel_npu/config/options.hpp"
#include "intel_npu/profiling.hpp"
#include "vpux/compiler/icompiler.hpp"
#ifdef ENABLE_PROFILING_ITT
#include <ittnotify.h>  // __itt_release_resources
#endif

#include "vpux/compiler/NPU37XX/backend_pipeline_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU40XX/backend_pipeline_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU50XX/backend_pipeline_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/compilation_options.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/ELF/IR/export.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/export.hpp"
#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/network_description.hpp"
#include "vpux/compiler/dialect/bytecode/IR/serialize.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/constant_transformations_control.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_in_background.hpp"
#include "vpux/compiler/dialect/net/IR/dialect.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/frontend/IE.hpp"
#include "vpux/compiler/frontend/model_preprocessor.hpp"
#include "vpux/compiler/frontend/ov_batch_detection.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"
#include "vpux/compiler/init/hw_strategy_registry.hpp"
#include "vpux/compiler/init/interfaces_registry.hpp"
#include "vpux/compiler/init/singleton_initializer.hpp"
#include "vpux/compiler/pipelines/developer_config.hpp"
#include "vpux/compiler/pipelines/options_mapper.hpp"
#include "vpux/compiler/profiling.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/pipeline_strategies.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/memory_usage.hpp"
#include "vpux/utils/ov/itt.hpp"
#include "vpux/utils/ov/private_properties.hpp"
#include "vpux/utils/ov/profiling_utils.hpp"
#include "vpux/utils/profiling/reports/api.hpp"

#include "npu_bytecode_utils/magic_number.hpp"
#include "npu_bytecode_utils/network_description.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/Timing.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/ManagedStatic.h>  // llvm_shutdown
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ThreadPool.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>

#include <openvino/core/dimension.hpp>
#include <openvino/opsets/opset.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/runtime/iplugin.hpp>
#include <openvino/runtime/shared_buffer.hpp>
#include <openvino/util/xml_parse_utils.hpp>

#include <transformations/common_optimizations/dimension_tracking.hpp>
#include <transformations/init_node_info.hpp>

using namespace vpux;

namespace {

constexpr std::string_view UNSUPPORTED_PLATFORM_ERROR_MESSAGE =
        "Unsupported platform: '{0}'\nThe current version of the compiler is unable to compile on the given platform. "
        "If you're using the compiler inside the plugin, please try using the '{1}' configuration option to set a "
        "supported platform explicitly.";

void checkPlatformSupportedForCompilation(const std::string_view platform) {
    const std::unordered_set supportedPlatforms{ov::intel_npu::Platform::NPU3720, ov::intel_npu::Platform::NPU5010,
                                                ov::intel_npu::Platform::NPU5020, ov::intel_npu::Platform::NPU4000};

    if (!supportedPlatforms.count(ov::intel_npu::Platform::standardize(platform))) {
        VPUX_THROW(UNSUPPORTED_PLATFORM_ERROR_MESSAGE.data(), platform, intel_npu::PLATFORM::key());
    }
}

//
// createDialectPipelineStrategyFn
//

StrategyFactoryFn createDialectPipelineStrategyFn(const intel_npu::Config& config) {
    auto arch = getArchKind(config);
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return [&](config::CompilationMode compilationMode) {
            return createDialectPipelineStrategy37XX(compilationMode, config);
        };
    case config::ArchKind::NPU40XX:
        return [&](config::CompilationMode compilationMode) {
            return createDialectPipelineStrategy40XX(compilationMode, config);
        };
    case config::ArchKind::NPU50XX:
        return [&](config::CompilationMode compilationMode) {
            return createDialectPipelineStrategy50XX(compilationMode, config);
        };
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}

//
// buildPipeline
//

void buildPipeline(const intel_npu::Config& config, mlir::PassManager& pm, Logger log) {
    auto pipelineStrategyFn = createDialectPipelineStrategyFn(config);

    const auto compilationMode = getCompilationMode(config);
    auto pipelineFactory = createPipelineFactory(compilationMode, std::move(pipelineStrategyFn), log);
    pipelineFactory->buildPipeline(pm);
}

//
// createBackendPipelineStrategy
//

std::unique_ptr<IBackendPipelineStrategy> createBackendPipelineStrategy(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<BackendPipelineStrategy37XX>();
    case config::ArchKind::NPU40XX:
        return std::make_unique<BackendPipelineStrategy40XX>();
    case config::ArchKind::NPU50XX:
        return std::make_unique<BackendPipelineStrategy50XX>();
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}

auto importNetwork(mlir::MLIRContext* ctx, const std::shared_ptr<ov::Model>& model,
                   const std::vector<std::shared_ptr<const ov::Node>>& originalParameters,
                   const std::vector<std::shared_ptr<const ov::Node>>& originalResults, const intel_npu::Config& config,
                   const DeveloperConfig& devConf, mlir::TimingScope& rootTiming, Logger log,
                   bool enableWeightsSeparationPath = false) {
    auto importTiming = rootTiming.nest("Import network");

    IE::ImportNetworkConfig importCfg;
    importCfg.sharedConstants = devConf.useSharedConstants();
    importCfg.enableProfiling = config.get<intel_npu::PERF_COUNT>();
    importCfg.stubLayers = getDummyOpReplacement(config).value_or(DummyOpMode::DISABLED);
    importCfg.dynamicShapeToStatic = config.get<intel_npu::DYNAMIC_SHAPE_TO_STATIC>();
    importCfg.enableWeightsSeparationPath = enableWeightsSeparationPath;
    importCfg.enableDecomposeSDPA = getEnableDecomposeSDPA(config).value_or(false);
    importCfg.executionModeAccuracy = config.get<intel_npu::EXECUTION_MODE_HINT>() == ov::hint::ExecutionMode::ACCURACY;
    importCfg.ioWithDynamicStrides = getIoWithDynamicStrides(config);

    return IE::importNetwork(ctx, model, originalParameters, originalResults, importTiming, importCfg, log.nest());
}

mlir::LogicalResult compileNetwork(mlir::ModuleOp module, mlir::PassManager& pm, mlir::TimingScope& nestTiming) {
    pm.enableTiming(nestTiming);
    return pm.run(module);
}

void middleendCompilation(mlir::OwningOpRef<mlir::ModuleOp>& module, const DeveloperConfig& devConf,
                          const intel_npu::Config& config, mlir::TimingScope& compileNetworkTiming, Logger log) {
    auto initIEtoVPUIPTiming = compileNetworkTiming.nest("IE to VPUIP pipeline");

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    devConf.setup(pm, config);

    buildPipeline(config, pm, log);

    auto compileResult = compileNetwork(module.get(), pm, initIEtoVPUIPTiming);
    VPUX_THROW_WHEN(mlir::failed(compileResult), "Compilation failed");

    devConf.dump(pm);
}

void backendHostCompilation(mlir::OwningOpRef<mlir::ModuleOp>& hostModule, const DeveloperConfig& devConf,
                            const intel_npu::Config& config, mlir::TimingScope& compileNetworkTiming, Logger log) {
    auto hostExecTiming = compileNetworkTiming.nest("HostExec pipeline");
    mlir::PassManager hostExecPm(hostModule.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    devConf.setup(hostExecPm, config);

    auto enablePipelinedCmdListRecording =
            getEnablePipelinedCmdListRecording(config).value_or(HostExec::defaultEnablePipelinedCmdListRecording);
    vpux::HostExec::buildHostExecPipeline(hostExecPm, enablePipelinedCmdListRecording, log);

    auto compileResult = compileNetwork(hostModule.get(), hostExecPm, hostExecTiming);
    VPUX_THROW_WHEN(mlir::failed(compileResult), "Compilation failed");
}

void backendBytecodeCompilation(mlir::OwningOpRef<mlir::ModuleOp>& hostModule, const DeveloperConfig& devConf,
                                const intel_npu::Config& config, mlir::TimingScope& compileNetworkTiming, Logger log) {
    auto bytecodeTiming = compileNetworkTiming.nest("Bytecode backend pipeline");
    mlir::PassManager bytecodePm(hostModule.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    devConf.setup(bytecodePm, config);
    vpux::HostExec::buildBytecodeBackendPipeline(bytecodePm, log);

    auto compileResult = compileNetwork(hostModule.get(), bytecodePm, bytecodeTiming);
    VPUX_THROW_WHEN(mlir::failed(compileResult), "Compilation failed");
}

void backendCompilation(mlir::OwningOpRef<mlir::ModuleOp>& vpuipModule, const DeveloperConfig& devConf,
                        const intel_npu::Config& config, mlir::TimingScope& compileNetworkTiming, Logger log,
                        bool nested = false) {
    auto elfTiming = compileNetworkTiming.nest("ELF pipeline");
    mlir::PassManager elfPm(vpuipModule.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    devConf.setup(elfPm, config, /*isSubPipeline=*/nested);
    auto backendPipelineStrategy = createBackendPipelineStrategy(getArchKind(config));
    if (nested) {
        backendPipelineStrategy->buildELFPipeline(elfPm.nest<mlir::ModuleOp>(), config, elfTiming, log);
    } else {
        backendPipelineStrategy->buildELFPipeline(elfPm, config, elfTiming, log);
    }
    const auto compileResult = compileNetwork(vpuipModule.get(), elfPm, elfTiming);
    VPUX_THROW_WHEN(mlir::failed(compileResult), "Compilation failed");
}

auto exportToELF(mlir::ModuleOp module, Logger& log) {
    const auto arch = config::getArch(module);
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return vpux::ELFNPU37XX::exportToELF(module, log);
    default:
        return vpux::ELF::exportToELF(module, log);
    }
}

auto exportToELF(mlir::ModuleOp module, Logger log, BlobAllocator& allocator, std::string& compatibilityString,
                 bool allocateCompatibilityString = false) {
    const auto arch = config::getArch(module);
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return ELFNPU37XX::exportToELF(module, allocator, compatibilityString, log, allocateCompatibilityString);
    default:
        return ELF::exportToELF(module, allocator, compatibilityString, log, allocateCompatibilityString);
    }
}

vpux::BlobView exportLLVM(mlir::ModuleOp module, BlobAllocator& allocator) {
    std::string streamStr;
    llvm::raw_string_ostream output(streamStr);
    output << *module;

    auto blob = allocator.allocate(vpux::Byte{static_cast<int64_t>(streamStr.size())});
    memcpy(blob, streamStr.c_str(), streamStr.size());

    return {blob, static_cast<uint64_t>(streamStr.size())};
}

auto exportLLVM(mlir::ModuleOp module) {
    std::string streamStr;
    llvm::raw_string_ostream output(streamStr);

    output << *module;

    auto blob = std::vector<uint8_t>(streamStr.size());
    memcpy(blob.data(), streamStr.c_str(), streamStr.size());

    return blob;
}

BlobView exportBytecode(mlir::ModuleOp module, BlobAllocator& allocator, Logger /*log*/) {
    std::string streamStr;
    llvm::raw_string_ostream output(streamStr);

    VPUX_THROW_WHEN(mlir::failed(bytecode::serializeTo(module, output)), "Failed to serialize bytecode module");
    output.flush();

    auto blob = allocator.allocate(vpux::Byte{static_cast<int64_t>(streamStr.size())});
    memcpy(blob, streamStr.c_str(), streamStr.size());

    return {blob, static_cast<uint64_t>(streamStr.size())};
}

auto exportBytecode(mlir::ModuleOp module, Logger /*log*/) {
    std::string streamStr;
    llvm::raw_string_ostream output(streamStr);

    VPUX_THROW_WHEN(mlir::failed(bytecode::serializeTo(module, output)), "Failed to serialize bytecode module");
    output.flush();

    auto blob = std::vector<uint8_t>(streamStr.size());
    memcpy(blob.data(), streamStr.c_str(), streamStr.size());

    return blob;
}

// Temporary compatibility adapter for HostCompile_Interpreter metadata path.
// OpenVINO and VPUX compiler currently keep equivalent metadata structures in separate namespaces.
// These local converters bridge the boundary until a follow-up refactor unifies metadata ownership.

vpux::IODescriptor toVPUXDescriptor(const intel_npu::vm::IODescriptor& src) {
    vpux::IODescriptor dst;
    dst.nameFromCompiler = src.nameFromCompiler;
    dst.precision = src.precision;
    dst.shapeFromCompiler = src.shapeFromCompiler;
    dst.isStateInput = src.isStateInput;
    dst.isStateOutput = src.isStateOutput;
    dst.isShapeTensor = src.isShapeTensor;
    dst.isInitInputWeights = src.isInitInputWeights;
    dst.isInitOutputWeights = src.isInitOutputWeights;
    dst.isMainInputWeights = src.isMainInputWeights;
    dst.relatedDescriptorIndex = src.relatedDescriptorIndex;
    dst.nodeFriendlyName = src.nodeFriendlyName;
    dst.outputTensorNames = src.outputTensorNames;
    dst.shapeFromIRModel = src.shapeFromIRModel;
    dst.indexUsedByDriver = src.indexUsedByDriver;
    dst.supportsStridedLayout = src.supportsStridedLayout;
    return dst;
}

vpux::NetworkMetadata toVPUXMetadata(const intel_npu::vm::NetworkMetadata& src) {
    vpux::NetworkMetadata dst;
    dst.name = src.name;
    dst.numStreams = src.numStreams;

    dst.inputs.reserve(src.inputs.size());
    for (const auto& desc : src.inputs) {
        dst.inputs.push_back(toVPUXDescriptor(desc));
    }

    dst.outputs.reserve(src.outputs.size());
    for (const auto& desc : src.outputs) {
        dst.outputs.push_back(toVPUXDescriptor(desc));
    }

    dst.profilingOutputs.reserve(src.profilingOutputs.size());
    for (const auto& desc : src.profilingOutputs) {
        dst.profilingOutputs.push_back(toVPUXDescriptor(desc));
    }

    return dst;
}

NetworkDescription exportNetwork(mlir::ModuleOp module, const intel_npu::Config& config, Logger log) {
    const auto compilationMode = getCompilationMode(config);
    NetworkMetadata meta;
    std::vector<uint8_t> blob;
    switch (compilationMode) {
    case config::CompilationMode::HostCompile:
    case config::CompilationMode::HostCompile_JIT:
        blob = exportLLVM(module);
        meta = VPUMI37XX::getHostCompileNetworkMetadata(module);
        break;
    case config::CompilationMode::HostCompile_Interpreter: {
        blob = exportBytecode(module, log);
        auto metadata = intel_npu::vm::parseNetworkMetadata(blob.data(), blob.size());
        VPUX_THROW_UNLESS(metadata.has_value(), "Failed to parse metadata from bytecode");
        meta = toVPUXMetadata(metadata.value());
        break;
    }
    default:
        blob = exportToELF(module, log);
        meta = VPUMI37XX::getNetworkMetadata(blob);
    }
    return NetworkDescription(std::move(blob), std::move(meta));
}

NetworkDescriptionView exportNetwork(mlir::ModuleOp module, const intel_npu::Config& config, Logger log,
                                     BlobAllocator& allocator, bool allocateCompatibilityString = false) {
    const auto compilationMode = getCompilationMode(config);

    switch (compilationMode) {
    case config::CompilationMode::HostCompile:
    case config::CompilationMode::HostCompile_JIT: {
        auto blob = exportLLVM(module, allocator);
        auto meta = VPUMI37XX::getHostCompileNetworkMetadata(module);
        return NetworkDescriptionView(blob, std::move(meta));
    }
    case config::CompilationMode::HostCompile_Interpreter: {
        auto blob = exportBytecode(module, allocator, log);
        auto metadata = intel_npu::vm::parseNetworkMetadata(blob.ptr, static_cast<size_t>(blob.size));
        VPUX_THROW_UNLESS(metadata.has_value(), "Failed to parse metadata from bytecode");
        auto meta = toVPUXMetadata(metadata.value());
        return NetworkDescriptionView(blob, std::move(meta));
    }
    default: {
        std::string compatibilityString;
        auto blobView = exportToELF(module, log, allocator, compatibilityString, allocateCompatibilityString);
        auto meta = VPUMI37XX::getNetworkMetadata(
                mlir::ArrayRef(blobView.first.ptr, static_cast<size_t>(blobView.first.size)));
        meta.compatibilityString = std::move(compatibilityString);
        return NetworkDescriptionView(blobView.first, blobView.second, std::move(meta));
    }
    }
}

std::optional<size_t> getModelBatchPartitionIfPossible(const std::shared_ptr<ov::Model>& model,
                                                       const intel_npu::Config& config) {
    std::set<ov::Output<const ov::Node>> batchedInputs;
    std::set<ov::Output<const ov::Node>> batchedOutputs;
    std::set<size_t> sBatchSize;

    vpux::Logger logger("getBatchSize", getLogLevel(config));

    if (!config.has<intel_npu::BATCH_MODE>()) {
        return std::make_optional<size_t>(1);
    }

    if (config.get<intel_npu::BATCH_MODE>() == ov::intel_npu::BatchMode::COMPILER) {
        return std::make_optional<size_t>();
    }

    std::stringstream sstreamInconsistencyDescr;
    if (!checkCfgOnBatchOptionConsistency(config, sstreamInconsistencyDescr)) {
        VPUX_THROW("Incompatible options have been detected: {0}", sstreamInconsistencyDescr.str());
    }

    std::shared_ptr<ov::Model> testBatchModel = model->clone();
    // Find the batch dim
    ov::pass::Manager passManager;
    passManager.register_pass<ov::pass::InitNodeInfo>();
    passManager.register_pass<ov::pass::FindBatch>();
    passManager.run_passes(testBatchModel);
    // Do not reshape/re-batch originally batched networks and when there are no inputs with the N* layouts
    // input(s) should have the batch dim as the first dim (current limitation of the auto-batching impl)
    const auto& params = testBatchModel->get_parameters();
    for (size_t input_id = 0; input_id < params.size(); input_id++) {
        const auto& input = params[input_id];
        const auto& shape = input->get_partial_shape();
        ov::Layout layout = ov::layout::get_layout(input);
        // Batching on plugin is working only when batching is found on 0th dimension
        if ((shape.size() && shape[0].has_symbol()) ||
            (ov::layout::has_batch(layout) && ov::layout::batch_idx(layout) == 0)) {
            const auto& staticShape = shape.is_dynamic() ? shape.get_max_shape() : input->get_shape();
            batchedInputs.insert(params[input_id]->output(0));
            if (shape.rank().is_dynamic()) {
                VPUX_THROW("Shapes with dynamic rank are not supported.");
            } else {
                sBatchSize.insert(staticShape[0]);
            }
        } else {
            // gather some diagnostic info
            std::optional<int> batch_dim_index_detected;
            for (size_t i = 1; i < shape.size(); i++) {
                if (shape[i].has_symbol()) {
                    batch_dim_index_detected = i;
                    break;
                }
            }
            std::stringstream sstream;
            sstream << "Only networks with inputs batched by 0th dimension are supported. ";
            if (batch_dim_index_detected.has_value()) {
                sstream << "The batch has been detected on: " << batch_dim_index_detected.value()
                        << " dimension instead. ";
            } else {
                sstream << "The batch hasn't been detected at all. ";
            }
            sstream << "Please check input id: " << input_id << " by the name: " << input->get_friendly_name()
                    << ", layout: " << layout.to_string() << ", is_dynamic: " << shape.is_dynamic();
            logger.info("{0}", sstream.str());
            return std::nullopt;
        }
    }
    for (const auto& output : testBatchModel->get_results()) {
        const auto& shape = output->get_output_partial_shape(0);
        ov::Layout layout = ov::layout::get_layout(output);
        // Batching on plugin is working only when batching is found on 0th dimension
        if ((shape.size() && shape[0].has_symbol()) ||
            (ov::layout::has_batch(layout) && ov::layout::batch_idx(layout) == 0)) {
            const auto& node = output->input_value(0);
            const auto& staticShape = shape.is_dynamic() ? shape.get_max_shape() : output->get_shape();
            batchedOutputs.insert(ov::Output<const ov::Node>(node.get_node(), node.get_index()));
            if (shape.rank().is_dynamic()) {
                VPUX_THROW("Shapes with dynamic rank are not supported.");
            } else {
                sBatchSize.insert(staticShape[0]);
            }
        } else {
            logger.info("Only networks with outputs batched by 0th dimension are supported. Please check an output by "
                        "the name: {0}, layout: {1}",
                        output->get_friendly_name(), layout.to_string());
            return std::nullopt;
        }
    }
    if (!batchedInputs.size() || !batchedOutputs.size()) {
        logger.info("Only networks with inputs/outputs featuring batched dim are supported! Got inputs: {0}, "
                    "outputs: {1}",
                    batchedInputs.size(), batchedOutputs.size());
        return std::nullopt;
    }

    if (sBatchSize.size() != 1) {
        logger.info("Batching size shall have same value for all tensors! Got unique batch sizes number: {0}",
                    sBatchSize.size());
        return std::nullopt;
    }

    auto node_info_printer = [&logger](const auto& ov_node, std::string_view nodeType) {
        logger.debug("{0}: {1} has shape value: {2}", nodeType, ov_node.get_any_name(),
                     ov_node.get_partial_shape().to_string());
    };
    for (const auto& ov_node : batchedInputs) {
        node_info_printer(ov_node, "Input");
    }
    for (const auto& ov_node : batchedOutputs) {
        node_info_printer(ov_node, "Output");
    }
    auto it = sBatchSize.begin();
    return *it;
}

bool isTypeSupportedNPU37xx(ov::element::Type_t elemType) {
    switch (elemType) {
    case ov::element::Type_t::dynamic:
    case ov::element::Type_t::boolean:
    case ov::element::Type_t::bf16:
    case ov::element::Type_t::f16:
    case ov::element::Type_t::f32:
    case ov::element::Type_t::f64:
    case ov::element::Type_t::i4:
    case ov::element::Type_t::i8:
    case ov::element::Type_t::i16:
    case ov::element::Type_t::i32:
    case ov::element::Type_t::i64:
    case ov::element::Type_t::u4:
    case ov::element::Type_t::u8:
    case ov::element::Type_t::u16:
    case ov::element::Type_t::u32:
    case ov::element::Type_t::u64:
        return true;
    default:
        return false;
    }
}

// Supported NPU40xx types: dynamic, boolean, bf16, f16, f32, f64, i4, i8, i16, i32, i64, u2, u4, u8, u16, u32, u64,
// nf4
bool isTypeSupportedNPU40xx(ov::element::Type_t elemType) {
    switch (elemType) {
    case ov::element::Type_t::u2:
    case ov::element::Type_t::nf4:
        return true;
    default:
        return isTypeSupportedNPU37xx(elemType);
    }
}

// Supported NPU50xx types: dynamic, boolean, bf16, f16, f32, f64, i4, i8, i16, i32, i64, u2, u4, u8, u16, u32, u64,
//                          nf4, f8e4m3, f8e5m2
bool isTypeSupportedNPU50xx(ov::element::Type_t elemType) {
    switch (elemType) {
    case ov::element::Type_t::f8e4m3:
    case ov::element::Type_t::f8e5m2:
        return true;
    default:
        return isTypeSupportedNPU40xx(elemType);
    }
}

bool isTypeSupported(config::ArchKind arch, ov::element::Type_t elemType) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return isTypeSupportedNPU37xx(elemType);
    case config::ArchKind::NPU40XX:
        return isTypeSupportedNPU40xx(elemType);
    case config::ArchKind::NPU50XX:
        return isTypeSupportedNPU50xx(elemType);
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
};

void checkDataTypes(const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config) {
    const auto archKind = getArchKind(config);
    for (auto& input : model->inputs()) {
        auto elemType = input.get_element_type();
        VPUX_THROW_UNLESS(isTypeSupported(archKind, elemType), "Unsupported data type '{0}'", elemType.get_type_name());
    }
    for (auto& op : model->get_ops()) {
        auto elemType = op->get_output_element_type(0);
        VPUX_THROW_UNLESS(isTypeSupported(archKind, elemType), "Unsupported data type '{0}'", elemType.get_type_name());
    }
}

mlir::OwningOpRef<mlir::ModuleOp> compileModel(mlir::MLIRContext& ctx, const std::shared_ptr<ov::Model>& model,
                                               const std::vector<std::shared_ptr<const ov::Node>>& originalParameters,
                                               const std::vector<std::shared_ptr<const ov::Node>>& originalResults,
                                               DeveloperConfig& devConf, mlir::TimingScope& rootTiming,
                                               const intel_npu::Config& config, vpux::Logger& log) {
    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "compileModel");

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "importNetwork");

    checkDataTypes(model, config);

    mlir::OwningOpRef<mlir::ModuleOp> module =
            importNetwork(&ctx, model, originalParameters, originalResults, config, devConf, rootTiming, log,
                          /*enableWeightsSeparationPath=*/false);

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "PassManager");

#ifdef BACKGROUND_FOLDING_ENABLED
    const auto foldingConfig = getConstantFoldingInBackground(config);

    std::unique_ptr<vpux::Const::BackgroundConstantFolding> foldingManager;
    if (foldingConfig.has_value() && foldingConfig.value().foldingInBackgroundEnabled) {
        foldingManager = std::make_unique<vpux::Const::BackgroundConstantFolding>(
                &ctx, foldingConfig.value().maxConcurrentTasks, foldingConfig.value().collectStatistics,
                foldingConfig.value().memoryUsageLimit, foldingConfig.value().cacheCleanThreshold, log);
    }
#endif

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "compileNetwork");

    // applies each pass in the pipeline
    auto compileNetworkTiming = rootTiming.nest("Compile network");

    middleendCompilation(module, devConf, config, compileNetworkTiming, log);

    const auto compilationMode = getCompilationMode(config);
    if (config::isHostCompileMode(compilationMode)) {
        // Host compilation pipeline requires lowering VPUIP to ELF for submodules as a graph is
        // split into subgraphs. The ELF pipeline is nested under a child ModuleOp anchor so MLIR
        // dispatches it to each submodule concurrently when multi-threading is enabled.
        backendCompilation(module, devConf, config, compileNetworkTiming, log, /*nested=*/true);
        if (compilationMode == config::CompilationMode::HostCompile_Interpreter) {
            backendBytecodeCompilation(module, devConf, config, compileNetworkTiming, log);
        } else {
            backendHostCompilation(module, devConf, config, compileNetworkTiming, log);
        }
    } else {
        backendCompilation(module, devConf, config, compileNetworkTiming, log);
    }

    return module;
}

struct CompilationResult {
    // OV model needs be alive until serialization (export) of moduleOp finishes
    // as serialization may access constants directly from OV model
    // store & return from compileImpl these 2 together to ensure correct lifetime
    mlir::OwningOpRef<mlir::ModuleOp> moduleOp;
    std::shared_ptr<ov::Model> ovModel;

    CompilationResult(mlir::OwningOpRef<mlir::ModuleOp> mlirModule, std::shared_ptr<ov::Model> model)
            : moduleOp(std::move(mlirModule)), ovModel(std::move(model)) {
    }
};

auto createContext(mlir::DialectRegistry& registry, config::Platform platform) {
    auto interfacesRegistry = createInterfacesRegistry(platform);
    interfacesRegistry->registerInterfaces(registry);
    return std::make_unique<mlir::MLIRContext>(registry, mlir::MLIRContext::Threading::DISABLED);
}

std::unique_ptr<llvm::DefaultThreadPool> createThreadpool(const Logger& log, const intel_npu::Config& config) {
    // Set the number of threads in the pool to be the total number of threads of the compilation minus one: one for
    // the main thread and the rest for the MLIR thread pool. If user didn't specify the number of threads, default
    // to 8 threads for the pool. By default MLIR will attempt to use all of the threads available on the system
    // which might cause large peak memory usage during constant-related passes such as constant-folding, hence a
    // limit is set
    const bool hasThreadLimit = config.has<intel_npu::COMPILATION_NUM_THREADS>();
    const auto totalThreadCount = hasThreadLimit ? config.get<intel_npu::COMPILATION_NUM_THREADS>() : 9;

    log.info("Total number of threads available for compilation: {0}", totalThreadCount);

    if (totalThreadCount <= 1) {
        return nullptr;
    }

    llvm::ThreadPoolStrategy strategy;
    strategy.ThreadsRequested = totalThreadCount - 1;
    strategy.Limit = true;  // limits number of threads to the number of physical threads

    return std::make_unique<llvm::DefaultThreadPool>(strategy);
}

struct CompilerSetup {
    std::unique_ptr<llvm::DefaultThreadPool> threadPool;  // The thread pool must outlive the context
    mlir::DialectRegistry registry;
    std::unique_ptr<mlir::MLIRContext> ctx;

    static std::unique_ptr<CompilerSetup> create(const intel_npu::Config& config, Logger& log);
    ~CompilerSetup() = default;

    CompilerSetup(CompilerSetup&&) = default;
    CompilerSetup& operator=(CompilerSetup&&) = default;

private:
    CompilerSetup(const intel_npu::Config& config, Logger& log);
    CompilerSetup(const CompilerSetup&) = delete;
    CompilerSetup& operator=(const CompilerSetup&) = delete;
    CompilerSetup& operator()(const CompilerSetup&) = delete;
};

std::unique_ptr<CompilerSetup> CompilerSetup::create(const intel_npu::Config& config, Logger& log) {
    return std::unique_ptr<CompilerSetup>(new CompilerSetup(config, log));
}

CompilerSetup::CompilerSetup(const intel_npu::Config& config, Logger& log) {
    registry = createDialectRegistry(getDummyOpReplacement(config).value_or(DummyOpMode::DISABLED));
    auto platform = getPlatform(config);
    config::registerConstraints(registry, platform);
    IE::registerStrategies(registry, platform);
    VPU::initializeSingletons(registry, platform);
    VPU::registerStrategies(registry, platform);
    VPUIP::registerStrategies(registry, platform);

    ctx = createContext(registry, platform);
    addLogging(*ctx, log);

    auto threadPoolLog = log.nest("compiler-threading");
    if ((threadPool = createThreadpool(threadPoolLog, config))) {
        threadPoolLog.info("Created an MLIR thread pool with {0} workers", threadPool->getMaxConcurrency());
        ctx->setThreadPool(*threadPool);
    }

    threadPoolLog.info("MLIR multi-threading is {0}", ctx->isMultithreadingEnabled() ? "enabled" : "disabled");
}

std::tuple<std::shared_ptr<ov::Model>, const intel_npu::Config> debatchModel(const std::shared_ptr<ov::Model>& model,
                                                                             const intel_npu::Config& config,
                                                                             size_t partitionCount, Logger& log) {
    log.info("A batched model with batch: {0} is about to be processed by the plugin",
             partitionCount == ov::Interval::s_max ? "<Inf>" : std::to_string(partitionCount));
    // When batching is handled by the plugin we need to modify performance_mode property to Throughput mode
    auto configPerformanceMode = config;
    if (configPerformanceMode.get<intel_npu::PERFORMANCE_HINT>() == ov::hint::PerformanceMode::LATENCY) {
        log.info("Override performance mode to THROUGHPUT");
        std::stringstream strStream;
        strStream << ov::hint::PerformanceMode::THROUGHPUT;
        configPerformanceMode.update({{ov::hint::performance_mode.name(), strStream.str()}});
    }

    // If fallback and handle batching on the compiler is needed we will use the original model
    auto batchModel = model->clone();
    try {
        ov::set_batch(batchModel, 1);
    } catch (const std::exception& ex) {
        log.warning("The plugin couldn't resize a batched model due to exception: {0}.\nProbably, the "
                    "model is a dynamic model and layout hasn't been specified. Trying to debatch it...",
                    ex.what());
        batchModel = debatchDynamicModel(batchModel, log);
        if (!batchModel) {
            VPUX_THROW("Cannot debatch a model");
        }
        log.info("The model has been debatched successfully");
    }
    return make_tuple(batchModel, configPerformanceMode);
}

// leave reference to const std::shared_ptr<ov::Model> instead of taking std::shared_ptr<ov::Model> by value
// as in case of batching we don't copy pointer to ov::Model, we clone it and use clone afterwards
// taking by-value would mean extra copy of std::shared_ptr for no reason in this case, even though
// it's fine for "regular" scenario without batching (just 1 copy anyway)
CompilationResult compileImpl(std::unique_ptr<CompilerSetup>& setup, const std::shared_ptr<ov::Model>& model,
                              const intel_npu::Config& config, Logger& log) {
    checkPlatformSupportedForCompilation(config.get<intel_npu::PLATFORM>());

    DeveloperConfig devConf(log);

    mlir::DefaultTimingManager tm;
    devConf.setup(tm);

    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "compileImpl");

    auto rootTiming = tm.getRootScope();

    // Save the original model parameters and results before batching
    const auto originalParameters = IE::buildOVParams(model);
    const auto originalResults = IE::buildOVResults(model);

    try {
        auto partitionCount = getModelBatchPartitionIfPossible(model, config);
        if (partitionCount.has_value()) {
            if (*partitionCount > 1) {
                auto [batchModel, configPerformanceMode] = debatchModel(model, config, *partitionCount, log);
                auto moduleOp = compileModel(*setup->ctx, batchModel, originalParameters, originalResults, devConf,
                                             rootTiming, configPerformanceMode, log);
                return CompilationResult{std::move(moduleOp), std::move(batchModel)};
            }
        } else {
            const auto& batchType = config.get<intel_npu::BATCH_MODE>();
            if (batchType == ov::intel_npu::BatchMode::AUTO) {
                log.info("Batching is handled by the compiler");
            } else if (batchType == ov::intel_npu::BatchMode::PLUGIN) {
                VPUX_THROW("This model is not supported when handling batching on the plugin.");
            }
        }
    } catch (const std::exception& ex) {
        const auto& batchType = config.get<intel_npu::BATCH_MODE>();
        if (batchType == ov::intel_npu::BatchMode::AUTO) {
            log.info("An error occurred during network compilation so fallback on compiler batch mode {0}", ex.what());
        } else {
            VPUX_THROW(ex.what());
        }
    }

    auto [newConfig, autoBatchingEnabled] = autoDetectBatchedModelIfPossible(model, config);
    if (autoBatchingEnabled) {
        try {
            // Try method "debatch" at first, because it supports more real models providing better performance
            // numbers than "unroll"
            auto moduleOp = compileModel(*setup->ctx, model, originalParameters, originalResults, devConf, rootTiming,
                                         newConfig, log);
            if (!moduleOp) {
                throw std::runtime_error("unknown error(a result model is empty)");
            }
            return CompilationResult{std::move(moduleOp), model};
        } catch (const std::exception& ex) {
            log.warning("Cannot compile a model using auto-batch compiler detection method, error: {0}\nTrying "
                        "default...",
                        ex.what());
            // TODO E####-160706
            // For simplicity we create a new MLIRContext as the old one may be spoiled and inconsisted as it is not
            // exception safety
            setup = CompilerSetup::create(config, log);
            auto moduleOp = compileModel(*setup->ctx, model, originalParameters, originalResults, devConf, rootTiming,
                                         config, log);
            return CompilationResult{std::move(moduleOp), model};
        }
    }
    auto moduleOp =
            compileModel(*setup->ctx, model, originalParameters, originalResults, devConf, rootTiming, config, log);
    return CompilationResult{std::move(moduleOp), model};
}

std::shared_ptr<ModelData> prepareAndPreprocessModelFromDesc(const vcl_executable_desc_t& desc,
                                                             const vcl_compiler_properties_t& compilerProp,
                                                             vcl_compiler_desc_t& compilerDesc,
                                                             vcl_device_desc_t& deviceDesc, intel_npu::Config& config,
                                                             bool isDeviceDescEmpty, Logger& log) {
    VPUX_THROW_WHEN(!desc.modelIRData || desc.modelIRSize == 0, "Invalid model IR data");

    /// To avoid access violation, need to convert to string
    std::string descOptions(desc.options, desc.optionsSize);
    log.info("config: {0}", descOptions);

    auto modelData = std::make_shared<ModelData>();
    /// Parse user descriptions and store the input && output settings, compilation configs
    auto [precisions, layouts] =
            prepareBuildFlags(descOptions, compilerDesc, compilerProp, deviceDesc, config, isDeviceDescEmpty);
    modelData->precisions = std::move(precisions);
    modelData->layouts = std::move(layouts);
    modelData->model = prepareModel(desc.modelIRData, desc.modelIRSize, config, compilerProp);

    [[maybe_unused]] const auto scopedTimer = startScopedTimer(config, [&log](double deltaMs) {
        log.info("Model preprocessing time: {0} ms", deltaMs);
    });
    preprocessModel(modelData);

    return modelData;
}

}  // namespace

CompilerImpl::CompilerImpl() {
    // Ensure llvm_shutdown is called to free LLVM's managed static objects.
    // llvm_shutdown must be called in a single thread while no LLVM APIs are in use
    // but CompilerImpl dtor and vclCompilerDestroy may be called from multiple threads.
    // Use an object with static lifetime to get llvm_shutdown called on DLL unload
    // While this could same well be a global object, block scoped static brings more
    // deterministic destruction order akin to calling llvm_shutdown from main as in LLVM
    // samples.
    [[maybe_unused]] static llvm::llvm_shutdown_obj shutdown;
#ifndef VPUX_DEVELOPER_BUILD
    [[maybe_unused]] static auto ignoreStdIOErr = llvm::make_scope_exit([]() {
        // Ignoring standard I/O errors; permission to write to stdout/stderr might be restricted (e.g. by ONNX
        // Runtime). If these errors persist, they can cause the program to exit unexpectedly. clearing them ensures a
        // clean shutdown.
        llvm::errs().clear_error();
        llvm::outs().clear_error();
    });
#endif
#ifdef ENABLE_PROFILING_ITT
    // Release ITT resources on DLL unload
    [[maybe_unused]] static auto ittResourceReleaser = llvm::make_scope_exit(__itt_release_resources);
#endif
}

//
// CompilerImpl::query
//

ov::SupportedOpsMap vpux::CompilerImpl::query(const std::shared_ptr<const ov::Model>& model,
                                              const intel_npu::Config& config) const {
    Logger log("vpux-compiler", getLogLevel(config));
    log.setName("vpux::CompilerImpl::query");

    ov::SupportedOpsMap result;

    const std::string plugin_name = DEVICE_NAME;

    DeveloperConfig devConf(log);
    mlir::DefaultTimingManager tm;
    devConf.setup(tm);
    auto rootTiming = tm.getRootScope();
    // query is executed with the default values of the config
    IE::ImportNetworkConfig importCfg;

    log.trace("Get supported nodes.");
    auto supportedNodes = ov::get_supported_nodes(
            model,
            [&](const std::shared_ptr<ov::Model>& model) {
                log.trace("Run common nGraph passes.");
                IE::NGraphPasses::runNGraphPasses(model, rootTiming, importCfg);
            },
            [&](const std::shared_ptr<ov::Node>& op) {
                log.trace("Get supported operations list.");
                return IE::NGraphImporter::isOpSupported(op);
            });

    for (auto&& layerName : supportedNodes) {
        result.emplace(layerName, plugin_name);
    }

    return result;
}

//
// CompilerImpl::queryFromDesc
//

ov::SupportedOpsMap vpux::CompilerImpl::queryFromDesc(const vcl_query_desc_t& desc, vcl_compiler_desc_t& compilerDesc,
                                                      vcl_compiler_properties_t& compilerProp,
                                                      vcl_device_desc_t& deviceDesc, intel_npu::Config& config,
                                                      bool isDeviceDescEmpty) const {
    Logger log("vpux-compiler", getLogLevel(config));
    log.setName("vpux::CompilerImpl::queryFromDesc");

    VPUX_THROW_WHEN(!desc.modelIRData || desc.modelIRSize == 0, "Invalid model IR data");

    /// To avoid access violation, need to convert to string
    std::string descOptions(desc.options, desc.optionsSize);
    log.info("config: {0}", descOptions);

    /// Parse and update the config
    prepareConfig(descOptions, compilerDesc, deviceDesc, config, isDeviceDescEmpty);
    auto model = prepareModel(desc.modelIRData, desc.modelIRSize, config, compilerProp);
    return query(model, config);
}

//
// CompilerImpl::compile
//

NetworkDescription CompilerImpl::compile(const std::shared_ptr<ov::Model>& model,
                                         const intel_npu::Config& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compile");
    checkPlatformSupportedForCompilation(config.get<intel_npu::PLATFORM>());
    checkCompilerOptions(config);

    Logger log("vpux-compiler", getLogLevel(config));

    [[maybe_unused]] const auto scopedTimer = startScopedTimer(config, [&log](double deltaMs) {
        log.info("Compile net time: {0} ms", deltaMs);
    });

    auto setup = CompilerSetup::create(config, log);
    auto peakMemStart = getPeakMemoryUsage();
    auto compilationResult = compileImpl(setup, model, config, log);

    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "exportNetwork");
    auto networkDescription = exportNetwork(compilationResult.moduleOp.get(), config, log);

    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);

    auto peakMemEnd = getPeakMemoryUsage();
    log.debug("Start of compilation memory usage: Peak {0} KB", peakMemStart.count());
    log.debug("End of compilation memory usage: Peak {0} KB", peakMemEnd.count());
    // Note: Following log is parsed by CI. Take care when modifying it.
    log.info("Compilation memory usage: Peak {0} KB", peakMemEnd.count() - peakMemStart.count());

    return networkDescription;
}

NetworkDescriptionView CompilerImpl::compileFromDesc(const vcl_executable_desc_t& desc,
                                                     const vcl_compiler_properties_t& compilerProp,
                                                     vcl_compiler_desc_t& compilerDesc, vcl_device_desc_t& deviceDesc,
                                                     intel_npu::Config& config, bool isDeviceDescEmpty,
                                                     BlobAllocator& allocator, bool generateCompatibilityString) const {
    Logger log("vpux-compiler", getLogLevel(config));

    auto modelData = prepareAndPreprocessModelFromDesc(desc, compilerProp, compilerDesc, deviceDesc, config,
                                                       isDeviceDescEmpty, log);

    return config.get<intel_npu::ENABLE_WEIGHTLESS>()
                   ? compileWsIterative(modelData->model, config, config.get<intel_npu::WS_COMPILE_CALL_NUMBER>(),
                                        allocator)
                   : compile(modelData->model, config, allocator, generateCompatibilityString);
}

NetworkDescriptionView CompilerImpl::compile(const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config,
                                             BlobAllocator& allocator, bool allocateCompatibilityString) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compile");
    checkPlatformSupportedForCompilation(config.get<intel_npu::PLATFORM>());
    checkCompilerOptions(config);

    Logger log("vpux-compiler", getLogLevel(config));

    [[maybe_unused]] const auto scopedTimer = startScopedTimer(config, [&log](double deltaMs) {
        log.info("Compile net time: {0} ms", deltaMs);
    });

    auto setup = CompilerSetup::create(config, log);
    auto peakMemStart = getPeakMemoryUsage();
    auto compilationResult = compileImpl(setup, model, config, log);

    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "exportNetwork");
    auto allocatedCompiledNetwork =
            exportNetwork(compilationResult.moduleOp.get(), config, log, allocator, allocateCompatibilityString);
    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);

    auto peakMemEnd = getPeakMemoryUsage();

    log.debug("Start of compilation memory usage: Peak {0} KB", peakMemStart.count());
    log.debug("End of compilation memory usage: Peak {0} KB", peakMemEnd.count());
    // Note: Following log is parsed by CI. Take care when modifying it.
    log.info("Compilation memory usage: Peak {0} KB", peakMemEnd.count() - peakMemStart.count());

    return allocatedCompiledNetwork;
}

NetworkDescriptionView CompilerImpl::compile(const std::shared_ptr<const ov::Model>& origModel,
                                             const intel_npu::Config& config, BlobAllocator& allocator,
                                             bool allocateCompatibilityString) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compile");
    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "clone_model");

    // NGraph pipeline modifies the model so need to clone here
    auto model = origModel->clone();

    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);

    return compile(model, config, allocator, allocateCompatibilityString);
}

NetworkDescription CompilerImpl::compile(const std::shared_ptr<const ov::Model>& origModel,
                                         const intel_npu::Config& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compile");
    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "clone_model");

    // NGraph pipeline modifies the model so need to clone here
    auto model = origModel->clone();

    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);

    return compile(model, config);
}

NetworkDescription CompilerImpl::compileFromDesc(const vcl_executable_desc_t& desc,
                                                 const vcl_compiler_properties_t& compilerProp,
                                                 vcl_compiler_desc_t& compilerDesc, vcl_device_desc_t& deviceDesc,
                                                 intel_npu::Config& config, bool isDeviceDescEmpty) const {
    Logger log("vpux-compiler", getLogLevel(config));

    auto modelData = prepareAndPreprocessModelFromDesc(desc, compilerProp, compilerDesc, deviceDesc, config,
                                                       isDeviceDescEmpty, log);

    return config.get<intel_npu::ENABLE_WEIGHTLESS>()
                   ? compileWsIterative(modelData->model, config, config.get<intel_npu::WS_COMPILE_CALL_NUMBER>())
                   : compile(modelData->model, config);
}

namespace ws {

void compileIEtoVPU(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp,
                    std::unique_ptr<IDialectPipelineStrategy>& pipelineStrategy, const DeveloperConfig& devConf,
                    const intel_npu::Config& config, const std::optional<std::string>& wsExtractionMode,
                    std::optional<int64_t> initPart, std::optional<Byte> memLimit, mlir::TimingScope& rootTiming,
                    Logger log) {
    mlir::PassManager ieToVPUpm(moduleOp.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    devConf.setup(ieToVPUpm, config);

    // The regular compilation flow is used to compile the model in the same way as the defaul pipeline to get Module in
    // the VPU dialect.
    const bool isRegularCompilationFlow = !wsExtractionMode.has_value();
    if (isRegularCompilationFlow) {
        pipelineStrategy->initializePipeline(ieToVPUpm, log);
    } else {
        VPUX_THROW_WHEN(wsExtractionMode.value() != "gen-init",
                        "Following pipeline extracts Init schedule, but wsExtractionMode has wrong value: {0}",
                        wsExtractionMode.value());
        ieToVPUpm.addPass(VPU::createConstructWsAnalysisPass(log));
        ieToVPUpm.addPass(VPU::createIntroduceInitFunctionPass(wsExtractionMode.value(), initPart, memLimit, log));
        ieToVPUpm.addPass(VPU::createConcatInitResultsPass(wsExtractionMode.value(), initPart, memLimit, log));
        ieToVPUpm.addPass(VPU::createDestructWsAnalysisPass(log));

        const auto grc = getDefaultGreedyRewriteConfig();
        ieToVPUpm.addPass(mlir::createCanonicalizerPass(grc));
    }

    auto ieToVPUTiming = rootTiming.nest("IE to VPU pipeline");
    pipelineStrategy->buildIEPipeline(ieToVPUpm, log);
    pipelineStrategy->buildLowerIE2VPUPipeline(ieToVPUpm, log);
    pipelineStrategy->buildVPUPipeline(ieToVPUpm, log);

    if (!wsExtractionMode.has_value()) {
        ieToVPUpm.addPass(VPU::createQueryWSInfoPass(memLimit, log));
    }

    VPUX_THROW_WHEN(mlir::failed(compileNetwork(moduleOp.get(), ieToVPUpm, ieToVPUTiming)), "Compilation failed");
}

void compileVPUIP(mlir::OwningOpRef<mlir::ModuleOp>& vpuModule,
                  std::unique_ptr<IDialectPipelineStrategy>& pipelineStrategy, const DeveloperConfig& devConf,
                  const intel_npu::Config& config, const std::optional<std::string>& wsExtractionMode,
                  std::optional<Byte> memLimit, mlir::TimingScope& nestTiming, Logger log) {
    mlir::PassManager vpuipPM(vpuModule.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    devConf.setup(vpuipPM, config);

    const bool isMainScheduleCompilation = wsExtractionMode.has_value();
    if (isMainScheduleCompilation) {
        VPUX_THROW_WHEN(wsExtractionMode.value() != "gen-main",
                        "Following pipeline extracts Main schedule, but wsExtractionMode has wrong value: {0}",
                        wsExtractionMode.value());
        vpuipPM.addPass(VPU::createConstructWsAnalysisPass(log));
        vpuipPM.addPass(VPU::createIntroduceInitFunctionPass(wsExtractionMode.value(), /* initPart = */ std::nullopt,
                                                             memLimit, log));
        vpuipPM.addPass(VPU::createConcatInitResultsPass(wsExtractionMode.value(), /* initPart = */ std::nullopt,
                                                         memLimit, log));
        vpuipPM.addPass(VPU::createDestructWsAnalysisPass(log));
    }

    pipelineStrategy->buildLowerVPU2VPUIPPipeline(vpuipPM, log);
    pipelineStrategy->buildVPUIPPipeline(vpuipPM, log);

    VPUX_THROW_WHEN(mlir::failed(compileNetwork(vpuModule.get(), vpuipPM, nestTiming)), "Compilation failed");
}

std::vector<CompilationResult> compileImplWsOneShot(
        std::unique_ptr<CompilerSetup>& setup, const std::vector<std::shared_ptr<const ov::Node>>& originalParameters,
        const std::vector<std::shared_ptr<const ov::Node>>& originalResults, const std::shared_ptr<ov::Model>& model,
        const intel_npu::Config& config, Logger& log) {
    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compileWsOneShot",
                      "ws::compileImplWsOneShot");

    checkPlatformSupportedForCompilation(config.get<intel_npu::PLATFORM>());

    mlir::MLIRContext* ctx = setup->ctx.get();
    DeveloperConfig devConf(log);

    mlir::DefaultTimingManager tm;
    devConf.setup(tm);
    Const::setLazyFoldingOptions(ctx, Const::getWsFoldingOptions());

    auto rootTiming = tm.getRootScope();

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "importNetwork");
    mlir::OwningOpRef<mlir::ModuleOp> moduleMain =
            importNetwork(ctx, model, originalParameters, originalResults, config, devConf, rootTiming, log,
                          /*enableWeightsSeparationPath=*/true);

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "compile");
    auto factoryMethod = createDialectPipelineStrategyFn(config);

    // This value determines how many Init schedules will be generated to enable the Init pipelining feature.
    // For more details, please refer to weights_separation.md.
    // E#176573: remove this property
    auto hardcodedMemoryLimit = vpux::Byte(800_MB);
    log.info("Init pipelining memory limit: {0}", hardcodedMemoryLimit);

    auto mainPipelineStrategy = factoryMethod(config::CompilationMode::WSMain);
    ws::compileIEtoVPU(moduleMain, mainPipelineStrategy, devConf, config, /* wsExtractionMode = */ std::nullopt,
                       /* initPart = */ std::nullopt, hardcodedMemoryLimit, rootTiming, log);

    auto totalInitPartCount = mlir::cast<mlir::IntegerAttr>(moduleMain.get()->getAttr("VPU.WsTotalInitPartCount"))
                                      .getValue()
                                      .getSExtValue();

    log.info("Total number of Inits: {0}", totalInitPartCount);

    std::vector<CompilationResult> results;
    results.reserve(totalInitPartCount + 1);  // all inits + main

    auto initPipelineStrategy = factoryMethod(config::CompilationMode::WSInit);
    for (int64_t initPart = 0; initPart < totalInitPartCount; ++initPart) {
        log.info("Compile Init[{0}]", initPart);
        auto moduleInit = mlir::OwningOpRef<mlir::ModuleOp>(moduleMain.get().clone());
        moduleInit->setSymName(formatv("{0}_init_{1}", moduleMain->getSymName().value_or("module"), initPart).str());

        mlir::DefaultTimingManager initTm;
        devConf.setup(initTm);
        auto rootInitTiming = initTm.getRootScope();

        {
            auto initIEtoVPUIPTiming = rootInitTiming.nest("IE to VPUIP pipeline for Init");
            ws::compileIEtoVPU(moduleInit, initPipelineStrategy, devConf, config, std::string("gen-init"), initPart,
                               hardcodedMemoryLimit, initIEtoVPUIPTiming, log);
            ws::compileVPUIP(moduleInit, initPipelineStrategy, devConf, config,
                             /* wsExtractionMode = */ std::nullopt, /* memLimit = */ std::nullopt, initIEtoVPUIPTiming,
                             log);
        }

        backendCompilation(moduleInit, devConf, config, rootInitTiming, log);

        results.emplace_back(std::move(moduleInit), model);
    }

    log.info("Compile Main");
    {
        auto mainVPUIPTiming = rootTiming.nest("VPUIP pipeline for Main");
        ws::compileVPUIP(moduleMain, mainPipelineStrategy, devConf, config, std::string("gen-main"),
                         hardcodedMemoryLimit, mainVPUIPTiming, log);
    }

    backendCompilation(moduleMain, devConf, config, rootTiming, log);

    results.emplace_back(std::move(moduleMain), model);
    return results;
}

void compileModelWsIterative(DeveloperConfig& devConf, mlir::TimingScope& rootTiming, const intel_npu::Config& config,
                             vpux::Logger log, std::optional<int64_t> initPart, std::optional<Byte> memLimit,
                             mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "ws::compileImplWsIterative",
                      "ws::compileModelWsIterative");

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "compile");
    auto factoryMethod = createDialectPipelineStrategyFn(config);

    if (initPart.has_value()) {
        log.info("Compile Init[{0}]", initPart.value());
        auto initPipelineStrategy = factoryMethod(config::CompilationMode::WSInit);
        moduleOp->setSymName(
                formatv("{0}_init_{1}", moduleOp->getSymName().value_or("module"), initPart.value()).str());

        mlir::DefaultTimingManager initTm;
        devConf.setup(initTm);
        auto rootInitTiming = initTm.getRootScope();
        {
            auto initIEtoVPUIPTiming = rootInitTiming.nest("IE to VPUIP pipeline for Init");
            ws::compileIEtoVPU(moduleOp, initPipelineStrategy, devConf, config, std::string("gen-init"),
                               initPart.value(), memLimit, initIEtoVPUIPTiming, log);
            ws::compileVPUIP(moduleOp, initPipelineStrategy, devConf, config,
                             /* wsExtractionMode = */ std::nullopt, /* memLimit = */ std::nullopt, initIEtoVPUIPTiming,
                             log);
        }

        backendCompilation(moduleOp, devConf, config, rootInitTiming, log);
        return;
    }

    log.info("Compile Main");
    auto mainPipelineStrategy = factoryMethod(config::CompilationMode::WSMain);

    {
        auto mainVPUIPTiming = rootTiming.nest("VPUIP pipeline for Main");
        ws::compileVPUIP(moduleOp, mainPipelineStrategy, devConf, config, std::string("gen-main"), memLimit,
                         mainVPUIPTiming, log);
    }

    backendCompilation(moduleOp, devConf, config, rootTiming, log);
}

std::tuple<mlir::OwningOpRef<mlir::ModuleOp>, bool> compileImplWsIterative(
        std::unique_ptr<CompilerSetup>& setup, const std::vector<std::shared_ptr<const ov::Node>>& originalParameters,
        const std::vector<std::shared_ptr<const ov::Node>>& originalResults, const std::shared_ptr<ov::Model>& model,
        const intel_npu::Config& config, size_t callIdx, Logger& log) {
    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compileImplWsIterative",
                      "ws::compileImplWsIterative");

    mlir::MLIRContext* ctx = setup->ctx.get();
    DeveloperConfig devConf(log);

    mlir::DefaultTimingManager tm;
    devConf.setup(tm);
    auto rootTiming = tm.getRootScope();

    Const::setLazyFoldingOptions(ctx, Const::getWsFoldingOptions());

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "importNetwork");
    auto moduleOp = importNetwork(ctx, model, originalParameters, originalResults, config, devConf, rootTiming, log,
                                  /*enableWeightsSeparationPath=*/true);

    auto factoryMethod = createDialectPipelineStrategyFn(config);
    auto mainPipelineStrategy = factoryMethod(config::CompilationMode::WSMain);

    // This value determines how many Init schedules will be generated to enable the Init pipelining feature.
    // For more details, please refer to weights_separation.md.
    // E#176573: remove this property
    auto hardcodedMemoryLimit = vpux::Byte(800_MB);
    log.info("Init pipelining memory limit: {0}", hardcodedMemoryLimit);

    ws::compileIEtoVPU(moduleOp, mainPipelineStrategy, devConf, config, /* wsExtractionMode = */ std::nullopt,
                       /* initPart = */ std::nullopt, hardcodedMemoryLimit, rootTiming, log);

    auto totalInitPartCount = mlir::cast<mlir::IntegerAttr>(moduleOp.get()->getAttr("VPU.WsTotalInitPartCount"))
                                      .getValue()
                                      .getSExtValue();
    log.info("Total number of Inits: {0}", totalInitPartCount);

    auto processedInitPartCount = checked_cast<int64_t>(callIdx);
    const bool compileInit = processedInitPartCount < totalInitPartCount;

    if (compileInit) {
        ws::compileModelWsIterative(devConf, rootTiming, config, log, callIdx, hardcodedMemoryLimit, moduleOp);
    } else {
        ws::compileModelWsIterative(devConf, rootTiming, config, log, /* initPart = */ std::nullopt,
                                    /* memLimit = */ hardcodedMemoryLimit, moduleOp);
    }

    return std::make_tuple(std::move(moduleOp), compileInit);
}

}  // namespace ws

template <typename CompiledT>
std::tuple<CompiledT, intel_npu::Config> tryCompileDebatchedModel(
        const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config, vpux::Logger& log,
        FuncRef<CompiledT(const std::shared_ptr<ov::Model>&, const std::vector<std::shared_ptr<const ov::Node>>&,
                          const std::vector<std::shared_ptr<const ov::Node>>&, const intel_npu::Config&)>
                callCompilation) {
    const auto originalParameters = IE::buildOVParams(model);
    const auto originalResults = IE::buildOVResults(model);

    auto isCompatibleWithWSPipeline = [&](ov::intel_npu::BatchMode batchType) {
        // Debatch method is not supported for the WS pipeline. Continue compilation only for unroll one.
        auto [newConfig, needsDebatchingInCompiler] = autoDetectBatchedModelIfPossible(model, config);
        return !needsDebatchingInCompiler && batchType != ov::intel_npu::BatchMode::PLUGIN;
    };
    try {
        auto partitionCount = getModelBatchPartitionIfPossible(model, config);
        if (partitionCount.has_value()) {
            if (*partitionCount > 1) {
                auto [batchModel, configPerformanceMode] = debatchModel(model, config, *partitionCount, log);
                return std::make_tuple(
                        callCompilation(batchModel, originalParameters, originalResults, configPerformanceMode),
                        configPerformanceMode);
            }
        } else {
            const auto& batchType = config.get<intel_npu::BATCH_MODE>();
            if (isCompatibleWithWSPipeline(batchType)) {
                log.info("Batching doesn't need handling. Flag is not found or batch dimension if 1.");
            } else {
                VPUX_THROW("This model is not supported when handling batching.");
            }
        }
    } catch (const std::exception& ex) {
        const auto& batchType = config.get<intel_npu::BATCH_MODE>();
        if (isCompatibleWithWSPipeline(batchType)) {
            log.info("An error occurred during network compilation so fallback on compiler batch mode {0}. Batching in "
                     "compiler can be handled.",
                     ex.what());
        } else {
            VPUX_THROW(ex.what());
        }
    }

    return std::make_tuple(callCompilation(model, originalParameters, originalResults, config), config);
}

std::vector<std::shared_ptr<NetworkDescription>> CompilerImpl::compileWsOneShot(const std::shared_ptr<ov::Model>& model,
                                                                                const intel_npu::Config& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compileWsOneShot");
    checkPlatformSupportedForCompilation(config.get<intel_npu::PLATFORM>());

    Logger log("vpux-compiler", getLogLevel(config));
    log.info("Start oneshot WS compilation");

    [[maybe_unused]] const auto scopedTimer = startScopedTimer(config, [&log](double deltaMs) {
        log.info("Compile net time: {0} ms", deltaMs);
    });

    auto setup = CompilerSetup::create(config, log);

    using CompilationReturnType = std::vector<CompilationResult>;
    auto getCompilationResult = [&](const std::shared_ptr<ov::Model>& debatchedModel,
                                    const std::vector<std::shared_ptr<const ov::Node>>& originalParameters,
                                    const std::vector<std::shared_ptr<const ov::Node>>& originalResults,
                                    const intel_npu::Config& debatchedConfig) -> CompilationReturnType {
        return ws::compileImplWsOneShot(setup, originalParameters, originalResults, debatchedModel, debatchedConfig,
                                        log);
    };
    auto [compilationResults, compiledConfig] =
            tryCompileDebatchedModel<CompilationReturnType>(model, config, log, getCompilationResult);

    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compileWsOneShot",
                      "exportNetwork");
    std::vector<std::shared_ptr<NetworkDescription>> networkDescrs;
    networkDescrs.reserve(compilationResults.size());
    for (const auto& result : compilationResults) {
        networkDescrs.emplace_back(
                std::make_shared<NetworkDescription>(exportNetwork(result.moduleOp.get(), compiledConfig, log)));
    }
    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);

    // Plugin will collect the compilation memory usage
    return networkDescrs;
}

std::vector<std::shared_ptr<NetworkDescriptionView>> CompilerImpl::compileFromDescWsOneShot(
        const vcl_executable_desc_t& desc, const vcl_compiler_properties_t& compilerProp,
        vcl_compiler_desc_t& compilerDesc, vcl_device_desc_t& deviceDesc, intel_npu::Config& config,
        bool isDeviceDescEmpty, BlobAllocator& allocator) const {
    Logger log("vpux-compiler", getLogLevel(config));

    auto modelData = prepareAndPreprocessModelFromDesc(desc, compilerProp, compilerDesc, deviceDesc, config,
                                                       isDeviceDescEmpty, log);

    return compileWsOneShot(modelData->model, config, allocator);
}

std::vector<std::shared_ptr<NetworkDescriptionView>> CompilerImpl::compileWsOneShot(
        const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config, BlobAllocator& allocator) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compileWsOneShot");
    checkPlatformSupportedForCompilation(config.get<intel_npu::PLATFORM>());

    Logger log("vpux-compiler", getLogLevel(config));
    log.info("Start oneshot WS compilation");

    [[maybe_unused]] const auto scopedTimer = startScopedTimer(config, [&log](double deltaMs) {
        log.info("Compile net time: {0} ms", deltaMs);
    });

    auto setup = CompilerSetup::create(config, log);

    using CompilationReturnType = std::vector<CompilationResult>;
    auto getCompilationResult = [&](const std::shared_ptr<ov::Model>& debatchedModel,
                                    const std::vector<std::shared_ptr<const ov::Node>>& originalParameters,
                                    const std::vector<std::shared_ptr<const ov::Node>>& originalResults,
                                    const intel_npu::Config& debatchedConfig) -> CompilationReturnType {
        return ws::compileImplWsOneShot(setup, originalParameters, originalResults, debatchedModel, debatchedConfig,
                                        log);
    };
    auto [compilationResults, compiledConfig] =
            tryCompileDebatchedModel<CompilationReturnType>(model, config, log, getCompilationResult);

    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compileWsOneShot",
                      "exportNetwork");
    std::vector<std::shared_ptr<NetworkDescriptionView>> networkDescrs;
    networkDescrs.reserve(compilationResults.size());
    for (const auto& result : compilationResults) {
        networkDescrs.emplace_back(std::make_shared<NetworkDescriptionView>(
                exportNetwork(result.moduleOp.get(), compiledConfig, log, allocator)));
    }
    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);

    // Plugin will collect the compilation memory usage
    return networkDescrs;
}

NetworkDescription CompilerImpl::compileWsIterative(const std::shared_ptr<ov::Model>& model,
                                                    const intel_npu::Config& config, size_t callIdx) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compileWsIterative");
    checkPlatformSupportedForCompilation(config.get<intel_npu::PLATFORM>());

    Logger log("vpux-compiler", getLogLevel(config));

    [[maybe_unused]] const auto scopedTimer = startScopedTimer(config, [&log](double deltaMs) {
        log.info("Compile net time: {0} ms", deltaMs);
    });

    auto setup = CompilerSetup::create(config, log);

    using CompilationReturnType = std::tuple<mlir::OwningOpRef<mlir::ModuleOp>, bool>;
    auto getCompilationResult = [&](const std::shared_ptr<ov::Model>& debatchedModel,
                                    const std::vector<std::shared_ptr<const ov::Node>>& originalParameters,
                                    const std::vector<std::shared_ptr<const ov::Node>>& originalResults,
                                    const intel_npu::Config& debatchedConfig) -> CompilationReturnType {
        return ws::compileImplWsIterative(setup, originalParameters, originalResults, debatchedModel, debatchedConfig,
                                          callIdx, log);
    };
    auto [compilationResult, compiledConfig] =
            tryCompileDebatchedModel<CompilationReturnType>(model, config, log, getCompilationResult);

    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compileWsIterative",
                      "exportNetwork");
    auto compiledModel = std::get<0>(compilationResult).get();
    auto dscr = exportNetwork(compiledModel, compiledConfig, log);
    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);

    // Plugin will collect the compilation memory usage
    return dscr;
}

NetworkDescriptionView CompilerImpl::compileWsIterative(const std::shared_ptr<ov::Model>& originModel,
                                                        const intel_npu::Config& config, size_t callIdx,
                                                        BlobAllocator& allocator) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compileWsIterative");
    checkPlatformSupportedForCompilation(config.get<intel_npu::PLATFORM>());

    Logger log("vpux-compiler", getLogLevel(config));

    [[maybe_unused]] const auto scopedTimer = startScopedTimer(config, [&log](double deltaMs) {
        log.info("Compile net time: {0} ms", deltaMs);
    });

    auto setup = CompilerSetup::create(config, log);

    using CompilationReturnType = std::tuple<mlir::OwningOpRef<mlir::ModuleOp>, bool>;
    auto getCompilationResult = [&](const std::shared_ptr<ov::Model>& debatchedModel,
                                    const std::vector<std::shared_ptr<const ov::Node>>& originalParameters,
                                    const std::vector<std::shared_ptr<const ov::Node>>& originalResults,
                                    const intel_npu::Config& debatchedConfig) -> CompilationReturnType {
        return ws::compileImplWsIterative(setup, originalParameters, originalResults, debatchedModel, debatchedConfig,
                                          callIdx, log);
    };
    auto [compilationResult, compiledConfig] =
            tryCompileDebatchedModel<CompilationReturnType>(originModel, config, log, getCompilationResult);

    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compileWsIterative",
                      "exportNetwork");
    auto compiledModel = std::get<0>(compilationResult).get();
    auto dscr = exportNetwork(compiledModel, compiledConfig, log, allocator);
    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);

    // Plugin will collect the compilation memory usage
    return dscr;
}

//
// CompilerImpl::parse
//
enum class BlobFormat {
    UNKNOWN,
    ELF,       //<- ELF binary for static models
    LLVM_TXT,  //<- LLVM text generated from HostCompile mode for dynamic models
    BYTECODE   //<- LLVM bytecode generated from HostCompile mode for dynamic models
};

inline BlobFormat getBlobFormat(const uint8_t* data, size_t dataSize) {
    if (data) {
        constexpr size_t header_size = 20;

        // Temporarily use 20 as header length
        size_t headerSize = dataSize > header_size ? header_size : dataSize;
        std::string header(reinterpret_cast<const char*>(data), headerSize);

        if (header.find("ELF") != std::string::npos) {
            return BlobFormat::ELF;
        } else if (header.find("module") != std::string::npos) {
            return BlobFormat::LLVM_TXT;
        } else {
            if (headerSize >= intel_npu::vm::MAGIC_NUMBER_SIZE) {
                intel_npu::vm::Span<uint8_t> header(const_cast<uint8_t*>(data), intel_npu::vm::MAGIC_NUMBER_SIZE);
                if (std::equal(intel_npu::vm::MAGIC_NUMBER.begin(), intel_npu::vm::MAGIC_NUMBER.end(), header.begin(),
                               header.end())) {
                    return BlobFormat::BYTECODE;
                }
            }
        }
    }

    return BlobFormat::UNKNOWN;
}

NetworkMetadata CompilerImpl::parse(const std::vector<uint8_t>& compiledNetwork, const intel_npu::Config&) const {
    BlobFormat format = getBlobFormat(compiledNetwork.data(), compiledNetwork.size());
    switch (format) {
    case BlobFormat::ELF:
        return VPUMI37XX::getNetworkMetadata(mlir::ArrayRef(compiledNetwork));
    case BlobFormat::LLVM_TXT: {
        mlir::DialectRegistry registry;
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);
        auto context = std::make_unique<mlir::MLIRContext>(registry);

        // Metadata<METADATA_VERSION_X_X> is stored after LLVM code in CompiledModel::export_model
        // So, the file size needs to be adjusted to avoid compilation error
        auto getLLVMIRSize = [](const std::vector<uint8_t>& llvmIR) {
            if (llvmIR.empty()) {
                return static_cast<uint64_t>(0);
            }
            for (int64_t index = static_cast<int64_t>(llvmIR.size()) - 1; index >= 0LL; --index) {
                if (llvmIR[index] == static_cast<uint8_t>('}')) {
                    return static_cast<uint64_t>(index + 1LL);
                }
            }

            return static_cast<uint64_t>(0);
        };

        llvm::SMDiagnostic err;
        llvm::StringRef content(reinterpret_cast<const char*>(compiledNetwork.data()), getLLVMIRSize(compiledNetwork));
        std::unique_ptr<llvm::MemoryBuffer> buffer = llvm::MemoryBuffer::getMemBufferCopy(content, "LLVMBlob");
        auto sourceMgr = std::make_shared<llvm::SourceMgr>();
        sourceMgr->AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
        mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(*sourceMgr, context.get());

        return VPUMI37XX::getHostCompileNetworkMetadata(module.get());
    }
    case BlobFormat::BYTECODE: {
        auto metadata = intel_npu::vm::parseNetworkMetadata(compiledNetwork.data(), compiledNetwork.size());
        VPUX_THROW_UNLESS(metadata.has_value(), "Failed to parse metadata from bytecode");
        return toVPUXMetadata(metadata.value());
    }
    case BlobFormat::UNKNOWN:
    default:
        VPUX_THROW("Unknown blob format");
    }
}

//
// CompilerImpl::processProfilingOutput
//

std::vector<ov::ProfilingInfo> CompilerImpl::processProfilingOutput(const std::vector<uint8_t>& profData,
                                                                    const std::vector<uint8_t>& network,
                                                                    const intel_npu::Config&) const {
    auto layerInfo = profiling::getLayerProfilingInfoHook(profData, network);
    return intel_npu::profiling::convertLayersToIeProfilingInfo(layerInfo);
}

std::vector<ze::ze_profiling_layer_info> CompilerImpl::getLayerInfo(const uint8_t* blobData, uint64_t blobSize,
                                                                    const uint8_t* profData, uint64_t profSize) const {
    return getLayerInfoImpl(blobData, blobSize, profData, profSize);
}

std::vector<ze::ze_profiling_task_info> CompilerImpl::getTaskInfo(const uint8_t* blobData, uint64_t blobSize,
                                                                  const uint8_t* profData, uint64_t profSize) const {
    return getTaskInfoImpl(blobData, blobSize, profData, profSize);
}

BlobView::BlobView(uint8_t* _ptr, uint64_t _size): ptr(_ptr), size(_size) {
}

NetworkDescriptionView::NetworkDescriptionView(BlobView blob, NetworkMetadata&& meta)
        : compiledNetwork(std::move(blob)), compatibilityString{nullptr, 0}, metadata(std::move(meta)) {
}

NetworkDescriptionView::NetworkDescriptionView(BlobView blob, BlobView compatibilityString, NetworkMetadata&& meta)
        : compiledNetwork(std::move(blob)),
          compatibilityString(std::move(compatibilityString)),
          metadata(std::move(meta)) {
}
