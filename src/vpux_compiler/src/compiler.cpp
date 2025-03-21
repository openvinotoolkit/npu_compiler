//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/compiler.hpp"

#include "intel_npu/config/common.hpp"
#include "intel_npu/config/compiler.hpp"
#include "intel_npu/profiling.hpp"

#include "vpux/compiler/NPU37XX/pipeline_strategy.hpp"
#include "vpux/compiler/NPU37XX/pipelines.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/export.hpp"
#include "vpux/compiler/NPU40XX/pipeline_strategy.hpp"
#include "vpux/compiler/NPU40XX/pipelines.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/export.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/network_description.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_in_background.hpp"
#include "vpux/compiler/frontend/IE.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/interfaces_registry.hpp"
#include "vpux/compiler/options_mapper.hpp"
#include "vpux/compiler/utils/dot_printer.hpp"
#include "vpux/compiler/utils/function_statistics_instrumentation.hpp"
#include "vpux/compiler/utils/locations_verifier.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/memory_usage_collector.hpp"

#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/IE/private_properties.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/memory_usage.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/profiling/reports/api.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/Timing.h>

#include <llvm/Support/ManagedStatic.h>  // llvm_shutdown
#include <llvm/Support/Regex.h>
#include <llvm/Support/ThreadPool.h>
#include <llvm/Support/raw_ostream.h>

#include <openvino/core/dimension.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/runtime/iplugin.hpp>

#include <transformations/common_optimizations/dimension_tracking.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include <algorithm>
#include <regex>

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
#include "vpux/compiler/core/developer_build_utils.hpp"
#endif

using namespace vpux;

using intel_npu::ICompiler;
using intel_npu::NetworkDescription;
using intel_npu::NetworkMetadata;

namespace {

constexpr std::string_view UNSUPPORTED_PLATFORM_ERROR_MESSAGE =
        "Unsupported platform: '{0}'\nThe current version of the compiler is unable to compile on the given platform. "
        "If you're using the compiler inside the plugin, please try using the '{1}' configuration option to set a "
        "supported platform explicitly.";

void checkPlatformSupportedForCompilation(const std::string_view platform) {
    const std::unordered_set supportedPlatforms{ov::intel_npu::Platform::NPU3720, ov::intel_npu::Platform::NPU4000};

    if (!supportedPlatforms.count(ov::intel_npu::Platform::standardize(platform))) {
        VPUX_THROW(UNSUPPORTED_PLATFORM_ERROR_MESSAGE.data(), platform, intel_npu::PLATFORM::key());
    }
}
constexpr uint32_t SUPPORTED_OPSET = 11;

//
// createPipelineStrategy
//

std::unique_ptr<IPipelineStrategy> createPipelineStrategy(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return std::make_unique<PipelineStrategy37XX>();
    case VPU::ArchKind::NPU40XX:
        return std::make_unique<PipelineStrategy40XX>();
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}

//
// DeveloperConfig
//

class DeveloperConfig final {
public:
    explicit DeveloperConfig(Logger log);
    DeveloperConfig(const DeveloperConfig& other) = delete;
    DeveloperConfig& operator=(const DeveloperConfig& other) = delete;
    ~DeveloperConfig();

    void setup(mlir::DefaultTimingManager& tm) const;
    void setup(mlir::PassManager& pm, const intel_npu::Config& config, bool isSubPipeline = false) const;
    void dump(mlir::PassManager& pm) const;

    bool useSharedConstants() const {
        return _useSharedConstants;
    }

private:
    Logger _log;

    std::string _crashReproducerFile;
    bool _localReproducer = true;

    std::string _irPrintingFilter;
    std::string _irPrintingFile;
    std::string _irPrintingOrderStr;
    bool _printFullIR = false;
    bool _printFullConstant = false;
    bool _useSharedConstants = true;
    bool _allowPrintingHexConstant = true;
    bool _printDebugInfo = false;
    bool _printDebugInfoPrettyForm = false;
    std::string _printAsTextualPipelineFilePath = "";
    std::string _printDotOptions;

    llvm::raw_ostream* _timingStream = nullptr;

    std::unique_ptr<llvm::Regex> _irDumpFilter;
    std::unique_ptr<llvm::raw_fd_ostream> _irDumpFile;
    llvm::raw_ostream* _irDumpStream = nullptr;
    IRPrintingOrder _irPrintingOrder = IRPrintingOrder::AFTER;
};

DeveloperConfig::DeveloperConfig(Logger log): _log(log) {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    parseEnv("IE_NPU_CRASH_REPRODUCER_FILE", _crashReproducerFile);
    parseEnv("IE_NPU_GEN_LOCAL_REPRODUCER", _localReproducer);

    parseEnv("IE_NPU_IR_PRINTING_FILTER", _irPrintingFilter);
    parseEnv("IE_NPU_IR_PRINTING_FILE", _irPrintingFile);
    parseEnv("IE_NPU_IR_PRINTING_ORDER", _irPrintingOrderStr);
    parseEnv("IE_NPU_PRINT_FULL_IR", _printFullIR);
    parseEnv("IE_NPU_PRINT_FULL_CONSTANT", _printFullConstant);
    parseEnv("IE_NPU_USE_SHARED_CONSTANTS", _useSharedConstants);
    parseEnv("IE_NPU_PRINT_HEX_CONSTANT", _allowPrintingHexConstant);
    parseEnv("IE_NPU_PRINT_DEBUG_INFO", _printDebugInfo);
    parseEnv("IE_NPU_PRINT_DEBUG_INFO_PRETTY_FORM", _printDebugInfoPrettyForm);
    parseEnv("IE_NPU_PRINT_AS_TEXTUAL_PIPELINE_FILE", _printAsTextualPipelineFilePath);

    parseEnv("IE_NPU_PRINT_DOT", _printDotOptions);
#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

    if (_log.isActive(LogLevel::Info)) {
        _timingStream = &Logger::getBaseStream();
    }

    if (!_irPrintingOrderStr.empty()) {
        auto orderString = _irPrintingOrderStr;
        std::transform(orderString.begin(), orderString.end(), orderString.begin(), [](unsigned char c) {
            return std::toupper(c);
        });
        if (orderString == "BEFORE") {
            _irPrintingOrder = IRPrintingOrder::BEFORE;
        } else if (orderString == "AFTER") {
            _irPrintingOrder = IRPrintingOrder::AFTER;
        } else if (orderString == "BEFORE_AFTER") {
            _irPrintingOrder = IRPrintingOrder::BEFORE_AFTER;
        } else {
            VPUX_THROW("Invalid IR printing order: {0}.\nValid cases are: before, after and before_after. They are not "
                       "case-sensitive.\nExample: IE_NPU_IR_PRINTING_ORDER=Before",
                       _irPrintingOrderStr);
        }
    }

    if (!_irPrintingFilter.empty()) {
        _irDumpFilter = std::make_unique<llvm::Regex>(_irPrintingFilter, llvm::Regex::IgnoreCase);

        std::string regexErr;
        if (!_irDumpFilter->isValid(regexErr)) {
            VPUX_THROW("Invalid regular expression '{0}' : {1}", _irPrintingFilter, regexErr);
        }

        if (_irPrintingFile.empty()) {
            _irDumpStream = &Logger::getBaseStream();
        } else {
            std::error_code err;
            _irDumpFile = std::make_unique<llvm::raw_fd_ostream>(_irPrintingFile, err);
            if (err) {
                VPUX_THROW("Failed to open file '{0}' for write : {1}", _irPrintingFile, err.message());
            }

            _irDumpStream = _irDumpFile.get();
        }
    }
}

DeveloperConfig::~DeveloperConfig() {
    if (_timingStream != nullptr) {
        _timingStream->flush();
    }

    if (_irDumpStream != nullptr) {
        _irDumpStream->flush();
    }
}

void DeveloperConfig::setup(mlir::DefaultTimingManager& tm) const {
    if (_timingStream == nullptr) {
        tm.setEnabled(false);
    } else {
        tm.setEnabled(true);
        tm.setDisplayMode(mlir::DefaultTimingManager::DisplayMode::Tree);
        tm.setOutput(*_timingStream);
    }
}

void DeveloperConfig::setup(mlir::PassManager& pm, const intel_npu::Config& config, bool isSubPipeline) const {
    addLogging(pm, _log);

    // Crash reproducer
    if (!_crashReproducerFile.empty()) {
        // In case the pass manager represents a sub-pipeline (e.g. for the backend), multithreading cannot be safely
        // disabled since the context could be in a multithreading execution context
        if (_localReproducer && !isSubPipeline) {
            pm.getContext()->disableMultithreading();
        }

        pm.enableCrashReproducerGeneration(_crashReproducerFile, _localReproducer);
    }

    // IR printing
    if (_irDumpFilter != nullptr) {
        const bool printAfterOnlyOnChange = false;
        const bool printAfterOnlyOnFailure = false;

        const auto shouldPrintBeforePass = [&](mlir::Pass* pass, mlir::Operation*) {
            return (_irDumpFilter->match(pass->getName()) || _irDumpFilter->match(pass->getArgument())) &&
                   (_irPrintingOrder == IRPrintingOrder::BEFORE || _irPrintingOrder == IRPrintingOrder::BEFORE_AFTER);
        };
        const auto shouldPrintAfterPass = [&](mlir::Pass* pass, mlir::Operation*) {
            return (_irDumpFilter->match(pass->getName()) || _irDumpFilter->match(pass->getArgument())) &&
                   (_irPrintingOrder == IRPrintingOrder::AFTER || _irPrintingOrder == IRPrintingOrder::BEFORE_AFTER);
        };

        if (_printFullIR && !isSubPipeline) {
            pm.getContext()->disableMultithreading();
        }

        mlir::OpPrintingFlags flags;
        if (!_printFullConstant) {
            flags.elideLargeElementsAttrs();
            flags.elideLargeResourceString();
        }
        if (!_allowPrintingHexConstant) {
            flags.setAllowPrintingElementsAttrAsHex(false);
        }
        if (_printDebugInfo) {
            flags.enableDebugInfo(true, _printDebugInfoPrettyForm);
        }

        pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, _printFullIR, printAfterOnlyOnChange,
                            printAfterOnlyOnFailure, *_irDumpStream, flags);
    }

    // Dot printing
    if (!_printDotOptions.empty()) {
        addDotPrinter(pm, _printDotOptions);
    }
    // Locations verifier
    addLocationsVerifier(pm);

    const auto shouldEnableFunctionStatistics = getEnableFunctionStatisticsInstrumentation(config).value_or(false);
    if (shouldEnableFunctionStatistics) {
        _log.info("The function statistics instrumentation is enabled");
        addFunctionStatisticsInstrumentation(pm, _log);
    }

    // Memory usage instrumentation
    const auto shouldEnableMemoryCollector = getEnableMemoryUsageCollector(config).value_or(false);
    _log.info("The memory usage collector is {0}", shouldEnableMemoryCollector ? "enabled" : "disabled");
    if (shouldEnableMemoryCollector) {
        addMemoryUsageCollector(pm, _log);
    }

    // Enable pass verifiers
    const auto shouldEnableVerifiers = getEnableVerifiers(config).value_or(false);
    _log.info("Verifiers are {0}", shouldEnableVerifiers ? "enabled" : "disabled");
    pm.enableVerifier(shouldEnableVerifiers);
}

void DeveloperConfig::dump(mlir::PassManager& pm) const {
    if (!_printAsTextualPipelineFilePath.empty()) {
        std::error_code err;
        auto passesDumpFile = std::make_unique<llvm::raw_fd_ostream>(_printAsTextualPipelineFilePath, err);
        if (err) {
            VPUX_THROW("Failed to open file '{0}' for write : {1}", _printAsTextualPipelineFilePath, err.message());
        }
        pm.printAsTextualPipeline(*passesDumpFile);
    }
}
}  // namespace

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

    log.trace("Get supported nodes.");
    auto supportedNodes = ov::get_supported_nodes(
            model,
            [&](const std::shared_ptr<ov::Model>& model) {
                log.trace("Run common nGraph passes.");
                IE::NGraphPasses::runNGraphPasses(model, rootTiming);
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
// CompilerImpl::compile
//

namespace {

auto importNetwork(mlir::MLIRContext* ctx, const std::shared_ptr<ov::Model>& model,
                   const std::vector<std::shared_ptr<const ov::Node>>& originalParameters,
                   const std::vector<std::shared_ptr<const ov::Node>>& originalResults, const DeveloperConfig& devConf,
                   mlir::TimingScope& rootTiming, bool enableProfiling, vpux::DummyOpMode stubLayers,
                   bool dynamicShapeToStatic, Logger log) {
    auto importTiming = rootTiming.nest("Import network");
    return IE::importNetwork(ctx, model, originalParameters, originalResults, devConf.useSharedConstants(),
                             importTiming, enableProfiling, stubLayers, dynamicShapeToStatic, log.nest());
}

mlir::LogicalResult compileNetwork(mlir::ModuleOp module, mlir::PassManager& pm, mlir::TimingScope& rootTiming) {
    auto compileTiming = rootTiming.nest("Compile network");
    pm.enableTiming(compileTiming);
    return pm.run(module);
}

auto exportToELF(mlir::ModuleOp module, Logger log) {
    const auto arch = VPU::getArch(module);
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return vpux::ELFNPU37XX::exportToELF(module, log);
    default:
        return vpux::ELF::exportToELF(module, log);
    }
}

auto exportToELF(mlir::ModuleOp module, Logger log, BlobAllocator& allocator) {
    const auto arch = VPU::getArch(module);
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return vpux::ELFNPU37XX::exportToELF(module, allocator, log);
    default:
        return vpux::ELF::exportToELF(module, allocator, log);
    }
}

NetworkDescription exportNetwork(mlir::ModuleOp module, Logger log) {
    auto blob = exportToELF(module, log);
    auto meta = VPUMI37XX::getNetworkMetadata(blob);

    return NetworkDescription(std::move(blob), std::move(meta));
}

NetworkDescriptionView exportNetwork(mlir::ModuleOp module, Logger log, BlobAllocator& allocator) {
    auto blobView = exportToELF(module, log, allocator);
    return NetworkDescriptionView(
            blobView, VPUMI37XX::getNetworkMetadata(mlir::ArrayRef(blobView.ptr, static_cast<size_t>(blobView.size))));
}

std::optional<size_t> getBatchSize(const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config) {
    std::set<ov::Output<const ov::Node>> batchedInputs;
    std::set<ov::Output<const ov::Node>> batchedOutputs;
    std::set<size_t> sBatchSize;

    vpux::Logger logger("getBatchSize", getLogLevel(config));

    if (!config.has<intel_npu::BATCH_MODE>()) {
        return 1;
    }

    if (config.get<intel_npu::BATCH_MODE>() == ov::intel_npu::BatchMode::COMPILER) {
        return 1;
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
        if (shape.is_dynamic()) {
            logger.info("Dynamic networks are not supported when batching is handled by the plugin");
            return std::nullopt;
        }
        // Batching on plugin is working only when batching is found on 0th dimension
        if (shape.size() && shape[0].has_symbol()) {
            const auto& static_shape = input->get_shape();
            batchedInputs.insert(params[input_id]->output(0));
            sBatchSize.insert(static_shape[0]);
        } else {
            logger.info("Only networks with inputs batched by 0th dimension are supported");
            return std::nullopt;
        }
    }
    for (const auto& output : testBatchModel->get_results()) {
        const auto& shape = output->get_output_partial_shape(0);
        if (shape.is_dynamic()) {
            logger.info("Dynamic networks are not supported when batching is handled by the plugin");
            return std::nullopt;
        }
        // Batching on plugin is working only when batching is found on 0th dimension
        if (shape.size() && shape[0].has_symbol()) {
            const auto& node = output->input_value(0);
            const auto& static_shape = output->get_shape();
            batchedOutputs.insert(ov::Output<const ov::Node>(node.get_node(), node.get_index()));
            sBatchSize.insert(static_shape[0]);
        } else {
            logger.info("Only networks with outputs batched by 0th dimension are supported");
            return std::nullopt;
        }
    }
    if (!batchedInputs.size() || !batchedOutputs.size()) {
        logger.info("Only networks with inputs/outputs featuring batched dim are supported!");
        return std::nullopt;
    }

    if (sBatchSize.size() != 1) {
        logger.info("Batching size shall have same value for all tensors!");
        return std::nullopt;
    }

    auto it = sBatchSize.begin();
    return *it;
}

mlir::OwningOpRef<mlir::ModuleOp> compileModel(mlir::MLIRContext& ctx, const std::shared_ptr<ov::Model>& model,
                                               const std::vector<std::shared_ptr<const ov::Node>>& originalParameters,
                                               const std::vector<std::shared_ptr<const ov::Node>>& originalResults,
                                               DeveloperConfig& devConf, mlir::TimingScope& rootTiming,
                                               const intel_npu::Config& config, vpux::Logger& log) {
    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "compileModel");
    const auto arch = getArchKind(config);

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "importNetwork");

    const auto dynamicShapeToStatic = config.get<intel_npu::DYNAMIC_SHAPE_TO_STATIC>();
    const auto dummyOpReplacement = getDummyOpReplacement(config).value_or(DummyOpMode::DISABLED);
    mlir::OwningOpRef<mlir::ModuleOp> module =
            importNetwork(&ctx, model, originalParameters, originalResults, devConf, rootTiming,
                          config.get<intel_npu::PERF_COUNT>(), dummyOpReplacement, dynamicShapeToStatic, log);

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "PassManager");

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    devConf.setup(pm, config);

    auto pipelineFactory = createPipelineStrategy(arch);

    // TODO: somehow protect non-target cases
    pipelineFactory->buildPipeline(pm, config, rootTiming, log);

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

    // Load VPUIP dialect before first compilation to set initial WlmStatus based on config
    ctx.loadDialect<VPUIP::VPUIPDialect>();
    auto wlmEnabled = getWlmEnabled(config).value_or(false);
    vpux::VPUIP::setWlmStatus(module.get(),
                              wlmEnabled ? vpux::VPUIP::WlmStatus::ENABLED : vpux::VPUIP::WlmStatus::DISABLED);

    // applies each pass in the pipeline
    auto compileResult = compileNetwork(module.get(), pm, rootTiming);
    VPUX_THROW_WHEN(mlir::failed(compileResult), "Compilation failed");

    mlir::PassManager elfPm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    devConf.setup(elfPm, config);
    auto wlmStatus = vpux::VPUIP::getWlmStatus(module.get());
    auto wlmStillEnabled = wlmStatus == vpux::VPUIP::WlmStatus::ENABLED;
    pipelineFactory->buildELFPipeline(elfPm, config, rootTiming, log, wlmStillEnabled);
    if (getWlmRollback(config).value_or(false)) {
        auto backup_module = mlir::OwningOpRef<mlir::ModuleOp>(module.get().clone());
        // We moved away from the exception-based fallback mechanism because the MLIRContext remained in an invalid
        // state when the exception was thrown, it assumed that it was still executing the pass leading to broken
        // compile time stats. Now we rely on the PassManager::run result and WLM status attribute to decide if we need
        // to rollback. This allows MLIR to run the pass instrumentation and set the context to the correct state.
        compileResult = compileNetwork(module.get(), elfPm, rootTiming);
        wlmStatus = vpux::VPUIP::getWlmStatus(module.get());
        if (mlir::failed(compileResult) && wlmStatus == vpux::VPUIP::WlmStatus::FAILED) {
            log.warning("Failed to export to ELF with current config, reverting to simple ELF pipeline");
            module = std::move(backup_module);
            mlir::PassManager simpleElfPm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
            devConf.setup(simpleElfPm, config, /*isSubPipeline=*/true);
            pipelineFactory->buildELFPipeline(simpleElfPm, config, rootTiming, log, /*useWlm=*/false);
            vpux::VPUIP::setWlmStatus(module.get(), vpux::VPUIP::WlmStatus::DISABLED);
            VPUX_THROW_UNLESS(mlir::succeeded(compileNetwork(module.get(), simpleElfPm, rootTiming)),
                              "Compilation failed");
        } else {
            VPUX_THROW_WHEN(mlir::failed(compileResult), "Compilation failed");
        }
    } else {
        compileResult = compileNetwork(module.get(), elfPm, rootTiming);
        VPUX_THROW_WHEN(mlir::failed(compileResult), "Compilation failed");
    }

    devConf.dump(pm);

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

// leave reference to const std::shared_ptr<ov::Model> instead of taking std::shared_ptr<ov::Model> by value
// as in case of batching we don't copy pointer to ov::Model, we clone it and use clone afterwards
// taking by-value would mean extra copy of std::shared_ptr for no reason in this case, even though
// it's fine for "regular" scenario without batching (just 1 copy anyway)
CompilationResult compileImpl(mlir::MLIRContext& ctx, const std::shared_ptr<ov::Model>& model,
                              const intel_npu::Config& config, Logger& log) {
    checkPlatformSupportedForCompilation(config.get<intel_npu::PLATFORM>());

    DeveloperConfig devConf(log);

    mlir::DefaultTimingManager tm;
    devConf.setup(tm);

    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "compileImpl");

    addLogging(ctx, log);
    auto rootTiming = tm.getRootScope();

    // Save the original model parameters and results before batching
    const auto originalParameters = IE::buildOVParams(model);
    const auto originalResults = IE::buildOVResults(model);

    try {
        auto batchSize = getBatchSize(model, config);

        if (batchSize.has_value()) {
            if (*batchSize > 1) {
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
                ov::set_batch(batchModel, 1);

                auto moduleOp = compileModel(ctx, batchModel, originalParameters, originalResults, devConf, rootTiming,
                                             configPerformanceMode, log);
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

    auto moduleOp = compileModel(ctx, model, originalParameters, originalResults, devConf, rootTiming, config, log);
    return CompilationResult{std::move(moduleOp), model};
}

auto createContext(mlir::DialectRegistry& registry, const intel_npu::Config& config) {
    auto interfacesRegistry = createInterfacesRegistry(getArchKind(config));
    interfacesRegistry->registerInterfaces(registry);
    return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

auto enableMultithreading(mlir::MLIRContext& context, const intel_npu::Config& config)
        -> std::unique_ptr<llvm::ThreadPool> {
    // Set the number of threads in the pool to be the total number of threads of the compilation minus one: one for the
    // main thread and the rest for the MLIR thread pool. If user didn't specify the number of threads, default to 8
    // threads for the pool. By default MLIR will attempt to use all of the threads available on the system which might
    // cause large peak memory usage during constant-related passes such as constant-folding, hence a limit is set
    const bool hasThreadLimit = config.has<intel_npu::COMPILATION_NUM_THREADS>();
    const auto totalThreadCount = hasThreadLimit ? config.get<intel_npu::COMPILATION_NUM_THREADS>() : 9;

    if (totalThreadCount <= 1) {
        return nullptr;
    }

    llvm::ThreadPoolStrategy strategy;
    strategy.ThreadsRequested = totalThreadCount - 1;
    strategy.Limit = true;  // limits number of threads to the number of physical threads

    auto threadPool = std::make_unique<llvm::ThreadPool>(strategy);
    context.setThreadPool(*threadPool);

    return threadPool;
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
}

uint32_t CompilerImpl::getSupportedOpsetVersion() const {
    return SUPPORTED_OPSET;
}

NetworkDescription CompilerImpl::compile(const std::shared_ptr<ov::Model>& model,
                                         const intel_npu::Config& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compile");
    checkPlatformSupportedForCompilation(config.get<intel_npu::PLATFORM>());

    Logger log("vpux-compiler", getLogLevel(config));

    const auto enableExtraShapeBoundOps = getEnableExtraShapeBoundOps(config);
    auto registry = createDialectRegistry(getDummyOpReplacement(config).value_or(DummyOpMode::DISABLED),
                                          enableExtraShapeBoundOps);
    auto ctx = createContext(registry, config);
    auto threadPool = enableMultithreading(ctx, config);

    auto peakMemStart = getPeakMemoryUsage();
    auto compilationResult = compileImpl(ctx, model, config, log);

    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "exportNetwork");
    auto networkDescription = exportNetwork(compilationResult.moduleOp.get(), log);
    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);

    auto peakMemEnd = getPeakMemoryUsage();
    log.debug("Start of compilation memory usage: Peak {0} KB", peakMemStart.count());
    log.debug("End of compilation memory usage: Peak {0} KB", peakMemEnd.count());
    // Note: Following log is parsed by CI. Take care when modifying it.
    log.info("Compilation memory usage: Peak {0} KB", peakMemEnd.count() - peakMemStart.count());

    return networkDescription;
}

NetworkDescriptionView CompilerImpl::compile(const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config,
                                             BlobAllocator& allocator) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compile");
    checkPlatformSupportedForCompilation(config.get<intel_npu::PLATFORM>());

    Logger log("vpux-compiler", getLogLevel(config));

    const auto enableExtraShapeBoundOps = getEnableExtraShapeBoundOps(config);
    auto registry = createDialectRegistry(getDummyOpReplacement(config).value_or(DummyOpMode::DISABLED),
                                          enableExtraShapeBoundOps);
    auto ctx = createContext(registry, config);
    auto threadPool = enableMultithreading(ctx, config);

    auto peakMemStart = getPeakMemoryUsage();
    auto compilationResult = compileImpl(ctx, model, config, log);

    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "exportNetwork");
    auto allocatedCompliedNetwork = exportNetwork(compilationResult.moduleOp.get(), log, allocator);
    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);

    auto peakMemEnd = getPeakMemoryUsage();

    log.debug("Start of compilation memory usage: Peak {0} KB", peakMemStart.count());
    log.debug("End of compilation memory usage: Peak {0} KB", peakMemEnd.count());
    // Note: Following log is parsed by CI. Take care when modifying it.
    log.info("Compilation memory usage: Peak {0} KB", peakMemEnd.count() - peakMemStart.count());

    return allocatedCompliedNetwork;
}

NetworkDescriptionView CompilerImpl::compile(const std::shared_ptr<const ov::Model>& origModel,
                                             const intel_npu::Config& config, BlobAllocator& allocator) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compile");
    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "clone_model");

    // NGraph pipeline modifies the model so need to clone here
    auto model = origModel->clone();

    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);

    return compile(std::move(model), config, allocator);
}

NetworkDescription CompilerImpl::compile(const std::shared_ptr<const ov::Model>& origModel,
                                         const intel_npu::Config& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compile");
    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "clone_model");

    // NGraph pipeline modifies the model so need to clone here
    auto model = origModel->clone();

    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);

    return compile(std::move(model), config);
}

//
// CompilerImpl::parse
//

NetworkMetadata CompilerImpl::parse(const std::vector<uint8_t>& compiledNetwork, const intel_npu::Config&) const {
    return VPUMI37XX::getNetworkMetadata(mlir::ArrayRef(compiledNetwork));
}

//
// CompilerImpl::process_profiling_output
//

std::vector<ov::ProfilingInfo> CompilerImpl::process_profiling_output(const std::vector<uint8_t>& profData,
                                                                      const std::vector<uint8_t>& network,
                                                                      const intel_npu::Config&) const {
    auto layerInfo = profiling::getLayerProfilingInfoHook(profData, network);
    return intel_npu::profiling::convertLayersToIeProfilingInfo(layerInfo);
}

//
// CreateNPUCompiler
//

#ifndef OPENVINO_STATIC_LIBRARY
OPENVINO_PLUGIN_API void CreateNPUCompiler(std::shared_ptr<ICompiler>& obj) {
    obj = std::make_shared<CompilerImpl>();
}
#endif

BlobView::BlobView(uint8_t* _ptr, uint64_t _size): ptr(_ptr), size(_size) {
}

NetworkDescriptionView::NetworkDescriptionView(BlobView blob, NetworkMetadata&& meta)
        : compiledNetwork(std::move(blob)), metadata(std::move(meta)) {
}
