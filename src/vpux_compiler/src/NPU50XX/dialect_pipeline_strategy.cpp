//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU50XX/pipeline_options.hpp"

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/pipelines/options_setup.hpp"

using namespace vpux;

namespace {

//
// OptionsSetup50XX
//

class DefaultHWSetup50XX : public OptionsSetupBase<DefaultHWSetup50XX, DefaultHWOptions50XX> {
public:
    using Base = OptionsSetupBase<DefaultHWSetup50XX, DefaultHWOptions50XX>;
    using Base::Base;
    // Expose setupOptionsImpl() to OptionsSetup
    friend Base::Base;

    static void setupLitTestOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        Base::setupLitTestOptionsImpl(options, initCompilerOptions);
        setupOptionsCommon(options);
    }

    static void setupOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const intel_npu::Config& config) {
        Base::setupOptionsImpl(options, initCompilerOptions, config);
        if (config.get<intel_npu::TURBO>()) {
            overwriteIfUnset(options.enableReduceNumTilesForSmallModelsPass, true);
            overwriteIfUnset(options.workloadManagementMode, WorkloadManagementMode::FWLM_V1_PAGES);
        }
        setupOptionsCommon(options, getLogLevel(config));
    }

    static void setupOptionsCommon(DefaultHWOptions50XX& options, LogLevel logLevel = LogLevel::None) {
        setupPWLMParams50XX(options, logLevel);
    }
};

class ReferenceSWSetup50XX : public OptionsSetupBase<ReferenceSWSetup50XX, DefaultHWOptions50XX> {
public:
    using Base = OptionsSetupBase<ReferenceSWSetup50XX, DefaultHWOptions50XX>;
    using Base::Base;

    static void setupOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const intel_npu::Config& config) {
        Base::setupOptionsImpl(options, initCompilerOptions, config);
        setupOptionsCommon(options, getLogLevel(config));
    }

    static void setupOptionsCommon(DefaultHWOptions50XX& options, LogLevel logLevel = LogLevel::None) {
        setupPWLMParams50XX(options, logLevel);
        // ReferenceSW specific values
        overwriteIfUnset(options.enableForceZMajorConcat, false);
        overwriteIfUnset(options.enableSwapTransposeWithFQ, false);
        overwriteIfUnset(options.enableAlignScales, false);
        overwriteIfUnset(options.fuseMvn6ScaleBias, false);
        overwriteIfUnset(options.enableConvertFCToConv, false);
        overwriteIfUnset(options.enableAdjustNonZeroFakeQuant, false);
        overwriteIfUnset(options.enableExtraStaticShapeOps, false);
        overwriteIfUnset(options.enableOptimizeReorders, false);
        overwriteIfUnset(options.enableVPUNNPreSplit, false);
        overwriteIfUnset(options.enableRuntimeDequant, false);

        overwriteIfUnset(options.enableConvertFFTToConv, false);
        overwriteIfUnset(options.enableConvertToSdpaExtended, false);
        overwriteIfUnset(options.enableDecomposeGRUSequence, false);
        overwriteIfUnset(options.enableAutoPaddingIDU, false);
        overwriteIfUnset(options.enableAutoPaddingODU, false);
        overwriteIfUnset(options.enableIsReduceSupported, false);
    }
};

class HostCompileSetup50XX : public OptionsSetupBase<HostCompileSetup50XX, DefaultHWOptions50XX> {
public:
    using Base = OptionsSetupBase<HostCompileSetup50XX, DefaultHWOptions50XX>;
    using Base::Base;
    // Expose setupOptionsImpl() to OptionsSetup
    friend Base::Base;

private:
    static void setupLitTestOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        // DefaultHW options
        DefaultHWSetup50XX::setupLitTestOptionsImpl(options, initCompilerOptions);

        setupOptionsCommon(options);
    }

    static void setupOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const intel_npu::Config& config) {
        // DefaultHW options
        DefaultHWSetup50XX::setupOptionsImpl(options, initCompilerOptions, config);

        // HostCompileSetup50XX common options
        setupOptionsCommon(options, getLogLevel(config));
    }

    static void setupOptionsCommon(DefaultHWOptions50XX& options, LogLevel logLevel = LogLevel::None) {
        // DefaultHW specific options
        DefaultHWSetup50XX::setupOptionsCommon(options, logLevel);

        // HostCompile specific options
        overwriteIfUnset(options.enableDynamicShapeTransformationsPipeline, false);
        overwriteIfUnset(options.enableSCFTiling, true);
        overwriteIfUnset(options.enableScfComputeOpsOutlining, true);
        overwriteIfUnset(options.useMemrefForHostFunctionBufferization, true);
        overwriteIfUnset(options.disablePassOnEntryFunctionForHostCompile, true);
        overwriteIfUnset(options.setMemorySpaceForFunctionBoundaries, false);

        // the below options enable DepthToSpace as a SHAVE operator
        overwriteIfUnset(options.enableD2SToTransposedConvConversion, false);
        overwriteIfUnset(options.enableFuseD2SExpand, true);
        overwriteIfUnset(options.enableOpsAsDMA, false);
        overwriteIfUnset(options.enableConvertExpandToConvPass, false);

        // tiling over channels is not supported for HostCompile, so we disable propagation of permute through eltwise
        overwriteIfUnset(options.enablePropagateMemPermuteThroughEltwise, false);
        overwriteIfUnset(options.enableAdjustMemPermuteAroundOp, false);
    }
};

class WSInitSetup50XX : public WSInitSetupBase<WSInitSetup50XX, DefaultHWOptions50XX> {
public:
    using Base = WSInitSetupBase<WSInitSetup50XX, DefaultHWOptions50XX>;
    using Base::Base;
    friend Base::Base;

private:
    static void setupLitTestOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        Base::setupLitTestOptionsImpl(options, initCompilerOptions);
        setupPWLMParams50XX(options);
    }

    static void setupOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const intel_npu::Config& config) {
        Base::setupOptionsImpl(options, initCompilerOptions, config);
        setupPWLMParams50XX(options, getLogLevel(config));
    }
};

class WSMainSetup50XX final : public WSMainSetupBase<WSMainSetup50XX, DefaultHWOptions50XX> {
public:
    using Base = WSMainSetupBase<WSMainSetup50XX, DefaultHWOptions50XX>;
    using Base::Base;
    // Expose setupOptionsImpl() to OptionsSetup
    friend Base::Base;

private:
    static void setupLitTestOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        Base::setupLitTestOptionsImpl(options, initCompilerOptions);
        setupOptionsCommon(options);
    }

    static void setupOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const intel_npu::Config& config) {
        Base::setupOptionsImpl(options, initCompilerOptions, config);
        setupOptionsCommon(options, getLogLevel(config));
    }

    static void setupOptionsCommon(DefaultHWOptions50XX& options, LogLevel logLevel = LogLevel::None) {
        setupPWLMParams50XX(options, logLevel);
    }
};

//
// DialectPipelineStrategy50XX
//

template <class OptionsContainerType>
class DialectPipelineStrategy50XX final : public IDialectPipelineStrategy {
public:
    explicit DialectPipelineStrategy50XX(const intel_npu::Config& config)
            : _optionsContainer(std::make_unique<OptionsContainerType>(config)) {
    }

    explicit DialectPipelineStrategy50XX(std::unique_ptr<OptionsContainerType> optionsContainer)
            : _optionsContainer(std::move(optionsContainer)) {
    }

    void initializePipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::buildInitCompilerPipeline(pm, _optionsContainer->getInitCompilerOptions(), log.nest());
        // TRACK: E#179877
        // This is needed for the HostCompile pipeline to properly set up outlined NPU compute ops
        if (_optionsContainer->getPipelineOptions().enableProfiling) {
            pm.addPass(VPU::createCloneReservedResourcesFromTopModulePass(log));
        }
    }

    void buildIEPipeline(mlir::OpPassManager& pm, Logger log) override {
        IE::arch50xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        // Lowering to VPU
        vpux::buildLowerIE2VPUPipeline(pm, log);
    }

    void buildVPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::arch50xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::buildLowerVPU2VPUIPPipeline(pm, _optionsContainer->getPipelineOptions().enableInPlaceBufferization,
                                          _optionsContainer->getPipelineOptions().useMemrefForHostFunctionBufferization,
                                          log);
    }

    void buildVPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPUIP::arch50xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

private:
    std::unique_ptr<OptionsContainerType> _optionsContainer;
};

//
// DialectPipelineStrategy50XX: [ReferenceSW]
// This implementation will be chosen if OptionsContainerType contains ReferenceSWOptions
//

class DialectPipelineStrategyReferenceSW50XX final : public IDialectPipelineStrategy {
public:
    explicit DialectPipelineStrategyReferenceSW50XX(const intel_npu::Config& config)
            : _optionsContainer(std::make_unique<ReferenceSWSetup50XX>(config)) {
    }

    explicit DialectPipelineStrategyReferenceSW50XX(std::unique_ptr<ReferenceSWSetup50XX> optionsContainer)
            : _optionsContainer(std::move(optionsContainer)) {
    }

    void initializePipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::buildInitCompilerPipeline(pm, _optionsContainer->getInitCompilerOptions(), log.nest());
    }

    void buildIEPipeline(mlir::OpPassManager& pm, Logger log) override {
        IE::arch50xx::buildReferenceSWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::buildLowerIE2VPUPipeline(pm, log);
    }

    void buildVPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::arch37xx::buildReferenceSWPipeline(
                pm, VPU::arch37xx::DefaultHWOptions(_optionsContainer->getPipelineOptions()), log);
    }

    void buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::buildLowerVPU2VPUIPPipeline(pm, _optionsContainer->getPipelineOptions().enableInPlaceBufferization,
                                          /*useMemrefForHostFunctionBufferization*/ false, log);
    }

    void buildVPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPUIP::arch50xx::buildReferenceSWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

private:
    std::unique_ptr<ReferenceSWSetup50XX> _optionsContainer;
};

}  // namespace

//
// createDialectPipelineStrategy50XX
//

std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy50XX(
        config::CompilationMode compilationMode, const intel_npu::Config& config) {
    switch (compilationMode) {
    case config::CompilationMode::DefaultHW: {
        return std::make_unique<DialectPipelineStrategy50XX<DefaultHWSetup50XX>>(config);
    }
    case config::CompilationMode::ReferenceSW: {
        return std::make_unique<DialectPipelineStrategyReferenceSW50XX>(config);
    }
    case config::CompilationMode::HostCompile: {
        return std::make_unique<DialectPipelineStrategy50XX<HostCompileSetup50XX>>(config);
    }
    case config::CompilationMode::WSInit: {
        return std::make_unique<DialectPipelineStrategy50XX<WSInitSetup50XX>>(config);
    }
    case config::CompilationMode::WSMain: {
        return std::make_unique<DialectPipelineStrategy50XX<WSMainSetup50XX>>(config);
    }
    default:
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }
}

//
// createDialectPipelineStrategy50XX [lit-tests]
//

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy50XX(
        const VPU::InitCompilerOptions* initCompilerOptions, const DefaultHWOptions50XX* options) {
    auto wrapper = std::make_unique<DefaultHWSetup50XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategy50XX<DefaultHWSetup50XX>>(std::move(wrapper));
}

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy50XXReferenceSW(
        const VPU::InitCompilerOptions* initCompilerOptions, const DefaultHWOptions50XX* options) {
    auto wrapper = std::make_unique<ReferenceSWSetup50XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategyReferenceSW50XX>(std::move(wrapper));
}

/// The reason this method is separate from the default and reference compilation modes is that it has to *copy* the
/// options in order to override them.
template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy50XXWS(
        config::CompilationMode compilationMode, const VPU::InitCompilerOptions* initCompilerOptions,
        const DefaultHWOptions50XX* options) {
    switch (compilationMode) {
    case config::CompilationMode::WSInit: {
        auto wrapper = std::make_unique<WSInitSetup50XX>(initCompilerOptions, options);
        return std::make_unique<DialectPipelineStrategy50XX<WSInitSetup50XX>>(std::move(wrapper));
    }
    case config::CompilationMode::WSMain: {
        auto wrapper = std::make_unique<WSMainSetup50XX>(initCompilerOptions, options);
        return std::make_unique<DialectPipelineStrategy50XX<WSMainSetup50XX>>(std::move(wrapper));
    }
    default:
        VPUX_THROW("Unsupported compilation mode {0} for Monolithic WS.", config::stringifyEnum(compilationMode));
        return {};
    }
}

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy50XXHostCompile(
        config::CompilationMode compilationMode, const VPU::InitCompilerOptions* initCompilerOptions,
        const DefaultHWOptions50XX* options) {
    VPUX_THROW_UNLESS(compilationMode == config::CompilationMode::HostCompile,
                      "Unsupported compilation mode {0} for Host Compile.", config::stringifyEnum(compilationMode));

    auto wrapper = std::make_unique<HostCompileSetup50XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategy50XX<HostCompileSetup50XX>>(std::move(wrapper));
}
