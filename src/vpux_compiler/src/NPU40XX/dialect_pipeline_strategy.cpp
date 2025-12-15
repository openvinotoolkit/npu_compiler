//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU40XX/pipeline_options.hpp"

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/pipelines/options_setup.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

namespace {

//
// OptionsSetup40XX
//

class DefaultHWSetup40XX final : public OptionsSetupBase<DefaultHWSetup40XX, DefaultHWOptions40XX> {
public:
    using Base = OptionsSetupBase<DefaultHWSetup40XX, DefaultHWOptions40XX>;
    using Base::Base;
    // Expose setupOptionsImpl() to OptionsSetup
    friend Base::Base;

    static void setupLitTestOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        Base::setupLitTestOptionsImpl(options, initCompilerOptions);
        setupOptionsCommon(options);
    }

    static void setupOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const intel_npu::Config& config) {
        Base::setupOptionsImpl(options, initCompilerOptions, config);
        if (config.get<intel_npu::TURBO>()) {
            overwriteIfUnset(options.optimizationLevel, 3);
        }
        setupOptionsCommon(options, getLogLevel(config));
    }

    static void setupOptionsCommon(DefaultHWOptions40XX& options, LogLevel logLevel = LogLevel::None) {
        setupParamsAccordingToOptimizationLevel(options.optimizationLevel, options, options.workloadManagementEnable);
        setupPWLMParams(options, logLevel);
    }
};

class ReferenceSWSetup40XX : public OptionsSetupBase<ReferenceSWSetup40XX, DefaultHWOptions40XX> {
public:
    using Base = OptionsSetupBase<ReferenceSWSetup40XX, DefaultHWOptions40XX>;
    using Base::Base;

    static void setupOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const intel_npu::Config& config) {
        Base::setupOptionsImpl(options, initCompilerOptions, config);
        setupOptionsCommon(options);
        setupPWLMParams(options, getLogLevel(config));
    }

    static void setupOptionsCommon(DefaultHWOptions40XX& options) {
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
    }
};

class HostCompileSetup40XX : public OptionsSetupBase<HostCompileSetup40XX, DefaultHWOptions40XX> {
public:
    using Base = OptionsSetupBase<HostCompileSetup40XX, DefaultHWOptions40XX>;
    using Base::Base;
    // Expose setupOptionsImpl() to OptionsSetup
    friend Base::Base;

private:
    static void setupLitTestOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        // DefaultHW options
        DefaultHWSetup40XX::setupLitTestOptionsImpl(options, initCompilerOptions);

        setupOptionsCommon(options);
    }

    static void setupOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const intel_npu::Config& config) {
        // DefaultHW options
        DefaultHWSetup40XX::setupOptionsImpl(options, initCompilerOptions, config);

        // HostCompileSetup40XX common options
        setupOptionsCommon(options, getLogLevel(config));
    }

    static void setupOptionsCommon(DefaultHWOptions40XX& options, LogLevel logLevel = LogLevel::None) {
        // DefaultHW specific options
        DefaultHWSetup40XX::setupOptionsCommon(options, logLevel);

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

class WSInitSetup40XX : public WSInitSetupBase<WSInitSetup40XX, DefaultHWOptions40XX> {
public:
    using Base = WSInitSetupBase<WSInitSetup40XX, DefaultHWOptions40XX>;
    using Base::Base;
    friend Base::Base;

private:
    static void setupLitTestOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        Base::setupLitTestOptionsImpl(options, initCompilerOptions);
        setupOptionsCommon(options);
    }

    static void setupOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const intel_npu::Config& config) {
        Base::setupOptionsImpl(options, initCompilerOptions, config);
        setupOptionsCommon(options, getLogLevel(config));
    }

    static void setupOptionsCommon(DefaultHWOptions40XX& options, LogLevel logLevel = LogLevel::None) {
        setupParamsAccordingToOptimizationLevel(options.optimizationLevel, options, options.workloadManagementEnable);
        setupPWLMParams(options, logLevel);
    }
};

class WSMainSetup40XX final : public WSMainSetupBase<WSMainSetup40XX, DefaultHWOptions40XX> {
public:
    using Base = WSMainSetupBase<WSMainSetup40XX, DefaultHWOptions40XX>;
    using Base::Base;
    friend Base::Base;

private:
    static void setupLitTestOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        Base::setupLitTestOptionsImpl(options, initCompilerOptions);
        setupOptionsCommon(options);
    }

    static void setupOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const intel_npu::Config& config) {
        Base::setupOptionsImpl(options, initCompilerOptions, config);
        setupOptionsCommon(options, getLogLevel(config));
    }

    static void setupOptionsCommon(DefaultHWOptions40XX& options, LogLevel logLevel = LogLevel::None) {
        setupParamsAccordingToOptimizationLevel(options.optimizationLevel, options, options.workloadManagementEnable);
        setupPWLMParams(options, logLevel);
    }
};

//
// DialectPipelineStrategy40XX
//

template <class OptionsContainerType, class Enable = void>
class DialectPipelineStrategy40XX final : public IDialectPipelineStrategy {
public:
    explicit DialectPipelineStrategy40XX(const intel_npu::Config& config)
            : _optionsContainer(std::make_unique<OptionsContainerType>(config)) {
    }

    explicit DialectPipelineStrategy40XX(std::unique_ptr<OptionsContainerType> optionsContainer)
            : _optionsContainer(std::move(optionsContainer)) {
    }

    void initializePipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::buildInitCompilerPipeline(pm, _optionsContainer->getInitCompilerOptions(), log.nest());

        // TRACK: E#179877
        // This is needed for the HostCompile pipeline to properly set up outlined NPU compute ops
        pm.addPass(VPU::createCloneReservedResourcesFromTopModulePass(log));
    }

    void buildIEPipeline(mlir::OpPassManager& pm, Logger log) override {
        IE::arch40xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::buildLowerIE2VPUPipeline(pm, log);
    }

    void buildVPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::arch40xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::buildLowerVPU2VPUIPPipeline(pm, _optionsContainer->getPipelineOptions().enableInPlaceBufferization,
                                          _optionsContainer->getPipelineOptions().useMemrefForHostFunctionBufferization,
                                          log);
    }

    void buildVPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPUIP::arch40xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

private:
    std::unique_ptr<OptionsContainerType> _optionsContainer;
};

//
// DialectPipelineStrategy40XX: [ReferenceSW]
// This implementation will be chosen if OptionsContainerType contains ReferenceSWOptions
//

class DialectPipelineStrategyReferenceSW40XX final : public IDialectPipelineStrategy {
public:
    explicit DialectPipelineStrategyReferenceSW40XX(const intel_npu::Config& config)
            : _optionsContainer(std::make_unique<ReferenceSWSetup40XX>(config)) {
    }

    explicit DialectPipelineStrategyReferenceSW40XX(std::unique_ptr<ReferenceSWSetup40XX> optionsContainer)
            : _optionsContainer(std::move(optionsContainer)) {
    }

    void initializePipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::buildInitCompilerPipeline(pm, _optionsContainer->getInitCompilerOptions(), log.nest());
    }

    void buildIEPipeline(mlir::OpPassManager& pm, Logger log) override {
        IE::arch40xx::buildReferenceSWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
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
        VPUIP::arch40xx::buildReferenceSWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

private:
    std::unique_ptr<ReferenceSWSetup40XX> _optionsContainer;
};

}  // namespace

//
// createDialectPipelineStrategy40XX
//

std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy40XX(
        config::CompilationMode compilationMode, const intel_npu::Config& config) {
    switch (compilationMode) {
    case config::CompilationMode::DefaultHW: {
        return std::make_unique<DialectPipelineStrategy40XX<DefaultHWSetup40XX>>(config);
    }
    case config::CompilationMode::ReferenceSW: {
        // return std::make_unique<DialectPipelineStrategy40XX<ReferenceSWSetup40XX>>(config);
        return std::make_unique<DialectPipelineStrategyReferenceSW40XX>(config);
    }
    case config::CompilationMode::HostCompile: {
        return std::make_unique<DialectPipelineStrategy40XX<HostCompileSetup40XX>>(config);
    }
    case config::CompilationMode::WSInit: {
        return std::make_unique<DialectPipelineStrategy40XX<WSInitSetup40XX>>(config);
    }
    case config::CompilationMode::WSMain: {
        return std::make_unique<DialectPipelineStrategy40XX<WSMainSetup40XX>>(config);
    }
    default:
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }
}

//
// createDialectPipelineStrategy40XX [lit-tests]
//

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy40XX(
        const VPU::InitCompilerOptions* initCompilerOptions, const DefaultHWOptions40XX* options) {
    auto wrapper = std::make_unique<DefaultHWSetup40XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategy40XX<DefaultHWSetup40XX>>(std::move(wrapper));
}

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy40XXReferenceSW(
        const VPU::InitCompilerOptions* initCompilerOptions, const DefaultHWOptions40XX* options) {
    auto wrapper = std::make_unique<ReferenceSWSetup40XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategyReferenceSW40XX>(std::move(wrapper));
}

/// The reason this method is separate from the default and reference compilation modes is that it has to *copy* the
/// options in order to override them.
template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy40XXWS(
        config::CompilationMode compilationMode, const VPU::InitCompilerOptions* initCompilerOptions,
        const DefaultHWOptions40XX* options) {
    switch (compilationMode) {
    case config::CompilationMode::WSInit: {
        auto wrapper = std::make_unique<WSInitSetup40XX>(initCompilerOptions, options);
        return std::make_unique<DialectPipelineStrategy40XX<WSInitSetup40XX>>(std::move(wrapper));
    }
    case config::CompilationMode::WSMain: {
        auto wrapper = std::make_unique<WSMainSetup40XX>(initCompilerOptions, options);
        return std::make_unique<DialectPipelineStrategy40XX<WSMainSetup40XX>>(std::move(wrapper));
    }
    default:
        VPUX_THROW("Unsupported compilation mode {0} for Monolithic WS.", config::stringifyEnum(compilationMode));
        return {};
    }
}

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy40XXHostCompile(
        config::CompilationMode compilationMode, const VPU::InitCompilerOptions* initCompilerOptions,
        const DefaultHWOptions40XX* options) {
    VPUX_THROW_UNLESS(compilationMode == config::CompilationMode::HostCompile,
                      "Unsupported compilation mode {0} for Host Compile.", config::stringifyEnum(compilationMode));

    auto wrapper = std::make_unique<HostCompileSetup40XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategy40XX<HostCompileSetup40XX>>(std::move(wrapper));
}
