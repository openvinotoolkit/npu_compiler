//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU40XX/pipeline_options.hpp"

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
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

    static void setupLitTestOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        Base::setupLitTestOptionsImpl(options, initCompilerOptions);
        setupOptionsCommon(options);
    }

    static void setupOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const vpux::OV::Config& config) {
        Base::setupOptionsImpl(options, initCompilerOptions, config);
        if (config.get<vpux::OV::TURBO>()) {
            overwriteIfUnset(options.optimizationLevel, 3);
        }
        setupOptionsCommon(options, config.get<vpux::OV::LOG_LEVEL>());
    }

    static void setupOptionsCommon(DefaultHWOptions40XX& options, LogLevel logLevel = LogLevel::None) {
        setupParamsAccordingToOptimizationLevel(options.optimizationLevel, options);
        setupPWLMParams(options, logLevel);
        if (options.enableSCFTiling) {
            overwriteIfUnset(options.enableBoundedTensorsToDynamicDimsMask, false);
        }
        // NPU40XX has no native SDPA execution, so every Attention op is decomposed.
        overwriteIfUnset(options.forceAttentionDecomposition, true);
    }
};

class ReferenceSWSetup40XX final : public OptionsSetupBase<ReferenceSWSetup40XX, DefaultHWOptions40XX> {
public:
    using Base = OptionsSetupBase<ReferenceSWSetup40XX, DefaultHWOptions40XX>;
    using Base::Base;

    static void setupOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const vpux::OV::Config& config) {
        Base::setupOptionsImpl(options, initCompilerOptions, config);
        setupOptionsCommon(options);
        setupPWLMParams(options, config.get<vpux::OV::LOG_LEVEL>());
    }

    static void setupOptionsCommon(DefaultHWOptions40XX& options) {
        // ReferenceSW specific values
        overwriteIfUnset(options.enableForceZMajorConcat, false);
        overwriteIfUnset(options.enableSwapTransposeWithFQ, false);
        overwriteIfUnset(options.enableAlignScales, false);
        overwriteIfUnset(options.enableConvertFCToConv, false);
        overwriteIfUnset(options.enableAdjustNonZeroFakeQuant, false);
        overwriteIfUnset(options.enableExtraStaticShapeOps, false);
        overwriteIfUnset(options.enableOptimizeReorders, false);
        overwriteIfUnset(options.enableVPUNNPreSplit, false);
        overwriteIfUnset(options.enableODULocalRegion, false);
        overwriteIfUnset(options.enableRuntimeDequant, false);

        overwriteIfUnset(options.enableConvertFFTToConv, false);
        overwriteIfUnset(options.enableConvertToAttention, false);
        // NPU40XX has no native SDPA execution, so every Attention op is decomposed.
        overwriteIfUnset(options.forceAttentionDecomposition, true);
        overwriteIfUnset(options.enableConvertToReduceSquare, true);
        overwriteIfUnset(options.enableDecomposeGRUSequence, false);
        overwriteIfUnset(options.workloadManagementMode, WorkloadManagementMode::PWLM_V0_1_PAGES);
    }
};

class HostCompileSetup40XX final : public OptionsSetupBase<HostCompileSetup40XX, DefaultHWOptions40XX> {
public:
    using Base = OptionsSetupBase<HostCompileSetup40XX, DefaultHWOptions40XX>;
    using Base::Base;

    static void setupLitTestOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        overwriteIfUnset(options.enableDynamicDimAlignment, false);
        setupHostPipelineOptionsCommon<DefaultHWOptions40XX>(options);
        DefaultHWSetup40XX::setupLitTestOptionsImpl(options, initCompilerOptions);
    }

    static void setupOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const vpux::OV::Config& config) {
        overwriteIfUnset(options.enableDynamicDimAlignment, false);
        setupHostPipelineOptionsCommon<DefaultHWOptions40XX>(options);
        DefaultHWSetup40XX::setupOptionsImpl(options, initCompilerOptions, config);
    }
};

class WSInitSetup40XX final : public OptionsSetupBase<WSInitSetup40XX, DefaultHWOptions40XX> {
public:
    using Base = OptionsSetupBase<WSInitSetup40XX, DefaultHWOptions40XX>;
    using Base::Base;

    static void setupLitTestOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        setupWSInitOptionsCommon<DefaultHWOptions40XX>(options);
        DefaultHWSetup40XX::setupLitTestOptionsImpl(options, initCompilerOptions);
    }

    static void setupOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const vpux::OV::Config& config) {
        setupWSInitOptionsCommon<DefaultHWOptions40XX>(options);
        DefaultHWSetup40XX::setupOptionsImpl(options, initCompilerOptions, config);
    }
};

class WSMainSetup40XX final : public OptionsSetupBase<WSMainSetup40XX, DefaultHWOptions40XX> {
public:
    using Base = OptionsSetupBase<WSMainSetup40XX, DefaultHWOptions40XX>;
    using Base::Base;

    static void setupLitTestOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        setupWSMainOptionsCommon<DefaultHWOptions40XX>(options);
        DefaultHWSetup40XX::setupLitTestOptionsImpl(options, initCompilerOptions);
    }

    static void setupOptionsImpl(DefaultHWOptions40XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const vpux::OV::Config& config) {
        setupWSMainOptionsCommon<DefaultHWOptions40XX>(options);
        DefaultHWSetup40XX::setupOptionsImpl(options, initCompilerOptions, config);
    }
};

//
// DialectPipelineStrategy40XX
//

template <class OptionsContainerType, class Enable = void>
class DialectPipelineStrategy40XX final : public IDialectPipelineStrategy {
public:
    explicit DialectPipelineStrategy40XX(const vpux::OV::Config& config)
            : _optionsContainer(std::make_unique<OptionsContainerType>(config)) {
    }

    explicit DialectPipelineStrategy40XX(std::unique_ptr<OptionsContainerType> optionsContainer)
            : _optionsContainer(std::move(optionsContainer)) {
    }

    void initializePipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::buildInitCompilerPipeline(pm, _optionsContainer->getInitCompilerOptions(), log.nest());
    }

    void buildDebatcherPipeline(mlir::OpPassManager& pm, Logger log) override {
        IE::buildDebatcherPipeline(pm, _optionsContainer->getPipelineOptions().getBatchCompileAdapter(), log);
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
    explicit DialectPipelineStrategyReferenceSW40XX(const vpux::OV::Config& config)
            : _optionsContainer(std::make_unique<ReferenceSWSetup40XX>(config)) {
    }

    explicit DialectPipelineStrategyReferenceSW40XX(std::unique_ptr<ReferenceSWSetup40XX> optionsContainer)
            : _optionsContainer(std::move(optionsContainer)) {
    }

    void initializePipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::buildInitCompilerPipeline(pm, _optionsContainer->getInitCompilerOptions(), log.nest());
    }

    void buildDebatcherPipeline(mlir::OpPassManager&, Logger log) override {
        log.warning("Debatching is not supported");
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
        config::CompilationMode compilationMode, const vpux::OV::Config& config) {
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

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy40XXHostCompile(
        config::CompilationMode compilationMode, const VPU::InitCompilerOptions* initCompilerOptions,
        const DefaultHWOptions40XX* options) {
    VPUX_THROW_UNLESS(compilationMode == config::CompilationMode::HostCompile,
                      "Unsupported compilation mode {0} for Host Compile.", config::stringifyEnum(compilationMode));

    auto wrapper = std::make_unique<HostCompileSetup40XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategy40XX<HostCompileSetup40XX>>(std::move(wrapper));
}

//
// createOptionsDefaultHW [unit-tests]
//

template <>
std::tuple<std::unique_ptr<VPU::InitCompilerOptions>, std::unique_ptr<DefaultHWOptions40XX>>
vpux::createOptionsDefaultHW(const vpux::OV::Config& config) {
    // NOTE: DefaultHWSetup40XX is defined in this file which is why helper is called
    auto defaultHWSetup = std::make_unique<DefaultHWSetup40XX>(config);
    return createOptionsDefaultHWHelper<DefaultHWSetup40XX, DefaultHWOptions40XX>(std::move(defaultHWSetup));
}
