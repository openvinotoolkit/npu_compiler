//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU50XX/pipeline_options.hpp"

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/pipelines/options_setup.hpp"

using namespace vpux;

namespace {

//
// OptionsSetup50XX
//

class DefaultHWSetup50XX final : public OptionsSetupBase<DefaultHWSetup50XX, DefaultHWOptions50XX> {
public:
    using Base = OptionsSetupBase<DefaultHWSetup50XX, DefaultHWOptions50XX>;
    using Base::Base;

    static void setupLitTestOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        Base::setupLitTestOptionsImpl(options, initCompilerOptions);
        setupOptionsCommon(options, initCompilerOptions);
    }

    static void setupOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const vpux::OV::Config& config) {
        Base::setupOptionsImpl(options, initCompilerOptions, config);
        if (config.get<vpux::OV::TURBO>()) {
            overwriteIfUnset(options.enableReduceNumTilesForSmallModelsPass, true);
            overwriteIfUnset(options.workloadManagementMode, WorkloadManagementMode::FWLM_V1_PAGES);
        }
        setupOptionsCommon(options, initCompilerOptions, config.get<vpux::OV::LOG_LEVEL>());

        const auto dynamicQuantization = getCompilerDynamicQuantization(config);
        if ((dynamicQuantization.has_value() && dynamicQuantization.value())) {
            options.weightsTableReuseMode = vpux::WeightsTableReuseMode::ENABLED;
        }
    }

    static void setupOptionsCommon(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                   LogLevel logLevel = LogLevel::None) {
        setupPWLMParams50XX(options, logLevel);
        if (options.enableSCFTiling) {
            overwriteIfUnset(options.enableBoundedTensorsToDynamicDimsMask, false);
        }

        const auto& platformOpt = initCompilerOptions.platform;
        if (platformOpt.hasValue()) {
            const auto platformOption = platformOpt.getValue();
            const auto platform = config::symbolizeEnum<config::Platform>(platformOption);
            VPUX_THROW_UNLESS(platform.has_value(), "Unsupported platform: {0}", platformOption);
            if (platform.value() == config::Platform::NPU5010) {
                // NPU5010 prefers spatial (H W) alignment of 8 for better DPU efficiency.
                overwriteIfUnset(options.preferredSpatialAlignment, static_cast<int64_t>(8));
                overwriteIfUnset(initCompilerOptions.preferredSpatialAlignment, static_cast<int64_t>(8));
            }
            // PTL supports pipeline-aware convolution split-over-IC in EnsureNCEOpsSizeRequirements, while WCL still
            // needs the legacy heuristic split pass scheduled in the tiling pipeline. TODO: E#208499 - remove the
            // platform split once IC splitting is handled uniformly by EnsureNCEOpsSizeRequirements and cost-based
            // tiling search.
            overwriteIfUnset(options.enableLegacyConvSplitOverIC,
                             !VPU::isPipelineAwareConvSplitOverICSupported(platform.value()));
        }
    }
};

class ReferenceSWSetup50XX : public OptionsSetupBase<ReferenceSWSetup50XX, DefaultHWOptions50XX> {
public:
    using Base = OptionsSetupBase<ReferenceSWSetup50XX, DefaultHWOptions50XX>;
    using Base::Base;

    static void setupOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const vpux::OV::Config& config) {
        Base::setupOptionsImpl(options, initCompilerOptions, config);
        setupOptionsCommon(options, config.get<vpux::OV::LOG_LEVEL>());
    }

    static void setupOptionsCommon(DefaultHWOptions50XX& options, LogLevel logLevel = LogLevel::None) {
        setupPWLMParams50XX(options, logLevel);
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
        overwriteIfUnset(options.enableConvertToReduceSquare, true);
        overwriteIfUnset(options.enableDecomposeGRUSequence, false);
        overwriteIfUnset(options.enableAutoPaddingIDU, false);
        overwriteIfUnset(options.enableAutoPaddingODU, false);
        overwriteIfUnset(options.enableIsReduceSupported, false);
    }
};

class HostCompileSetup50XX final : public OptionsSetupBase<HostCompileSetup50XX, DefaultHWOptions50XX> {
public:
    using Base = OptionsSetupBase<HostCompileSetup50XX, DefaultHWOptions50XX>;
    using Base::Base;

    static void setupLitTestOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        setupHostPipelineOptionsCommon<DefaultHWOptions50XX>(options);
        DefaultHWSetup50XX::setupLitTestOptionsImpl(options, initCompilerOptions);
    }

    static void setupOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const vpux::OV::Config& config) {
        setupHostPipelineOptionsCommon<DefaultHWOptions50XX>(options);
        DefaultHWSetup50XX::setupOptionsImpl(options, initCompilerOptions, config);
    }
};

class WSInitSetup50XX final : public OptionsSetupBase<WSInitSetup50XX, DefaultHWOptions50XX> {
public:
    using Base = OptionsSetupBase<WSInitSetup50XX, DefaultHWOptions50XX>;
    using Base::Base;

    static void setupLitTestOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        setupWSInitOptionsCommon<DefaultHWOptions50XX>(options);
        DefaultHWSetup50XX::setupLitTestOptionsImpl(options, initCompilerOptions);
    }

    static void setupOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const vpux::OV::Config& config) {
        setupWSInitOptionsCommon<DefaultHWOptions50XX>(options);
        DefaultHWSetup50XX::setupOptionsImpl(options, initCompilerOptions, config);
    }
};
class WSMainSetup50XX final : public OptionsSetupBase<WSMainSetup50XX, DefaultHWOptions50XX> {
public:
    using Base = OptionsSetupBase<WSMainSetup50XX, DefaultHWOptions50XX>;
    using Base::Base;

    static void setupLitTestOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        setupWSMainOptionsCommon<DefaultHWOptions50XX>(options);
        DefaultHWSetup50XX::setupLitTestOptionsImpl(options, initCompilerOptions);
    }

    static void setupOptionsImpl(DefaultHWOptions50XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const vpux::OV::Config& config) {
        setupWSMainOptionsCommon<DefaultHWOptions50XX>(options);
        DefaultHWSetup50XX::setupOptionsImpl(options, initCompilerOptions, config);
    }
};

//
// DialectPipelineStrategy50XX
//

template <class OptionsContainerType>
class DialectPipelineStrategy50XX final : public IDialectPipelineStrategy {
public:
    explicit DialectPipelineStrategy50XX(const vpux::OV::Config& config)
            : _optionsContainer(std::make_unique<OptionsContainerType>(config)) {
    }

    explicit DialectPipelineStrategy50XX(std::unique_ptr<OptionsContainerType> optionsContainer)
            : _optionsContainer(std::move(optionsContainer)) {
    }

    void initializePipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::buildInitCompilerPipeline(pm, _optionsContainer->getInitCompilerOptions(), log.nest());
    }

    void buildDebatcherPipeline(mlir::OpPassManager& pm, Logger log) override {
        IE::buildDebatcherPipeline(pm, _optionsContainer->getPipelineOptions().getBatchCompileAdapter(), log);
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
    explicit DialectPipelineStrategyReferenceSW50XX(const vpux::OV::Config& config)
            : _optionsContainer(std::make_unique<ReferenceSWSetup50XX>(config)) {
    }

    explicit DialectPipelineStrategyReferenceSW50XX(std::unique_ptr<ReferenceSWSetup50XX> optionsContainer)
            : _optionsContainer(std::move(optionsContainer)) {
    }

    void initializePipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::buildInitCompilerPipeline(pm, _optionsContainer->getInitCompilerOptions(), log.nest());
    }

    void buildDebatcherPipeline(mlir::OpPassManager&, Logger log) override {
        log.warning("Debatching is not supported");
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
        config::CompilationMode compilationMode, const vpux::OV::Config& config) {
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

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy50XXHostCompile(
        config::CompilationMode compilationMode, const VPU::InitCompilerOptions* initCompilerOptions,
        const DefaultHWOptions50XX* options) {
    VPUX_THROW_UNLESS(compilationMode == config::CompilationMode::HostCompile,
                      "Unsupported compilation mode {0} for Host Compile.", config::stringifyEnum(compilationMode));

    auto wrapper = std::make_unique<HostCompileSetup50XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategy50XX<HostCompileSetup50XX>>(std::move(wrapper));
}

//
// createOptionsDefaultHW [unit-tests]
//

template <>
std::tuple<std::unique_ptr<VPU::InitCompilerOptions>, std::unique_ptr<DefaultHWOptions50XX>>
vpux::createOptionsDefaultHW(const vpux::OV::Config& config) {
    // NOTE: DefaultHWSetup50XX is defined in this file which is why helper is called
    auto defaultHWSetup = std::make_unique<DefaultHWSetup50XX>(config);
    return createOptionsDefaultHWHelper<DefaultHWSetup50XX, DefaultHWOptions50XX>(std::move(defaultHWSetup));
}
