//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU60XX/pipeline_options.hpp"

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/pipelines/options_setup.hpp"

using namespace vpux;

namespace {

//
// OptionsSetup60XX
//

class DefaultHWSetup60XX final : public OptionsSetupBase<DefaultHWSetup60XX, DefaultHWOptions60XX> {
public:
    using Base = OptionsSetupBase<DefaultHWSetup60XX, DefaultHWOptions60XX>;
    using Base::Base;
    static void setupLitTestOptionsImpl(DefaultHWOptions60XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        Base::setupLitTestOptionsImpl(options, initCompilerOptions);
        setupOptionsCommon(options, initCompilerOptions);
    }

    static void setupOptionsImpl(DefaultHWOptions60XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const vpux::OV::Config& config) {
        Logger log("options-parser", config.get<vpux::OV::LOG_LEVEL>());
        Base::setupOptionsImpl(options, initCompilerOptions, config);
        if (config.get<vpux::OV::TURBO>()) {
            overwriteIfUnset(options.enableReduceNumTilesForSmallModelsPass, true);
        }
        setupOptionsCommon(options, initCompilerOptions);
        setupSparsityOptions60XX(options, log);
    }

    static void setupOptionsCommon(DefaultHWOptions60XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                   [[maybe_unused]] LogLevel logLevel = LogLevel::None) {
        setupPWLMParams60XX(options);
        setupDefaultHWOptionsCommon<DefaultHWOptions60XX>(options);
        // NPU60XX prefers spatial (H W) alignment of 8 for better DPU efficiency (numClusters * 8 HW tiling alignment).
        overwriteIfUnset(options.preferredSpatialAlignment, static_cast<int64_t>(8));
        overwriteIfUnset(initCompilerOptions.preferredSpatialAlignment, static_cast<int64_t>(8));
    }
};

class ReferenceSWSetup60XX final : public OptionsSetupBase<ReferenceSWSetup60XX, DefaultHWOptions60XX> {
public:
    using Base = OptionsSetupBase<ReferenceSWSetup60XX, DefaultHWOptions60XX>;
    using Base::Base;

    static void setupOptionsImpl(DefaultHWOptions60XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const vpux::OV::Config& config) {
        Base::setupOptionsImpl(options, initCompilerOptions, config);
        setupOptionsCommon(options);
    }

    static void setupOptionsCommon(DefaultHWOptions60XX& options) {
        setupPWLMParams60XX(options);
        setupReferenceSWOptionsCommon<DefaultHWOptions60XX>(options);

        // NPU60XX specific values
        overwriteIfUnset(options.enableConvertToAttention, false);
        overwriteIfUnset(options.enableConvertToReduceSquare, true);
        overwriteIfUnset(options.enableAutoPaddingIDU, false);
        overwriteIfUnset(options.enableAutoPaddingODU, false);
        overwriteIfUnset(options.enableAsymmetricPerTensorZP, false);
        overwriteIfUnset(options.enableAsymmetricPerChannelZP, false);
        overwriteIfUnset(options.enableIsReduceSupported, false);
        overwriteIfUnset(options.useMemrefForHostFunctionBufferization, false);
    }
};

class HostCompileSetup60XX final : public OptionsSetupBase<HostCompileSetup60XX, DefaultHWOptions60XX> {
public:
    using Base = OptionsSetupBase<HostCompileSetup60XX, DefaultHWOptions60XX>;
    using Base::Base;

    static void setupLitTestOptionsImpl(DefaultHWOptions60XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        setupHostPipelineOptionsCommon<DefaultHWOptions60XX>(options);
        DefaultHWSetup60XX::setupLitTestOptionsImpl(options, initCompilerOptions);
    }

    static void setupOptionsImpl(DefaultHWOptions60XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const vpux::OV::Config& config) {
        setupHostPipelineOptionsCommon<DefaultHWOptions60XX>(options);
        DefaultHWSetup60XX::setupOptionsImpl(options, initCompilerOptions, config);
    }
};

class WSInitSetup60XX final : public OptionsSetupBase<WSInitSetup60XX, DefaultHWOptions60XX> {
public:
    using Base = OptionsSetupBase<WSInitSetup60XX, DefaultHWOptions60XX>;
    using Base::Base;

    static void setupLitTestOptionsImpl(DefaultHWOptions60XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        setupWSInitOptionsCommon<DefaultHWOptions60XX>(options);
        DefaultHWSetup60XX::setupLitTestOptionsImpl(options, initCompilerOptions);
    }

    static void setupOptionsImpl(DefaultHWOptions60XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const vpux::OV::Config& config) {
        setupWSInitOptionsCommon<DefaultHWOptions60XX>(options);
        DefaultHWSetup60XX::setupOptionsImpl(options, initCompilerOptions, config);
    }
};

class WSMainSetup60XX final : public OptionsSetupBase<WSMainSetup60XX, DefaultHWOptions60XX> {
public:
    using Base = OptionsSetupBase<WSMainSetup60XX, DefaultHWOptions60XX>;
    using Base::Base;

    static void setupLitTestOptionsImpl(DefaultHWOptions60XX& options, VPU::InitCompilerOptions& initCompilerOptions) {
        setupWSMainOptionsCommon<DefaultHWOptions60XX>(options);
        DefaultHWSetup60XX::setupLitTestOptionsImpl(options, initCompilerOptions);
    }

    static void setupOptionsImpl(DefaultHWOptions60XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const vpux::OV::Config& config) {
        setupWSMainOptionsCommon<DefaultHWOptions60XX>(options);
        DefaultHWSetup60XX::setupOptionsImpl(options, initCompilerOptions, config);
    }
};

//
// DialectPipelineStrategy60XX
//

template <class OptionsContainerType, class Enable = void>
class DialectPipelineStrategy60XX final : public IDialectPipelineStrategy {
public:
    explicit DialectPipelineStrategy60XX(const vpux::OV::Config& config)
            : _optionsContainer(std::make_unique<OptionsContainerType>(config)) {
    }

    explicit DialectPipelineStrategy60XX(std::unique_ptr<OptionsContainerType> optionsContainer)
            : _optionsContainer(std::move(optionsContainer)) {
    }

    void initializePipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::buildInitCompilerPipeline(pm, _optionsContainer->getInitCompilerOptions(), log.nest());
    }

    void buildDebatcherPipeline(mlir::OpPassManager& pm, Logger log) override {
        IE::buildDebatcherPipeline(pm, _optionsContainer->getPipelineOptions().getBatchCompileAdapter(), log);
    }

    void buildIEPipeline(mlir::OpPassManager& pm, Logger log) override {
        IE::arch60xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::buildLowerIE2VPUPipeline(pm, log);
    }

    void buildVPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::arch60xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::buildLowerVPU2VPUIPPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildVPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPUIP::arch60xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

private:
    std::unique_ptr<OptionsContainerType> _optionsContainer;
};

//
// DialectPipelineStrategy60XX: [ReferenceSW]
// This implementation will be chosen if OptionsContainerType contains ReferenceSWOptions
//

class DialectPipelineStrategyReferenceSW60XX final : public IDialectPipelineStrategy {
public:
    explicit DialectPipelineStrategyReferenceSW60XX(const vpux::OV::Config& config)
            : _optionsContainer(std::make_unique<ReferenceSWSetup60XX>(config)) {
    }

    explicit DialectPipelineStrategyReferenceSW60XX(std::unique_ptr<ReferenceSWSetup60XX> optionsContainer)
            : _optionsContainer(std::move(optionsContainer)) {
    }

    void initializePipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::buildInitCompilerPipeline(pm, _optionsContainer->getInitCompilerOptions(), log.nest());
    }

    void buildDebatcherPipeline(mlir::OpPassManager&, Logger log) override {
        log.warning("Debatching is not supported");
    }

    void buildIEPipeline(mlir::OpPassManager& pm, Logger log) override {
        IE::arch60xx::buildReferenceSWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::buildLowerIE2VPUPipeline(pm, log);
    }

    void buildVPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::arch37xx::buildReferenceSWPipeline(
                pm, VPU::arch37xx::DefaultHWOptions(_optionsContainer->getPipelineOptions()), log);
    }

    void buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::buildLowerVPU2VPUIPPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildVPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPUIP::arch60xx::buildReferenceSWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

private:
    std::unique_ptr<ReferenceSWSetup60XX> _optionsContainer;
};

}  // namespace

//
// createDialectPipelineStrategy60XX
//

std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy60XX(
        config::CompilationMode compilationMode, const vpux::OV::Config& config) {
    switch (compilationMode) {
    case config::CompilationMode::DefaultHW: {
        return std::make_unique<DialectPipelineStrategy60XX<DefaultHWSetup60XX>>(config);
    }
    case config::CompilationMode::ReferenceSW: {
        return std::make_unique<DialectPipelineStrategyReferenceSW60XX>(config);
    }
    case config::CompilationMode::HostCompile_Interpreter: {
        return std::make_unique<DialectPipelineStrategy60XX<HostCompileSetup60XX>>(config);
    }
    case config::CompilationMode::WSInit: {
        return std::make_unique<DialectPipelineStrategy60XX<WSInitSetup60XX>>(config);
    }
    case config::CompilationMode::WSMain: {
        return std::make_unique<DialectPipelineStrategy60XX<WSMainSetup60XX>>(config);
    }
    default:
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }
}

//
// createDialectPipelineStrategy60XX [lit-tests]
//

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy60XX(
        const VPU::InitCompilerOptions* initCompilerOptions, const DefaultHWOptions60XX* options) {
    auto wrapper = std::make_unique<DefaultHWSetup60XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategy60XX<DefaultHWSetup60XX>>(std::move(wrapper));
}

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy60XXReferenceSW(
        const VPU::InitCompilerOptions* initCompilerOptions, const DefaultHWOptions60XX* options) {
    auto wrapper = std::make_unique<ReferenceSWSetup60XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategyReferenceSW60XX>(std::move(wrapper));
}

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy60XXHostCompile(
        config::CompilationMode compilationMode, const VPU::InitCompilerOptions* initCompilerOptions,
        const DefaultHWOptions60XX* options) {
    VPUX_THROW_UNLESS(compilationMode == config::CompilationMode::HostCompile_Interpreter,
                      "Unsupported compilation mode {0} for Host Compile.", config::stringifyEnum(compilationMode));

    auto wrapper = std::make_unique<HostCompileSetup60XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategy60XX<HostCompileSetup60XX>>(std::move(wrapper));
}

//
// createOptionsDefaultHW [unit-tests]
//

template <>
std::tuple<std::unique_ptr<VPU::InitCompilerOptions>, std::unique_ptr<DefaultHWOptions60XX>>
vpux::createOptionsDefaultHW(const vpux::OV::Config& config) {
    // NOTE: DefaultHWSetup60XX is defined in this file which is why helper is called
    auto defaultHWSetup = std::make_unique<DefaultHWSetup60XX>(config);
    return createOptionsDefaultHWHelper<DefaultHWSetup60XX, DefaultHWOptions60XX>(std::move(defaultHWSetup));
}
