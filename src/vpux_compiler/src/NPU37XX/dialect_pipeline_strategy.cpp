//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU37XX/conversion.hpp"
#include "vpux/compiler/NPU37XX/pipeline_options.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/pipelines/options_setup.hpp"

using namespace vpux;

namespace {

//
// OptionsSetup37XX
//

class DefaultHWSetup37XX : public OptionsSetupBase<DefaultHWSetup37XX, DefaultHWOptions37XX> {
public:
    using Base = OptionsSetupBase<DefaultHWSetup37XX, DefaultHWOptions37XX>;
    using Base::Base;
};

class ReferenceSWSetup37XX : public OptionsSetupBase<ReferenceSWSetup37XX, DefaultHWOptions37XX> {
public:
    using Base = OptionsSetupBase<ReferenceSWSetup37XX, DefaultHWOptions37XX>;
    using Base::Base;

    static void setupOptionsImpl(DefaultHWOptions37XX& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const intel_npu::Config& config) {
        Base::setupOptionsImpl(options, initCompilerOptions, config);
        setupOptionsCommon(options);
    }

    static void setupOptionsCommon(DefaultHWOptions37XX& options) {
        overwriteIfUnset(options.enableDummyOpReplacement, false);
        overwriteIfUnset(options.constantFoldingInBackground, false);
        overwriteIfUnset(options.enableMergeFakeQuant, true);
        overwriteIfUnset(options.enableOptimizeReorders, false);
        overwriteIfUnset(options.enableExperimentalSEPtrsOperations, false);
        overwriteIfUnset(options.enableFuseClampOperations, false);
        overwriteIfUnset(options.enableConvertPrecisionToFP16, true);
        overwriteIfUnset(options.enableConvertNonConstantPadToSliceAndConcat, true);
        overwriteIfUnset(options.enableSimpleSchedule, true);
        overwriteIfUnset(options.reduceParallelControlFlows, true);
        overwriteIfUnset(options.enableGroupedMatMul, false);
        overwriteIfUnset(options.fuseScalesToAccumulate, false);
        overwriteIfUnset(options.enableFP16CompressedConvolution, false);
        overwriteIfUnset(options.enableVPUNNPreSplit, false);
        overwriteIfUnset(options.enableInPlaceBufferization, false);
        overwriteIfUnset(options.useMemrefForHostFunctionBufferization, false);
        overwriteIfUnset(options.enableRuntimeDequant, false);

        // ReferenceSW specific values
        overwriteIfUnset(options.enableForceZMajorConcat, false);
        overwriteIfUnset(options.enableSwapTransposeWithFQ, false);
        overwriteIfUnset(options.enableAlignScales, false);
        overwriteIfUnset(options.fuseMvn6ScaleBias, false);
        overwriteIfUnset(options.enableConvertFCToConv, false);
        overwriteIfUnset(options.enableAdjustNonZeroFakeQuant, false);
        overwriteIfUnset(options.enableExtraStaticShapeOps, false);

        overwriteIfUnset(options.enableConvertFFTToConv, false);
        overwriteIfUnset(options.enableConvertToSdpaExtended, false);
        overwriteIfUnset(options.enableDecomposeGRUSequence, false);
    }
};

class WSInitSetup37XX : public WSInitSetupBase<WSInitSetup37XX, DefaultHWOptions37XX> {
public:
    using Base = WSInitSetupBase<WSInitSetup37XX, DefaultHWOptions37XX>;
    using Base::Base;
};

class WSMainSetup37XX : public WSMainSetupBase<WSMainSetup37XX, DefaultHWOptions37XX> {
public:
    using Base = WSMainSetupBase<WSMainSetup37XX, DefaultHWOptions37XX>;
    using Base::Base;
};

//
// DialectPipelineStrategy37XX
//

template <class OptionsContainerType, class Enable = void>
class DialectPipelineStrategy37XX final : public IDialectPipelineStrategy {
public:
    explicit DialectPipelineStrategy37XX(const intel_npu::Config& config)
            : _optionsContainer(std::make_unique<OptionsContainerType>(config)) {
    }

    explicit DialectPipelineStrategy37XX(std::unique_ptr<OptionsContainerType> optionsContainer)
            : _optionsContainer(std::move(optionsContainer)) {
    }

    void initializePipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::buildInitCompilerPipeline(pm, _optionsContainer->getInitCompilerOptions(), log.nest());
    }

    void buildIEPipeline(mlir::OpPassManager& pm, Logger log) override {
        IE::arch37xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::buildLowerIE2VPUPipeline(pm, log);
    }

    void buildVPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::arch37xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::buildLowerVPU2VPUIPPipeline(pm, _optionsContainer->getPipelineOptions().enableInPlaceBufferization,
                                          _optionsContainer->getPipelineOptions().useMemrefForHostFunctionBufferization,
                                          log);
    }

    void buildVPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPUIP::arch37xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

private:
    std::unique_ptr<OptionsContainerType> _optionsContainer;
};

//
// DialectPipelineStrategy37XX: [ReferenceSW]
// This implementation will be chosen if we have ReferenceSW setup
//

class DialectPipelineStrategyReferenceSW37XX final : public IDialectPipelineStrategy {
public:
    explicit DialectPipelineStrategyReferenceSW37XX(const intel_npu::Config& config)
            : _optionsContainer(std::make_unique<ReferenceSWSetup37XX>(config)) {
    }

    explicit DialectPipelineStrategyReferenceSW37XX(std::unique_ptr<ReferenceSWSetup37XX> optionsContainer)
            : _optionsContainer(std::move(optionsContainer)) {
    }

    void initializePipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::buildInitCompilerPipeline(pm, _optionsContainer->getInitCompilerOptions(), log.nest());
    }

    void buildIEPipeline(mlir::OpPassManager& pm, Logger log) override {
        IE::arch37xx::buildReferenceSWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::arch37xx::buildLowerIE2VPUPipelineReferenceSW(pm, log);
    }

    void buildVPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::arch37xx::buildReferenceSWPipeline(pm, log);
    }

    void buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::buildLowerVPU2VPUIPPipeline(pm, _optionsContainer->getPipelineOptions().enableInPlaceBufferization,
                                          /*useMemrefForHostFunctionBufferization*/ false, log);
    }

    void buildVPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPUIP::arch37xx::buildReferenceSWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

private:
    std::unique_ptr<ReferenceSWSetup37XX> _optionsContainer;
};

}  // namespace

//
// createDialectPipelineStrategy37XX
//

std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy37XX(
        config::CompilationMode compilationMode, const intel_npu::Config& config) {
    switch (compilationMode) {
    case config::CompilationMode::DefaultHW: {
        return std::make_unique<DialectPipelineStrategy37XX<DefaultHWSetup37XX>>(config);
    }
    case config::CompilationMode::ReferenceSW: {
        return std::make_unique<DialectPipelineStrategyReferenceSW37XX>(config);
    }
    case config::CompilationMode::WSInit: {
        return std::make_unique<DialectPipelineStrategy37XX<WSInitSetup37XX>>(config);
    }
    case config::CompilationMode::WSMain: {
        return std::make_unique<DialectPipelineStrategy37XX<WSMainSetup37XX>>(config);
    }
    default:
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }
}

//
// createDialectPipelineStrategy37XX [lit-tests]
//

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy37XX(
        const VPU::InitCompilerOptions* initCompilerOptions, const DefaultHWOptions37XX* options) {
    auto wrapper = std::make_unique<DefaultHWSetup37XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategy37XX<DefaultHWSetup37XX>>(std::move(wrapper));
}

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy37XXReferenceSW(
        const VPU::InitCompilerOptions* initCompilerOptions, const DefaultHWOptions37XX* options) {
    auto wrapper = std::make_unique<ReferenceSWSetup37XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategy37XX<ReferenceSWSetup37XX>>(std::move(wrapper));
}
