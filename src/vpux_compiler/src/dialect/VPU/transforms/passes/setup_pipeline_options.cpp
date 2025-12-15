//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/ppe_factory.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/impl/ppe_factory.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/factories/cost_model_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/config/utils/setup_pipeline_options_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_SETUPPIPELINEOPTIONS
#define GEN_PASS_DEF_SETUPPIPELINEOPTIONS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// SetupPipelineOptionsPass
//

class SetupPipelineOptionsPass final : public VPU::impl::SetupPipelineOptionsBase<SetupPipelineOptionsPass> {
public:
    SetupPipelineOptionsPass() = default;
    SetupPipelineOptionsPass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(initCompilerOptions);

        initializeFromOptions();
    }

private:
    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;
    void safeRunOnModule() final;

private:
    config::ArchKind getArch();
    // Initialize fields from pass options
    void initializeFromOptions();

private:
    bool _allowCustomValues = false;
};

mlir::LogicalResult SetupPipelineOptionsPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }

    initializeFromOptions();

    return mlir::success();
}

config::ArchKind SetupPipelineOptionsPass::getArch() {
    VPUX_THROW_WHEN(platformOpt.hasValue() && archOpt.hasValue(), "Either 'platform' or 'vpu-arch' shall be set.");
    if (platformOpt.hasValue()) {
        const auto platform = config::symbolizeEnum<config::Platform>(platformOpt.getValue());
        VPUX_THROW_UNLESS(platform.has_value(), "Unknown NPU platform : '{0}'", platformOpt.getValue());
        return config::getArch(platform.value());
    } else {
        auto arch_opt = config::symbolizeEnum<config::ArchKind>(archOpt.getValue());
        VPUX_THROW_UNLESS(arch_opt.has_value(), "Unknown NPU architecture : '{0}'", archOpt.getValue());
        return arch_opt.value();
    }
}

void SetupPipelineOptionsPass::initializeFromOptions() {
    const auto arch = getArch();

    if (allowCustomValues.hasValue()) {
        _allowCustomValues = allowCustomValues.getValue();
    }

    // Register the default PPE factory singleton
    const auto& ppeVersion = ppeVersionOpt.getValue();
    if (ppeVersion == "Auto") {
        if (arch == config::ArchKind::NPU37XX || arch == config::ArchKind::NPU40XX) {
            VPU::PpeVersionConfig::setFactory<VPU::arch37xx::PpeFactory>();
            _log.info("Auto target PPE version set to: 'IntPPE'");
        } else {
            VPU::PpeVersionConfig::setFactory<VPU::arch50xx::PpeFactory>();
            _log.info("Auto target PPE version set to: 'FpPPE'");
        }
    } else if (ppeVersion == "IntPPE") {
        VPU::PpeVersionConfig::setFactory<VPU::arch37xx::PpeFactory>();
    } else if (ppeVersion == "FpPPE") {
        VPU::PpeVersionConfig::setFactory<VPU::arch50xx::PpeFactory>();
    } else {
        _log.error("Unknown PPE version name: '{0}'", ppeVersion);
    }

    // Register the default cost model factory singleton
    VPU::CostModelConfig::setFactory(arch);
    // Create the ShaveUtil based on how Factory was generated
    VPU::CostModelConfig::setCMShaveUtils(arch);
}

void SetupPipelineOptionsPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto moduleOp = getModuleOp(getOperation());

    const auto hasPipelineOptions =
            moduleOp.lookupSymbol<config::PipelineOptionsOp>(config::PIPELINE_OPTIONS) != nullptr;
    VPUX_THROW_WHEN(!_allowCustomValues && hasPipelineOptions,
                    "PipelineOptions operation is already defined, probably you run '--init-compiler' twice");

    if (hasPipelineOptions) {
        return;
    }

    auto optionsBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());
    auto pipelineOptionsOp =
            optionsBuilder.create<config::PipelineOptionsOp>(mlir::UnknownLoc::get(&ctx), config::PIPELINE_OPTIONS);
    pipelineOptionsOp.getOptions().emplaceBlock();
}

}  // namespace

//
// createSetupPipelineOptionsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupPipelineOptionsPass() {
    return std::make_unique<SetupPipelineOptionsPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupPipelineOptionsPass(
        const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
    return std::make_unique<SetupPipelineOptionsPass>(initCompilerOptions, log);
}
