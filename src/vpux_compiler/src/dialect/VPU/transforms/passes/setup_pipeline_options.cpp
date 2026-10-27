//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
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

void SetupPipelineOptionsPass::initializeFromOptions() {
    if (allowCustomValues.hasValue()) {
        _allowCustomValues = allowCustomValues.getValue();
    }
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
            config::PipelineOptionsOp::create(optionsBuilder, mlir::UnknownLoc::get(&ctx), config::PIPELINE_OPTIONS);
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
