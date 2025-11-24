//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_constraint_utils.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_SETUPTILINGCONSTRAINT
#define GEN_PASS_DEF_SETUPTILINGCONSTRAINT
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// SetupTilingConstraintPass
//

class SetupTilingConstraintPass final : public VPU::impl::SetupTilingConstraintBase<SetupTilingConstraintPass> {
public:
    SetupTilingConstraintPass() = default;
    SetupTilingConstraintPass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
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

void addConstant(mlir::OpBuilder optionsBuilder, config::PipelineOptionsOp pipelineOptionsOp,
                 mlir::StringRef constantName, double constantValue, bool allowCustomValues) {
    auto hasPipelineOption = pipelineOptionsOp.lookupSymbol<config::OptionOp>(constantName) != nullptr;
    VPUX_THROW_WHEN(!allowCustomValues && hasPipelineOption,
                    "Kernel size constant is already defined, probably you run '--init-compiler' twice");

    if (hasPipelineOption) {
        return;
    }
    auto* ctx = optionsBuilder.getContext();
    auto sizeType = mlir::Float32Type::get(ctx);
    const auto constantAttr = mlir::StringAttr::get(ctx, constantName);
    optionsBuilder.create<config::OptionOp>(optionsBuilder.getUnknownLoc(), constantAttr,
                                            mlir::FloatAttr::get(sizeType, constantValue));
}

mlir::LogicalResult SetupTilingConstraintPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }

    initializeFromOptions();

    return mlir::success();
}

void SetupTilingConstraintPass::initializeFromOptions() {
    if (allowCustomValues.hasValue()) {
        _allowCustomValues = allowCustomValues.getValue();
    }
}

void SetupTilingConstraintPass::safeRunOnModule() {
    auto moduleOp = getModuleOp(getOperation());
    auto optionsBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());
    auto pipelineOptionsOp = config::getPipelineOptionsOp(getContext(), moduleOp);
    optionsBuilder =
            mlir::OpBuilder::atBlockBegin(&pipelineOptionsOp.getOptions().front(), optionsBuilder.getListener());

    auto largeFilterRatio =
            vpux::VPU::getFragmentationAvoidRatioPipeliningLargeWeights(config::getArch(getOperation()));

    addConstant(optionsBuilder, pipelineOptionsOp, config::FRAGMENTATION_AVOID_RATIO_PIPELINING_LARGE_WEIGHTS,
                largeFilterRatio, _allowCustomValues);
}

}  // namespace

//
// createSetupTilingConstraintPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupTilingConstraintPass() {
    return std::make_unique<SetupTilingConstraintPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupTilingConstraintPass(
        const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
    return std::make_unique<SetupTilingConstraintPass>(initCompilerOptions, log);
}
