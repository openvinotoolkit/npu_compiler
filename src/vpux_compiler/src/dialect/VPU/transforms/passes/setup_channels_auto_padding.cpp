//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

namespace {

//
// SetupChannelsAutoPaddingPass
//

class SetupChannelsAutoPaddingPass final : public VPU::SetupChannelsAutoPaddingBase<SetupChannelsAutoPaddingPass> {
public:
    SetupChannelsAutoPaddingPass() = default;
    SetupChannelsAutoPaddingPass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log)
            : _enableAutoPaddingODU(enableAutoPaddingODU), _enableAutoPaddingIDU(enableAutoPaddingIDU) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(initCompilerOptions);

        initializeFromOptions();
    }

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    void safeRunOnModule() final;

private:
    // Initialize fields from pass options
    void initializeFromOptions();

private:
    bool _enableAutoPaddingODU = false;
    bool _enableAutoPaddingIDU = false;
    bool _allowCustomValues = false;
};

void addOption(mlir::OpBuilder optionsBuilder, IE::PipelineOptionsOp pipelineOptionsOp, mlir::StringRef optionName,
               size_t optionValue, bool allowCustomValues) {
    auto hasPipelineOption = pipelineOptionsOp.lookupSymbol<IE::OptionOp>(optionName) != nullptr;
    VPUX_THROW_WHEN(!allowCustomValues && hasPipelineOption,
                    "ODU auto padding is already defined, probably you run '--init-compiler' twice");

    if (hasPipelineOption) {
        return;
    }
    auto* ctx = optionsBuilder.getContext();
    const auto constraintAttr = mlir::StringAttr::get(ctx, optionName);
    optionsBuilder.create<IE::OptionOp>(optionsBuilder.getUnknownLoc(), constraintAttr,
                                        mlir::BoolAttr::get(ctx, optionValue));
}

mlir::LogicalResult SetupChannelsAutoPaddingPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    initializeFromOptions();

    return mlir::success();
}

void SetupChannelsAutoPaddingPass::initializeFromOptions() {
    if (enableAutoPaddingODU.hasValue()) {
        _log.trace("Overloading the default value {0} of the '_enableAutoPaddingODU' field to the value {1} "
                   "of the pass option 'enableAutoPaddingODU' generated by MLIR",
                   _enableAutoPaddingODU, enableAutoPaddingODU);
        _enableAutoPaddingODU = enableAutoPaddingODU;
    }

    if (enableAutoPaddingIDU.hasValue()) {
        _log.trace("Overloading the default value {0} of the '_enableAutoPaddingIDU' field to the value {1} "
                   "of the pass option 'enableAutoPaddingIDU' generated by MLIR",
                   _enableAutoPaddingIDU, enableAutoPaddingIDU);
        _enableAutoPaddingIDU = enableAutoPaddingIDU;
    }

    if (allowCustomValues.hasValue()) {
        _allowCustomValues = allowCustomValues.getValue();
    }
}

void SetupChannelsAutoPaddingPass::safeRunOnModule() {
    auto moduleOp = getModuleOp(getOperation());
    auto optionsBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());
    auto pipelineOptionsOp = VPU::getPipelineOptionsOp(getContext(), moduleOp);
    optionsBuilder =
            mlir::OpBuilder::atBlockBegin(&pipelineOptionsOp.getOptions().front(), optionsBuilder.getListener());

    addOption(optionsBuilder, pipelineOptionsOp, VPU::AUTO_PADDING_ODU, _enableAutoPaddingODU, _allowCustomValues);
    addOption(optionsBuilder, pipelineOptionsOp, VPU::AUTO_PADDING_IDU, _enableAutoPaddingIDU, _allowCustomValues);
}

}  // namespace

//
// createSetupChannelsAutoPaddingPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupChannelsAutoPaddingPass() {
    return std::make_unique<SetupChannelsAutoPaddingPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupChannelsAutoPaddingPass(
        const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
    return std::make_unique<SetupChannelsAutoPaddingPass>(initCompilerOptions, log);
}
