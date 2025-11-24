//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/max_kernel_size_constant.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_SETUPMAXKERNELSIZE
#define GEN_PASS_DEF_SETUPMAXKERNELSIZE
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// SetupMaxKernelSizePass
//

class SetupMaxKernelSizePass final : public VPU::impl::SetupMaxKernelSizeBase<SetupMaxKernelSizePass> {
public:
    SetupMaxKernelSizePass() = default;
    SetupMaxKernelSizePass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
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
                 mlir::StringRef constantName, int64_t constantValue, bool allowCustomValues) {
    auto hasPipelineOption = pipelineOptionsOp.lookupSymbol<config::OptionOp>(constantName) != nullptr;
    VPUX_THROW_WHEN(!allowCustomValues && hasPipelineOption,
                    "Kernel size constant is already defined, probably you run '--init-compiler' twice");

    if (hasPipelineOption) {
        return;
    }
    auto* ctx = optionsBuilder.getContext();
    mlir::IntegerType sizeType = mlir::IntegerType::get(ctx, sizeof(void*) * 8, mlir::IntegerType::Signed);
    const auto constantAttr = mlir::StringAttr::get(ctx, constantName);
    optionsBuilder.create<config::OptionOp>(optionsBuilder.getUnknownLoc(), constantAttr,
                                            mlir::IntegerAttr::get(sizeType, constantValue));
}

mlir::LogicalResult SetupMaxKernelSizePass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }

    initializeFromOptions();

    return mlir::success();
}

void SetupMaxKernelSizePass::initializeFromOptions() {
    if (allowCustomValues.hasValue()) {
        _allowCustomValues = allowCustomValues.getValue();
    }
}

void SetupMaxKernelSizePass::safeRunOnModule() {
    auto moduleOp = getModuleOp(getOperation());
    auto optionsBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());
    auto pipelineOptionsOp = config::getPipelineOptionsOp(getContext(), moduleOp);
    optionsBuilder =
            mlir::OpBuilder::atBlockBegin(&pipelineOptionsOp.getOptions().front(), optionsBuilder.getListener());

    auto maxKernelSizeConstant = vpux::VPU::getMaxKernelSizeConstant(config::getArch(getOperation()));
    auto maxKernelSize = maxKernelSizeConstant.getMaxKernelSize();

    addConstant(optionsBuilder, pipelineOptionsOp, config::MAX_KERNEL_SIZE, maxKernelSize, _allowCustomValues);
}

}  // namespace

//
// createSetupMaxKernelSizePass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupMaxKernelSizePass() {
    return std::make_unique<SetupMaxKernelSizePass>();
}

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupMaxKernelSizePass(const VPU::InitCompilerOptions& initCompilerOptions,
                                                                    Logger log) {
    return std::make_unique<SetupMaxKernelSizePass>(initCompilerOptions, log);
}
