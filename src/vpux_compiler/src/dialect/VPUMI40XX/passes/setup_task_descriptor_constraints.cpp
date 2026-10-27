//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/config/utils/setup_pipeline_options_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_SETUPTASKDESCRIPTORCONSTRAINTS
#define GEN_PASS_DEF_SETUPTASKDESCRIPTORCONSTRAINTS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

class SetupTaskDescriptorConstraints :
        public VPUMI40XX::impl::SetupTaskDescriptorConstraintsBase<SetupTaskDescriptorConstraints> {
public:
    explicit SetupTaskDescriptorConstraints(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    explicit SetupTaskDescriptorConstraints(const VPUMI40XX::TaskDescriptorConstraintsPassArgs& args, Logger log) {
        dpuInvariantSize = static_cast<int64_t>(args.dpuInvariant);
        dpuVariantSize = static_cast<int64_t>(args.dpuVariant);
        actKernelInvocationSize = static_cast<int64_t>(args.actKernelInvocation);
        actKernelRangeSize = static_cast<int64_t>(args.actKernelRange);
        dmaSize = static_cast<int64_t>(args.dma);
        m2iSize = static_cast<int64_t>(args.m2i);
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void setConstraint(mlir::OpBuilder& builder, config::PipelineOptionsOp pipelineOptionsOp, mlir::StringRef name,
                   int64_t value) {
    if (pipelineOptionsOp.lookupSymbol<config::OptionOp>(name)) {
        return;
    }
    auto* ctx = builder.getContext();
    mlir::IntegerType i64 = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    config::OptionOp::create(builder, builder.getUnknownLoc(), mlir::StringAttr::get(ctx, name),
                             mlir::IntegerAttr::get(i64, value));
}

void SetupTaskDescriptorConstraints::safeRunOnModule() {
    auto moduleOp = getModuleOp(getOperation());
    auto arch = config::getArch(getOperation());
    auto pipelineOptionsOp = config::getPipelineOptionsOp(getContext(), moduleOp);
    auto builder = mlir::OpBuilder::atBlockBegin(&pipelineOptionsOp.getOptions().front());

    setConstraint(builder, pipelineOptionsOp, VPUMI40XX::TASK_BINARY_SIZE_DPU_INVARIANT_ATTR, dpuInvariantSize);
    setConstraint(builder, pipelineOptionsOp, VPUMI40XX::TASK_BINARY_SIZE_DPU_VARIANT_ATTR, dpuVariantSize);
    setConstraint(builder, pipelineOptionsOp, VPUMI40XX::TASK_BINARY_SIZE_ACT_KERNEL_INVOCATION_ATTR,
                  actKernelInvocationSize);
    setConstraint(builder, pipelineOptionsOp, VPUMI40XX::TASK_BINARY_SIZE_ACT_KERNEL_RANGE_ATTR, actKernelRangeSize);
    setConstraint(builder, pipelineOptionsOp, VPUMI40XX::TASK_BINARY_SIZE_DMA_ATTR, dmaSize);

    if (!config::isArchVPUX3XXX(arch)) {
        setConstraint(builder, pipelineOptionsOp, VPUMI40XX::TASK_BINARY_SIZE_M2I_ATTR, m2iSize);
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> VPUMI40XX::createSetupTaskDescriptorConstraintsPass(Logger log) {
    return std::make_unique<SetupTaskDescriptorConstraints>(log);
}

std::unique_ptr<mlir::Pass> VPUMI40XX::createSetupTaskDescriptorConstraintsPass(
        const TaskDescriptorConstraintsPassArgs& args, Logger log) {
    return std::make_unique<SetupTaskDescriptorConstraints>(args, log);
}
