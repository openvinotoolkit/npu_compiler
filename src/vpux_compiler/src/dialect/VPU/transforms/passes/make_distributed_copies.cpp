//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/overlap_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_MAKEDISTRIBUTEDCOPIES
#define GEN_PASS_DEF_MAKEDISTRIBUTEDCOPIES
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

class UnrolledTypeToCopyConversion final : public mlir::OpRewritePattern<VPU::UnrolledTypeOp> {
public:
    UnrolledTypeToCopyConversion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::UnrolledTypeOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::UnrolledTypeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UnrolledTypeToCopyConversion::matchAndRewrite(VPU::UnrolledTypeOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    auto isDistributedType = [](mlir::Value val) {
        auto distributedIf = val.getType().dyn_cast_or_null<VPU::DistributedTypeInterface>();
        return distributedIf != nullptr && distributedIf.containsDistributedTypes();
    };

    const bool isDistributedInput = isDistributedType(origOp.getInput());
    const bool isDistributedOutput = isDistributedType(origOp.getOutput());

    if (!isDistributedInput && !isDistributedOutput) {
        rewriter.replaceOp(origOp, origOp.getInput());
        return mlir::success();
    }

    IndexedSymbolAttr memSpace = nullptr;
    if (!isDistributedInput && isDistributedOutput) {
        memSpace = IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(MemoryKind::CMX_NN));
    }

    rewriter.replaceOpWithNewOp<VPU::CopyOp>(origOp, origOp.getType(), origOp.getInput(), memSpace);
    return mlir::success();
}

//
// MakeDistributedCopiesPass
//

class MakeDistributedCopiesPass final : public VPU::impl::MakeDistributedCopiesBase<MakeDistributedCopiesPass> {
public:
    MakeDistributedCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    };

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

void MakeDistributedCopiesPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPU::UnrolledTypeOp>();
    target.addLegalDialect<VPU::VPUDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalOp<VPU::CopyOp>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp, mlir::func::CallOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UnrolledTypeToCopyConversion>(&ctx, _log);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMakeDistributedCopiesPass
//

std::unique_ptr<mlir::Pass> VPU::createMakeDistributedCopiesPass(Logger log) {
    return std::make_unique<MakeDistributedCopiesPass>(log);
}
