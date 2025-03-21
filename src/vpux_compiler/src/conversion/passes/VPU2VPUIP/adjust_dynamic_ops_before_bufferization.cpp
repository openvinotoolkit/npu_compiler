//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinTypes.h>

#include <functional>
#include <utility>
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/utils/core/checked_cast.hpp"

namespace vpux {
#define GEN_PASS_DECL_ADJUSTDYNAMICOPSBEFOREBUFFERIZATION
#define GEN_PASS_DEF_ADJUSTDYNAMICOPSBEFOREBUFFERIZATION
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// UnsqueezeRewrite
//

class UnsqueezeRewrite final : public mlir::OpRewritePattern<VPU::UnsqueezeOp> {
public:
    UnsqueezeRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::UnsqueezeOp>(ctx), _log(std::move(log)) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::UnsqueezeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UnsqueezeRewrite::matchAndRewrite(VPU::UnsqueezeOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Rewriting '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto ctx = origOp->getContext();

    const auto outputShape = getShape(origOp.getOutput());
    const auto outputRank = checked_cast<int64_t>(outputShape.size());
    const auto si32Type = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
    const auto shapeType = mlir::RankedTensorType::get({outputRank}, si32Type);

    auto shapeValues = mlir::SmallVector<int32_t>(outputRank);
    const auto dynamicToZero = [](int64_t dim) -> int32_t {
        return (dim != mlir::ShapedType::kDynamic) ? checked_cast<int32_t>(dim) : int32_t{0};
    };
    llvm::transform(outputShape, std::begin(shapeValues), dynamicToZero);

    const auto shapeTensor = Const::createConst(rewriter, origOp->getLoc(), shapeType, ArrayRef(shapeValues));

    const auto outputShapeAttr = getIntArrayAttr(ctx, outputShape.raw());
    const auto outputBoundsAttr = vpux::getBounds(origOp.getOutput());
    rewriter.replaceOpWithNewOp<VPU::DynamicReshapeOp>(origOp, origOp.getType(), origOp.getInput(), shapeTensor,
                                                       outputShapeAttr, outputBoundsAttr);

    return mlir::success();
}

//
// AdjustDynamicOpsBeforeBufferizationPass
//

class AdjustDynamicOpsBeforeBufferizationPass final :
        public impl::AdjustDynamicOpsBeforeBufferizationBase<AdjustDynamicOpsBeforeBufferizationPass> {
private:
    void safeRunOnModule() final;
};

void AdjustDynamicOpsBeforeBufferizationPass::safeRunOnModule() {
    auto& ctx = getContext();

    const auto hasDynamicTensors = [](mlir::Operation* op) {
        const auto isDynamic = [](mlir::Value value) {
            return getShape(value).isDynamic();
        };

        const auto hasDynamicInputs = llvm::any_of(op->getOperands(), isDynamic);
        const auto hasDynamicOutputs = llvm::any_of(op->getResults(), isDynamic);

        return hasDynamicInputs || hasDynamicOutputs;
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addDynamicallyLegalOp<VPU::UnsqueezeOp>(std::not_fn(hasDynamicTensors));
    target.addLegalOp<VPU::DynamicReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UnsqueezeRewrite>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAdjustDynamicOpsBeforeBufferizationPass
//

std::unique_ptr<mlir::Pass> vpux::createAdjustDynamicOpsBeforeBufferizationPass() {
    return std::make_unique<AdjustDynamicOpsBeforeBufferizationPass>();
}
