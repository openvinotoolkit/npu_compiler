//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
#define GEN_PASS_DECL_DECOMPOSEAGGREGATEOPS
#define GEN_PASS_DEF_DECOMPOSEAGGREGATEOPS
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

static mlir::Operation* decomposeSoftmax(mlir::linalg::SoftmaxOp op) {
    mlir::OpBuilder builder(op->getContext());
    builder.setInsertionPoint(op);
    auto ctx = builder.getContext();
    auto loc = op.getLoc();
    auto ty = op.getInputOperandType();
    auto elTy = ty.getElementType();
    auto input = op.getOperand(0);
    auto output = op.getOutput();
    auto axis = checked_cast<int64_t>(op.getDimension());
    auto rank = ty.getRank();

    // Pre-compute iterator types and affine maps.
    //
    // The reductions keep the rank of the original operation and the reduced dimension becomes a unit
    // dimension in the result. We add an extra parallel iterator to the iteration
    // space and use it to index into the output reduced dimension. This keeps all output maps
    // as projected permutations, allowing the op to be tiled.
    const auto unitDim = rank;
    SmallVector<mlir::utils::IteratorType> reductionIterators(rank + 1, mlir::utils::IteratorType::parallel);
    reductionIterators[axis] = mlir::utils::IteratorType::reduction;
    SmallVector<mlir::AffineExpr> reductionInExprs;
    SmallVector<mlir::AffineExpr> reductionOutExprs;
    for (int64_t i = 0; i < rank; i++) {
        reductionInExprs.push_back(mlir::getAffineDimExpr(i, ctx));
        reductionOutExprs.push_back(mlir::getAffineDimExpr(i == axis ? unitDim : i, ctx));
    }
    auto reductionInMap = mlir::AffineMap::get(rank + 1, 0, reductionInExprs, ctx);
    auto reductionOutMap = mlir::AffineMap::get(rank + 1, 0, reductionOutExprs, ctx);

    SmallVector<mlir::utils::IteratorType> parallelIterators(rank, mlir::utils::IteratorType::parallel);
    auto parallelMap = mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
    SmallVector<mlir::AffineExpr> broadcastExprs(parallelMap.getResults().begin(), parallelMap.getResults().end());
    broadcastExprs[axis] = mlir::getAffineConstantExpr(0, ctx);
    auto broadcastMap = mlir::AffineMap::get(rank, 0, broadcastExprs, ctx);

    SmallVector<mlir::AffineMap> reductionMaps{reductionInMap, reductionOutMap};
    SmallVector<mlir::AffineMap> broadcastMaps{parallelMap, broadcastMap, parallelMap};
    SmallVector<mlir::OpFoldResult> dims = mlir::tensor::getMixedSizes(builder, loc, input);
    dims[axis] = builder.getIndexAttr(1);

    // REDUCE MAX
    auto maxFillVal = mlir::arith::getIdentityValue(mlir::arith::AtomicRMWKind::maximumf, elTy, builder, loc,
                                                    /*useOnlyFiniteValue=*/true);
    auto maxEmpt = mlir::tensor::EmptyOp::create(builder, loc, dims, elTy);
    auto maxInit = mlir::linalg::FillOp::create(builder, loc, mlir::ValueRange{maxFillVal}, mlir::ValueRange(maxEmpt))
                           .result();
    auto max = mlir::linalg::GenericOp::create(
                       builder, loc, maxEmpt.getType(), input, maxInit, reductionMaps, reductionIterators,
                       [&](mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange args) {
                           auto result = mlir::arith::MaximumFOp::create(
                                                 b, loc, args[0], args[1],
                                                 mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz)
                                                 .getResult();
                           mlir::linalg::YieldOp::create(b, loc, result);
                       })
                       ->getResult(0);

    // SUB+EXP
    auto subExp =
            mlir::linalg::GenericOp::create(
                    builder, loc, output.getType(), mlir::ValueRange{input, max}, output, broadcastMaps,
                    parallelIterators,
                    [&](mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange args) {
                        auto result = mlir::arith::SubFOp::create(b, loc, args[0], args[1]).getResult();
                        result = mlir::math::ExpOp::create(b, loc, result, mlir::arith::FastMathFlags::afn).getResult();
                        mlir::linalg::YieldOp::create(b, loc, result);
                    })
                    ->getResult(0);

    // REDUCE ADD
    mlir::Type accumElTy = elTy;
    auto addEmpt = maxEmpt;
    if (elTy.getIntOrFloatBitWidth() < 32) {
        accumElTy = mlir::Float32Type::get(ctx);
        addEmpt = mlir::tensor::EmptyOp::create(builder, loc, dims, accumElTy);
    }
    auto zero = mlir::arith::getIdentityValue(mlir::arith::AtomicRMWKind::addf, accumElTy, builder, loc,
                                              /*useOnlyFiniteValue=*/true);
    auto reduceAddInit =
            mlir::linalg::FillOp::create(builder, loc, mlir::ValueRange{zero}, mlir::ValueRange{addEmpt}).result();
    auto reduceAdd = mlir::linalg::GenericOp::create(
                             builder, loc, addEmpt.getType(), mlir::ValueRange{subExp}, reduceAddInit, reductionMaps,
                             reductionIterators,
                             [&](mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange args) {
                                 mlir::Value result = args[0];
                                 if (accumElTy != elTy) {
                                     result = mlir::arith::ExtFOp::create(b, loc, accumElTy, args[0]).getResult();
                                 }
                                 result = mlir::arith::AddFOp::create(b, loc, args[1], result,
                                                                      mlir::arith::FastMathFlags::reassoc)
                                                  .getResult();
                                 mlir::linalg::YieldOp::create(b, loc, result);
                             })
                             ->getResult(0);

    // DIV
    return mlir::linalg::GenericOp::create(
            builder, loc, output.getType(), mlir::ValueRange{subExp, reduceAdd}, output, broadcastMaps,
            parallelIterators, [&](mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange args) {
                mlir::Value result = args[0];
                if (accumElTy != elTy) {
                    result = mlir::arith::ExtFOp::create(b, loc, accumElTy, result).getResult();
                }
                result = mlir::arith::DivFOp::create(b, loc, result, args[1], mlir::arith::FastMathFlags::arcp)
                                 .getResult();
                if (accumElTy != elTy) {
                    result = mlir::arith::TruncFOp::create(b, loc, elTy, result).getResult();
                }
                mlir::linalg::YieldOp::create(b, loc, result);
            });
}

class DecomposeAggregateOpsPass final : public impl::DecomposeAggregateOpsBase<DecomposeAggregateOpsPass> {
public:
    explicit DecomposeAggregateOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

class SoftmaxDecompose : public mlir::OpRewritePattern<mlir::linalg::SoftmaxOp> {
public:
    using mlir::OpRewritePattern<mlir::linalg::SoftmaxOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::linalg::SoftmaxOp op, mlir::PatternRewriter& rewriter) const final {
        rewriter.replaceOp(op, decomposeSoftmax(op));
        return mlir::success();
    }
};

void DecomposeAggregateOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<mlir::linalg::SoftmaxOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SoftmaxDecompose>(&ctx);
    const mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    getOperation()->walk<mlir::WalkOrder::PreOrder>([&](IE::CodeGenCapsuleOp capsuleOp) {
        if (mlir::failed(mlir::applyPartialConversion(capsuleOp, target, frozenPatterns))) {
            signalPassFailure();
        }
        // Avoid walking the capsule contents.
        return mlir::WalkResult::skip();
    });
}

}  // namespace

std::unique_ptr<mlir::Pass> ShaveCodeGen::createDecomposeAggregateOpsPass(Logger log) {
    return std::make_unique<DecomposeAggregateOpsPass>(log);
}
