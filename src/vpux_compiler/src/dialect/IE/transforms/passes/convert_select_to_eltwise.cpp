//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTSELECTTOELTWISE
#define GEN_PASS_DEF_CONVERTSELECTTOELTWISE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool isSplatZeroConst(IE::SelectOp selectOp) {
    const auto trueConstOp = selectOp.getInput2().getDefiningOp<Const::DeclareOp>();
    if (trueConstOp == nullptr || !trueConstOp.getContentAttr().isSplat()) {
        return false;
    }
    return isDoubleEqual(trueConstOp.getContent().getSplatValue<double>(), 0.0);
}

//
// Helper: build a float mask from a constant mask with float or integer elements.
// keepNonZero=true  (Pattern A, LogicalNot):   const nonzero -> LogicalNot=false -> data kept  -> mask=1.0f
// keepNonZero=false (Pattern B, direct const): const nonzero -> cond true -> zero selected     -> mask=0.0f
//
mlir::Value buildFloatMask(mlir::PatternRewriter& rewriter, mlir::Location loc, Const::DeclareOp boolConstOp,
                           mlir::Type outElemType, vpux::ShapeRef maskShape, bool keepNonZero) {
    const auto maskType = mlir::RankedTensorType::get(maskShape.raw(), outElemType);
    const auto boolContent = boolConstOp.getContent();

    const auto toMaskVal = [keepNonZero](bool isNonZero) -> float {
        return (isNonZero == keepNonZero) ? 1.0f : 0.0f;
    };

    const auto constElemType = mlir::cast<vpux::NDTypeInterface>(boolConstOp.getOutput().getType()).getElementType();

    if (boolConstOp.getContentAttr().isSplat()) {
        const bool isNonZero = constElemType.isInteger() ? boolContent.getSplatValue<bool>()
                                                         : boolContent.getSplatValue<float>() != 0.0f;
        return Const::createFloatConst(rewriter, appendLoc(loc, "_mask"), maskType,
                                       ArrayRef<float>{toMaskVal(isNonZero)});
    }
    SmallVector<float> maskValues(static_cast<size_t>(maskType.getNumElements()));
    auto* ctx = rewriter.getContext();
    if (constElemType.isInteger()) {
        const auto boolVals = boolContent.getValues<bool>();
        loop_1d(LoopExecPolicy::Parallel, ctx, maskValues.size(), [&](size_t i) {
            maskValues[i] = toMaskVal(boolVals[i]);
        });
    } else {
        const auto floatVals = boolContent.getValues<float>();
        loop_1d(LoopExecPolicy::Parallel, ctx, maskValues.size(), [&](size_t i) {
            maskValues[i] = toMaskVal(floatVals[i] != 0.0f);
        });
    }
    return Const::createFloatConst(rewriter, appendLoc(loc, "_mask"), maskType, ArrayRef<float>(maskValues));
}

//
// ConvertSelectToMultiply
//
// Fuses the subgraph into IE.Multiply:
//
//
//   bool_const zero_const  data
//       |          |         |
// (IE.LogicalNot)  |    IE.Broadcast
//       \          |        /
//               IE.Select
//                  |
//                output
//
// Into:
//
//   float_mask  data
//       |         |
//       IE.Multiply
//           |
//         output
//

class ConvertSelectToMultiply final : public mlir::OpRewritePattern<IE::SelectOp> {
public:
    ConvertSelectToMultiply(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SelectOp>(ctx), _log(log) {
        setDebugName("ConvertSelectToMultiply");
    }

    mlir::LogicalResult matchAndRewrite(IE::SelectOp selectOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertSelectToMultiply::matchAndRewrite(IE::SelectOp selectOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", selectOp->getName(), selectOp->getLoc());

    // Only transform for f16/f32 — createFloatConst only supports these types.
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(selectOp.getOutput().getType()).getElementType();
    if (!mlir::isa<mlir::Float16Type, mlir::Float32Type>(outElemType)) {
        return mlir::failure();
    }

    if (selectOp.getAutoBroadcast() != IE::AutoBroadcastType::NUMPY) {
        _log.trace("IE.Select auto_broadcast is not NUMPY — skipping");
        return mlir::failure();
    }

    if (!isSplatZeroConst(selectOp)) {
        _log.trace("True branch is not a splat-zero constant");
        return mlir::failure();
    }

    auto broadcastOp = selectOp.getInput3().getDefiningOp<IE::BroadcastOp>();
    if (broadcastOp == nullptr || !broadcastOp->hasOneUse()) {
        _log.trace("False branch is not from IE.Broadcast or has multiple users — skipping");
        return mlir::failure();
    }

    // Only NUMPY and BIDIRECTIONAL modes are safe to replace with IE.Multiply(NUMPY).
    const auto broadcastMode = broadcastOp.getMode().value_or(IE::BroadcastType::NUMPY);
    if (broadcastMode != IE::BroadcastType::NUMPY && broadcastMode != IE::BroadcastType::BIDIRECTIONAL) {
        _log.trace("IE.Broadcast mode is not NUMPY or BIDIRECTIONAL — skipping");
        return mlir::failure();
    }

    // Guard: broadcast input must not tile any non-1 dimension.
    // If any input dim != 1 and != output dim, replacing Broadcast with Multiply(NUMPY)
    // would produce a different output shape — skip the transformation.
    // Input and output ranks may differ (NUMPY aligns from the right; leading dims are treated as 1).
    const auto broadcastInShape = getShape(broadcastOp.getInput());
    const auto broadcastOutShape = getShape(broadcastOp.getOutput());
    const auto inRank = broadcastInShape.size();
    const auto outRank = broadcastOutShape.size();

    if (inRank > outRank) {
        _log.trace("IE.Broadcast input rank exceeds output rank — malformed op, skipping");
        return mlir::failure();
    }

    const auto rankDiff = outRank - inRank;
    bool safeToBroadcast = true;
    for (size_t i = 0; i < outRank; ++i) {
        // Leading dims absent from input are implicitly 1 — always safe.
        if (i < rankDiff) {
            continue;
        }
        const auto inDim = broadcastInShape[Dim(i - rankDiff)];
        const auto outDim = broadcastOutShape[Dim(i)];
        if (inDim != 1 && inDim != outDim) {
            safeToBroadcast = false;
            break;
        }
    }
    if (!safeToBroadcast) {
        _log.trace("IE.Broadcast tiles a non-1 dimension — skipping");
        return mlir::failure();
    }

    // Determine mask source: resolve the constant and its shape before building the mask.
    // Pattern A: input1 = LogicalNot(const) — keepNonZero=true: bool nonzero means cond=false -> data kept ->
    // mask=1.0f. Pattern B: input1 = const directly   — keepNonZero=false: const nonzero means cond=true -> zero
    // selected -> mask=0.0f.
    Const::DeclareOp maskConstOp;
    vpux::ShapeRef maskShape;
    bool keepNonZero = false;
    if (auto logicalNotOp = selectOp.getInput1().getDefiningOp<IE::LogicalNotOp>()) {
        if (!logicalNotOp->hasOneUse()) {
            _log.trace("IE.LogicalNot has multiple users — skipping");
            return mlir::failure();
        }
        maskConstOp = logicalNotOp.getInput1().getDefiningOp<Const::DeclareOp>();
        if (maskConstOp == nullptr) {
            _log.trace("LogicalNot input is not a constant");
            return mlir::failure();
        }
        maskShape = getShape(logicalNotOp.getOutput());
        keepNonZero = true;
    } else if (auto condConstOp = selectOp.getInput1().getDefiningOp<Const::DeclareOp>()) {
        maskConstOp = condConstOp;
        maskShape = getShape(condConstOp.getOutput());
        keepNonZero = false;
    } else {
        _log.trace("Condition is neither IE.LogicalNot nor a compile-time constant");
        return mlir::failure();
    }

    const auto maskConstElemType =
            mlir::cast<vpux::NDTypeInterface>(maskConstOp.getOutput().getType()).getElementType();
    if (!maskConstElemType.isInteger() && !mlir::isa<mlir::Float16Type, mlir::Float32Type>(maskConstElemType)) {
        _log.trace("Mask constant element type is not supported — skipping");
        return mlir::failure();
    }

    // Verify that NUMPY broadcasting between data and mask produces the expected output shape.
    // If both data and mask have dim=1 on some axis while the select output has dim=N,
    // Multiply(NUMPY) would infer dim=1 there — incorrect.
    const auto dataShape = getShape(broadcastOp.getInput());
    const auto expectedOutShape = getShape(selectOp.getOutput());
    const auto dataRank = dataShape.size();
    const auto maskRank = maskShape.size();
    const auto expectedRank = expectedOutShape.size();
    const auto dataRankDiff = expectedRank - dataRank;
    const auto maskRankDiff = expectedRank - maskRank;
    bool shapeCompatible = true;
    for (size_t i = 0; i < expectedRank; ++i) {
        const auto dataDim = (i < dataRankDiff) ? int64_t(1) : dataShape[Dim(i - dataRankDiff)];
        const auto maskDim = (i < maskRankDiff) ? int64_t(1) : maskShape[Dim(i - maskRankDiff)];
        const auto inferredDim = std::max(dataDim, maskDim);
        if (inferredDim != expectedOutShape[Dim(i)]) {
            shapeCompatible = false;
            break;
        }
    }
    if (!shapeCompatible) {
        _log.trace("Multiply(NUMPY) cannot reproduce the Select output shape — skipping");
        return mlir::failure();
    }

    const auto maskOperand =
            buildFloatMask(rewriter, selectOp.getLoc(), maskConstOp, outElemType, maskShape, keepNonZero);

    const auto numpyBroadcastAttr = IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), IE::AutoBroadcastType::NUMPY);
    rewriter.replaceOpWithNewOp<IE::MultiplyOp>(selectOp, broadcastOp.getInput(), maskOperand, numpyBroadcastAttr,
                                                nullptr, nullptr, nullptr, nullptr);

    _log.trace("Replaced IE.Select with IE.Multiply done");
    return mlir::success();
}

//
// ConvertSelectToEltwisePass
//

class ConvertSelectToEltwisePass final : public IE::impl::ConvertSelectToEltwiseBase<ConvertSelectToEltwisePass> {
public:
    explicit ConvertSelectToEltwisePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertSelectToEltwisePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertSelectToMultiply>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createConvertSelectToEltwisePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertSelectToEltwisePass(Logger log) {
    return std::make_unique<ConvertSelectToEltwisePass>(log);
}
