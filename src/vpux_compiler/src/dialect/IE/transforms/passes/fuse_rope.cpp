//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEROPE
#define GEN_PASS_DEF_FUSEROPE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseRoPEPass
//

class FuseRoPEPass final : public IE::impl::FuseRoPEBase<FuseRoPEPass> {
public:
    explicit FuseRoPEPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

mlir::Operation* getSliceOrStridedSliceOp(mlir::Operation* op, bool& isInterleaved) {
    if (mlir::isa_and_nonnull<IE::SliceOp, IE::StridedSliceOp>(op)) {
        return op;
    }
    if (mlir::isa_and_nonnull<IE::SplitOp>(op)) {
        isInterleaved = true;
        return op;
    }

    return nullptr;
}

bool isSliceOrStridedSliceOp(mlir::Operation* op) {
    return mlir::isa_and_nonnull<IE::SliceOp, IE::StridedSliceOp>(op);
}

bool isPairwiseInterleaved(mlir::Operation* lhsDataDef, mlir::Operation* rhsDataDef) {
    auto lhsSlice = mlir::dyn_cast_or_null<IE::StridedSliceOp>(lhsDataDef);
    auto rhsSlice = mlir::dyn_cast_or_null<IE::StridedSliceOp>(rhsDataDef);
    if (!lhsSlice || !rhsSlice) {
        return false;
    }

    if (!lhsSlice.getBeginsAttr().has_value() || !rhsSlice.getBeginsAttr().has_value() ||
        !lhsSlice.getStridesAttr().has_value() || !rhsSlice.getStridesAttr().has_value()) {
        return false;
    }

    const auto lhsBegins = parseIntArrayAttr<int64_t>(lhsSlice.getBeginsAttr().value());
    const auto rhsBegins = parseIntArrayAttr<int64_t>(rhsSlice.getBeginsAttr().value());
    const auto lhsStrides = parseIntArrayAttr<int64_t>(lhsSlice.getStridesAttr().value());
    const auto rhsStrides = parseIntArrayAttr<int64_t>(rhsSlice.getStridesAttr().value());

    if (lhsBegins.size() != rhsBegins.size() || lhsStrides.size() != rhsStrides.size() ||
        lhsBegins.size() != lhsStrides.size() || lhsBegins.empty()) {
        return false;
    }

    if (!std::equal(lhsStrides.begin(), lhsStrides.end(), rhsStrides.begin())) {
        return false;
    }

    if (!std::all_of(lhsStrides.begin(), lhsStrides.end() - 1, [](int64_t stride) {
            return stride == 1;
        })) {
        return false;
    }

    if (lhsStrides.back() != 2) {
        return false;
    }

    if (!std::equal(lhsBegins.begin(), lhsBegins.end() - 1, rhsBegins.begin())) {
        return false;
    }

    const auto lhsLastBegin = lhsBegins.back();
    const auto rhsLastBegin = rhsBegins.back();
    return (lhsLastBegin == 0 && rhsLastBegin == 1) || (lhsLastBegin == 1 && rhsLastBegin == 0);
}

struct MulDataAndTrig {
    mlir::Value data;
    mlir::Value trig;
};

bool extractMulDataAndTrig(IE::MultiplyOp mulOp, MulDataAndTrig& out) {
    auto lhsDef = mulOp.getOperand(0).getDefiningOp();
    auto rhsDef = mulOp.getOperand(1).getDefiningOp();

    const auto lhsIsData = isSliceOrStridedSliceOp(lhsDef);
    const auto rhsIsData = isSliceOrStridedSliceOp(rhsDef);
    if (lhsIsData == rhsIsData) {
        return false;
    }

    out.data = lhsIsData ? mulOp.getOperand(0) : mulOp.getOperand(1);
    out.trig = lhsIsData ? mulOp.getOperand(1) : mulOp.getOperand(0);
    return true;
}

mlir::Operation* skipReshapeIfPresent(mlir::Operation* op) {
    if (!mlir::isa_and_nonnull<IE::AffineReshapeOp, IE::ReshapeOp>(op)) {
        return op;
    }
    if (!op->hasOneUse()) {
        return nullptr;
    }

    return op->getOperand(0).getDefiningOp();
}

mlir::LogicalResult fuseRoPEPattern(IE::AddOp addOp, mlir::PatternRewriter& rewriter, Logger log) {
    bool isInterleaved = false;

    auto mulOp1 = mlir::dyn_cast_or_null<IE::MultiplyOp>(skipReshapeIfPresent(addOp->getOperand(0).getDefiningOp()));
    auto mulOp2 = mlir::dyn_cast_or_null<IE::MultiplyOp>(addOp.getOperand(1).getDefiningOp());
    if (!mulOp1 || !mulOp2) {
        return mlir::failure();
    }

    auto concatOp = mlir::dyn_cast_or_null<IE::ConcatOp>(skipReshapeIfPresent(mulOp2.getOperand(0).getDefiningOp()));
    if (!concatOp || concatOp.getInputs().size() != 2) {
        return mlir::failure();
    }

    auto mulOp3 = mlir::dyn_cast_or_null<IE::MultiplyOp>(skipReshapeIfPresent(concatOp.getOperand(0).getDefiningOp()));
    auto stridedSliceOp2 = getSliceOrStridedSliceOp(concatOp.getOperand(1).getDefiningOp(), isInterleaved);
    if (!mulOp3 || !stridedSliceOp2) {
        return mlir::failure();
    }

    auto stridedSliceOp1 =
            getSliceOrStridedSliceOp(skipReshapeIfPresent(mulOp3.getOperand(0).getDefiningOp()), isInterleaved);
    if (!stridedSliceOp1) {
        return mlir::failure();
    }

    auto input = stridedSliceOp1->getOperand(0);
    auto inputCos = mulOp1.getOperand(1);
    auto inputSin = mulOp2.getOperand(1);

    // For interleaving, before SplitOp, the input is reshaped to <NxCxHxWx2>,
    // so the input has to be taken from the MultiplyOp.
    if (isInterleaved) {
        input = mulOp1->getOperand(0);
    }

    auto tensorType = mlir::dyn_cast<vpux::NDTypeInterface>(input.getType());
    if (!tensorType || tensorType.getRank() != 4) {
        return mlir::failure();
    }

    const auto shape = tensorType.getShape();

    // For avoiding performance decrease for certain networks, we limit the cases below for H = 1.
    // Follow next ticket for updates on generalizing the pass: E#162922
    constexpr int64_t unsupportedH = 1;
    const auto channelAndWidth = SmallVector<SmallVector<int64_t>>{{1, 64}, {64, 64}, {16, 128}, {2, 128}};

    if (shape[Dims4D::Act::H] == unsupportedH) {
        auto it = std::find(channelAndWidth.begin(), channelAndWidth.end(),
                            SmallVector<int64_t>{shape[Dims4D::Act::C], shape[Dims4D::Act::W]});
        if (it == channelAndWidth.end()) {
            return mlir::failure();
        }
    }

    log.trace("RoPE pattern matched for operation {0} at {1}", addOp->getName(), addOp->getLoc());
    auto builder = mlir::OpBuilder(addOp);
    auto cosShape = mlir::cast<mlir::RankedTensorType>(inputCos.getType()).getShape();
    auto sinShape = mlir::cast<mlir::RankedTensorType>(inputSin.getType()).getShape();
    if (cosShape != sinShape) {
        inputCos = builder.create<IE::ReshapeOp>(appendLoc(addOp->getLoc(), "cos_reshape"), inputCos,
                                                 getIntArrayAttr(builder, sinShape));
        log.trace("Reshaped input_cos to match input_sin shape");
    }

    const auto ropeMode = IE::RoPEModeAttr::get(addOp.getContext(),
                                                isInterleaved ? IE::RoPEMode::INTERLEAVED : IE::RoPEMode::SPLIT_HALF);
    auto ropeOp = builder.create<IE::RoPEOp>(appendLoc(addOp->getLoc(), "rope"), input, inputCos, inputSin, ropeMode);
    rewriter.replaceOp(addOp, ropeOp.getOutput());

    return mlir::success();
}

mlir::LogicalResult fuseRoPEPairwisePattern(IE::ConcatOp concatOp, mlir::PatternRewriter& rewriter, Logger log) {
    if (concatOp.getInputs().size() != 2) {
        return mlir::failure();
    }

    auto subOp = mlir::dyn_cast_or_null<IE::SubtractOp>(skipReshapeIfPresent(concatOp.getOperand(0).getDefiningOp()));
    auto addOp = mlir::dyn_cast_or_null<IE::AddOp>(skipReshapeIfPresent(concatOp.getOperand(1).getDefiningOp()));
    if (!subOp || !addOp) {
        return mlir::failure();
    }

    auto subMulLhs = mlir::dyn_cast_or_null<IE::MultiplyOp>(skipReshapeIfPresent(subOp.getOperand(0).getDefiningOp()));
    auto subMulRhs = mlir::dyn_cast_or_null<IE::MultiplyOp>(skipReshapeIfPresent(subOp.getOperand(1).getDefiningOp()));
    auto addMulLhs = mlir::dyn_cast_or_null<IE::MultiplyOp>(skipReshapeIfPresent(addOp.getOperand(0).getDefiningOp()));
    auto addMulRhs = mlir::dyn_cast_or_null<IE::MultiplyOp>(skipReshapeIfPresent(addOp.getOperand(1).getDefiningOp()));
    if (!subMulLhs || !subMulRhs || !addMulLhs || !addMulRhs) {
        return mlir::failure();
    }

    MulDataAndTrig subLhs;
    MulDataAndTrig subRhs;
    MulDataAndTrig addLhs;
    MulDataAndTrig addRhs;
    if (!extractMulDataAndTrig(subMulLhs, subLhs) || !extractMulDataAndTrig(subMulRhs, subRhs) ||
        !extractMulDataAndTrig(addMulLhs, addLhs) || !extractMulDataAndTrig(addMulRhs, addRhs)) {
        return mlir::failure();
    }

    // Match:
    // first = data0 * cos - data1 * sin
    // second = data0 * sin + data1 * cos
    const auto directOrder = addLhs.data == subLhs.data && addLhs.trig == subRhs.trig && addRhs.data == subRhs.data &&
                             addRhs.trig == subLhs.trig;
    const auto swappedAddOrder = addRhs.data == subLhs.data && addRhs.trig == subRhs.trig &&
                                 addLhs.data == subRhs.data && addLhs.trig == subLhs.trig;
    if (!directOrder && !swappedAddOrder) {
        return mlir::failure();
    }

    auto lhsDataDef = subLhs.data.getDefiningOp();
    auto rhsDataDef = subRhs.data.getDefiningOp();
    if (!isSliceOrStridedSliceOp(lhsDataDef) || !isSliceOrStridedSliceOp(rhsDataDef)) {
        return mlir::failure();
    }

    auto input = lhsDataDef->getOperand(0);
    if (rhsDataDef->getOperand(0) != input) {
        return mlir::failure();
    }

    auto tensorType = mlir::dyn_cast<vpux::NDTypeInterface>(input.getType());
    if (!tensorType || tensorType.getRank() != 4) {
        return mlir::failure();
    }

    // For avoiding specific network regressions, we limit the case below
    const auto inputShape = tensorType.getShape();
    if (!(inputShape[Dims4D::Act::C] == 256 && (inputShape[Dims4D::Act::H] == 1 || inputShape[Dims4D::Act::H] == 3))) {
        return mlir::failure();
    }

    const auto isInterleaved = isPairwiseInterleaved(lhsDataDef, rhsDataDef);
    if (isInterleaved) {
        return mlir::failure();
    }

    auto inputCos = subLhs.trig;
    auto inputSin = subRhs.trig;
    auto cosShape = mlir::cast<mlir::RankedTensorType>(inputCos.getType()).getShape();
    auto sinShape = mlir::cast<mlir::RankedTensorType>(inputSin.getType()).getShape();

    // For RoPE Pairwise, sin and cos width should be inputW/2
    const auto inputWidth = inputShape[Dims4D::Act::W];
    const auto cosWidth = cosShape[Dims4D::Act::W.ind()];
    const auto sinWidth = sinShape[Dims4D::Act::W.ind()];
    if (cosWidth != inputWidth / 2 || sinWidth != inputWidth / 2) {
        return mlir::failure();
    }

    auto builder = mlir::OpBuilder(concatOp);
    log.trace("RoPE Pairwise pattern matched for operation {0} at {1}", concatOp->getName(), concatOp->getLoc());
    if (cosShape != sinShape) {
        inputCos = builder.create<IE::ReshapeOp>(appendLoc(concatOp->getLoc(), "cos_reshape"), inputCos,
                                                 getIntArrayAttr(builder, sinShape));
        log.trace("Reshaped input_cos to match input_sin shape");
    }

    const auto ropeMode = IE::RoPEModeAttr::get(concatOp.getContext(), IE::RoPEMode::PAIRWISE);
    auto ropeOp =
            builder.create<IE::RoPEOp>(appendLoc(concatOp->getLoc(), "rope"), input, inputCos, inputSin, ropeMode);
    rewriter.replaceOp(concatOp, ropeOp.getOutput());

    return mlir::success();
}

class FuseRoPEAddPattern final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    FuseRoPEAddPattern(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
        setDebugName("FuseRoPEAddPattern");
    }

    mlir::LogicalResult matchAndRewrite(IE::AddOp addOp, mlir::PatternRewriter& rewriter) const final {
        return fuseRoPEPattern(addOp, rewriter, _log);
    }

private:
    Logger _log;
};

class FuseRoPEConcatPattern final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    FuseRoPEConcatPattern(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("FuseRoPEConcatPattern");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConcatOp concatOp, mlir::PatternRewriter& rewriter) const final {
        return fuseRoPEPairwisePattern(concatOp, rewriter, _log);
    }

private:
    Logger _log;
};

//
// safeRunOnFunc
//

// Match RoPE Split-Half pattern
// Input --------> IE.Multiply --------------------------------------------------
//       |                                                                      |
//       --> IE.StridedSlice -> IE.Multiply                                     |   -> IE.Add
//       |                                 | ---> IE.Concat ---> IE.Multiply ----
//       --> IE.StridedSlice ---------------                                              ^
//   |                                                                                    |
//   |                                                                                    |
//    -------------------------------------------------------------------------------------
// Or
// Input --------> IE.Multiply ---------------IE.AffineReshape-----------------------------------------
//       |                                                                                            |
//       |                    |--> IE.StridedSlice -> IE.Multiply                                     |   -> IE.Add
//       IE.AffineReshape ----|                                  | ---> IE.Concat ---> IE.Multiply ----
//                            |--> IE.StridedSlice ---------------                                              ^
//   |                                                                                                          |
//   |                                                                                                          |
//    -----------------------------------------------------------------------------------------------------------

// Match RoPE Interleaved pattern
// Input --------> IE.Multiply -----------------------------------------------------------------------
//       |                                                                                           |
//       |              |--> IE.Split -> IE.Reshape -> IE.Multiply -> IE.Reshape--|                  |
//       IE.Reshape ----|                                                         |---> IE.Concat ---| -> IE.Add
//                      |--> IE.Split --------------------------------------------|                         ^
//   |                                                                                                      |
//   --------------------------------------------------------------------------------------------------------

// Match RoPE Pairwise pattern
// Input ---> IE.Slice --> IE.Multiply (cos) --|
//       |                                      |--> IE.Subtract --|
//       |--> IE.Slice --> IE.Multiply (sin) --|                  |
//       |                                                         |--> IE.Concat
//       |--> IE.Slice --> IE.Multiply (sin) --|                  |
//       |                                      |--> IE.Add -------|
//       ---> IE.Slice --> IE.Multiply (cos) --|

void FuseRoPEPass::safeRunOnFunc() {
    auto func = getOperation();

    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseRoPEAddPattern>(&ctx, _log);
    patterns.add<FuseRoPEConcatPattern>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createFuseRoPEPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseRoPEPass(Logger log) {
    return std::make_unique<FuseRoPEPass>(log);
}
