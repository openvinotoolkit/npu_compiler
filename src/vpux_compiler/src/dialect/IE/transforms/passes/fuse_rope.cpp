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

struct RoPEConfig final {
    int64_t channel;
    int64_t width;
};

static const SmallVector<RoPEConfig> LEGAL_ROPE_CONFIGS = {{32, 128}, {1, 128}, {40, 128}, {8, 256},  {1, 256},
                                                           {4, 256},  {1, 64},  {64, 64},  {16, 128}, {2, 128}};
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

bool isInterleavedStridedSlice(mlir::Operation* lhsDataDef, mlir::Operation* rhsDataDef) {
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

// Walk through Slice/StridedSlice chains to find the underlying base tensor.
mlir::Value getRootTensor(mlir::Value val) {
    while (auto* defOp = val.getDefiningOp()) {
        if (!mlir::isa<IE::SliceOp, IE::StridedSliceOp>(defOp) || defOp->getNumOperands() == 0) {
            break;
        }
        val = defOp->getOperand(0);
    }
    return val;
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
    // Only skip reshape on concat operand 1 when it leads to StridedSlice/Slice (stride-2 pattern).
    // Skipping reshape unconditionally would also reach SplitOp and set isInterleaved=true for models
    auto concatInput1Op = concatOp.getOperand(1).getDefiningOp();
    if (mlir::isa_and_nonnull<IE::AffineReshapeOp, IE::ReshapeOp>(concatInput1Op) && concatInput1Op->hasOneUse()) {
        auto underlying = concatInput1Op->getOperand(0).getDefiningOp();
        if (mlir::isa_and_nonnull<IE::StridedSliceOp, IE::SliceOp>(underlying)) {
            concatInput1Op = underlying;
        }
    }
    auto stridedSliceOp2 = getSliceOrStridedSliceOp(concatInput1Op, isInterleaved);
    if (!mulOp3 || !stridedSliceOp2) {
        return mlir::failure();
    }

    auto stridedSliceOp1 =
            getSliceOrStridedSliceOp(skipReshapeIfPresent(mulOp3.getOperand(0).getDefiningOp()), isInterleaved);
    if (!stridedSliceOp1) {
        return mlir::failure();
    }

    if (!isInterleaved) {
        isInterleaved = isInterleavedStridedSlice(stridedSliceOp1, stridedSliceOp2);
    }

    auto inputCos = mulOp1.getOperand(1);
    auto inputSin = mulOp2.getOperand(1);

    const auto cosWidth = mlir::cast<mlir::RankedTensorType>(inputCos.getType()).getShape().back();
    const auto sinWidth = mlir::cast<mlir::RankedTensorType>(inputSin.getType()).getShape().back();

    // Determine the canonical RoPE input.
    // For INTERLEAVED mode, use the cos-multiply operand directly (the sin path uses
    // Reshape+Split, so stridedSliceOp1->getOperand(0) would be a reshape output).
    // For SPLIT_HALF, prefer stridedSliceOp1->getOperand(0) to correctly handle models
    // that insert a reshape before slicing. Fall back to mulOp1.getOperand(0) when a
    // prior pass (e.g. ResolveStridedSlice) has merged cascaded slice chains, leaving
    // stridedSliceOp1's parent wider than the RoPE region (width != cosWidth).
    mlir::Value input;
    if (isInterleaved) {
        input = mulOp1.getOperand(0);
    } else {
        const auto sliceParent = stridedSliceOp1->getOperand(0);
        const auto sliceParentType = mlir::dyn_cast<vpux::NDTypeInterface>(sliceParent.getType());
        if (sliceParentType && sliceParentType.getRank() == 4 && sliceParentType.getShape().back() == cosWidth) {
            input = sliceParent;
        } else {
            input = mulOp1.getOperand(0);
        }
    }

    auto tensorType = mlir::dyn_cast<vpux::NDTypeInterface>(input.getType());
    if (!tensorType || tensorType.getRank() != 4) {
        return mlir::failure();
    }
    // Verify that input, cos and sin last-dimension widths agree. A mismatch means the
    // pattern was only partially matched; fusing would produce incorrect results.
    const auto inputWidth = tensorType.getShape().back();
    if (inputWidth != cosWidth || inputWidth != sinWidth) {
        log.trace("RoPE fusion rejected for {0} at {1}: width mismatch (input={2}, cos={3}, sin={4})", addOp->getName(),
                  addOp->getLoc(), inputWidth, cosWidth, sinWidth);
        return mlir::failure();
    }
    // Verify that both sin-path slices share the same base tensor as the RoPE input.
    // Both slices must come from the same source, and that source must trace back to
    // the same root as input through any Slice/StridedSlice chain (e.g. in the
    // partial-width case: input = Slice[0:W](root), sin-path slices come from root).
    if (!isInterleaved) {
        const auto sinRoot = stridedSliceOp1->getOperand(0);
        if (sinRoot != stridedSliceOp2->getOperand(0)) {
            log.trace("RoPE fusion rejected for {0} at {1}: sin-path slices come from different sources",
                      addOp->getName(), addOp->getLoc());
            return mlir::failure();
        }
        if (getRootTensor(input) != getRootTensor(sinRoot)) {
            log.trace("RoPE fusion rejected for {0} at {1}: sin-path root is inconsistent with input source",
                      addOp->getName(), addOp->getLoc());
            return mlir::failure();
        }
    }
    // Currently we only support floating-point tensors for RoPE fusion
    const auto isFloatTensor = [](mlir::Value value) {
        const auto type = mlir::dyn_cast<vpux::NDTypeInterface>(value.getType());
        return type != nullptr && mlir::isa<mlir::FloatType>(type.getElementType());
    };

    if (!isFloatTensor(input) || !isFloatTensor(inputCos) || !isFloatTensor(inputSin)) {
        log.trace("RoPE fusion skipped for operation {0} at {1}: non-floating-point operands", addOp->getName(),
                  addOp->getLoc());
        return mlir::failure();
    }
    const auto shape = tensorType.getShape();

    constexpr int64_t unsupportedH = 1;
    if (shape[Dims4D::Act::H] == unsupportedH) {
        const auto channel = shape[Dims4D::Act::C];
        const auto width = shape[Dims4D::Act::W];

        const auto it =
                std::find_if(LEGAL_ROPE_CONFIGS.begin(), LEGAL_ROPE_CONFIGS.end(), [&](const RoPEConfig& config) {
                    return config.channel == channel && config.width == width;
                });
        if (it == LEGAL_ROPE_CONFIGS.end()) {
            return mlir::failure();
        }
    }

    // The RoPE op output inherits input's type and directly replaces addOp.
    if (addOp.getType() != input.getType()) {
        log.trace("RoPE fusion rejected for {0} at {1}: addOp result type incompatible with input type",
                  addOp->getName(), addOp->getLoc());
        return mlir::failure();
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

// Try to merge two adjacent IE::SliceOps that differ only in last-dim offset/size into a
// single wider SliceOp covering both regions. The slices must share the same source and
// agree on offsets and sizes in every dimension except the last. Returns the merged result
// Value on success, or a null Value on failure.
mlir::Value tryMergeAdjacentSlices(IE::SliceOp lhsSlice, IE::SliceOp rhsSlice, mlir::OpBuilder& builder, Logger log) {
    const auto lhsOffsets = parseIntArrayAttr<int64_t>(lhsSlice.getStaticOffsets());
    const auto lhsSizes = parseIntArrayAttr<int64_t>(lhsSlice.getStaticSizes());
    const auto rhsOffsets = parseIntArrayAttr<int64_t>(rhsSlice.getStaticOffsets());
    const auto rhsSizes = parseIntArrayAttr<int64_t>(rhsSlice.getStaticSizes());

    if (lhsOffsets.size() != rhsOffsets.size() || lhsOffsets.empty()) {
        return {};
    }

    const auto rank = static_cast<int64_t>(lhsOffsets.size());
    for (int64_t i = 0; i < rank - 1; ++i) {
        if (lhsOffsets[i] != rhsOffsets[i] || lhsSizes[i] != rhsSizes[i]) {
            return {};
        }
    }

    const auto lhsLastOffset = lhsOffsets[rank - 1];
    const auto lhsLastSize = lhsSizes[rank - 1];
    const auto rhsLastOffset = rhsOffsets[rank - 1];
    const auto rhsLastSize = rhsSizes[rank - 1];

    int64_t mergedLastOffset = 0;
    if (lhsLastOffset + lhsLastSize == rhsLastOffset) {
        mergedLastOffset = lhsLastOffset;
    } else if (rhsLastOffset + rhsLastSize == lhsLastOffset) {
        mergedLastOffset = rhsLastOffset;
    } else {
        return {};
    }

    SmallVector<int64_t> mergedOffsets(lhsOffsets.begin(), lhsOffsets.end());
    SmallVector<int64_t> mergedSizes(lhsSizes.begin(), lhsSizes.end());
    mergedOffsets[rank - 1] = mergedLastOffset;
    mergedSizes[rank - 1] = lhsLastSize + rhsLastSize;

    log.trace("Synthesized merged Slice for RoPE Pairwise: offsets=[{0}], sizes=[{1}]",
              llvm::make_range(mergedOffsets.begin(), mergedOffsets.end()),
              llvm::make_range(mergedSizes.begin(), mergedSizes.end()));
    return builder
            .create<IE::SliceOp>(lhsSlice.getLoc(), lhsSlice.getSource(), getIntArrayAttr(builder, mergedOffsets),
                                 getIntArrayAttr(builder, mergedSizes))
            .getResult();
}

mlir::LogicalResult fuseRoPEPairwisePattern(IE::ConcatOp concatOp, mlir::PatternRewriter& rewriter, Logger log) {
    if (concatOp.getInputs().size() != 2) {
        return mlir::failure();
    }

    SmallVector<mlir::Operation*> trailingReshapes;

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

    const auto inputShape = tensorType.getShape();
    const auto isInterleaved = isInterleavedStridedSlice(lhsDataDef, rhsDataDef);
    if (isInterleaved) {
        for (auto* user : concatOp->getUsers()) {
            if (!mlir::isa<IE::AffineReshapeOp, IE::ReshapeOp>(user)) {
                return mlir::failure();
            }

            if (user->getNumResults() != 1 || user->getResult(0).getType() != input.getType()) {
                return mlir::failure();
            }

            trailingReshapes.push_back(user);
        }

        if (trailingReshapes.empty()) {
            return mlir::failure();
        }

        auto concatType = mlir::dyn_cast<vpux::NDTypeInterface>(concatOp.getType());
        if (concatType == nullptr || concatType.getRank() != 5) {
            return mlir::failure();
        }
    }

    auto builder = mlir::OpBuilder(concatOp);

    // When the data slices cut both channels and width from a wider parent, `input` is the full
    // parent and its type does not match the concat result type. Merge the two half-width slices
    // into one covering the full RoPE width:
    //   Slice %src [0,40,0, 0][1,10,H, 64] --+
    //                                          +--> Slice %src [0,40,0,0][1,10,H,128]
    //   Slice %src [0,40,0,64][1,10,H, 64] --+
    if (!isInterleaved && concatOp.getType() != input.getType()) {
        auto lhsSliceOp = mlir::dyn_cast<IE::SliceOp>(lhsDataDef);
        auto rhsSliceOp = mlir::dyn_cast<IE::SliceOp>(rhsDataDef);
        if (!lhsSliceOp || !rhsSliceOp) {
            log.trace("RoPE Pairwise fusion rejected for {0} at {1}: type mismatch and data ops are not plain SliceOps",
                      concatOp->getName(), concatOp->getLoc());
            return mlir::failure();
        }
        auto mergedInput = tryMergeAdjacentSlices(lhsSliceOp, rhsSliceOp, builder, log);
        if (!mergedInput || mergedInput.getType() != concatOp.getType()) {
            log.trace("RoPE Pairwise fusion rejected for {0} at {1}: could not merge adjacent width slices",
                      concatOp->getName(), concatOp->getLoc());
            return mlir::failure();
        }
        input = mergedInput;
        tensorType = mlir::dyn_cast<vpux::NDTypeInterface>(input.getType());
    }

    auto inputCos = subLhs.trig;
    auto inputSin = subRhs.trig;
    auto cosShape = mlir::cast<mlir::RankedTensorType>(inputCos.getType()).getShape();
    auto sinShape = mlir::cast<mlir::RankedTensorType>(inputSin.getType()).getShape();

    // For RoPE Pairwise, sin and cos width should be inputW/2
    const auto inputWidth = inputShape[Dims4D::Act::W];
    const auto cosWidth = cosShape.back();
    const auto sinWidth = sinShape.back();
    if (cosWidth != inputWidth / 2 || sinWidth != inputWidth / 2) {
        return mlir::failure();
    }

    log.trace("RoPE Pairwise pattern matched for operation {0} at {1}", concatOp->getName(), concatOp->getLoc());
    if (cosShape != sinShape) {
        inputCos = builder.create<IE::ReshapeOp>(appendLoc(concatOp->getLoc(), "cos_reshape"), inputCos,
                                                 getIntArrayAttr(builder, sinShape));
        log.trace("Reshaped input_cos to match input_sin shape");
    }

    const auto ropeMode = IE::RoPEModeAttr::get(
            concatOp.getContext(), isInterleaved ? IE::RoPEMode::PAIRWISE_INTERLEAVED : IE::RoPEMode::PAIRWISE);
    auto ropeOp =
            builder.create<IE::RoPEOp>(appendLoc(concatOp->getLoc(), "rope"), input, inputCos, inputSin, ropeMode);

    // Pairwise-interleaved uses Concat only as a temporary 5D packing step; the trailing reshapes materialize the
    // real result shape, so rewrite those users directly. Plain pairwise produces the final result at Concat.
    if (isInterleaved) {
        for (auto* user : trailingReshapes) {
            rewriter.replaceOp(user, ropeOp.getOutput());
        }
    } else {
        rewriter.replaceOp(concatOp, ropeOp.getOutput());
    }

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
