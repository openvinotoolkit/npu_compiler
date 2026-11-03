//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters/fuse_conv_like_ops_with_slice_rewriter.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEMATMULWITHSLICE
#define GEN_PASS_DEF_FUSEMATMULWITHSLICE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// MoveSliceBeforeDynamicDequantize
//

/*
    Moves Slice consumer(s) above a DynamicDequantize producer.
    Slices can be introduced by the FuseWithSlice transformations above. Propagating the Slices
    further, through DynamicDequantize, will preserve DynDequant -> ConvLikeOp expected pattern for
    subsequent passes.

      Input(AxBx!qElemType)   Scale(Ax1xf16)
                    \           /
                DynamicDequantize(AxBxf16)
                   /            \
           Slice(A1xBxf16)   Slice(A2xBxf16)

      To:

      Input(AxBx!qElemType)                Input(AxBx!qElemType)
            |                                      |
      Slice(A1xBx!qElemType)              Slice(A2xBx!qElemType)
       |          \                        |           \
       |      Scale(A1x1xf16)              |       Scale(A2x1xf16)
       |      /                            |       /
      DynamicDequantize(A1xBxf16)         DynamicDequantize(A2xBxf16)
*/
class MoveSliceBeforeDynamicDequantize final : public mlir::OpRewritePattern<IE::DynamicDequantizeOp> {
public:
    MoveSliceBeforeDynamicDequantize(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DynamicDequantizeOp>(ctx), _log(log) {
        setDebugName("MoveSliceBeforeDynamicDequantize");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DynamicDequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveSliceBeforeDynamicDequantize::matchAndRewrite(IE::DynamicDequantizeOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("Got DynamicDequantize at '{0}'", origOp->getLoc());
    auto nestedLogger = _log.nest();

    if (origOp->use_empty()) {
        return matchFailed(nestedLogger, rewriter, origOp, "DynamicDequantize has no uses");
    }

    SmallVector<IE::SliceOp> slices;
    for (auto* user : origOp->getUsers()) {
        auto slice = mlir::dyn_cast_if_present<IE::SliceOp>(user);
        if (slice == nullptr) {
            return matchFailed(nestedLogger, rewriter, origOp, "Not all users are Slice ops");
        }
        slices.push_back(slice);
    }

    const auto outShape = getShape(origOp.getOutput());
    const auto scaleShape = getShape(origOp.getScale());
    if (outShape.isDynamic() || scaleShape.isDynamic()) {
        return matchFailed(nestedLogger, rewriter, origOp, "Output or scale shape is dynamic.");
    }

    const auto isScalarScale = scaleShape.totalSize() == 1;

    const auto zp = origOp.getZp();
    if (zp != nullptr && getShape(zp).isDynamic()) {
        return matchFailed(nestedLogger, rewriter, origOp, "Zero point shape is dynamic.");
    }

    const auto zpShape = zp != nullptr ? getShape(zp).raw() : ArrayRef<int64_t>{};
    const auto isScalarZp = zp != nullptr && getShape(zp).totalSize() == 1;

    // find the single dim each Slice narrows; it must be the same dim across all of them, and
    // (unless the scale/zp is a scalar) the scale/zp must vary over that same dim so it can be
    // sliced the same way as the input
    std::optional<int64_t> slicedDim;
    for (auto slice : slices) {
        const auto sliceSizes = parseIntArrayAttr<int64_t>(slice.getStaticSizes());
        const auto currentDim = IE::getSingleDiffAxis(ShapeRef(sliceSizes), ShapeRef(outShape));

        if (!currentDim.has_value()) {
            return matchFailed(nestedLogger, rewriter, origOp, "Slice at '{0}' does not narrow any dim",
                               slice->getLoc());
        }
        if (slicedDim.has_value() && slicedDim.value() != currentDim.value().ind()) {
            return matchFailed(nestedLogger, rewriter, origOp, "Slices narrow different dims");
        }
        slicedDim = currentDim.value().ind();
    }

    const auto dim = slicedDim.value();
    if (!isScalarScale &&
        (static_cast<size_t>(dim) >= scaleShape.size() || scaleShape[Dim(dim)] != outShape[Dim(dim)])) {
        return matchFailed(nestedLogger, rewriter, origOp, "Scale does not vary over the same dim the Slice(s) narrow");
    }
    if (zp != nullptr && !isScalarZp && zpShape[dim] != outShape[Dim(dim)]) {
        return matchFailed(nestedLogger, rewriter, origOp,
                           "ZeroPoint does not vary over the same dim the Slice(s) narrow");
    }

    // If slices are overlapped or they are invalid (e.g. different rank, nullptr etc.),
    // don't apply transformation.
    if (IE::slicesHaveOverlap(slices, Dim(dim)).value_or(true)) {
        return matchFailed(nestedLogger, rewriter, origOp, "Slice users overlap");
    }

    auto ctx = origOp->getContext();
    auto sliceQuantParameter = [&](mlir::Value param, ArrayRef<int64_t> offsets, ArrayRef<int64_t> shapes,
                                   mlir::Location loc) -> mlir::Value {
        const auto paramShape = getShape(param);
        const auto isScalar = paramShape.totalSize() == 1;
        if (isScalar) {
            return param;
        }

        auto scaleOffsets = SmallVector<int64_t>(paramShape.size(), 0);
        auto scaleSizes = SmallVector<int64_t>(paramShape.begin(), paramShape.end());
        scaleOffsets[dim] = offsets[dim];
        scaleSizes[dim] = shapes[dim];
        return rewriter.createOrFold<IE::SliceOp>(loc, param, getIntArrayAttr(ctx, scaleOffsets),
                                                  getIntArrayAttr(ctx, scaleSizes));
    };

    for (auto sliceIdx : irange(slices.size())) {
        auto slice = slices[sliceIdx];
        const auto sliceOffsetsAttr = slice.getStaticOffsets();
        const auto sliceSizesAttr = slice.getStaticSizes();
        const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOffsetsAttr);
        const auto sliceSizes = parseIntArrayAttr<int64_t>(sliceSizesAttr);

        mlir::IRMapping mapper;
        auto newInput = rewriter.createOrFold<IE::SliceOp>(appendLoc(slice->getLoc(), "slice_input_{0}", sliceIdx),
                                                           origOp.getInput(), sliceOffsetsAttr, sliceSizesAttr);
        mapper.map(origOp.getInput(), newInput);

        mlir::Value newScale = origOp.getScale();
        if (!isScalarScale) {
            newScale = sliceQuantParameter(origOp.getScale(), sliceOffsets, sliceSizes,
                                           appendLoc(slice->getLoc(), "slice_scale_{0}", sliceIdx));
            mapper.map(origOp.getScale(), newScale);
        }

        mlir::Value newZp = zp;
        if (zp != nullptr && !isScalarZp) {
            newZp = sliceQuantParameter(zp, sliceOffsets, sliceSizes,
                                        appendLoc(slice->getLoc(), "slice_zp_{0}", sliceIdx));

            mapper.map(zp, newZp);
        }

        auto newDequant = mlir::cast<IE::DynamicDequantizeOp>(rewriter.clone(*origOp.getOperation(), mapper));
        newDequant->setLoc(appendLoc(origOp->getLoc(), "{0}", sliceIdx));
        newDequant->getResult(0).setType(slice.getType());
        newDequant.setDstElemType(mlir::cast<NDTypeInterface>(slice.getType()).getElementType());

        rewriter.replaceOp(slice, newDequant.getOutput());
    }

    nestedLogger.trace("Moved Slice(s) above DynamicDequantize at '{0}'", origOp->getLoc());
    rewriter.eraseOp(origOp);
    return mlir::success();
}

//
// FuseMatMulWithSlicePass
//

class FuseMatMulWithSlicePass final : public IE::impl::FuseMatMulWithSliceBase<FuseMatMulWithSlicePass> {
public:
    explicit FuseMatMulWithSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseMatMulWithSlicePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<IE::FuseMatMulWithSlice>(&ctx, _log);
    patterns.add<IE::FuseFullyConnectedWithSlice>(&ctx, _log);
    collectOpsAndApplyPatterns(func, std::move(patterns));

    patterns.clear();
    patterns.add<MoveSliceBeforeDynamicDequantize>(&ctx, _log);
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createFuseMatMulWithSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseMatMulWithSlicePass(Logger log) {
    return std::make_unique<FuseMatMulWithSlicePass>(log);
}
