//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/optimize_slice_expand.hpp"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/expand_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Pass/PassManager.h>

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZESLICEEXPAND
#define GEN_PASS_DEF_OPTIMIZESLICEEXPAND
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

// IsLegal functions for support operations
bool IE::isMiddleOpLegal(IE::SliceOp sliceOp, mlir::Operation* op, IE::ExpandOp expandOp) {
    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<IE::ConcatOp>([&](IE::ConcatOp concatOp) {
                return isConcatLegal(sliceOp, concatOp, expandOp);
            })
            .Case<IE::PReluOp>([&](IE::PReluOp preluOp) {
                return isPReluLegal(sliceOp, preluOp, expandOp);
            })
            .Default([](mlir::Operation*) -> bool {
                return false;
            });
}

bool IE::isPReluLegal(IE::SliceOp sliceOp, IE::PReluOp preluOp, IE::ExpandOp expandOp) {
    if (preluOp == nullptr || preluOp->getNumResults() != 1 || !preluOp->hasOneUse()) {
        return false;
    }

    const auto expandAxis = IE::getExpandAxis(expandOp);
    if (!expandAxis.has_value()) {
        return false;
    }

    auto patternInShape = getShape(sliceOp.getSource());
    auto sliceOutShape = getShape(sliceOp.getResult());
    auto sliceAxis = IE::getSingleDiffAxis(patternInShape, sliceOutShape);
    if (!sliceAxis.has_value()) {
        return false;
    }

    const auto sliceAxisVal = sliceAxis.value();
    const auto expandAxisVal = expandAxis.value();

    if (sliceAxisVal != expandAxisVal) {
        return false;
    }

    auto patternOutShape = getShape(expandOp.getResult());
    if (patternInShape[sliceAxisVal] != patternOutShape[expandAxisVal]) {
        return false;
    }

    for (auto index : irange<unsigned>(1, preluOp->getOperands().size())) {
        auto input = preluOp.getOperand(index);
        if (!mlir::isa_and_nonnull<Const::DeclareOp>(input.getDefiningOp())) {
            auto partialSliceOp = input.getDefiningOp<IE::SliceOp>();
            if (partialSliceOp == nullptr) {
                return false;
            }
            const auto partialInShape = getShape(partialSliceOp.getSource());
            if (partialInShape[Dims4D::Act::C] != patternOutShape[Dims4D::Act::C]) {
                return false;
            }
        }
    }

    return true;
}

bool IE::isConcatLegal(IE::SliceOp maybeSliceOp, IE::ConcatOp concatOp, IE::ExpandOp expandOp) {
    if (concatOp == nullptr || concatOp->getNumResults() != 1 || !concatOp->hasOneUse()) {
        return false;
    }

    SmallVector<Dim> sliceAxes;
    SmallVector<std::pair<int32_t, mlir::Operation*>> sliceOpInfos;
    for (const auto& concatInput : concatOp.getInputs() | indexed) {
        auto inputOp = concatInput.value().getDefiningOp();
        if (mlir::isa_and_nonnull<Const::DeclareOp>(inputOp)) {
            sliceOpInfos.push_back(std::pair<int32_t, mlir::Operation*>(concatInput.index(), inputOp));
            continue;
        }

        auto sliceOp = maybeSliceOp;
        if (sliceOp == nullptr) {
            sliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(inputOp);
            if (sliceOp == nullptr) {
                continue;
            }
        }

        auto sliceAxis = IE::getSingleDiffAxis(getShape(sliceOp.getSource()), getShape(sliceOp.getResult()));
        if (!sliceAxis.has_value()) {
            return false;
        }

        if (sliceAxes.empty() || sliceAxis.value() != sliceAxes.back()) {
            sliceAxes.push_back(sliceAxis.value());
        }

        sliceOpInfos.push_back(std::pair<int32_t, mlir::Operation*>(concatInput.index(), sliceOp));
    }

    const auto concatAxis = getConcatAxis(concatOp);
    const auto expandAxis = getExpandAxis(expandOp);

    if (sliceAxes.size() != 1 || !concatAxis.has_value() || !expandAxis.has_value()) {
        return false;
    }

    const auto sliceAxisVal = sliceAxes.front();
    const auto concatAxisVal = concatAxis.value();
    const auto expandAxisVal = expandAxis.value();

    if (sliceAxisVal != expandAxisVal) {
        return false;
    }

    const auto expandOutShape = to_small_vector(getShape(expandOp.getResult()));
    const auto expandPadsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
    const auto expandPadsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEnd());

    // Only consider the 'slice' and 'expand' can be completely eliminated currently
    // TODO(E#95438): Remove part of 'slice' or 'expand' Op
    const auto checkDim = sliceAxisVal.ind();
    if (concatAxisVal != sliceAxisVal) {
        const auto isLegalSliceOp = [&](const auto& sliceOpInfo) {
            auto op = sliceOpInfo.second;
            if (mlir::isa_and_nonnull<Const::DeclareOp>(op)) {
                return true;
            }
            auto sliceOp = mlir::cast<IE::SliceOp>(op);
            const auto sliceInShape = to_small_vector(getShape(sliceOp.getSource()));
            const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
            return sliceOffsets[checkDim] == expandPadsBegin[checkDim] &&
                   sliceInShape[checkDim] == expandOutShape[checkDim];
        };

        if (concatOp.getInputs().size() != sliceOpInfos.size() || !llvm::all_of(sliceOpInfos, isLegalSliceOp)) {
            return false;
        }
    }

    if (concatAxisVal == sliceAxisVal) {
        const auto isLegalSliceOp = [&](const auto& sliceOpInfo) {
            auto inputIdx = sliceOpInfo.first;
            auto op = sliceOpInfo.second;
            if (mlir::isa_and_nonnull<Const::DeclareOp>(op)) {
                return true;
            }
            auto sliceOp = mlir::cast<IE::SliceOp>(op);
            const auto sliceInShape = to_small_vector(getShape(sliceOp.getSource()));
            const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
            const auto sliceStaticSizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
            if (inputIdx == 0) {
                return sliceOffsets[checkDim] == expandPadsBegin[checkDim] &&
                       sliceInShape[checkDim] == sliceOffsets[checkDim] + sliceStaticSizes[checkDim];
            } else if (inputIdx == checked_cast<int64_t>(concatOp.getInputs().size()) - 1) {
                return sliceOffsets[checkDim] == 0 &&
                       sliceInShape[checkDim] == expandPadsEnd[checkDim] + sliceStaticSizes[checkDim];
            } else {
                return false;
            }
        };

        if (!llvm::all_of(sliceOpInfos, isLegalSliceOp)) {
            return false;
        }
    }

    return true;
}

mlir::Value createNewConstValue(Const::DeclareOp constOp, Dim expandAxisVal, ShapeRef expandOutShape,
                                mlir::PatternRewriter& rewriter) {
    const auto constShape = getShape(constOp);
    int64_t padding = expandOutShape[expandAxisVal] - constShape[expandAxisVal];
    SmallVector<int64_t> padBegin(constShape.size(), 0);
    SmallVector<int64_t> padEnd(constShape.size(), 0);
    padEnd[expandAxisVal.ind()] = padding;
    auto contentAttr = constOp.transformContentAttr().padWithZero(ShapeRef(padBegin), ShapeRef(padEnd)).get();
    return rewriter.create<Const::DeclareOp>(constOp->getLoc(), contentAttr.getType(), std::move(contentAttr))
            .getResult();
}

IE::FuseMode vpux::IE::getFuseMode(ShapeRef patternInShape, ShapeRef patternOutShape) {
    VPUX_THROW_UNLESS(patternInShape.size() == patternOutShape.size(),
                      "The size of the input '{0}' and output '{1}' tensors does not match", patternInShape.size(),
                      patternOutShape.size());
    const auto inOutShapes = zip(patternInShape, patternOutShape);
    const auto isAllInShapeLargerThanOut = llvm::all_of(inOutShapes, [](const auto& inOutShape) {
        return std::get<0>(inOutShape) >= std::get<1>(inOutShape);
    });
    return isAllInShapeLargerThanOut ? IE::FuseMode::CONVERT_TO_SLICE : IE::FuseMode::CONVERT_TO_EXPAND;
}

// Pattern 1: 'SliceOp -> Implicit(optional) -> ExpandOp' convert to 'SliceOp' that should has following limitations:
// 1. padBegin < = sliceOffset
// 2. sliceOffset + sliceStaticSize + padEnd < = inputLen
// And we can get:
// newSliceOffset = sliceOffset - padBegin
// newSliceStaticSize = padBegin + sliceStaticSize + padEnd
//
// InData: |------------------------------------|
//                         inputLen
//                                                           InData: |------------------------------------|
// Slice:  |         |------------------|                                         inputLen
//         sliceOffset  sliceStaticSize
//                                                   ->      Slice:  |    |----------------------------|
// Expand:      |----|------------------|----|                   newSliceOffset   newSliceStaticSize
//           padBegin + sliceStaticSize + padEnd
//                                                           OutData:     |----------------------------|
// OutData:     |----------------------------|                                      outputLen
//                         outputLen
//
// Pattern 2: 'SliceOp -> Implicit(optional) -> ExpandOp' convert to 'ExpandOp' that should has following limitations:
// 1. padBegin > = sliceOffset
// 2. sliceOffset + sliceStaticSize + padEnd > = inputLen
// And we can get:
// newPadBegin = padBegin - sliceOffset
// newPadEnd = padEnd - (inputLen - sliceOffset - sliceStaticSize)
//
// InData:       |----------------------------|
//                          inputLen
//                                                           InData:       |----------------------------|
// Slice:        |   |--------------------|                                         inputLen
//           sliceOffset sliceStaticSize
//                                                     ->    Expand:  |----|----------------------------|---|
// Expand:  |--------|--------------------|-------|                newPadBegin        inputLen        newPadEnd
//           padBegin   sliceStaticSize     padEnd
//                                                           OutData: |-------------------------------------|
// OutData: |-------------------------------------|                                    outputLen
//                         outputLen
//
mlir::FailureOr<std::tuple<Shape, Shape, IE::FuseMode>> vpux::IE::getSliceExpandFusedParameters(IE::SliceOp sliceOp,
                                                                                                IE::ExpandOp expandOp) {
    const auto patternInShape = getShape(sliceOp.getSource());
    const auto patternOutShape = getShape(expandOp.getResult());
    const auto rank = patternInShape.size();

    const auto fuseMode = getFuseMode(patternInShape, patternOutShape);

    const auto expandPadsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
    const auto expandPadsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEnd());
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
    const auto sliceStaticSizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());

    // CONVERT_TO_SLICE:  the 'firstShapeRef' is 'newSliceOffsets'; the 'secondShapeRef' is 'newSliceStaticSizes'
    // CONVERT_TO_EXPAND: the 'firstShapeRef' is 'newPadsBegin'; the 'secondShapeRef' is 'newPadsEnd'
    SmallVector<int64_t> firstShapeRef(rank, 0);
    SmallVector<int64_t> secondShapeRef(rank, 0);
    for (auto idx : irange(rank)) {
        const auto inputLen = patternInShape[Dim(idx)];
        const auto sliceOffset = sliceOffsets[idx];
        const auto sliceStaticSize = sliceStaticSizes[idx];
        const auto padBegin = expandPadsBegin[idx];
        const auto padEnd = expandPadsEnd[idx];

        const auto outDataMaxRange = sliceOffset + sliceStaticSize + padEnd;
        if (fuseMode == IE::FuseMode::CONVERT_TO_SLICE && padBegin <= sliceOffset && outDataMaxRange <= inputLen) {
            firstShapeRef[idx] = sliceOffset - padBegin;
            secondShapeRef[idx] = padBegin + sliceStaticSize + padEnd;
        } else if (fuseMode == IE::FuseMode::CONVERT_TO_EXPAND && padBegin >= sliceOffset &&
                   outDataMaxRange >= inputLen) {
            firstShapeRef[idx] = padBegin - sliceOffset;
            secondShapeRef[idx] = padEnd - (inputLen - sliceOffset - sliceStaticSize);
        } else {
            return mlir::failure();
        }
    }

    return std::tuple<Shape, Shape, IE::FuseMode>(firstShapeRef, secondShapeRef, fuseMode);
}

// Pattern 1: 'ExpandOp -> Implicit(optional) -> SliceOp' convert to 'SliceOp' that should has following limitations:
// 1. padBegin < = sliceOffset
// 2. padBegin + inputLen > = sliceOffset + sliceStaticSize
// And we can get:
// newSliceOffset = sliceOffset - padBegin
// newSliceStaticSize = sliceStaticSize
//
// InData:       |-----------------|
//                    inputLen
//                                                           InData:       |-----------------|
// Expand:  |----|-----------------|------|                                      inputLen
//         padBegin   inputLen      padEnd
//                                                   ->      Slice:        |     |--------|
// Slice:   |          |--------|                                   newSliceOffset  newSliceStaticSize
//        sliceOffset sliceStaticSize
//                                                           OutData:            |--------|
// OutData:            |--------|                                                 outputLen
//                      outputLen
//
// Pattern 2: 'ExpandOp -> Implicit(optional) -> SliceOp' convert to 'Expand' that should has following limitations:
// 1. padBegin > = sliceOffset
// 2. padBegin + inputLen < = sliceOffset + sliceStaticSize
// And we can get:
// newPadBegin = padBegin - sliceOffset
// newPadEnd = sliceOffset + sliceStaticSize - padBegin - inputLen
//
// InData:       |-----------------|
//                    inputLen
//                                                           InData:       |-----------------|
// Expand:  |----|-----------------|------|                                      inputLen
//         padBegin   inputLen      padEnd
//                                                   ->      Expand:     |-|-----------------|--|
// Slice:   |  |----------------------|                              newPadBegin inputLen newPadEnd
//       sliceOffset sliceStaticSize
//                                                           OutData:    |----------------------|
// OutData:    |----------------------|                                           outputLen
//                     outputLen
//
mlir::FailureOr<std::tuple<Shape, Shape, IE::FuseMode>> vpux::IE::getExpandSliceFusedParameters(IE::ExpandOp expandOp,
                                                                                                IE::SliceOp sliceOp) {
    const auto patternInShape = getShape(expandOp.getInput());
    const auto patternOutShape = getShape(sliceOp.getResult());
    const auto rank = patternInShape.size();

    const auto fuseMode = getFuseMode(patternInShape, patternOutShape);

    const auto expandPadsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
    const auto expandPadsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEnd());
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
    const auto sliceStaticSizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());

    SmallVector<int64_t> firstShapeRef(rank, 0);
    SmallVector<int64_t> secondShapeRef(rank, 0);
    for (auto idx : irange(rank)) {
        const auto inputLen = patternInShape[Dim(idx)];
        const auto sliceOffset = sliceOffsets[idx];
        const auto sliceStaticSize = sliceStaticSizes[idx];
        const auto padBegin = expandPadsBegin[idx];

        const auto expandDataRange = padBegin + inputLen;
        const auto sliceDataRange = sliceOffset + sliceStaticSize;
        if (fuseMode == IE::FuseMode::CONVERT_TO_SLICE && padBegin <= sliceOffset &&
            expandDataRange >= sliceDataRange) {
            firstShapeRef[idx] = sliceOffset - padBegin;
            secondShapeRef[idx] = sliceStaticSize;
        } else if (fuseMode == IE::FuseMode::CONVERT_TO_EXPAND && padBegin >= sliceOffset &&
                   expandDataRange <= sliceDataRange) {
            firstShapeRef[idx] = padBegin - sliceOffset;
            secondShapeRef[idx] = sliceOffset + sliceStaticSize - padBegin - inputLen;
        } else {
            return mlir::failure();
        }
    }

    return std::tuple<Shape, Shape, IE::FuseMode>(firstShapeRef, secondShapeRef, fuseMode);
}

//
// OptimizeSliceExpand
//

mlir::LogicalResult vpux::IE::OptimizeSliceExpand::matchAndRewrite(IE::ExpandOp expandOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), expandOp->getName(), expandOp->getLoc());
    const auto innerLog = _log.nest();

    auto sliceOp = expandOp.getInput().getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        innerLog.trace("'Expand' at '{0}' input is not 'SliceOp'", expandOp->getLoc());
        return mlir::failure();
    }

    const auto sliceExpandFusedParameters = getSliceExpandFusedParameters(sliceOp, expandOp);
    if (mlir::failed(sliceExpandFusedParameters)) {
        innerLog.trace("Illegal to fuse 'Slice' at '{0}' and 'Expand' at '{1}'", sliceOp->getLoc(), expandOp->getLoc());
        return mlir::failure();
    }

    // It is specific cases for Eltwise NCE Op
    // This Add can be futher reshaped to avoid expand by AdjustInputShapePass
    // TODO(E#95919): Create Sub Pipeline to check dependency between those two passes
    // In1(1x12x64x64) -> Slice(1x3x64x64) -> Expand(1x16x64x64)
    //                                                           -> Add(1x16x64x64) -> Slice(1x3x64x64)
    // In2(1x12x64x64) -> Slice(1x3x64x64) -> Expand(1x16x64x64)
    auto isEltwiseOp = mlir::isa<IE::AddOp, IE::MultiplyOp>(*(expandOp.getOutput().getUsers().begin()));
    auto eltwiseOp = *(expandOp.getOutput().getUsers().begin());
    auto quantizeCastOp = mlir::dyn_cast_or_null<IE::QuantizeCastOp>(*(expandOp.getOutput().getUsers().begin()));
    if (quantizeCastOp != nullptr) {
        isEltwiseOp = mlir::isa<IE::AddOp, IE::MultiplyOp>(*(quantizeCastOp.getOutput().getUsers().begin()));
        eltwiseOp = *(quantizeCastOp.getOutput().getUsers().begin());
    }
    // E#93789: Follow up task to continue keep slice-expand for Eltwise if expand has multi users
    if (expandOp.getOutput().hasOneUse() && isEltwiseOp) {
        auto newExpandedShapeResult = getShapeCastExpandedShape(eltwiseOp, getShape(expandOp.getOutput()),
                                                                getShape(expandOp.getInput()), _log.nest());
        if (!mlir::failed(newExpandedShapeResult)) {
            innerLog.trace("Expand channel for Eltwise, skip this optimization");
            return mlir::failure();
        }
    }

    const auto sliceExpandFusedParametersVal = sliceExpandFusedParameters.value();
    const auto padsBeginOrOffsetsAttr =
            getIntArrayAttr(expandOp.getContext(), std::get<0>(sliceExpandFusedParametersVal));
    const auto padsEndOrStaticSizesAttr =
            getIntArrayAttr(expandOp.getContext(), std::get<1>(sliceExpandFusedParametersVal));
    const auto fuseMode = std::get<2>(sliceExpandFusedParametersVal);

    if (fuseMode == IE::FuseMode::CONVERT_TO_EXPAND) {
        innerLog.trace("Convert to 'Expand' completed successfully at '{0}'", expandOp->getLoc());
        rewriter.replaceOpWithNewOp<IE::ExpandOp>(expandOp, sliceOp.getSource(), padsBeginOrOffsetsAttr,
                                                  padsEndOrStaticSizesAttr);
        return mlir::success();
    }

    if (fuseMode == IE::FuseMode::CONVERT_TO_SLICE) {
        innerLog.trace("Convert to 'Slice' completed successfully at '{0}'", expandOp->getLoc());
        rewriter.replaceOpWithNewOp<IE::SliceOp>(expandOp, sliceOp.getSource(), padsBeginOrOffsetsAttr,
                                                 padsEndOrStaticSizesAttr);
        return mlir::success();
    }

    return mlir::failure();
}

//
// OptimizeSliceLayoutCastExpand
//

mlir::LogicalResult vpux::IE::OptimizeSliceLayoutCastExpand::matchAndRewrite(IE::ExpandOp expandOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), expandOp->getName(), expandOp->getLoc());
    const auto innerLog = _log.nest();

    auto layoutCastOp = mlir::dyn_cast_or_null<IE::LayoutCastOp>(expandOp.getInput().getDefiningOp());
    if (layoutCastOp == nullptr || !layoutCastOp->hasOneUse()) {
        innerLog.trace("'Expand' at '{0}' input is not 'LayoutCast'", expandOp->getLoc());
        return mlir::failure();
    }

    auto sliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(layoutCastOp.getInput().getDefiningOp());
    if (sliceOp == nullptr) {
        innerLog.trace("'Expand' at '{0}' input is not 'SliceOp'", expandOp->getLoc());
        return mlir::failure();
    }

    auto inputSliceShape = mlir::cast<vpux::NDTypeInterface>(sliceOp.getInput().getType()).getShape();
    auto outputExpandedShape = mlir::cast<vpux::NDTypeInterface>(expandOp.getOutput().getType()).getShape();
    if (inputSliceShape != outputExpandedShape) {
        innerLog.trace("Output 'Expand' at '{0}' and input 'Slice' has different shape'", expandOp->getLoc());
        return mlir::failure();
    }

    auto newExpand = rewriter.create<IE::ExpandOp>(expandOp->getLoc(), sliceOp.getOutput(), expandOp.getPadsBegin(),
                                                   expandOp.getPadsEnd());

    rewriter.replaceOpWithNewOp<IE::LayoutCastOp>(expandOp, newExpand.getOutput(), layoutCastOp.getDstOrder());

    return mlir::success();
}

//
// OptimizeExpandSlice
//

mlir::LogicalResult vpux::IE::OptimizeExpandSlice::matchAndRewrite(IE::ExpandOp expandOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), expandOp->getName(), expandOp->getLoc());
    const auto innerLog = _log.nest();

    auto sliceOp = mlir::dyn_cast<IE::SliceOp>(*expandOp.getOutput().getUsers().begin());

    if (sliceOp == nullptr) {
        innerLog.trace("'Expand' at '{0}' user is not 'SliceOp'", expandOp->getLoc());
        return mlir::failure();
    }

    const auto expandSliceFusedParameters = getExpandSliceFusedParameters(expandOp, sliceOp);
    if (mlir::failed(expandSliceFusedParameters)) {
        innerLog.trace("Illegal to fuse 'Expand' at '{0}' and 'Slice' at '{1}'", expandOp->getLoc(), sliceOp->getLoc());
        return mlir::failure();
    }

    const auto expandSliceFusedParametersVal = expandSliceFusedParameters.value();
    const auto padsBeginOrOffsetsAttr =
            getIntArrayAttr(expandOp.getContext(), std::get<0>(expandSliceFusedParametersVal));
    const auto padsEndOrStaticSizesAttr =
            getIntArrayAttr(expandOp.getContext(), std::get<1>(expandSliceFusedParametersVal));
    const auto fuseMode = std::get<2>(expandSliceFusedParametersVal);

    if (fuseMode == IE::FuseMode::CONVERT_TO_EXPAND) {
        innerLog.trace("Convert to 'Expand' completed successfully at '{0}'", expandOp->getLoc());
        rewriter.replaceOpWithNewOp<IE::ExpandOp>(sliceOp, expandOp.getInput(), padsBeginOrOffsetsAttr,
                                                  padsEndOrStaticSizesAttr);
        return mlir::success();
    }

    if (fuseMode == IE::FuseMode::CONVERT_TO_SLICE) {
        innerLog.trace("Convert to 'Slice' completed successfully at '{0}'", expandOp->getLoc());
        rewriter.replaceOpWithNewOp<IE::SliceOp>(sliceOp, expandOp.getInput(), padsBeginOrOffsetsAttr,
                                                 padsEndOrStaticSizesAttr);
        return mlir::success();
    }

    return mlir::failure();
}

//
// OptimizeSliceImplicitExpand
//

mlir::LogicalResult vpux::IE::genericOptimizeSliceImplicitExpand(IE::ExpandOp expandOp, mlir::Operation* implicitOp,
                                                                 bool hasCalculationCost,
                                                                 mlir::PatternRewriter& rewriter, Logger innerLog) {
    if (implicitOp == nullptr || implicitOp->getNumOperands() != 1 || implicitOp->getNumResults() != 1 ||
        !implicitOp->hasOneUse()) {
        return mlir::failure();
    }

    const auto hasConvUser = llvm::any_of(expandOp.getOutput().getUsers(), [&](mlir::Operation* userOp) {
        return mlir::isa_and_present<IE::ConvolutionOp>(userOp);
    });
    if (hasConvUser && VPU::inputCompatibleWithAutoPad(expandOp.getInput().getType()) &&
        VPU::hasAutoPaddingIDU(getModuleOp(expandOp))) {
        innerLog.trace("Skip as user can use autopad");
        return mlir::failure();
    }

    auto sliceOp = implicitOp->getOperand(0).getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        innerLog.trace("Cannot get 'Slice' before '{0}'", implicitOp->getName());
        return mlir::failure();
    }

    const auto patternInShape = getShape(sliceOp.getSource());
    const auto patternOutShape = getShape(expandOp.getResult());
    // If the implicitOp has calculation cost
    // Only consider the 'slice' and 'expand' can be completely eliminated currently
    // Otherwise not ensure for case that reserve one 'slice' or 'expand' will get the performance benefit
    // Due to the computational size of the SW layer become larger
    // It is possible to remove restrictions on SW layers that has the calculation cost in the future
    // depend on the execution efficiency
    if (hasCalculationCost && patternInShape != patternOutShape) {
        innerLog.trace("'{0}' has calculation cost and 'Slice' and 'Expand' cannot be completely eliminated",
                       implicitOp->getName());
        return mlir::failure();
    }

    const auto sliceExpandFusedParameters = getSliceExpandFusedParameters(sliceOp, expandOp);
    if (mlir::failed(sliceExpandFusedParameters)) {
        innerLog.trace("Illegal to fuse Slice at '{0}' and Expand at '{1}'", sliceOp->getLoc(), expandOp->getLoc());
        return mlir::failure();
    }

    const auto sliceExpandFusedParametersVal = sliceExpandFusedParameters.value();
    const auto padsBeginOrOffsetsAttr =
            getIntArrayAttr(expandOp.getContext(), std::get<0>(sliceExpandFusedParametersVal));
    const auto padsEndOrStaticSizesAttr =
            getIntArrayAttr(expandOp.getContext(), std::get<1>(sliceExpandFusedParametersVal));
    const auto fuseMode = std::get<2>(sliceExpandFusedParametersVal);

    if (fuseMode == IE::FuseMode::CONVERT_TO_EXPAND) {
        innerLog.trace("Convert to 'Expand' completed successfully at '{0}'", expandOp->getLoc());
        rewriter.setInsertionPointAfter(implicitOp);
        implicitOp->getOpOperand(0).set(sliceOp.getSource());
        vpux::inferReturnTypes(implicitOp, vpux::InferShapedTypeMode::SHAPE);
        rewriter.replaceOpWithNewOp<IE::ExpandOp>(expandOp, implicitOp->getResults()[0], padsBeginOrOffsetsAttr,
                                                  padsEndOrStaticSizesAttr);
        return mlir::success();
    }

    if (fuseMode == IE::FuseMode::CONVERT_TO_SLICE) {
        innerLog.trace("Convert to 'Slice' completed successfully at '{0}'", expandOp->getLoc());
        rewriter.setInsertionPoint(implicitOp);
        auto newSliceOp = rewriter.create<IE::SliceOp>(expandOp.getLoc(), sliceOp.getSource(), padsBeginOrOffsetsAttr,
                                                       padsEndOrStaticSizesAttr);
        implicitOp->getOpOperand(0).set(newSliceOp.getResult());
        vpux::inferReturnTypes(implicitOp, vpux::InferShapedTypeMode::SHAPE);
        expandOp->replaceAllUsesWith(implicitOp);
        rewriter.eraseOp(expandOp);
        return mlir::success();
    }

    return mlir::failure();
}

//
// OptimizeSliceShapeCastExpand
//

// Only consider a simple pattern for now:
//   - input/output shape of ShapeCast has ony one dimension at which the shape is not one,
//     e.g. <Nx1x1x1>, <1x1xHx1> or <1x1x1xW>
//   - ExpandOp also has the same input/output shape pattern as ShapeCast
//
// It guarantees that ExpandOp can expands tensor at the correct axis after swap
// ShapeCast and Expand.

bool canSwapShapeCastAndExpand(IE::ShapeCastOp shapeCastOp, IE::ExpandOp expandOp) {
    const auto shapeNotOne = [](auto dimShape) -> bool {
        return dimShape != 1;
    };

    const auto isOpSingleShape = [&](ShapeRef inputShape, ShapeRef outputShape) -> bool {
        const auto isInputSingleDim = llvm::count_if(inputShape, shapeNotOne) == 1;
        const auto isOutputSingleDim = llvm::count_if(outputShape, shapeNotOne) == 1;
        return isInputSingleDim && isOutputSingleDim;
    };

    return isOpSingleShape(getShape(shapeCastOp.getSource()), getShape(shapeCastOp.getResult())) &&
           isOpSingleShape(getShape(expandOp.getInput()), getShape(expandOp.getResult()));
}

bool canEliminateSliceExpand(IE::ShapeCastOp shapeCastOp, IE::ExpandOp expandOp, ShapeRef sliceInputShape) {
    if (!canSwapShapeCastAndExpand(shapeCastOp, expandOp)) {
        return false;
    }

    // check if the input type of SliceOp and the output type of Expand are the same,
    // if yes, then Slice and Expand can be eliminated.
    // e.g.
    // <1x80x1x1> -> Slice -> <1x72x1x1> -> Sigmoid -> <1x72x1x1> -> ShapeCast -> <72x1x1x1> -> Expand -> <80x1x1x1>
    //
    // The new Expand output type is <1x80x1x1> which is the same as Slice input after swap ShapeCast and Expand
    // <1x80x1x1> -> Slice -> <1x72x1x1> -> Sigmoid -> <1x72x1x1> -> Expand -> <1x80x1x1> -> ShapeCast -> <80x1x1x1>
    //
    const auto shapeNotOne = [](auto dimShape) -> bool {
        return dimShape != 1;
    };

    const auto getDimShapeNotOne = [&](ShapeRef shape) {
        const auto shapeIt = llvm::find_if(shape, shapeNotOne);
        VPUX_THROW_WHEN(shapeIt == shape.end(), "ilegal shape {0}", shape);
        return std::distance(shape.begin(), shapeIt);
    };

    const auto shapeCastInputShape = getShape(shapeCastOp.getSource());
    const auto shapeCastInputShapeDim = getDimShapeNotOne(shapeCastInputShape);
    const auto shapeCastOutputShape = getShape(shapeCastOp.getResult());
    const auto shapeCastOutputShapeDim = getDimShapeNotOne(shapeCastOutputShape);

    const auto expandInputShape = getShape(expandOp.getInput());
    const auto expandInputShapeDim = getDimShapeNotOne(expandInputShape);
    const auto expandOutputShape = getShape(expandOp.getResult());
    const auto expandOutputShapeDim = getDimShapeNotOne(expandOutputShape);

    VPUX_THROW_UNLESS(expandInputShapeDim == expandOutputShapeDim, "not expand at the same axis: {0}, {1}",
                      expandInputShapeDim, expandOutputShapeDim);
    VPUX_THROW_UNLESS(shapeCastOutputShapeDim == expandInputShapeDim, "{0} not expand at the ShapeCast axis {1}",
                      expandInputShapeDim, shapeCastOutputShapeDim);

    auto expandOutShapeAfterSwap = Shape(to_small_vector(shapeCastInputShape));
    expandOutShapeAfterSwap[Dim(shapeCastInputShapeDim)] = expandOutputShape[Dim(expandOutputShapeDim)];

    return expandOutShapeAfterSwap == sliceInputShape;
}

// optimize the pattern below:
//   ->Slice->EltwiseLikeSW->ShapeCast->Expand
// to
//   ->EltwiseLikeSW->ShapeCast

mlir::LogicalResult vpux::IE::genericOptimizeSliceImplicitShapeCastExpand(IE::ExpandOp origOp,
                                                                          IE::ShapeCastOp shapeCastOp,
                                                                          mlir::Operation* implicitOp,
                                                                          mlir::PatternRewriter& rewriter,
                                                                          Logger innerLog) {
    if (implicitOp == nullptr || implicitOp->getNumOperands() != 1 || implicitOp->getNumResults() != 1 ||
        !implicitOp->hasOneUse()) {
        return mlir::failure();
    }

    auto sliceOp = implicitOp->getOperand(0).getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        innerLog.trace("Cannot get 'Slice' before '{0}'", implicitOp->getName());
        return mlir::failure();
    }
    if (!sliceOp->hasOneUse()) {
        return mlir::failure();
    }

    if (!canEliminateSliceExpand(shapeCastOp, origOp, getShape(sliceOp.getSource()))) {
        return mlir::failure();
    }

    // found the beneficial pattern and create new ops:
    //     EltwiseLikeSW->ShapeCast
    //

    implicitOp->getOpOperand(0).set(sliceOp.getSource());
    vpux::inferReturnTypes(implicitOp, vpux::InferShapedTypeMode::SHAPE);

    rewriter.setInsertionPointAfter(implicitOp);
    const auto newShapeCastOutShape = getShape(origOp.getResult());
    auto newShapeCastOp = rewriter.create<IE::ShapeCastOp>(origOp->getLoc(), origOp.getType(), implicitOp->getResult(0),
                                                           getIntArrayAttr(origOp.getContext(), newShapeCastOutShape));

    rewriter.replaceOp(origOp, newShapeCastOp.getResult());
    return mlir::success();
}

SmallVector<mlir::Value> vpux::IE::OptimizeSlicePReluExpand::updateInputsForOp(mlir::PatternRewriter& rewriter,
                                                                               IE::PReluOp origOp,
                                                                               IE::ExpandOp expandOp) const {
    SmallVector<mlir::Value> inputs;
    inputs.push_back(origOp.getInput().getDefiningOp<IE::SliceOp>().getSource());
    for (auto index : irange<unsigned>(1, origOp->getOperands().size())) {
        auto input = origOp.getOperand(index);
        if (auto constOp = input.getDefiningOp<Const::DeclareOp>()) {
            inputs.push_back(createNewConstValue(constOp, Dims4D::Act::C, getShape(expandOp.getResult()), rewriter));
        } else {
            inputs.push_back(input.getDefiningOp<IE::SliceOp>().getSource());
        }
    }
    return inputs;
}

SmallVector<mlir::Value> vpux::IE::OptimizeSliceConcatExpand::updateInputsForOp(mlir::PatternRewriter& rewriter,
                                                                                IE::ConcatOp origOp,
                                                                                IE::ExpandOp expandOp) const {
    const auto expandAxisVal = getExpandAxis(expandOp).value();
    SmallVector<mlir::Value> newConcatInputs;
    for (const auto& concatInput : origOp.getInputs()) {
        if (auto sliceOp = concatInput.getDefiningOp<IE::SliceOp>()) {
            newConcatInputs.push_back(sliceOp.getSource());
        } else if (auto constOp = concatInput.getDefiningOp<Const::DeclareOp>()) {
            newConcatInputs.push_back(
                    createNewConstValue(constOp, expandAxisVal, getShape(expandOp.getResult()), rewriter));
        } else {
            newConcatInputs.push_back(concatInput);
        }
    }
    return newConcatInputs;
}

/**
 * Fuse slice and expand when operations between them are supported with any order and numbers.
 * Insert Expand and SliceOp between ops, then we can get single SliceOp-Op-ExpandOp patterns and call related pattern
 * optimizations. It's easier to handle single SliceOp-Op-ExpandOp than many ops between SliceOp and ExpandOp.
 *
 *         SliceOp                         SliceOp                      op1
 *            |                               |                          |
 *           op1                             op1                        op2
 *            |                               |                          |
 *           op2             ->            ExpandOp         ->          op3
 *            |                               |
 *           op3                           SliceOp
 *            |                               |
 *         ExpandOp                          op2
 *                                            |
 *                                         ExpandOp
 *                                            |
 *                                         SliceOp
 *                                            |
 *                                           op3
 *                                            |
 *                                         ExpandOp
 *
 */
mlir::LogicalResult IE::OptimizeSliceOpsExpand::matchAndRewrite(IE::ExpandOp expandOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), expandOp->getName(), expandOp->getLoc());

    // TODO(E#126897) : support more middle ops
    auto isSupportOpType = [](mlir::Operation* op) -> bool {
        return mlir::isa_and_nonnull<IE::ConcatOp, IE::PReluOp>(op);
    };

    auto getNonConstConcatInput = [](IE::ConcatOp concatOp) -> mlir::FailureOr<mlir::Operation*> {
        SmallVector<mlir::Operation*> previousOpCandidates;
        for (const auto& concatInput : concatOp.getInputs() | indexed) {
            auto concatOplocal = concatInput.value().getDefiningOp();
            if (mlir::isa_and_nonnull<Const::DeclareOp>(concatOplocal)) {
                continue;
            }
            previousOpCandidates.push_back(concatOplocal);
        }
        if (previousOpCandidates.size() != 1) {
            return mlir::failure();
        }
        return previousOpCandidates.front();
    };

    SmallVector<mlir::Operation*> ops;
    auto inputOp = expandOp.getInput().getDefiningOp();
    while (isSupportOpType(inputOp)) {
        ops.push_back(inputOp);
        if (auto concatOp = mlir::dyn_cast_or_null<IE::ConcatOp>(inputOp)) {
            // TODO(E#126897) : support multi branch with non-const inputs for multi concat ops in middle
            auto inputOrFailure = getNonConstConcatInput(concatOp);
            if (mlir::failed(inputOrFailure)) {
                _log.trace("Ilegal IE::ConcatOp at '{0}', only one non-const input ConcatOp is supported Now.",
                           expandOp->getLoc());
                return mlir::failure();
            }
            inputOp = inputOrFailure.value();
        } else {
            inputOp = inputOp->getOperand(0).getDefiningOp();
        }
    }

    if (ops.size() <= 1) {
        _log.trace("Only one or no operations between slice and expand.", expandOp->getLoc());
        return mlir::failure();
    }

    auto sliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(inputOp);
    if (sliceOp == nullptr) {
        _log.trace("Cannot get 'Slice' in the front of the pattern.");
        return mlir::failure();
    }

    // Check if all the middle ops are feasible to be optimized.
    for (auto op : ops) {
        if (!isMiddleOpLegal(sliceOp, op, expandOp)) {
            _log.trace("Ilegal Middle operation at '{0}'.", op->getLoc());
            return mlir::failure();
        }
    }

    mlir::Value preOutput = sliceOp.getResult();
    auto padBegin = expandOp.getPadsBeginAttr();
    auto padEnd = expandOp.getPadsEndAttr();
    for (auto iter = ops.rbegin(); iter != ops.rend(); ++iter) {
        mlir::Operation* op = *iter;
        mlir::IRMapping mapper;
        if (auto concatOp = mlir::dyn_cast<IE::ConcatOp>(op)) {
            for (const auto& concatInput : concatOp.getInputs()) {
                if (!mlir::isa<Const::DeclareOp>(concatInput.getDefiningOp())) {
                    mapper.map(concatInput, preOutput);
                }
            }
        } else {
            mapper.map(op->getOperand(0), preOutput);
        }
        auto newOp = rewriter.clone(*op, mapper);
        vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);

        if (iter != ops.rend() - 1) {
            auto insertExpandOp =
                    rewriter.create<IE::ExpandOp>(expandOp->getLoc(), newOp->getResult(0), padBegin, padEnd);
            auto sliceOffset = sliceOp.getStaticOffsetsAttr();
            auto insertSliceOp =
                    rewriter.create<IE::SliceOp>(expandOp->getLoc(), insertExpandOp.getResult(), sliceOffset,
                                                 getIntArrayAttr(expandOp.getContext(), getShape(newOp->getResult(0))));
            preOutput = insertSliceOp.getResult();
        } else {
            preOutput = newOp->getResult(0);
        }
    }

    rewriter.replaceOpWithNewOp<IE::ExpandOp>(expandOp, expandOp.getType(), preOutput, padBegin, padEnd);
    return mlir::success();
}

namespace {

//
// OptimizeSliceSoftmaxExpand
//

class OptimizeSliceSoftmaxExpand final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSliceSoftmaxExpand(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeSliceSoftmaxExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
        const auto innerLog = _log.nest();

        auto implicitOp = origOp.getInput().getDefiningOp<IE::SoftMaxOp>();
        if (implicitOp == nullptr) {
            innerLog.trace("Expand '{0}' input is not 'SoftMaxOp'", origOp->getLoc());
            return mlir::failure();
        }

        auto inputType = mlir::cast<NDTypeInterface>(implicitOp.getInput().getType());
        auto order = inputType.getDimsOrder();
        auto outputShape = inputType.getShape();
        auto innerDim = getInnermostNonTrivialDim(outputShape, order);
        auto axisIdx = implicitOp.getAxisInd();

        if (innerDim != Dim(axisIdx)) {
            innerLog.trace("'SoftMaxOp' process axis should be innermost but got '{0}'", innerDim);
            return mlir::failure();
        }

        auto expandedShape = to_small_vector(getShape(origOp.getOutput()));
        auto implicitShape = to_small_vector(getShape(implicitOp->getResult(0)));
        int64_t expandedAxisSize = expandedShape[axisIdx] - implicitShape[axisIdx];
        const auto loc = origOp->getLoc();
        auto optimizeSuccess = genericOptimizeSliceImplicitExpand(origOp, implicitOp.getOperation(),
                                                                  /*hasCalculationCost=*/true, rewriter, innerLog);
        if (optimizeSuccess.failed()) {
            return mlir::failure();
        }
        // update necessary attribute
        implicitOp.setPadSizeAttr(getIntAttr(rewriter.getContext(), expandedAxisSize));
        innerLog.trace("Optimization completed successfully at '{0}'", loc);
        return mlir::success();
    }

private:
    Logger _log;
};

//
// OptimizeSliceExpandPass
//

class OptimizeSliceExpandPass final : public IE::impl::OptimizeSliceExpandBase<OptimizeSliceExpandPass> {
public:
    explicit OptimizeSliceExpandPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void OptimizeSliceExpandPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<IE::OptimizeSliceExpand>(&ctx, _log);
    patterns.add<IE::OptimizeExpandSlice>(&ctx, _log);
    patterns.add<IE::OptimizeSliceImplicitExpand<IE::QuantizeCastOp>>(&ctx, _log, /*hasCalculationCost=*/false);
    patterns.add<IE::OptimizeSliceImplicitExpand<IE::HSwishOp>>(&ctx, _log, /*hasCalculationCost=*/true);
    patterns.add<IE::OptimizeSliceImplicitExpand<IE::SwishOp>>(&ctx, _log, /*hasCalculationCost=*/true);
    patterns.add<IE::OptimizeSliceImplicitExpand<IE::GeluOp>>(&ctx, _log, /*hasCalculationCost=*/true);
    patterns.add<IE::OptimizeSliceImplicitExpand<IE::ClampOp>>(&ctx, _log, /*hasCalculationCost=*/true);
    patterns.add<OptimizeSliceSoftmaxExpand>(&ctx, _log);
    patterns.add<IE::OptimizeSliceLayoutCastExpand>(&ctx, _log);

    patterns.add<IE::OptimizeSliceShapeCastExpand<IE::HSwishOp>>(&ctx, _log);
    patterns.add<IE::OptimizeSliceShapeCastExpand<IE::SwishOp>>(&ctx, _log);
    patterns.add<IE::OptimizeSliceShapeCastExpand<IE::GeluOp>>(&ctx, _log);
    patterns.add<IE::OptimizeSliceShapeCastExpand<IE::SigmoidOp>>(&ctx, _log);

    // The middle op has multi inputs
    patterns.add<IE::OptimizeSliceConcatExpand>(&ctx, _log);
    patterns.add<IE::OptimizeSlicePReluExpand>(&ctx, _log);
    patterns.add<IE::OptimizeSliceEltwiseExpand<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<IE::OptimizeSliceEltwiseExpand<IE::AddOp>>(&ctx, _log);
    patterns.add<IE::OptimizeSliceEltwiseExpand<IE::SubtractOp>>(&ctx, _log);

    // Pattern slice-op1-op2-...-opN-expand
    patterns.add<IE::OptimizeSliceOpsExpand>(&ctx, _log);

    auto func = getOperation();
    // There is case for `OptimizeExpandSlice` that the iteration time larger than 10
    // Increase the default maxIterations value from 10 to 60
    auto greedyRewriteConfig = getDefaultGreedyRewriteConfig();
    greedyRewriteConfig.maxIterations = 60;

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), greedyRewriteConfig))) {
        signalPassFailure();
        return;
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeSliceExpandPass(Logger log) {
    return std::make_unique<OptimizeSliceExpandPass>(log);
}
