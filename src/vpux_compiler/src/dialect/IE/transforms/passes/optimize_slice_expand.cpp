//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/optimize_slice_expand.hpp"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/core/attributes/dims_order.hpp"
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
#include "vpux/compiler/dialect/IE/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
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
            .Case<IE::AddOp, IE::SubtractOp, IE::MultiplyOp, IE::DivideOp>([&](mlir::Operation* eltwiseOp) {
                return isSimpleOpLegal(sliceOp, eltwiseOp, expandOp);
            })
            .Case<IE::LayoutCastOp, IE::SoftMaxOp>([&](mlir::Operation* op) {
                // For simple middle operation
                return isSimpleOpLegal(sliceOp, op, expandOp);
            })
            .Default([](mlir::Operation*) -> bool {
                return false;
            });
}

bool isSliceAndExpandLegal(IE::SliceOp sliceOp, IE::ExpandOp expandOp) {
    auto sliceAxis = IE::getSingleDiffAxis(getShape(sliceOp.getSource()), getShape(sliceOp.getResult()));
    auto expandAxis = getExpandAxis(expandOp);

    if (!sliceAxis.has_value() || !expandAxis.has_value()) {
        return false;
    }

    if (sliceAxis.value() != expandAxis.value()) {
        return false;
    }

    auto patternInShape = getShape(sliceOp.getSource());
    auto patternOutShape = getShape(expandOp.getResult());

    return patternInShape[sliceAxis.value()] == patternOutShape[expandAxis.value()];
}

mlir::FailureOr<mlir::Operation*> getNonConstEltwiseInput(mlir::Operation* eltwiseOp) {
    auto parentOp1 = eltwiseOp->getOperand(0).getDefiningOp();
    auto parentOp2 = eltwiseOp->getOperand(1).getDefiningOp();
    bool isInput1Const = mlir::isa_and_nonnull<Const::DeclareOp>(parentOp1);
    bool isInput2Const = mlir::isa_and_nonnull<Const::DeclareOp>(parentOp2);

    if (isInput1Const && !isInput2Const) {
        return parentOp2;
    }
    if (!isInput1Const && isInput2Const) {
        return parentOp1;
    }
    return mlir::failure();
}

bool IE::isSimpleOpLegal(IE::SliceOp sliceOp, mlir::Operation* middleOp, IE::ExpandOp expandOp) {
    if (middleOp == nullptr || middleOp->getNumResults() != 1 || !middleOp->hasOneUse()) {
        return false;
    }

    return isSliceAndExpandLegal(sliceOp, expandOp);
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
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
    const auto expandPadsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
    if (sliceOp.getSource().getType() != expandOp.getResult().getType() || sliceOffsets != expandPadsBegin) {
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
// OptimizeSlicePermuteCastExpand
//

mlir::LogicalResult vpux::IE::OptimizeSlicePermuteCastExpand::matchAndRewrite(IE::ExpandOp expandOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), expandOp->getName(), expandOp->getLoc());
    const auto innerLog = _log.nest();

    auto permuteCastOp = mlir::dyn_cast_or_null<IE::PermuteCastOp>(expandOp.getInput().getDefiningOp());
    if (permuteCastOp == nullptr || !permuteCastOp->hasOneUse()) {
        innerLog.trace("'Expand' at '{0}' input is not 'PermuteCast'", expandOp->getLoc());
        return mlir::failure();
    }

    auto sliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(permuteCastOp.getInput().getDefiningOp());
    if (sliceOp == nullptr) {
        innerLog.trace("'PermuteCast' at '{0}' input is not 'SliceOp'", expandOp->getLoc());
        return mlir::failure();
    }

    auto expandInMemShape = Shape(getMemShape(expandOp.getInput()).raw());
    auto expandOutMemShape = Shape(getMemShape(expandOp.getResult()).raw());
    auto expandMemAxis = IE::getSingleDiffAxis(expandInMemShape, expandOutMemShape);
    if (!expandMemAxis.has_value()) {
        return mlir::failure();
    }
    auto sliceInMemShape = Shape(getMemShape(sliceOp.getSource()).raw());
    auto sliceOutMemShape = Shape(getMemShape(sliceOp.getResult()).raw());
    auto sliceMemAxis = IE::getSingleDiffAxis(sliceInMemShape, sliceOutMemShape);
    if (!sliceMemAxis.has_value()) {
        return mlir::failure();
    }

    auto isSupportedPermuteCastOp = [&](IE::PermuteCastOp permuteCastOp) {
        auto memPerm = permuteCastOp.getMemPerm();
        if (sliceOutMemShape[sliceMemAxis.value()] != 1) {
            // If the slice size is greater than 1 on the slice axis, the sliceMemIdx should be compatible with
            // expandMemIdx after PermuteCast.
            // [1,16,48,48] -> Slice -> [1,16,32,48] -> PermuteCast -> [16,32,1,48] -> expand -> [16,48,1,48]
            auto sliceInMemIdxAfterPermute = DimsOrder::fromAffineMap(memPerm).dimPos(sliceMemAxis.value());
            return static_cast<int32_t>(sliceInMemIdxAfterPermute) == expandMemAxis.value().ind();
        } else {
            // If the slice size is 1 on the slice axis, the expandMemAxis should be the same as the sliceMemAxis
            // and the memPerm should be NCHW
            // [1,16,48,48] -> Slice -> [1,16,1,48] -> PermuteCast -> [1,16,1,48] -> expand -> [1,16,48,48]
            return DimsOrder::fromAffineMap(memPerm) == DimsOrder::NCHW &&
                   expandMemAxis.value() == sliceMemAxis.value();
        }
    };
    if (!isSupportedPermuteCastOp(permuteCastOp)) {
        innerLog.trace("'PermuteCast' at '{0}' is not supported", permuteCastOp->getLoc());
        return mlir::failure();
    }

    // Calculate new padding by applying inverse permutation
    const auto sliceDstOrder = mlir::cast<vpux::NDTypeInterface>(sliceOp.getOutput().getType()).getDimsOrder();
    const auto expandDstOrder = DimsOrder::fromAffineMap(permuteCastOp.getDstOrder());
    const auto expandPadsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
    const auto expandPadsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEnd());
    const auto padsBeginMemShape = expandDstOrder.toMemoryOrder(Shape(expandPadsBegin));
    const auto padsEndMemShape = expandDstOrder.toMemoryOrder(Shape(expandPadsEnd));
    const auto newPadsBeginMemShape =
            applyPerm(padsBeginMemShape, mlir::inversePermutation(permuteCastOp.getMemPerm()));
    const auto newPadsEndMemShape = applyPerm(padsEndMemShape, mlir::inversePermutation(permuteCastOp.getMemPerm()));
    const auto newPadBeginLogicShape = sliceDstOrder.toLogicalOrder(newPadsBeginMemShape);
    const auto newPadEndLogicShape = sliceDstOrder.toLogicalOrder(newPadsEndMemShape);

    auto newExpandOp = rewriter.create<IE::ExpandOp>(expandOp->getLoc(), sliceOp.getOutput(),
                                                     getIntArrayAttr(expandOp.getContext(), newPadBeginLogicShape),
                                                     getIntArrayAttr(expandOp.getContext(), newPadEndLogicShape));

    rewriter.replaceOpWithNewOp<IE::PermuteCastOp>(expandOp, newExpandOp.getOutput(), permuteCastOp.getDstOrderAttr(),
                                                   permuteCastOp.getMemPermAttr());

    return mlir::success();
}

//
// OptimizeExpandSlice
//

mlir::LogicalResult vpux::IE::OptimizeExpandSlice::matchAndRewrite(IE::ExpandOp expandOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), expandOp->getName(), expandOp->getLoc());
    const auto innerLog = _log.nest();
    bool patternMatched = false;

    for (auto inputOp : llvm::make_early_inc_range(expandOp.getOutput().getUsers())) {
        auto sliceOp = mlir::dyn_cast<IE::SliceOp>(inputOp);
        if (sliceOp == nullptr) {
            innerLog.trace("'Expand' at '{0}' user is not 'SliceOp'", expandOp->getLoc());
            continue;
        }

        const auto expandSliceFusedParameters = getExpandSliceFusedParameters(expandOp, sliceOp);
        if (mlir::failed(expandSliceFusedParameters)) {
            innerLog.trace("Illegal to fuse 'Expand' at '{0}' and 'Slice' at '{1}'", expandOp->getLoc(),
                           sliceOp->getLoc());
            continue;
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
            patternMatched = true;
        } else if (fuseMode == IE::FuseMode::CONVERT_TO_SLICE) {
            innerLog.trace("Convert to 'Slice' completed successfully at '{0}'", expandOp->getLoc());
            rewriter.replaceOpWithNewOp<IE::SliceOp>(sliceOp, expandOp.getInput(), padsBeginOrOffsetsAttr,
                                                     padsEndOrStaticSizesAttr);
            patternMatched = true;
        }
    }

    return mlir::success(patternMatched);
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
        config::hasAutoPaddingIDU(getModuleOp(expandOp))) {
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
        VPUX_THROW_WHEN(shapeIt == shape.end(), "illegal shape {0}", shape);
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
 * Insert Expand and SliceOp between ops, then we can get single SliceOp-Op-ExpandOp patterns and call related
 * pattern optimizations. It's easier to handle single SliceOp-Op-ExpandOp than many ops between SliceOp and
 * ExpandOp.
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
        return mlir::isa_and_nonnull<IE::ConcatOp, IE::PReluOp, IE::LayoutCastOp, IE::SoftMaxOp, IE::AddOp,
                                     IE::MultiplyOp, IE::SubtractOp, IE::DivideOp>(op);
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
                _log.trace("Illegal IE::ConcatOp at '{0}', only one non-const input ConcatOp is supported currently.",
                           expandOp->getLoc());
                return mlir::failure();
            }
            inputOp = inputOrFailure.value();
        } else if (inputOp->hasTrait<IE::EltwiseOp>()) {
            auto inputOrFailure = getNonConstEltwiseInput(inputOp);
            if (mlir::failed(inputOrFailure)) {
                _log.trace("Illegal EltwiseOp at '{0}', only one non-const input EltwiseOp is supported currently.",
                           expandOp->getLoc());
                return mlir::failure();
            }
            inputOp = inputOrFailure.value();
        } else {
            inputOp = inputOp->getOperand(0).getDefiningOp();
        }
    }

    if (ops.size() <= 1) {
        _log.trace("Only one or no operations between slice and expand {0}", expandOp->getLoc());
        return mlir::failure();
    }

    auto sliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(inputOp);
    if (sliceOp == nullptr || !sliceOp->hasOneUse()) {
        _log.trace("Cannot get 'Slice' in the front of the pattern or Slice has multi-users.");
        return mlir::failure();
    }

    // Check if all the middle ops are feasible to be optimized.
    for (auto op : ops) {
        if (!isMiddleOpLegal(sliceOp, op, expandOp)) {
            _log.trace("Illegal Middle operation at '{0}'.", op->getLoc());
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
        } else if (op->hasTrait<IE::EltwiseOp>()) {
            for (auto operand : op->getOperands()) {
                if (!mlir::isa<Const::DeclareOp>(operand.getDefiningOp())) {
                    mapper.map(operand, preOutput);
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
    for (auto op : ops) {
        if (op != nullptr && op->use_empty()) {
            rewriter.eraseOp(op);
        }
    }
    return mlir::success();
}

/*
 *  Only consider the pattern that view-like ops can be almost completely ignored
 *     Slice        Slice
 *       |            |
 *  ViewLikeOps  ViewLikeOps        Slice       Slice
 *          \      /                     \     /
 *           Concat           =>          Concat
 *             |                            |
 *        ViewLikeOps                     Expand
 *             |                            |
 *           Expand                      ShapeCast
 */
mlir::LogicalResult IE::OptimizeSliceConcatExpandWithViewLikeOps::matchAndRewrite(
        IE::ExpandOp expandOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), expandOp->getName(), expandOp->getLoc());

    const auto maybeExpandAxis = IE::getExpandAxis(expandOp);
    if (!maybeExpandAxis.has_value()) {
        return mlir::failure();
    }
    const auto expandAxis = maybeExpandAxis.value();

    bool hasViewLikeOp = false;

    auto walkThroughViewLikeOps = [&hasViewLikeOp](mlir::Operation* currentOp) -> mlir::FailureOr<mlir::Operation*> {
        while (mlir::isa_and_nonnull<IE::ViewLikeOpInterface>(currentOp)) {
            hasViewLikeOp = true;
            if (!currentOp->hasOneUse()) {
                return mlir::failure();
            }
            currentOp = currentOp->getOperand(0).getDefiningOp();
        }

        return currentOp;
    };

    auto implicitOp = expandOp.getInput().getDefiningOp();
    auto implicitOpOrFailure = walkThroughViewLikeOps(implicitOp);
    if (mlir::failed(implicitOpOrFailure) || !hasViewLikeOp) {
        return mlir::failure();
    }
    implicitOp = implicitOpOrFailure.value();

    // Find concat through view-like op chain
    auto concatOp = mlir::dyn_cast_or_null<IE::ConcatOp>(implicitOp);
    if (concatOp == nullptr || !concatOp->hasOneUse()) {
        return mlir::failure();
    }

    const auto maybeConcatAxis = getConcatAxis(concatOp);
    if (!maybeConcatAxis.has_value()) {
        return mlir::failure();
    }
    const auto concatAxis = maybeConcatAxis.value();

    // Find slices through view-like op chain
    SmallVector<mlir::Value> newInputValues;
    auto expandDimOrder = DimsOrder::fromValue(expandOp.getResult());
    const auto concatDimOrder = DimsOrder::fromValue(concatOp.getResult());
    for (const auto& concatInput : concatOp.getInputs()) {
        if (mlir::isa<mlir::BlockArgument>(concatInput)) {
            return mlir::failure();
        }

        auto inputOp = concatInput.getDefiningOp();
        auto inputOpOrFailure = walkThroughViewLikeOps(inputOp);
        if (mlir::failed(inputOpOrFailure)) {
            return mlir::failure();
        }
        inputOp = inputOpOrFailure.value();

        if (auto sliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(inputOp)) {
            if (!sliceOp->hasOneUse()) {
                return mlir::failure();
            }

            const auto maybeSliceAxis =
                    IE::getSingleDiffAxis(getShape(sliceOp.getSource()), getShape(sliceOp.getResult()));
            if (!maybeSliceAxis.has_value() || maybeSliceAxis.value() != expandAxis) {
                return mlir::failure();
            }

            // Only consider the case that the layout of slice is the same as the expand
            auto sliceDimOrder = DimsOrder::fromValue(sliceOp.getResult());
            if (sliceDimOrder != expandDimOrder) {
                return mlir::failure();
            }

            // Only consider the case that the output `memShape` of the slice is the same as the input `memShape` of the
            // concat, so that we can ignore the view-like operations between them.
            const auto concatInMemShape = concatDimOrder.toMemoryOrder(getShape(concatInput));
            const auto sliceOutMemShape = sliceDimOrder.toMemoryOrder(getShape(sliceOp.getResult()));
            if (concatInMemShape != sliceOutMemShape) {
                return mlir::failure();
            }

            newInputValues.push_back(inputOp->getResult(0));
            continue;
        }

        return mlir::failure();
    }

    // Check the compatibility between the concat output and the expand input.
    // ExpandAxis should be the highest dim and that the corresponding size remains unchanged through view-like ops.
    auto expandInShape = getShape(expandOp.getInput());
    const auto expandInHighestNonTrivialDim = getHighestNonTrivialDim(expandInShape, expandDimOrder);
    if (!expandInHighestNonTrivialDim.has_value() || expandInHighestNonTrivialDim.value() != expandAxis) {
        return mlir::failure();
    }

    auto concatOutShape = getShape(concatOp.getResult());
    const auto concatOutHighestNonTrivialDim = getHighestNonTrivialDim(concatOutShape, concatDimOrder);
    if (!concatOutHighestNonTrivialDim.has_value()) {
        return mlir::failure();
    }

    if (concatOutShape[concatOutHighestNonTrivialDim.value()] != expandInShape[expandAxis]) {
        return mlir::failure();
    }

    auto ctx = rewriter.getContext();

    // Create new Concat
    const auto concatMemAxis = concatDimOrder.toMemDim(concatAxis);
    auto newConcatAxis = expandDimOrder.toDim(concatMemAxis);
    const auto newInputShapes = to_small_vector(llvm::map_range(newInputValues, getShape));
    const auto newConcatOffsetsAttr = inferConcatOffsets(newInputShapes, newConcatAxis, ctx);
    auto resultValue =
            rewriter.create<IE::ConcatOp>(concatOp.getLoc(), newInputValues, /*per_axis*/ nullptr, newConcatOffsetsAttr)
                    .getResult();

    const auto padsBeginVal = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin())[expandAxis.ind()];
    const auto padsEndVal = parseIntArrayAttr<int64_t>(expandOp.getPadsEnd())[expandAxis.ind()];
    SmallVector<int64_t> newPadsBegin(expandInShape.size(), 0);
    SmallVector<int64_t> newPadsEnd(expandInShape.size(), 0);
    auto newExpandAxis = getHighestNonTrivialDim(getShape(resultValue), expandDimOrder).value();
    newPadsBegin[newExpandAxis.ind()] = padsBeginVal;
    newPadsEnd[newExpandAxis.ind()] = padsEndVal;
    resultValue = rewriter.create<IE::ExpandOp>(expandOp.getLoc(), resultValue, getIntArrayAttr(ctx, newPadsBegin),
                                                getIntArrayAttr(ctx, newPadsEnd))
                          .getResult();

    auto outputShape = getShape(expandOp.getResult());
    if (getShape(resultValue) != outputShape) {
        resultValue =
                rewriter.create<IE::ShapeCastOp>(expandOp.getLoc(), resultValue, getIntArrayAttr(ctx, outputShape))
                        .getResult();
    }

    auto newElemType = mlir::cast<vpux::NDTypeInterface>(resultValue.getType()).getElementType();
    auto origElemType = mlir::cast<vpux::NDTypeInterface>(expandOp.getResult().getType()).getElementType();
    if (newElemType != origElemType) {
        resultValue = rewriter.create<IE::QuantizeCastOp>(expandOp.getLoc(), resultValue, origElemType).getOutput();
    }

    _log.trace("Optimization completed successfully at '{0}'", expandOp->getLoc());
    rewriter.replaceAllUsesWith(expandOp.getResult(), resultValue);
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
        const auto expandAxis = IE::getExpandAxis(origOp);
        const auto loc = origOp->getLoc();
        auto optimizeSuccess = genericOptimizeSliceImplicitExpand(origOp, implicitOp.getOperation(),
                                                                  /*hasCalculationCost=*/true, rewriter, innerLog);
        if (optimizeSuccess.failed()) {
            return mlir::failure();
        }
        // update necessary attribute
        if (expandAxis.has_value() && expandAxis.value() == Dim(axisIdx)) {
            int64_t expandedAxisSize = expandedShape[axisIdx] - implicitShape[axisIdx];
            implicitOp.setPadSizeAttr(getIntAttr(rewriter.getContext(), expandedAxisSize));
        }
        innerLog.trace("Optimization completed successfully at '{0}'", loc);
        return mlir::success();
    }

private:
    Logger _log;
};

//
// SliceAfterAddForLayoutCastExpandAddRewriter
//
// Original: Slice0 -> LayoutCast0 -> Expand -> Add -> Slice1 -> LayoutCast1
// To legalize Add on DPU, LayoutCast(ToNHWC) and Expand(for channel) are typically inserted before Add.
// Subsequently, Slice and LayoutCast are added after Add to restore the original layout and shape.
// However, this legalization blocks other optimizations. For instance, the input of Slice0 might
// directly satisfy the channel alignment requirements of Add via AdjustInputShape.
//
// This pattern optimizes the sequence by moving Slice0 to the end of the chain:
// LayoutCast0 -> Expand -> Add(large) -> Slice1(large) -> LayoutCast1(large) -> Slice0.
// This exposes opportunities for other rewriters (e.g., expandRewriter) to eliminate the Expand and Slice1
// operations.
class SliceAfterAddForLayoutCastExpandAddRewriter final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    SliceAfterAddForLayoutCastExpandAddRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
        setDebugName("SliceAfterAddForLayoutCastExpandAddRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp addOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SliceAfterAddForLayoutCastExpandAddRewriter::matchAndRewrite(
        IE::AddOp addOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), addOp->getName(), addOp->getLoc());
    if (!addOp->hasOneUse()) {
        return mlir::failure();
    }

    // 1. Identify Inputs: (Activation, Constant)
    auto getConstDeclare = [&](mlir::Value value) -> Const::DeclareOp {
        if (auto lc = value.getDefiningOp<IE::LayoutCastOp>()) {
            // Allow const wrapped by LayoutCast(s) only if they are shape-preserving.
            if (!lc.getOutput().hasOneUse()) {
                return nullptr;
            }
            value = lc.getInput();
        }
        return value.getDefiningOp<Const::DeclareOp>();
    };

    mlir::Value nonConstInput = nullptr;
    Const::DeclareOp constOp = nullptr;

    if (auto cst = getConstDeclare(addOp.getInput1())) {
        constOp = cst;
        nonConstInput = addOp.getInput2();
    } else if (auto cst2 = getConstDeclare(addOp.getInput2())) {
        constOp = cst2;
        nonConstInput = addOp.getInput1();
    } else {
        return mlir::failure();
    }

    // 2. Match Upstream Pattern: Slice -> LayoutCast -> Expand -> [Add]
    auto expandOp = nonConstInput.getDefiningOp<IE::ExpandOp>();
    if (expandOp == nullptr || !expandOp->hasOneUse()) {
        return mlir::failure();
    }

    auto preAddLayoutCastOp = expandOp.getInput().getDefiningOp<IE::LayoutCastOp>();
    if (preAddLayoutCastOp == nullptr || !preAddLayoutCastOp->hasOneUse()) {
        return mlir::failure();
    }

    const auto preLcInType = mlir::cast<vpux::NDTypeInterface>(preAddLayoutCastOp.getInput().getType());
    const auto preLcOutType = mlir::cast<vpux::NDTypeInterface>(preAddLayoutCastOp.getOutput().getType());
    if (preLcInType.getDimsOrder() != DimsOrder::NCHW || preLcOutType.getDimsOrder() != DimsOrder::NHWC) {
        return mlir::failure();
    }

    auto preAddSliceOp = preAddLayoutCastOp.getInput().getDefiningOp<IE::SliceOp>();
    if (preAddSliceOp == nullptr || !preAddSliceOp->hasOneUse()) {
        return mlir::failure();
    }

    const auto sliceAxis =
            IE::getSingleDiffAxis(getShape(preAddSliceOp.getSource()), getShape(preAddSliceOp.getResult()));
    const auto expandAxis = IE::getSingleDiffAxis(getShape(expandOp.getInput()), getShape(expandOp.getResult()));
    if (!sliceAxis.has_value() || !expandAxis.has_value() || sliceAxis.value() == expandAxis.value()) {
        return mlir::failure();
    }

    // 3. Match Downstream Pattern: [Add] -> Slice -> LayoutCast
    auto postAddSliceOp = mlir::dyn_cast<IE::SliceOp>(*addOp.getResult().getUsers().begin());
    if (postAddSliceOp == nullptr || !postAddSliceOp->hasOneUse()) {
        return mlir::failure();
    }

    const auto sliceOutAxis =
            IE::getSingleDiffAxis(getShape(postAddSliceOp.getSource()), getShape(postAddSliceOp.getResult()));
    if (!sliceOutAxis.has_value()) {
        return mlir::failure();
    }

    auto postAddLayoutCastOp = mlir::dyn_cast<IE::LayoutCastOp>(*postAddSliceOp.getResult().getUsers().begin());
    if (postAddLayoutCastOp == nullptr) {
        return mlir::failure();
    }

    const auto postLcOutType = mlir::cast<vpux::NDTypeInterface>(postAddLayoutCastOp.getOutput().getType());

    // Verify Loop-back properties: Input NCHW -> ... -> Output NCHW
    if (preLcInType.getDimsOrder() != postLcOutType.getDimsOrder() ||
        preLcInType.getShape() != postLcOutType.getShape()) {
        return mlir::failure();
    }

    // 4. Verify Slice Properties and Alignment
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(preAddSliceOp.getStaticOffsets());
    const auto sliceSizes = parseIntArrayAttr<int64_t>(preAddSliceOp.getStaticSizes());
    const auto sliceInShape = mlir::cast<vpux::NDTypeInterface>(preAddSliceOp.getSource().getType()).getShape();
    if (sliceOffsets.size() != 4 || sliceSizes.size() != 4 || sliceInShape.size() != 4) {
        return mlir::failure();
    }

    const auto alignment = VPU::NCEInvariant::getAlignment(
            mlir::cast<vpux::NDTypeInterface>(addOp.getOutput().getType()).getElementType());
    if (sliceInShape.totalSize() % alignment != 0 || sliceInShape[Dims4D::Act::C] % alignment == 0) {
        return mlir::failure();
    }

    // 5. Verify and Prepare Constant Transformations
    auto constContentAttr = constOp.getContentAttr();
    auto transformations = constContentAttr.getTransformations();
    if (transformations.empty()) {
        return mlir::failure();
    }

    // Check if we can inject padding into the constant transformations
    // The Slice removing spatial dimensions implies we need to pad the Const back to the full spatial size.
    SmallVector<int64_t> padBefore(sliceOffsets.begin(), sliceOffsets.end());
    SmallVector<int64_t> padAfter(sliceInShape.size(), 0);
    const auto axisInd = sliceAxis.value().ind();
    padAfter[axisInd] = sliceInShape[sliceAxis.value()] - sliceOffsets[axisInd] - sliceSizes[axisInd];

    bool hasLayoutCastAttr = false;
    bool hasPadAttr = false;
    auto currentType = mlir::cast<vpux::NDTypeInterface>(constContentAttr.getBaseContent().getType());
    SmallVector<Const::TransformAttrInterface> newTransformations;

    for (auto attr : transformations) {
        auto transformAttr = mlir::cast<Const::TransformAttrInterface>(attr);
        if (mlir::isa_and_nonnull<Const::LayoutCastAttr>(attr)) {
            // Insert padding before the first LayoutCast that acts on NCHW data
            if (currentType.getDimsOrder() == DimsOrder::NCHW) {
                auto padBeforeAttr = getIntArrayAttr(rewriter.getContext(), padBefore);
                auto padAfterAttr = getIntArrayAttr(rewriter.getContext(), padAfter);
                newTransformations.push_back(Const::PadWithZeroAttr::get(padBeforeAttr, padAfterAttr));
                hasLayoutCastAttr = true;
            }
        }

        hasPadAttr = hasPadAttr || mlir::isa_and_nonnull<Const::PadWithZeroAttr>(attr) ? true : false;
        newTransformations.push_back(transformAttr);
        currentType = transformAttr.inferOutputType(currentType);
    }

    if (!hasLayoutCastAttr || !hasPadAttr) {
        return mlir::failure();
    }

    _log.trace("Downstream match found, optimizing Slice position");
    rewriter.setInsertionPoint(addOp);

    // 6. Rewrite Sequence

    // 6a) LayoutCast on Slice source to match original LayoutCast dst order (NCHW -> NHWC usually)
    auto newLc0Type =
            mlir::cast<vpux::NDTypeInterface>(preAddLayoutCastOp.getOutput().getType()).changeShape(sliceInShape);
    auto newLc0 = rewriter.create<IE::LayoutCastOp>(preAddLayoutCastOp.getLoc(), newLc0Type, preAddSliceOp.getSource(),
                                                    preAddLayoutCastOp.getDstOrderAttr());

    // 6b) Expand on the new, larger layout cast output
    auto newExpand = rewriter.create<IE::ExpandOp>(expandOp.getLoc(), newLc0.getOutput(), expandOp.getPadsBeginAttr(),
                                                   expandOp.getPadsEndAttr());

    // 6c) Create new Const with injected padding
    auto newConstContentAttr = Const::ContentAttr::get(constContentAttr.getBaseContent(), newTransformations);
    auto newConstType = mlir::cast<vpux::NDTypeInterface>(constOp.getOutput().getType());
    auto newConstShape = Shape(newConstType.getShape().toValues());
    for (auto i : irange(newConstShape.size())) {
        newConstShape[Dim(i)] += padBefore[i] + padAfter[i];
    }
    newConstType = newConstType.changeShape(newConstShape);

    auto newConstOp = rewriter.create<Const::DeclareOp>(constOp.getLoc(), newConstType, newConstContentAttr);
    auto newConstValue = newConstOp.getOutput();

    // 6d) Add on expanded space
    const auto origType = mlir::cast<vpux::NDTypeInterface>(addOp.getType());
    const auto newType = origType.changeShape(getShape(newExpand.getOutput()));
    auto newAdd = rewriter.create<IE::AddOp>(addOp.getLoc(), newType, newExpand.getOutput(), newConstValue,
                                             addOp.getAutoBroadcast(), addOp.getPostOpAttr(), addOp.getClampAttr(),
                                             addOp.getOutputPaddingAttr(), addOp.getInputPaddingAttr());

    // 6e) Update the downstream Slice (postAddSliceOp) to handle the larger input
    // It was slicing the small Add output. Now it slices the large Add output.
    // We need to adjust its size to match the large Add output on non-slicing axes.
    auto newAddShape = getShape(newAdd.getOutput());
    auto nextSliceSizes = parseIntArrayAttr<int64_t>(postAddSliceOp.getStaticSizes());
    SmallVector<int64_t> newNextSliceSizes(nextSliceSizes.begin(), nextSliceSizes.end());

    for (auto i : irange(newNextSliceSizes.size())) {
        if (Dim(i) != sliceOutAxis.value()) {
            newNextSliceSizes[i] = newAddShape[Dim(i)];
        }
    }

    postAddSliceOp.setStaticSizesAttr(getIntArrayAttr(rewriter.getContext(), newNextSliceSizes));

    // Update postAddSliceOp Type
    auto nextSliceOutType = mlir::cast<vpux::NDTypeInterface>(postAddSliceOp.getResult().getType());
    auto newNextSliceOutShape = Shape(newNextSliceSizes);
    auto newNextSliceOutType = nextSliceOutType.changeShape(newNextSliceOutShape);
    postAddSliceOp.getResult().setType(mlir::cast<mlir::RankedTensorType>(newNextSliceOutType));

    // 6f) Update downstream LayoutCast (postAddLayoutCastOp) to handle larger input
    auto nextLcOutType = mlir::cast<vpux::NDTypeInterface>(postAddLayoutCastOp.getOutput().getType());
    auto newNextLcOutType = nextLcOutType.changeShape(newNextSliceOutShape);
    postAddLayoutCastOp.getOutput().setType(mlir::cast<mlir::RankedTensorType>(newNextLcOutType));

    // 6g) Insert the original pre-Add Slice at the very end
    rewriter.setInsertionPointAfter(postAddLayoutCastOp);
    SmallVector<int64_t> newSlice0Sizes(sliceSizes.begin(), sliceSizes.end());
    auto newSlice0 = rewriter.create<IE::SliceOp>(preAddSliceOp.getLoc(), postAddLayoutCastOp.getOutput(),
                                                  preAddSliceOp.getStaticOffsetsAttr(),
                                                  getIntArrayAttr(rewriter.getContext(), newSlice0Sizes));

    // Replace usages of the old downstream LayoutCast with the new final Slice
    postAddLayoutCastOp.getOutput().replaceAllUsesExcept(newSlice0.getResult(), newSlice0);
    rewriter.replaceOp(addOp, newAdd);

    return mlir::success();
}

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
    patterns.add<IE::OptimizeSlicePermuteCastExpand>(&ctx, _log);

    patterns.add<IE::OptimizeSliceShapeCastExpand<IE::HSwishOp>>(&ctx, _log);
    patterns.add<IE::OptimizeSliceShapeCastExpand<IE::SwishOp>>(&ctx, _log);
    patterns.add<IE::OptimizeSliceShapeCastExpand<IE::GeluOp>>(&ctx, _log);
    patterns.add<IE::OptimizeSliceShapeCastExpand<IE::SigmoidOp>>(&ctx, _log);

    // The middle op has multi inputs
    patterns.add<IE::OptimizeSliceConcatExpandWithViewLikeOps>(&ctx, _log);
    patterns.add<IE::OptimizeSliceConcatExpand>(&ctx, _log);
    patterns.add<IE::OptimizeSlicePReluExpand>(&ctx, _log);
    patterns.add<IE::OptimizeSliceEltwiseExpand<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<IE::OptimizeSliceEltwiseExpand<IE::AddOp>>(&ctx, _log);
    patterns.add<IE::OptimizeSliceEltwiseExpand<IE::SubtractOp>>(&ctx, _log);
    patterns.add<IE::OptimizeSliceEltwiseExpand<IE::DivideOp>>(&ctx, _log);
    patterns.add<SliceAfterAddForLayoutCastExpandAddRewriter>(&ctx, _log);

    // Pattern slice-op1-op2-...-opN-expand
    patterns.add<IE::OptimizeSliceOpsExpand>(&ctx, _log);

    auto func = getOperation();
    auto greedyRewriteConfig = getDefaultGreedyRewriteConfig();

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), greedyRewriteConfig))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeSliceExpandPass(Logger log) {
    return std::make_unique<OptimizeSliceExpandPass>(log);
}
