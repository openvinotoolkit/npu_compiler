//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_padding_utils.hpp"
#include "vpux/compiler/dialect/const/utils/affine_reshape.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace IE;

void vpux::IE::propagateElementTypeDown(IE::LayerDataInfo<mlir::Type>& info) {
    const auto inputElemType = info.getInput(0);

    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputElemType)) {
        // Do not propagate element type down in per channel case.
        return;
    }

    for (size_t outputInd = 0; outputInd < info.getNumOutputs(); ++outputInd) {
        info.setOutput(outputInd, inputElemType);
    }
}

void vpux::IE::propagateElementTypeUp(IE::LayerDataInfo<mlir::Type>& info) {
    const auto outputElemType = info.getOutput(0);

    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElemType)) {
        // Do not propagate element type up in per channel case.
        return;
    }

    for (size_t inputInd = 0; inputInd < info.getNumInputs(); ++inputInd) {
        info.setInput(inputInd, outputElemType);
    }
}

// Function for checking if Interpolate can be executed using Storage Element hardware.
// E#91549: Move this function to low-level dialect
bool vpux::IE::isSupportedNearestNCEInterpolate(IE::InterpolateOp interpolateOp, vpux::LogCb logCb) {
    const auto inputShape = getShape(interpolateOp.getInput());
    const auto outShape = getShape(interpolateOp.getOutput());

    if (interpolateOp.getAttr().getMode().getValue() == IE::InterpolateMode::NEAREST &&
        (outShape[Dims4D::Act::W] > inputShape[Dims4D::Act::W]) &&
        (outShape[Dims4D::Act::H] > inputShape[Dims4D::Act::H]) &&
        VPU::NCEInterpolateOp::isSupported(interpolateOp, logCb, /*checkLayout=*/false,
                                           /*checkChannelAlignment=*/false, /*checkBatch=*/false)) {
        return true;
    }

    return false;
}

bool vpux::IE::isSupportedElemTypeInfoCase(mlir::Operation* op, bool seOpsEnabled, vpux::LogCb logCb) {
    if (!mlir::isa<IE::ElemTypeInfoOpInterface>(op)) {
        return false;
    }

    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<IE::InterpolateOp>([&](IE::InterpolateOp origOp) {
                return seOpsEnabled && isSupportedNearestNCEInterpolate(origOp, logCb);
            })
            .Case<IE::PadOp>([&](IE::PadOp origOp) {
                return seOpsEnabled &&
                       VPU::isSupportedSEPPadOp(origOp, logCb, /*checkLayout=*/false, /*checkChannelAlignment=*/false);
            })
            .Default([](mlir::Operation*) {
                return true;
            });
}

void vpux::IE::propagateElemTypeDownForAffineReshapeOp(IE::AffineReshapeOp affineReshape,
                                                       IE::LayerDataInfo<mlir::Type>& info) {
    auto outputElemType = inferElemTypeAffineReshape(affineReshape, info.getInput(0));
    if (!outputElemType.has_value()) {
        return;
    }

    for (size_t outputInd = 0; outputInd < info.getNumOutputs(); ++outputInd) {
        info.setOutput(outputInd, outputElemType.value());
    }
}

void vpux::IE::propagateElemTypeDownForConcatOp(IE::ConcatOp concat, IE::LayerDataInfo<mlir::Type>& info) {
    auto loc = concat->getLoc();

    mlir::FailureOr<mlir::Type> outElemType;
    if (!concat.getPerAxis()) {
        const auto outShape = inferOutShapeWithOffsets(concat, loc);
        if (mlir::failed(outShape)) {
            return;
        }

        outElemType = inferOutElemTypeWithOffsets(info.getInputs(), concat, outShape.value());
    } else {
        outElemType = inferOutElemTypeWithAxis(info.getInputs(), concat);
    }

    if (mlir::failed(outElemType)) {
        return;
    }

    info.setOutput(0, outElemType.value());
}

void vpux::IE::propagateElemTypeDownForExpandDilatedOp(IE::ExpandDilatedOp expandDilated,
                                                       IE::LayerDataInfo<mlir::Type>& info) {
    const auto dilationsVal = parseIntArrayAttr<int64_t>(expandDilated.getDilations());
    const auto inputElemType = info.getInput(0);

    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputElemType) && dilationsVal.size() > 2) {
        return;
    }
    for (size_t outputInd = 0; outputInd < info.getNumOutputs(); ++outputInd) {
        info.setOutput(outputInd, inputElemType);
    }
}

void vpux::IE::propagateElemTypeDownForReorderOp(IE::ReorderOp reorder, IE::LayerDataInfo<mlir::Type>& info) {
    auto outputElemType = inferElemTypeReorder(reorder, info.getInput(0), reorder->getContext());

    for (size_t outputInd = 0; outputInd < info.getNumOutputs(); ++outputInd) {
        info.setOutput(outputInd, outputElemType);
    }
}

void vpux::IE::propagateElemTypeDownForTransposeOp(IE::TransposeOp transpose, IE::LayerDataInfo<mlir::Type>& info) {
    auto outputElemType = inferElemTypeTranspose(transpose.getOrderValue().value(), info.getInput(0));

    for (size_t outputInd = 0; outputInd < info.getNumOutputs(); ++outputInd) {
        info.setOutput(outputInd, outputElemType);
    }
}

void vpux::IE::propagateElemTypeUpForExpandDilatedOp(IE::ExpandDilatedOp expandDilated,
                                                     IE::LayerDataInfo<mlir::Type>& info) {
    const auto dilationsVal = parseIntArrayAttr<int64_t>(expandDilated.getDilations());
    const auto outputElemType = info.getOutput(0);

    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElemType) && dilationsVal.size() > 2) {
        return;
    }
    for (size_t inputInd = 0; inputInd < info.getNumInputs(); ++inputInd) {
        info.setInput(inputInd, outputElemType);
    }
}

//
// AffineReshapeOp
//

std::optional<mlir::Type> vpux::IE::inferElemTypeAffineReshape(AffineReshapeOpAdaptor affineReshapeOp,
                                                               mlir::Type inputElemType) {
    const auto inputType = mlir::cast<NDTypeInterface>(affineReshapeOp.getInput().getType());
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.getDimMapping());
    const auto shapeValue = parseIntArrayAttr<int64_t>(affineReshapeOp.getShapeValue());
    return Const::inferElemTypeAffineReshape(inputType.getShape(), inputElemType, dimMapping, shapeValue);
}

//
// ConcatOp
//

Dim vpux::IE::normalizeAxis(IE::ConcatOpAdaptor concat) {
    const auto inType = mlir::cast<mlir::ShapedType>(concat.getInputs().front().getType());
    const auto inRank = inType.getRank();

    auto axisInd = concat.getPerAxis().value().getAxis().getValue().getSExtValue();

    // Negative value means counting dimension from the end
    if (axisInd < 0) {
        axisInd += inRank;
    }

    VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Concat axis '{0}', out of range '{1}'", axisInd,
                      inRank);

    return Dim(axisInd);
}

mlir::FailureOr<Shape> vpux::IE::inferOutShapeWithOffsets(IE::ConcatOpAdaptor concat, mlir::Location loc) {
    if (!concat.getStaticOffsets().has_value()) {
        return errorAt(loc, "Missing static_offsets attribute");
    }

    const auto staticOffsets = concat.getStaticOffsets().value();
    if (staticOffsets.size() != concat.getInputs().size()) {
        return errorAt(loc, "Concat 'static_offsets' count '{0}' doesn't match inputs count '{1}'",
                       staticOffsets.size(), concat.getInputs().size());
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(concat.getInputs().front().getType());
    const auto allOffsets = staticOffsets.getAsRange<mlir::ArrayAttr>();

    Shape outShape(checked_cast<size_t>(inType.getRank()), 0);

    for (const auto& p : zip(concat.getInputs(), allOffsets)) {
        const auto curVal = std::get<0>(p);
        const auto curShape = getShape(curVal);

        if (curShape.size() != outShape.size()) {
            return errorAt(loc, "Concat inputs have mismatched ranks: '{0}' vs '{1}'", curShape.size(),
                           outShape.size());
        }

        const auto curOffsets = Shape(parseIntArrayAttr<int64_t>(std::get<1>(p)));

        if (curOffsets.size() != curShape.size()) {
            return errorAt(loc, "Concat 'static_offsets' rank doesn't match its input");
        }

        for (const auto ind : irange(outShape.size())) {
            const auto d = Dim(ind);

            outShape[d] = std::max(outShape[d], curOffsets[d] + curShape[d]);
        }
    }

    // TODO: validate that inputs+static_offsets fully covers the output without intersections

    return outShape;
}

mlir::FailureOr<mlir::Type> vpux::IE::inferOutElemTypeWithAxis(ArrayRef<mlir::Type> elemTypes,
                                                               IE::ConcatOpAdaptor concat, LogCb logCb) {
    const auto inElemType = elemTypes[0];

    const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(inElemType);
    SmallVector<mlir::quant::UniformQuantizedPerAxisType> inPerAxisQTypes;

    if (perAxisQType != nullptr) {
        const auto axis = normalizeAxis(concat);

        if (perAxisQType.getQuantizedDimension() == axis.ind()) {
            inPerAxisQTypes.push_back(perAxisQType);
        }
    }

    for (const auto& curElemType : elemTypes.drop_front()) {
        if (inPerAxisQTypes.empty()) {
            if (curElemType != inElemType) {
                logCb(formatv("Misaligned element types : '{0}' vs '{1}'", curElemType, inElemType));
                return mlir::failure();
            }
        } else {
            const auto curPerAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(curElemType);

            if (curPerAxisQType == nullptr) {
                logCb(formatv("Misaligned element types : not all of them are per-axis quantized : '{0}' vs '{1}'",
                              curElemType, inElemType));
                return mlir::failure();
            }

            if (curPerAxisQType.getQuantizedDimension() != perAxisQType.getQuantizedDimension()) {
                logCb(formatv(
                        "Misaligned element types : per-axis quantization is done on different axis : '{0}' vs '{1}'",
                        curPerAxisQType.getQuantizedDimension(), perAxisQType.getQuantizedDimension()));
                return mlir::failure();
            }

            if (!canBeMerged(curPerAxisQType, perAxisQType)) {
                logCb(formatv("Misaligned element types : per-axis quantization parameters can't be merged"));
                return mlir::failure();
            }

            inPerAxisQTypes.push_back(curPerAxisQType);
        }
    }

    return inPerAxisQTypes.empty() ? inElemType : concatScalesAndZP(inPerAxisQTypes);
}

std::unordered_set<Dim> vpux::IE::getConcatAxesFromOffsets(IE::ConcatOpAdaptor concat, ShapeRef outShape) {
    std::unordered_set<Dim> res;

    for (const auto& inVal : concat.getInputs()) {
        const auto curShape = getShape(inVal);

        for (const auto ind : irange(outShape.size())) {
            const auto d = Dim(ind);

            if (curShape[d] != outShape[d]) {
                res.insert(d);
            }
        }
    }

    return res;
}

mlir::FailureOr<mlir::Type> vpux::IE::inferOutElemTypeWithOffsets(ArrayRef<mlir::Type> elemTypes,
                                                                  IE::ConcatOpAdaptor concat, ShapeRef outShape,
                                                                  LogCb logCb) {
    const auto inElemType = elemTypes[0];

    const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(inElemType);

    const auto isConcatOverPerAxisQuantization = [&]() {
        if (perAxisQType == nullptr) {
            return false;
        }

        const auto qDim = Dim(perAxisQType.getQuantizedDimension());
        const auto concatAxes = getConcatAxesFromOffsets(concat, outShape);

        return concatAxes.count(qDim) != 0;
    }();

    if (!isConcatOverPerAxisQuantization) {
        for (const auto& curElemType : elemTypes.drop_front()) {
            if (curElemType != inElemType) {
                logCb(formatv("Misaligned element types : '{0}' vs '{1}'", curElemType, inElemType));
                return mlir::failure();
            }
        }

        return inElemType;
    }

    const auto qDim = perAxisQType.getQuantizedDimension();
    const auto allOffsets = concat.getStaticOffsets().value().getAsRange<mlir::ArrayAttr>();

    std::map<int64_t, mlir::quant::UniformQuantizedPerAxisType> perSliceQuantTypes;

    for (const auto& p : zip(elemTypes, allOffsets)) {
        const auto curElemType = std::get<0>(p);
        const auto curPerAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(curElemType);

        if (curPerAxisQType == nullptr) {
            logCb(formatv("Misaligned element types : not all of them are per-axis quantized : '{0}' vs '{1}'",
                          curElemType, inElemType));
            return mlir::failure();
        }

        if (curPerAxisQType.getQuantizedDimension() != qDim) {
            logCb(formatv("Misaligned element types : per-axis quantization is done on different axis : '{0}' vs '{1}'",
                          curPerAxisQType.getQuantizedDimension(), qDim));
            return mlir::failure();
        }

        const auto curOffsets = parseIntArrayAttr<int64_t>(std::get<1>(p));
        const auto sliceOffset = curOffsets[checked_cast<size_t>(qDim)];

        const auto it = perSliceQuantTypes.find(sliceOffset);
        if (it == perSliceQuantTypes.end()) {
            perSliceQuantTypes.insert({sliceOffset, curPerAxisQType});
        } else {
            if (curPerAxisQType != it->second) {
                logCb(formatv("Per-axis quantization is not aligned over non quantized axis : '{0}' vs '{1}'",
                              curPerAxisQType, it->second));
                return mlir::failure();
            }
        }
    }

    const auto inPerAxisQTypes = to_small_vector(perSliceQuantTypes | map_values);
    return concatScalesAndZP(inPerAxisQTypes);
}

//
// ReorderOp
//

mlir::Type vpux::IE::inferElemTypeReorder(IE::ReorderOpAdaptor reorder, mlir::Type inputElemType,
                                          mlir::MLIRContext* ctx) {
    const auto perAxisQType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(inputElemType);
    if (perAxisQType == nullptr) {
        return inputElemType;
    }

    const auto inputAxis = perAxisQType.getQuantizedDimension();

    const auto inOrder = DimsOrder::fromValue(reorder.getInput());
    const auto outOrder = DimsOrder::fromAffineMap(reorder.getDstOrder());
    const auto memPerm = vpux::getPermutationFromOrders(inOrder, outOrder, ctx);

    const auto outAxis = DimsOrder::fromAffineMap(memPerm).toPermutation()[inputAxis];

    if (const auto perAxisQuantileQType =
                mlir::dyn_cast_or_null<mlir::quant::QuantileQuantizedPerAxisType>(inputElemType)) {
        return mlir::quant::QuantileQuantizedPerAxisType::get(
                perAxisQuantileQType.getFlags(), perAxisQuantileQType.getStorageType(),
                perAxisQuantileQType.getQuantileType(), perAxisQuantileQType.getExpressedType(),
                perAxisQuantileQType.getQuantiles(), perAxisQuantileQType.getScales(),
                perAxisQuantileQType.getZeroPoints(), outAxis.ind(), perAxisQuantileQType.getStorageTypeMin(),
                perAxisQuantileQType.getStorageTypeMax());
    }

    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(),
            perAxisQType.getScales(), perAxisQType.getZeroPoints(), outAxis.ind(), perAxisQType.getStorageTypeMin(),
            perAxisQType.getStorageTypeMax());
}

//
// TransposeOp
//

mlir::Type vpux::IE::inferElemTypeTranspose(mlir::AffineMap map, mlir::Type inputElemType) {
    const auto perAxisQType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(inputElemType);
    if (perAxisQType == nullptr) {
        return inputElemType;
    }

    const auto origAxis = perAxisQType.getQuantizedDimension();
    const auto newAxis = DimsOrder::fromAffineMap(map).dimPos(Dim(origAxis));

    if (const auto perAxisQuantileQType =
                mlir::dyn_cast_or_null<mlir::quant::QuantileQuantizedPerAxisType>(inputElemType)) {
        return mlir::quant::QuantileQuantizedPerAxisType::get(
                perAxisQuantileQType.getFlags(), perAxisQuantileQType.getStorageType(),
                perAxisQuantileQType.getQuantileType(), perAxisQuantileQType.getExpressedType(),
                perAxisQuantileQType.getQuantiles(), perAxisQuantileQType.getScales(),
                perAxisQuantileQType.getZeroPoints(), newAxis, perAxisQuantileQType.getStorageTypeMin(),
                perAxisQuantileQType.getStorageTypeMax());
    }

    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(),
            perAxisQType.getScales(), perAxisQType.getZeroPoints(), newAxis, perAxisQType.getStorageTypeMin(),
            perAxisQType.getStorageTypeMax());
}
