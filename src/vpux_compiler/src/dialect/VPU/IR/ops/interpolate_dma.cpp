//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dynamic_shape_propagation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/interpolate_bound.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::InterpolateDMAOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::InterpolateDMAOpAdaptor adaptor(operands, attrs, prop);
    if (mlir::failed(adaptor.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(adaptor.getInput().getType());
    const auto inShape = getBoundedShape(adaptor.getInput());
    const auto axesVal = parseIntArrayAttr<int64_t>(adaptor.getAxesAttr());
    const auto beginPads = IE::extractIntVector(loc, nullptr, adaptor.getAttr().getPadsBegin());
    const auto endPads = IE::extractIntVector(loc, nullptr, adaptor.getAttr().getPadsEnd());

    // The two trailing axes (H, W) are resized, so at least 2 axes are required.
    // VPU_InterpolateDMAOp has no verifier enforcing this but IE_InterpolateOp does
    // and InterpolateDMAOp is only created from InterpolateOp, so this condition is guaranteed by construction.
    VPUX_THROW_UNLESS(axesVal.size() >= 2, "InterpolateDMA expects at least 2 axes, got {0}", axesVal.size());

    // Scales are runtime parameters — use bounded scales for compile-time shape inference
    auto scalesBound = SmallVector<double>(axesVal.size(), 1.0);
    const auto spatialScalesBound = getInterpolateScalesBound(inputType);
    scalesBound[scalesBound.size() - 1] = spatialScalesBound;
    scalesBound[scalesBound.size() - 2] = spatialScalesBound;

    const auto scalesElemType = mlir::cast<vpux::NDTypeInterface>(adaptor.getScales().getType()).getElementType();

    const auto outShapeVec =
            IE::inferInterpOutShape(loc, axesVal, inShape, beginPads, endPads, IE::InterpolateCalcMode::SCALES,
                                    mlir::FailureOr<ArrayRef<int64_t>>(mlir::failure()), ArrayRef<double>(scalesBound),
                                    scalesElemType, Logger::global());

    auto [outDesc, outShape] = callOnShapeOf(inputType, [&](const auto& shape) {
        using ShapeT = std::decay_t<decltype(shape)>;
        if constexpr (std::is_same_v<ShapeT, BoundedShape>) {
            auto desc =
                    vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace(), BoundsRef(outShapeVec));
            // An output dim is dynamic when:
            //   - its scale bound != 1.0 (spatial axes being resized), OR
            //   - the corresponding input dim is already dynamic (propagate input dynamism).
            // The second condition also covers identity-scale axes that are in axesVal.
            auto staticShape = outShapeVec;
            for (const auto& [axisIndex, axis] : llvm::enumerate(axesVal)) {
                if (scalesBound[axisIndex] != 1.0 || inputType.getShape()[Dim(axis)] == mlir::ShapedType::kDynamic) {
                    staticShape[axis] = mlir::ShapedType::kDynamic;
                }
            }
            for (int64_t i = 0; i < inputType.getRank(); ++i) {
                if (llvm::find(axesVal, i) == axesVal.end() &&
                    inputType.getShape()[Dim(i)] == mlir::ShapedType::kDynamic) {
                    staticShape[i] = mlir::ShapedType::kDynamic;
                }
            }
            return std::make_pair(desc, staticShape);
        } else if constexpr (std::is_same_v<ShapeT, DimsMaskedShape>) {
            auto mask = mlir::cast<Core::DynamicDimsMaskTensorType>(inputType).getDynamicDimsMask();
            auto outMask = SmallVector<int64_t>(mask.begin(), mask.end());
            for (const auto& [axisIndex, axis] : llvm::enumerate(axesVal)) {
                if (scalesBound[axisIndex] != 1.0) {
                    outMask[axis] = 1;
                }
            }
            auto desc = vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace(), {},
                                            DynamicDimsMaskRef(outMask));
            return std::make_pair(desc, outShapeVec);
        } else {
            // Static input — only variable-scale axes (scalesBound != 1.0) become dynamic with bounds.
            // Identity-scale axes (e.g. N, C) remain static.
            SmallVector<int64_t> outDynShape(outShapeVec.begin(), outShapeVec.end());
            for (const auto& [axisIndex, axis] : llvm::enumerate(axesVal)) {
                if (scalesBound[axisIndex] != 1.0) {
                    outDynShape[axis] = mlir::ShapedType::kDynamic;
                }
            }
            auto typeComponents =
                    TypeComponents().setDimsOrder(inputType.getDimsOrder()).setElementType(inputType.getElementType());
            assignDynamicTypeComponents(typeComponents, adaptor.getBoundsRepresentation(), outDynShape, outShapeVec);
            auto outType = inputType.changeTypeComponents(typeComponents);
            return std::make_pair(
                    mlir::dyn_cast_or_null<vpux::TensorAttr>(mlir::cast<mlir::RankedTensorType>(outType).getEncoding()),
                    SmallVector<int64_t>(mlir::cast<mlir::ShapedType>(outType).getShape()));
        }
    });

    auto outputType = mlir::RankedTensorType::get(outShape, inputType.getElementType(), outDesc);
    inferredReturnTypes.push_back(outputType);

    return mlir::success();
}

//
// ReifyRankedShapedTypeOpInterface
//

mlir::LogicalResult vpux::VPU::InterpolateDMAOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    const auto loc = getLoc();
    const auto outputShapedType = mlir::cast<mlir::ShapedType>(getOutput().getType());
    const auto axesVal = parseIntArrayAttr<int64_t>(getAxesAttr());

    return reifyInterpolateResultShape(builder, loc, getInput(), getScales(), std::nullopt, axesVal, outputShapedType,
                                       reifiedReturnShapes);
}
