//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::TransposedConvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::TransposedConvolutionOpAdaptor convBackpropData(operands, attrs, prop);
    if (mlir::failed(convBackpropData.verify(loc))) {
        return mlir::failure();
    }

    const auto featureType = mlir::cast<vpux::NDTypeInterface>(convBackpropData.getInput().getType());
    const auto featureShape = featureType.getShape().raw();
    const auto outputShape = convBackpropData.getOutputShape();
    const auto filterShape = mlir::cast<vpux::NDTypeInterface>(convBackpropData.getFilter().getType()).getShape().raw();

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(convBackpropData.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(convBackpropData.getPadsBegin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(convBackpropData.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(convBackpropData.getDilations());
    const auto outputPadding = parseIntArrayAttr<int64_t>(convBackpropData.getSpatialOutputPadding());

    if (outputShape != nullptr) {
        auto outputShapeConst = outputShape.getDefiningOp<Const::DeclareOp>();
        if (outputShapeConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for output_shape");
        }

        const auto outputShapeContent = outputShapeConst.getContent();
        const auto outputShapeVals = outputShapeContent.getValues<int64_t>();

        SmallVector<int64_t> mlirOutputShape;
        mlirOutputShape.push_back(featureShape[Dims4D::Act::N.ind()]);
        mlirOutputShape.push_back(filterShape[Dims4D::Filter::OC.ind()]);
        std::copy(outputShapeVals.begin(), outputShapeVals.end(), std::back_inserter(mlirOutputShape));

        auto outType = featureType.changeShape(ShapeRef(mlirOutputShape));
        inferredReturnTypes.push_back(outType);
    } else {
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(convBackpropData.getInput().getType());
        const auto filterType = mlir::cast<vpux::NDTypeInterface>(convBackpropData.getFilter().getType());

        const auto inShapeInfo = ShapeInfo::fromNDType(inputType);
        const auto filterShapeInfo = ShapeInfo::fromNDType(filterType);

        auto shapeInfo = inferTransposedConvBackpropOutputShapeInfo(inShapeInfo, filterShapeInfo, windowStrides,
                                                                    dataPaddingBelow, dataPaddingAbove, windowDilations,
                                                                    outputPadding);

        const auto outDesc =
                vpux::getTensorAttr(ctx, inputType.getDimsOrder(), /*memSpace=*/nullptr, BoundsRef(shapeInfo.bounds));
        const auto outType = mlir::RankedTensorType::get(shapeInfo.shape, inputType.getElementType(), outDesc);
        inferredReturnTypes.push_back(outType);
    }

    return mlir::success();
}
