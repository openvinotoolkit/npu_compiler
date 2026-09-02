//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dynamic_shape_propagation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <algorithm>

using namespace vpux;

namespace {

mlir::Type getAuxiliaryBufferType(mlir::Value inBoxCoords, mlir::FloatAttr softNmsSigmaValueAttr) {
    const auto inBoxCoordsType = mlir::cast<vpux::NDTypeInterface>(inBoxCoords.getType());
    auto elemType = inBoxCoordsType.getElementType();
    size_t elemTypeSize = Byte(vpux::getElemTypeSize(elemType)).count();
    const auto inputShape = inBoxCoordsType.getShape();
    const auto shapeInfo = ShapeInfo::fromNDType(inBoxCoordsType);
    auto numBoxes = inputShape[Dim(1)];
    if (shapeInfo.isDynamic()) {
        numBoxes = shapeInfo.bounds[Dim(1).ind()];
    }
    const auto softNmsSigma = softNmsSigmaValueAttr != nullptr ? softNmsSigmaValueAttr.getValueAsDouble() : 0.0;

    size_t offset = 0;

    // boxesPtrCMXBuffer should be allocated only if softNmsSigma is 0.0f
    size_t boxesPtrCMXBufferSize = 0;
    if (softNmsSigma == 0.0) {
        boxesPtrCMXBufferSize = 4 * numBoxes * elemTypeSize;
        offset += boxesPtrCMXBufferSize;
    }

    // scoresPtrCMX buffer
    size_t scoresPtrCMXbufferSize = numBoxes * elemTypeSize;
    offset += scoresPtrCMXbufferSize;
    offset = (offset + 3) & ~3;  // Align offset for boxIdxPtrCMX (int32_t)

    // boxIdxPtrCMX buffer
    size_t boxIdxPtrCMX = numBoxes * sizeof(int32_t);
    offset += boxIdxPtrCMX;

    const auto dataBufferSize = static_cast<int64_t>(offset);
    const auto auxBuffType =
            mlir::RankedTensorType::get({1, 1, 1, dataBufferSize}, getUInt8Type(inBoxCoords.getContext()));
    return auxBuffType;
}

}  // namespace

void VPU::NonMaxSuppressionOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                     mlir::Value inBoxCoords, mlir::Value inBoxScores,
                                     IE::BoxEncodingTypeAttr boxEncoding, mlir::UnitAttr sortResultDescending,
                                     mlir::IntegerAttr maxOutputBoxesPerClassValue, mlir::FloatAttr iouThresholdValue,
                                     mlir::FloatAttr scoreThresholdValue, mlir::FloatAttr softNmsSigmaValue,
                                     VPU::BoundsRepresentationAttr boundsRepresentation) {
    const auto auxBuffType = getAuxiliaryBufferType(inBoxCoords, softNmsSigmaValue);
    auto auxBuffer = VPU::createEmptyAuxiliaryBuffer(odsBuilder, odsState.location, auxBuffType);
    build(odsBuilder, odsState, inBoxCoords, inBoxScores, auxBuffer, boxEncoding, sortResultDescending,
          maxOutputBoxesPerClassValue, iouThresholdValue, scoreThresholdValue, softNmsSigmaValue, boundsRepresentation);
}

mlir::LogicalResult VPU::NonMaxSuppressionOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               std::optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::OpaqueProperties prop,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::NonMaxSuppressionOpAdaptor nms(operands, attrs, prop);
    if (mlir::failed(nms.verify(loc))) {
        return mlir::failure();
    }

    VPUX_THROW_UNLESS(nms.getMaxOutputBoxesPerClassValueAttr() != nullptr,
                      "VPU::NonMaxSuppression: max_output_boxes_per_class_value attribute is required");
    const int64_t maxOutputBoxesPerClass = nms.getMaxOutputBoxesPerClassValueAttr().getValue().getSExtValue();
    const auto inScoresType = mlir::cast<vpux::NDTypeInterface>(nms.getInBoxScores().getType());
    const auto inScoresShapeInfo = ShapeInfo::fromNDType(inScoresType);
    const auto hasBounds = inScoresShapeInfo.isDynamic();
    const auto& sizeSource = hasBounds ? inScoresShapeInfo.bounds : inScoresShapeInfo.shape;
    const auto numBatches = sizeSource[0];
    const auto numClasses = sizeSource[1];
    const auto numBoxes = maxOutputBoxesPerClass == 0 ? sizeSource[2] : std::min(sizeSource[2], maxOutputBoxesPerClass);

    const auto sInt32Type = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
    const SmallVector<int64_t> dynamicOutShape{mlir::ShapedType::kDynamic, 3};

    const auto memSpace = inScoresType.getMemSpace();
    const auto buildElemOutType = [&](mlir::Type elemType) -> mlir::Type {
        if (hasBounds) {
            const SmallVector<int64_t> boundsShape{numBatches * numClasses * numBoxes, 3};
            auto typeComponents = TypeComponents().setDimsOrder(DimsOrder::NC).setElementType(elemType);
            assignDynamicTypeComponents(typeComponents, nms.getBoundsRepresentation(), dynamicOutShape, boundsShape);
            const auto bounds =
                    typeComponents.bounds.has_value() ? BoundsRef(typeComponents.bounds.value()) : BoundsRef();
            const auto mask = typeComponents.dynamicDimsMask.has_value()
                                      ? DynamicDimsMaskRef(typeComponents.dynamicDimsMask.value())
                                      : DynamicDimsMaskRef();
            return getTensorType(ShapeRef(typeComponents.shape.value()), elemType, DimsOrder::NC, memSpace, bounds,
                                 mask);
        }
        const SmallVector<int64_t> staticOutShape{numBatches * numClasses * numBoxes, 3};
        return getTensorType(ShapeRef(staticOutShape), elemType, DimsOrder::NC, memSpace, BoundsRef(),
                             DynamicDimsMaskRef());
    };

    inferredReturnTypes.push_back(buildElemOutType(sInt32Type));
    inferredReturnTypes.push_back(buildElemOutType(inScoresType.getElementType()));
    inferredReturnTypes.push_back(mlir::RankedTensorType::get({1}, sInt32Type));
    return mlir::success();
}

llvm::LogicalResult VPU::NonMaxSuppressionOp::verify() {
    auto auxBufferType = mlir::cast<NDTypeInterface>(getDataBuffer().getType());
    auto expectedType =
            mlir::cast<NDTypeInterface>(getAuxiliaryBufferType(getInBoxCoords(), getSoftNmsSigmaValueAttr()));
    return VPU::compareTypes(getOperation()->getLoc(), auxBufferType, expectedType);
}

SmallVector<mlir::OpOperand*> VPU::NonMaxSuppressionOp::getAuxiliaryBuffers() {
    return {&getDataBufferMutable()};
}
