//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

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

void VPU::NonMaxSuppressionOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                     mlir::Value inBoxCoords, mlir::Value inBoxScores, mlir::Value iouThreshold,
                                     mlir::Value scoreThreshold, IE::BoxEncodingTypeAttr boxEncoding,
                                     mlir::UnitAttr sortResultDescending, mlir::IntegerAttr maxOutputBoxesPerClassValue,
                                     mlir::FloatAttr iouThresholdValue, mlir::FloatAttr scoreThresholdValue,
                                     mlir::FloatAttr softNmsSigmaValue) {
    const auto auxBuffType = getAuxiliaryBufferType(inBoxCoords, softNmsSigmaValue);
    auto auxBuffer = VPU::createEmptyAuxiliaryBuffer(odsBuilder, odsState.location, auxBuffType);
    build(odsBuilder, odsState, inBoxCoords, inBoxScores, iouThreshold, scoreThreshold, auxBuffer, boxEncoding,
          sortResultDescending, maxOutputBoxesPerClassValue, iouThresholdValue, scoreThresholdValue, softNmsSigmaValue);
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

    const int64_t maxOutputBoxesPerClass = nms.getMaxOutputBoxesPerClassValueAttr().getValue().getSExtValue();
    const auto inScoresType = mlir::cast<vpux::NDTypeInterface>(nms.getInBoxScores().getType());
    const auto inScoresShapeInfo = ShapeInfo::fromNDType(inScoresType);
    const auto numBatches = inScoresShapeInfo.shape[0];
    const auto numClasses = inScoresShapeInfo.shape[1];
    const auto numBoxes = std::min(inScoresShapeInfo.shape[2], maxOutputBoxesPerClass);
    SmallVector<int64_t> outShape{numBatches * numClasses * numBoxes, 3};
    TensorAttr outTensorAttr = nullptr;

    if (inScoresShapeInfo.isDynamic()) {
        // Handle dynamic shape case
        const auto numBatches = inScoresShapeInfo.bounds[0];
        const auto numClasses = inScoresShapeInfo.bounds[1];
        const auto numBoxes = std::min(inScoresShapeInfo.bounds[2], maxOutputBoxesPerClass);
        const Bounds bounds{numBatches * numClasses * numBoxes, 3};
        outTensorAttr = vpux::getTensorAttr(ctx, vpux::DimsOrder::NC, nullptr, bounds);
        outShape = SmallVector<int64_t>{mlir::ShapedType::kDynamic, 3};
    }

    const auto sInt32Type = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
    const auto outType0 = mlir::RankedTensorType::get(outShape, sInt32Type, outTensorAttr);
    const auto outType1 = mlir::RankedTensorType::get(outShape, inScoresType.getElementType(), outTensorAttr);
    const auto outType2 = mlir::RankedTensorType::get({1}, sInt32Type);

    inferredReturnTypes.push_back(outType0);
    inferredReturnTypes.push_back(outType1);
    inferredReturnTypes.push_back(outType2);
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
