//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

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

    const auto inType = mlir::cast<vpux::NDTypeInterface>(nms.getInBoxScores().getType());
    const auto sInt32Type = inType.changeElemType(mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed));

    int64_t maxOutputBoxesPerClass = nms.getMaxOutputBoxesPerClassValueAttr().getValue().getSExtValue();
    const auto inShape = inType.getShape().raw();  // nbatch*nclasses*nboxes
    const auto numBatches = inShape[0];
    const auto numClasses = inShape[1];
    const auto numBoxes = inShape[2];
    const auto minBoxes = std::min(numBoxes, maxOutputBoxesPerClass);
    const SmallVector<int64_t> outShape{minBoxes * numBatches * numClasses, 3};
    const SmallVector<int64_t> validOutputsShape{1};

    const auto outFloatType = inType.changeShape(ShapeRef(outShape));
    const auto outIntType = sInt32Type.changeShape(ShapeRef(outShape));
    const auto validOutputsType = sInt32Type.changeShape(ShapeRef(validOutputsShape));
    inferredReturnTypes.push_back(outIntType);
    inferredReturnTypes.push_back(outFloatType);
    inferredReturnTypes.push_back(validOutputsType);
    return mlir::success();
}

SmallVector<mlir::Value> VPU::NonMaxSuppressionOp::getAuxiliaryBuffers() {
    return {getDataBuffer()};
}

mlir::LogicalResult VPU::NonMaxSuppressionOp::setAuxiliaryBuffers(ArrayRef<mlir::Value> buffers) {
    if (buffers.size() != 1 || buffers.front() == nullptr) {
        return mlir::failure();
    }
    getDataBufferMutable().assign(buffers.front());
    return mlir::success();
}

SmallVector<mlir::Type> VPU::NonMaxSuppressionOp::getBufferTypes() {
    const auto inBoxCoordsType = mlir::cast<vpux::NDTypeInterface>(getInBoxCoords().getType());
    auto elemType = inBoxCoordsType.getElementType();
    size_t elemTypeSize = Byte(vpux::getElemTypeSize(elemType)).count();
    const auto inputShape = inBoxCoordsType.getShape();
    const auto numBoxes = inputShape[Dim(1)];
    auto softNmsSigmaAttr = getSoftNmsSigmaValueAttr();
    const auto softNmsSigma = softNmsSigmaAttr != nullptr ? softNmsSigmaAttr.getValueAsDouble() : 0.0;

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
    const auto auxBuffType = mlir::RankedTensorType::get({1, 1, 1, dataBufferSize}, getUInt8Type(getContext()));
    return {auxBuffType};
}
