//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::NonMaxSuppressionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
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
