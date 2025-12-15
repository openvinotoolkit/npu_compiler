//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::AdaptiveMaxPoolOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::AdaptiveMaxPoolOpAdaptor adaptiveMaxPool(operands, attrs, prop);
    if (mlir::failed(adaptiveMaxPool.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(adaptiveMaxPool.getInput().getType());
    const auto inputShape = inputType.getShape();

    if (inputShape.size() != 3 && inputShape.size() != 4 && inputShape.size() != 5) {
        return errorAt(loc, "Input shape should be 3D, 4D or 5D. Got {0}D", inputShape.size());
    }

    auto spatialDimData = VPU::extractConstData(loc, adaptiveMaxPool.getPooledSpatialShape());
    if (mlir::failed(spatialDimData)) {
        return mlir::failure();
    }

    auto pooledSpatialShape = spatialDimData.value();

    if (inputShape.size() != 2 + pooledSpatialShape.size()) {
        return errorAt(loc, "Input shape is not compatible with pooled shape size. Got {0}D and size {1}",
                       inputShape.size(), pooledSpatialShape.size());
    }

    SmallVector<int64_t> outputShape;
    outputShape.push_back(inputShape.raw()[0]);
    outputShape.push_back(inputShape.raw()[1]);
    for (size_t i = 0; i < pooledSpatialShape.size(); i++) {
        outputShape.push_back(pooledSpatialShape[i]);
    }

    const auto outType = inputType.changeShape(ShapeRef(outputShape));
    inferredReturnTypes.push_back(outType);

    const auto outType1 = outType.changeElemType(adaptiveMaxPool.getIndexElementType());
    inferredReturnTypes.push_back(outType1);

    return mlir::success();
}
