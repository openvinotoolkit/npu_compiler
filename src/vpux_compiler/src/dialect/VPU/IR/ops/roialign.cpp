//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ROIAlignOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            std::optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ROIAlignOpAdaptor roiAlign(operands, attrs, prop);
    if (mlir::failed(roiAlign.verify(loc))) {
        return mlir::failure();
    }

    const auto pooledH = roiAlign.getPooledH();
    const auto pooledW = roiAlign.getPooledW();
    const auto inTypeFeatureMap = mlir::cast<vpux::NDTypeInterface>(roiAlign.getInput().getType());
    const auto inShapeFeatureMap = inTypeFeatureMap.getShape();

    const auto inTypeCoord = mlir::cast<vpux::NDTypeInterface>(roiAlign.getCoords().getType());
    const auto inShapeCoord = inTypeCoord.getShape();

    if (inShapeFeatureMap.size() != 4) {
        return errorAt(loc, "Dimension of the feature maps input should be 4. Got {0} D tensor",
                       inShapeFeatureMap.size());
    }

    if (inShapeCoord.size() != 2) {
        return errorAt(loc, "Dimension of the ROIs input with box coordinates should be 2. Got {0} D tensor",
                       inShapeCoord.size());
    }

    if (pooledH <= 0) {
        return errorAt(loc, "pooledH should be positive. Got {0}", pooledH);
    }

    if (pooledW <= 0) {
        return errorAt(loc, "pooledW should be positive. Got {0}", pooledW);
    }

    SmallVector<int64_t> outputShape;
    outputShape.push_back(inShapeCoord.raw()[0]);
    outputShape.push_back(inShapeFeatureMap.raw()[1]);
    outputShape.push_back(pooledH);
    outputShape.push_back(pooledW);

    const auto outType = inTypeFeatureMap.changeShape(ShapeRef(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
