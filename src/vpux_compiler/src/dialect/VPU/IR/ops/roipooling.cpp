//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ROIPoolingOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ROIPoolingOpAdaptor roiPooling(operands, attrs, prop);
    if (mlir::failed(roiPooling.verify(loc))) {
        return mlir::failure();
    }

    const auto outputSize = parseIntArrayAttr<int64_t>(roiPooling.getOutputSize());
    const auto inTypeFeatureMap = mlir::cast<vpux::NDTypeInterface>(roiPooling.getInput().getType());
    const auto inShapeFeatureMap = inTypeFeatureMap.getShape();

    const auto inTypeCoord = mlir::cast<vpux::NDTypeInterface>(roiPooling.getCoords().getType());
    const auto inShapeCoord = inTypeCoord.getShape();

    if (inShapeFeatureMap.size() != 4) {
        return errorAt(loc, "Dimension of the feature maps input should be 4. Got {0} D tensor",
                       inShapeFeatureMap.size());
    }

    if (inShapeCoord.size() != 2) {
        return errorAt(loc, "Dimension of the ROIs input with box coordinates should be 2. Got {0} D tensor",
                       inShapeCoord.size());
    }

    if (outputSize.size() != 2) {
        return errorAt(loc, "Dimension of pooled size is expected to be equal to 2. Got {0}", outputSize.size());
    }

    if (outputSize[0] <= 0 && outputSize[1] <= 0) {
        return errorAt(loc, "Pooled size attributes pooled_h and pooled_w should should be positive.");
    }

    SmallVector<int64_t> outputShape;
    outputShape.push_back(inShapeCoord.raw()[0]);
    outputShape.push_back(inShapeFeatureMap.raw()[1]);
    outputShape.push_back(outputSize[0]);
    outputShape.push_back(outputSize[1]);

    const auto outType = inTypeFeatureMap.changeShape(ShapeRef(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
