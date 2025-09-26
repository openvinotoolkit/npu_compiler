//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

//
// inferReturnTypes
//

mlir::LogicalResult VPU::DetectionOutputCollectResultsOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputCollectResultsOpAdaptor collectResults(operands, attrs, prop);
    if (mlir::failed(collectResults.verify(loc))) {
        return mlir::failure();
    }

    const auto keepTopK = collectResults.getKeepTopK();

    const auto confidenceType = mlir::cast<vpux::NDTypeInterface>(collectResults.getConfidence().getType());
    const auto detectionSize = 7;
    const auto outputShape = SmallVector<int64_t>{1, 1, keepTopK, detectionSize};
    const auto outputType = confidenceType.changeShape(ShapeRef(outputShape));

    inferredReturnTypes.push_back(outputType);

    return mlir::success();
}
