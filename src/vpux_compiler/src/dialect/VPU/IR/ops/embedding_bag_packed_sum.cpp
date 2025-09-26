//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::EmbeddingBagPackedSumOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    VPU::EmbeddingBagPackedSumOpAdaptor embeddingBag(operands, attrs, prop);
    if (mlir::failed(embeddingBag.verify(loc))) {
        return mlir::failure();
    }

    const auto embTableType = mlir::cast<vpux::NDTypeInterface>(embeddingBag.getEmbTable().getType());
    const auto embTableShape = embTableType.getShape().raw();
    const auto indicesType = mlir::cast<vpux::NDTypeInterface>(embeddingBag.getIndices().getType());
    const auto indicesShape = indicesType.getShape().raw();
    int64_t batchSize = indicesShape[0];

    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < embTableShape.size(); i++) {
        outShape.emplace_back(embTableShape[i]);
    }
    outShape[0] = batchSize;
    const auto outType = embTableType.changeShape(ShapeRef(outShape));

    inferredReturnTypes.push_back(outType);
    return mlir::success();
}
