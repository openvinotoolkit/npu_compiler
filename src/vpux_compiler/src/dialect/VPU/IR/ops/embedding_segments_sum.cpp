//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::EmbeddingSegmentsSumOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::EmbeddingSegmentsSumOpAdaptor embeddingSegmentsSum(operands, attrs, prop);
    if (mlir::failed(embeddingSegmentsSum.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(embeddingSegmentsSum.getEmbTable().getType());

    auto outShape = to_small_vector(inType.getShape().raw());

    outShape[0] = checked_cast<int64_t>(embeddingSegmentsSum.getNumSegmentsValue());

    const auto outType = inType.changeShape(ShapeRef(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
