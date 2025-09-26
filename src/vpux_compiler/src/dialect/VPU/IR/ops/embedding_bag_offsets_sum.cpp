//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::EmbeddingBagOffsetsSumOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::EmbeddingBagOffsetsSumOpAdaptor embeddingBag(operands, attrs, prop);
    if (mlir::failed(embeddingBag.verify(loc))) {
        return mlir::failure();
    }

    const auto inTypeEmbTable = mlir::cast<vpux::NDTypeInterface>(embeddingBag.getEmbTable().getType());
    SmallVector<int64_t> outShape(to_small_vector(inTypeEmbTable.getShape().raw()));

    if (embeddingBag.getOffsets() != nullptr) {
        const auto inTypeOffsets = mlir::cast<vpux::NDTypeInterface>(embeddingBag.getOffsets().getType());
        SmallVector<int64_t> offsetsOutShape(to_small_vector(inTypeOffsets.getShape().raw()));
        outShape[0] = offsetsOutShape[0];
    } else if (embeddingBag.getOffsetsValue().has_value()) {
        const auto offsetsAttr = parseIntArrayAttr<int32_t>(embeddingBag.getOffsetsValue().value());
        outShape[0] = offsetsAttr.size();
    } else {
        return errorAt(loc, "Offsets input was not provided properly");
    }

    const auto outType = inTypeEmbTable.changeShape(ShapeRef(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
