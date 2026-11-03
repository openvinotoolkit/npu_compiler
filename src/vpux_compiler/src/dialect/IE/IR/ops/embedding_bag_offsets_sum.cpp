//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/import/embedding_bag_offsets_sum.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace vpux::IE::EmbeddingBagOffsetsSum {
int64_t parseDefaultIndex(mlir::ValueRange opInputs) {
    if (opInputs.size() < 4) {
        return -1;
    }

    auto defaultIndexConst = opInputs[3].getDefiningOp<Const::DeclareOp>();
    if (defaultIndexConst == nullptr) {
        return -1;
    }
    if (const auto& attr = defaultIndexConst.getContentAttr(); !attr.isSplat()) {
        return -1;
    }
    return defaultIndexConst.getContent().getSplatValue<int64_t>();
}
}  // namespace vpux::IE::EmbeddingBagOffsetsSum

mlir::LogicalResult vpux::IE::EmbeddingBagOffsetsSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    IE::EmbeddingBagOffsetsSumOpAdaptor embeddingBag(operands, attrs, prop);
    if (mlir::failed(embeddingBag.verify(loc))) {
        return mlir::failure();
    }

    const auto inTypeEmbTable = mlir::cast<mlir::ShapedType>(embeddingBag.getEmbTable().getType());

    SmallVector<int64_t> outShape(to_small_vector(inTypeEmbTable.getShape()));

    if (embeddingBag.getOffsets() != nullptr) {
        const auto inTypeOffsets = mlir::cast<mlir::ShapedType>(embeddingBag.getOffsets().getType());
        const auto offsetsShape = inTypeOffsets.getShape();
        outShape[0] = checked_cast<int64_t>(offsetsShape[0]);
    } else if (embeddingBag.getOffsetsValue().has_value()) {
        const auto offsetsAttr = parseIntArrayAttr<int32_t>(embeddingBag.getOffsetsValue().value());
        outShape[0] = offsetsAttr.size();
    } else {
        return errorAt(loc, "Offsets input was not provided properly");
    }

    inferredReturnShapes.emplace_back(outShape, inTypeEmbTable.getElementType());
    return mlir::success();
}
