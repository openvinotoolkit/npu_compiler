//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/import/embedding_segments_sum.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace vpux::IE::EmbeddingSegmentsSum {
llvm::FailureOr<int64_t> parseNumSegments(mlir::Location loc, mlir::ValueRange opInputs) {
    if (opInputs.size() < 4) {
        return errorAt(loc, "Invalid IE.EmbeddingSegmentsSum operation: expected at least 4 inputs got {0}",
                       opInputs.size());
    }

    auto numSegmentsConst = opInputs[3].getDefiningOp<Const::DeclareOp>();
    if (numSegmentsConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for numSegments");
    }

    if (const auto& attr = numSegmentsConst.getContentAttr(); !attr.isSplat()) {
        return errorAt(loc, "numSegments value must be a scalar");
    }

    const auto numSegmentsContent = numSegmentsConst.getContent();
    return numSegmentsContent.getSplatValue<int64_t>();
}

int64_t parseDefaultIndex(mlir::ValueRange opInputs) {
    if (opInputs.size() < 5) {
        return -1;
    }

    auto defaultIndexConst = opInputs[4].getDefiningOp<Const::DeclareOp>();
    if (defaultIndexConst == nullptr) {
        return -1;
    }
    if (const auto& attr = defaultIndexConst.getContentAttr(); !attr.isSplat()) {
        return -1;
    }
    return defaultIndexConst.getContent().getSplatValue<int64_t>();
}
}  // namespace vpux::IE::EmbeddingSegmentsSum

mlir::LogicalResult vpux::IE::EmbeddingSegmentsSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::EmbeddingSegmentsSumOpAdaptor embeddingSegmentsSum(operands, attrs, prop);
    if (mlir::failed(embeddingSegmentsSum.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(embeddingSegmentsSum.getEmbTable().getType());
    const int64_t numSegmentsVal = embeddingSegmentsSum.getNumSegmentsValue();

    SmallVector<int64_t> outShape(to_small_vector(inType.getShape()));
    outShape[0] = numSegmentsVal;

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}
