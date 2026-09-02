//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::AttentionDMAOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::AttentionDMAOpAdaptor attention(operands, attrs, prop);
    if (mlir::failed(attention.verify(loc))) {
        return mlir::failure();
    }

    SmallVector<int64_t> outShape;
    mlir::Attribute outEncoding;
    mlir::Type outElementType;
    inferAttentionOutputShapeComponents(ctx, attention.getInputQ(), attention.getInputK(), attention.getInputV(),
                                        outShape, outEncoding, outElementType);

    inferredReturnShapes.emplace_back(outShape, outElementType, outEncoding);
    return mlir::success();
}

mlir::LogicalResult vpux::IE::AttentionDMAOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    return reifyAttentionResultShape(builder, getInputQ(), getInputV(), getLoc(), reifiedReturnShapes);
}
