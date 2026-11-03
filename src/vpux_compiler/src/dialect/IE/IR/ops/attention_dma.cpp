//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
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

    // presentK/presentV keep the shape and element type of pastK/pastV:
    // the kernel writes the new entries back into the past cache layout.
    const auto appendCacheShape = [&](mlir::Value cache) {
        const auto cacheType = mlir::cast<NDTypeInterface>(cache.getType());
        mlir::Attribute cacheEncoding = nullptr;
        if (const auto rankedCacheType = mlir::dyn_cast<mlir::RankedTensorType>(cache.getType())) {
            cacheEncoding = rankedCacheType.getEncoding();
        }
        inferredReturnShapes.emplace_back(cacheType.getShape().raw(), cacheType.getElementType(), cacheEncoding);
    };
    if (attention.getPastK() != nullptr && attention.getPastV() != nullptr) {
        appendCacheShape(attention.getPastK());
        appendCacheShape(attention.getPastV());
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::AttentionDMAOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    if (mlir::failed(reifyAttentionResultShape(builder, getInputQ(), getInputK(), getInputV(), getLoc(),
                                               reifiedReturnShapes))) {
        return mlir::failure();
    }

    if (getPresentK() != nullptr && getPresentV() != nullptr) {
        reifiedReturnShapes.emplace_back(reifyTrivialTensor(builder, getPastK(), getLoc()));
        reifiedReturnShapes.emplace_back(reifyTrivialTensor(builder, getPastV(), getLoc()));
    }

    return mlir::success();
}
