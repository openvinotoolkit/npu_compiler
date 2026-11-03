//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

namespace {

// Bounds encoding for a dynamic result; nullptr when the result is static.
mlir::Attribute getBoundsEncoding(mlir::MLIRContext* ctx, ArrayRef<int64_t> shape, ArrayRef<int64_t> bounds) {
    if (ShapeRef(shape).isStatic()) {
        return nullptr;
    }
    return vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(shape.size()), /*memSpace=*/nullptr, BoundsRef(bounds));
}

int64_t mulDims(int64_t lhs, int64_t rhs) {
    return (lhs == mlir::ShapedType::kDynamic || rhs == mlir::ShapedType::kDynamic) ? mlir::ShapedType::kDynamic
                                                                                    : lhs * rhs;
}

int64_t addDims(int64_t lhs, int64_t rhs) {
    return (lhs == mlir::ShapedType::kDynamic || rhs == mlir::ShapedType::kDynamic) ? mlir::ShapedType::kDynamic
                                                                                    : lhs + rhs;
}

}  // namespace

mlir::LogicalResult vpux::IE::GroupQueryAttentionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GroupQueryAttentionOpAdaptor gqa(operands, attrs, prop);
    if (mlir::failed(gqa.verify(loc))) {
        return mlir::failure();
    }

    const auto queryType = mlir::cast<NDTypeInterface>(gqa.getQuery().getType());
    const auto pastKeyType = mlir::cast<NDTypeInterface>(gqa.getPastKey().getType());
    const auto queryShape = queryType.getShape().raw();
    const auto pastKeyShape = pastKeyType.getShape().raw();
    if (queryShape.size() != 4 || pastKeyShape.size() != 4) {
        return errorAt(loc, "GroupQueryAttention expects 4D query and past_key, got ranks '{0}' and '{1}'",
                       queryShape.size(), pastKeyShape.size());
    }

    const auto queryBounds = getBoundedShape(gqa.getQuery()).raw();
    const auto pastKeyBounds = getBoundedShape(gqa.getPastKey()).raw();
    const auto numHeads = gqa.getNumHeads();
    const auto kvNumHeads = gqa.getKvNumHeads();

    // output: [batch, seqLen, numHeads * headSize]
    const SmallVector<int64_t> outShape = {queryShape[0], queryShape[2], mulDims(numHeads, queryShape[3])};
    const SmallVector<int64_t> outBounds = {queryBounds[0], queryBounds[2], mulDims(numHeads, queryBounds[3])};
    inferredReturnShapes.emplace_back(outShape, queryType.getElementType(),
                                      getBoundsEncoding(ctx, outShape, outBounds));

    // present_key/present_value: [batch, kvNumHeads, cacheSeqLen, cacheHeadSize]; a non-windowed dynamic cache
    // grows by seqLen, a windowed one keeps its capacity.
    SmallVector<int64_t> kvShape = {queryShape[0], kvNumHeads, pastKeyShape[2], pastKeyShape[3]};
    SmallVector<int64_t> kvBounds = {queryBounds[0], kvNumHeads, pastKeyBounds[2], pastKeyBounds[3]};
    if (!gqa.getSlidingWindowCache() &&
        (pastKeyShape[2] == mlir::ShapedType::kDynamic || queryShape[2] == mlir::ShapedType::kDynamic)) {
        kvShape[2] = mlir::ShapedType::kDynamic;
        kvBounds[2] = addDims(pastKeyBounds[2], queryBounds[2]);
    }

    mlir::Attribute kvEncoding = getBoundsEncoding(ctx, kvShape, kvBounds);
    if (ShapeRef(kvShape).isStatic()) {
        if (const auto rankedPastKeyType = mlir::dyn_cast<mlir::RankedTensorType>(gqa.getPastKey().getType())) {
            kvEncoding = rankedPastKeyType.getEncoding();
        }
    }

    const auto kvElemType = pastKeyType.getElementType();
    inferredReturnShapes.emplace_back(kvShape, kvElemType, kvEncoding);
    inferredReturnShapes.emplace_back(kvShape, kvElemType, kvEncoding);

    return mlir::success();
}
