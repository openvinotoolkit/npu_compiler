//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <mlir/IR/BuiltinAttributes.h>

void inferPermuteReturnTypeComponents(mlir::Value input, mlir::AffineMap mem_perm, mlir::AffineMap dst_order,
                                      SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes,
                                      bool strictInfer) {
    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromAffineMap(dst_order);
    const auto inType = input.getType().cast<mlir::RankedTensorType>();

    const auto inShape = getShape(input);
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    const auto outMemShape = applyPerm(inMemShape, mem_perm);
    const auto outShape = outOrder.toLogicalOrder(outMemShape);

    const auto outBoundsAttr =
            permuteBounds(input.getContext(), inType.cast<vpux::BoundedTypeInterface>(), inOrder, outOrder, mem_perm);
    const auto outDesc =
            vpux::getTensorAttr(dst_order, strictInfer ? vpux::getMemorySpace(inType) : nullptr, outBoundsAttr);

    auto elemType = inType.getElementType();
    if (auto perAxisType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        // The code below reflects the shape inference logic.
        // inOrder.toMemoryOrder permutes the dimensions according to the input order:
        // 2x4x6x8 * NCHW = 2x4x6x8, 2x4x6x8 * NHWC = 2x6x8x4, 2x4x6x8 * NCWH = 2x4x8x6
        // applyPerm does the same thing, but according to the affine map:
        // 2x4x6x8 * (d0, d1, d2, d3) = 2x4x6x8, 2x4x6x8 * (d0, d2, d3, d1) = 2x6x8x4
        // Finally, outOrder.toLogicalOrder brings memory dimensions back to logical.
        // Let's suppose outOrder is NWCH. In that case the dimensions must be permuted to
        // NWCH.dimPos(N) = 0, NWCH.dimPos(C) = 2, NWCH.dimPos(H) = 3, NWCH.dimPos(W) = 1
        // this gives (d0, d2, d3, d1) = NHWC, therefore 2x4x8x16 * NHWC = 2x8x16x4
        //
        // When it comes to the axis propagation, let's consider the axis = 1 for input shape 2x4x8x16
        // NHCW.toMemoryOrder(2x4x8x16) yields 2x8x4x16
        // Which means that the axis (d1) moved from position 1 to position 2.
        // The position can be obtained via NHCW.dimPos(d1) = 2.
        // Same rules work for applyPerm logic.
        // However, for outOrder.toLogicalOrder it's a little bit different.
        // Since the permutation is inversed, the axis must be obtained via outOrder.dimAt(axis).
        // NWCH.dimAt(0) = N, NWCH.dimAt(1) = W, NWCH.dimAt(2) = C, NWCH.dimAt(3) = H
        // NWCH.toLogicalOrder(2x4x8x16) = 2x8x16x4
        // Axis = 4 moved from d1 to d3, NWCH.dimAt(1) = d3
        const auto origAxis = perAxisType.getQuantizedDimension();
        const auto inMemAxis = inOrder.dimPos(Dim(origAxis));
        const auto outMemAxis = DimsOrder::fromAffineMap(mem_perm).dimPos(Dim(inMemAxis));
        const auto outAxis = outOrder.dimAt(outMemAxis);
        elemType = changeAxis(perAxisType, outAxis.ind());
    }

    inferredReturnShapes.emplace_back(outShape.raw(), elemType, outDesc);
}

mlir::ArrayAttr permuteBounds(mlir::MLIRContext* ctx, vpux::BoundedTypeInterface boundedTensor, DimsOrder srcOrder,
                              DimsOrder dstOrder, mlir::AffineMap memPerm) {
    if (boundedTensor == nullptr || boundedTensor.getBounds() == nullptr) {
        return nullptr;
    }
    const auto boundValues = parseIntArrayAttr<int64_t>(boundedTensor.getBounds());
    const auto srcMemBounds = srcOrder.toMemoryOrder(ShapeRef(boundValues));
    const auto dstMemBounds = applyPerm(srcMemBounds, memPerm);
    const auto dstBounds = dstOrder.toLogicalOrder(dstMemBounds);
    return getIntArrayAttr(ctx, dstBounds.raw());
}
