//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

void inferPermuteReturnTypeComponents(mlir::Value input, mlir::AffineMap mem_perm, mlir::AffineMap dst_order,
                                      SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes,
                                      bool strictInfer) {
    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromAffineMap(dst_order);
    const auto inType = mlir::cast<mlir::RankedTensorType>(input.getType());

    const auto [outShape, outDesc] = callOnShapeOf(inType, [&](const auto& shape) {
        const auto inMemShape = inOrder.toMemoryOrder(shape);
        const auto outMemShape = applyPerm(inMemShape, mem_perm);
        const auto outAnyShape = outOrder.toLogicalOrder(outMemShape);

        auto outShape = extractShape(outAnyShape);
        const auto outDesc =
                getTensorAttr(inType, dst_order, strictInfer ? vpux::getMemorySpace(inType) : nullptr, outAnyShape);
        return std::make_pair(std::move(outShape), outDesc);
    });

    auto elemType = inType.getElementType();
    if (auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
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
