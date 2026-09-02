//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux {
class NDTypeInterface;

template <typename T, template <class> class Tag>
details::DimValues<MemDim, T, Tag> applyPerm(const details::DimValues<MemDim, T, Tag>& memShape,
                                             mlir::AffineMap memPerm) {
    const auto perm = DimsOrder::fromAffineMap(memPerm);
    VPUX_THROW_UNLESS(memShape.size() == perm.numDims(), "Permutation '{0}' is not compatible with shape '{1}'",
                      memPerm, memShape);

    details::DimValues<MemDim, T, Tag> outShape;
    outShape.resize(memShape.size(), 1);

    for (auto ind : irange(outShape.size())) {
        const auto outDim = MemDim(ind);
        const auto inDim = MemDim(perm.dimAt(ind).ind());
        outShape[outDim] = memShape[inDim];
    }

    return outShape;
}

SmallVector<int64_t> getPermutateDims(MemShapeRef inShape, mlir::AffineMap memPerm);
bool isTrivialPermute(MemShapeRef inShape, mlir::AffineMap memPerm);

bool areReshapedAxesPermutedIntegratedly(ArrayRef<SmallVector<int64_t>> dimMapping, ArrayRef<int64_t> permAxis,
                                         const DimsOrder& permInOrder, MemShapeRef memShape);
bool isTrivialReorder(const DimsOrder& inOrder, const DimsOrder& outOrder, ShapeRef shape);

mlir::AffineMap getPermutationFromOrders(const DimsOrder& inOrder, const DimsOrder& outOrder, mlir::MLIRContext* ctx);
DimsOrder applyPermutation(const DimsOrder& srcOrder, const DimsOrder& dstOrder);

DimsOrder moveD0ToTheFront(const DimsOrder& inOrder);

std::pair<SmallVector<uint32_t>, SmallVector<int64_t>> getMergedPermutationAndShape(NDTypeInterface input,
                                                                                    mlir::AffineMap permutation,
                                                                                    int64_t rank = 4);
void extendPermutationAndShape(SmallVector<uint32_t>& permutation, SmallVector<int64_t>& shape, int64_t targetRank);

NDTypeInterface inferNewTypeWithMemPerm(NDTypeInterface oldType, mlir::AffineMap memPerm, const DimsOrder& dstOrder);

std::optional<mlir::AffineMap> tryToFindPermutationForPermuteCast(NDTypeInterface inputType, const DimsOrder& outOrder,
                                                                  ShapeRef outShape, mlir::MLIRContext* ctx);

Dim inferDimAfterPermutation(Dim dim, const DimsOrder& srcOrder, const DimsOrder& dstOrder, mlir::AffineMap perm);
}  // namespace vpux
