//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux {
namespace IE {

void handleConsecutiveOnes(ArrayRef<int64_t> inShape, ArrayRef<int64_t> outShape, std::size_t& startIn,
                           std::size_t& startOut, SmallVector<SmallVector<int64_t>>& reassociationVec);
// Note: When having dims equal to 1 in one of the shapes that do not have a corresponding 1 in the other shape, there
// might be multiple dim associations possible.
// E.g.: 1 x 2 x 2 x 1 x 2 x 3 -> 1 x 4 x 6 has 2 possible mappings:
//      {0} -> {0}, {1, 2, 3} -> {1}, {4, 5} -> {2} (this one is computed by the fcn below)
//      {0} -> {0}, {1, 2} -> {1}, {3, 4, 5} -> {2} (this one is computed by the extended fcn below)
mlir::FailureOr<SmallVector<SmallVector<int64_t>>> getReassociationMap(ArrayRef<int64_t> inShape,
                                                                       ArrayRef<int64_t> outShape);
mlir::FailureOr<SmallVector<SmallVector<int64_t>>> getReassociationMapExtension(ArrayRef<int64_t> inShape,
                                                                                ArrayRef<int64_t> outShape);

bool isNotDimExpansionReshape(ShapeRef origShape, ShapeRef reshapeShape);
bool isNotDimShrinkReshape(ShapeRef origShape, ShapeRef reshapeShape);

IE::ShapeCastOp buildShapeCast(mlir::Location loc, mlir::Value input, ArrayRef<int64_t> targetShape,
                               mlir::PatternRewriter& rewriter);

bool isEligibleToFoldStrideKernel(vpux::NDTypeInterface inputType, vpux::NDTypeInterface outputType, int64_t kernelX,
                                  int64_t strideX, int64_t strideY, int64_t inAlignment, int64_t outAlignment,
                                  const Logger& log);

Shape getNewShapeAfterStrideFolding(ShapeRef origShape, int64_t SX);

mlir::Value createDynamicReshape(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                 BoundedShape outputShape);

bool allowsChannelsReshape(mlir::Operation* origOp);

/*
    @brief Returns the most performant output permutation a DPU op may have.
    @param initialDimOrder - output order of the DPU op
    @param nonBatchOneDims - vector of all dims that are equal to 1, but are not batch
    @param is32Bit - data type of the output is on 32 bits

    DPU ops can use the ODU's ability to permute data to output a different layout than NHWC, hence reducing the need of
    extra permute ops. However, not all possible ODU permutes have the same throughput. Best performance is obtained for
    C as innermost dim, then for H and finally W. For 32-bit output C-innermost and H-innermost have the same
    throughput, with W-innermost still being the worst one.

    This util tries to find an equivalent permutation for the output, such that the data is still the same in memory
   (i.e. no actual permutation op need to be inserted), but one with better element throughput.
*/
vpux::DimsOrder returnBestDimOrder(const vpux::DimsOrder& initialDimOrder, SmallVector<Dim>& nonBatchOneDims,
                                   bool is32Bit);
}  // namespace IE
}  // namespace vpux
