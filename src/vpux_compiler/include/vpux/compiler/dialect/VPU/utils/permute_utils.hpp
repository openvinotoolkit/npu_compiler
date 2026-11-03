//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::VPU {

Shape inferShapeThroughPermute(ShapeRef origShape, NDTypeInterface srcType, NDTypeInterface dstType,
                               mlir::AffineMap memPerm);

mlir::FailureOr<VPU::DistributionInfo> applyPermutationOnDistributionInfo(
        vpux::NDTypeInterface inType, const VPU::DistributionInfo& inDistribution, mlir::AffineMap memPerm,
        const DimsOrder& srcOrder, const DimsOrder& dstOrder, ShapeRef srcShape, ShapeRef dstShape);

// Helper for outlined-function flow: dynamic dims are tracked outside the function,
// while an input VPU.PermuteCast may appear inside the outlined function body.
// Remaps external dynamic dims to the layout used after that input PermuteCast.
// Supported cases are:
//   1) any number of tensor inputs without input VPU.PermuteCast,
//   2) any number of tensor inputs with input VPU.PermuteCast, provided all of
//      them perform the same permutation, or
//   3) all tensor inputs except one are consumed by compatible PermuteCast ops
//      with the same parameters, and the remaining tensor input is the shared-weights
//      input
// Any mixed pattern outside these cases remains unsupported because the mapping
// from external dynamic dims to internal axes becomes ambiguous.
mlir::FailureOr<SmallVector<Dim>> remapDimsThroughInputPermuteCast(mlir::func::FuncOp funcOp, ArrayRef<Dim> inputDims,
                                                                   mlir::func::CallOp callOp = nullptr);

template <typename T, std::enable_if_t<or_<std::is_same<VPU::DistributedTensorType, T>,
                                           std::is_same<VPUIP::DistributedBufferType, T>>::value,
                                       bool> = true>
mlir::FailureOr<VPU::DistributionInfoAttr> applyPermutationOnDistributionInfoAttr(
        T inDistributedType, mlir::AffineMap memPerm, const DimsOrder& srcOrder, const DimsOrder& dstOrder,
        ShapeRef srcShape, ShapeRef dstShape) {
    const auto inDistribution = VPU::DistributionInfo::getClassFromAttr(inDistributedType.getDistribution());

    auto distributionInfoOrFailure = applyPermutationOnDistributionInfo(inDistributedType, inDistribution, memPerm,
                                                                        srcOrder, dstOrder, srcShape, dstShape);
    if (mlir::failed(distributionInfoOrFailure)) {
        return mlir::failure();
    }

    return VPU::DistributionInfo::getAttrFromClass(inDistributedType.getContext(), distributionInfoOrFailure.value());
}

// Pre-computed plane sizes for sub-byte MemPermute alignment analysis.
struct MemPermuteAlignmentInfo {
    int64_t elemBitWidth;
    int64_t rank;
    DimsOrder outOrder;
    SmallVector<int64_t> permInv;
    SmallVector<int64_t> outInnerProducts;
    SmallVector<int64_t> inInnerProducts;

    // Tile alignment factor for a given memory-order position.
    // Returns 1 when both input and output planes are byte-aligned.
    int64_t getAlignmentForMemPos(int64_t memPos) const;
};

std::optional<MemPermuteAlignmentInfo> getMemPermuteAlignmentInfo(VPU::MemPermuteOp memPermuteOp);
std::optional<MemPermuteAlignmentInfo> getMemPermuteAlignmentInfo(NDTypeInterface inType, NDTypeInterface outType,
                                                                  mlir::AffineMap memPerm);
bool doesMemPermuteNeedAlignment(VPU::MemPermuteOp memPermuteOp);
std::optional<SmallVector<int64_t>> getSubbyteAlignmentForMemPermute(VPU::MemPermuteOp memPermuteOp, ShapeRef divisors);

}  // namespace vpux::VPU
