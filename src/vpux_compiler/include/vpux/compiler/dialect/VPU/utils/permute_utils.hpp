//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
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
// Current limitation: when there are multiple tensor inputs and at least one has
// an input VPU.PermuteCast, this utility returns failure.
// Reason: mapping becomes ambiguous because different inputs can have different
// permute chains, so external dynamic dims may map to different internal axes.
// The utility currently supports only the single-tensor-input case where the
// remapping path is deterministic.
mlir::FailureOr<SmallVector<Dim>> remapDimsThroughInputPermuteCast(mlir::func::FuncOp funcOp, ArrayRef<Dim> inputDims);

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

}  // namespace vpux::VPU
