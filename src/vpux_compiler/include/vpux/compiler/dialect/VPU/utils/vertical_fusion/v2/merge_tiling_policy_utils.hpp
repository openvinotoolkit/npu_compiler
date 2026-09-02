//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/merge_tiling_policy.hpp"

namespace vpux::VPU::VF::v2::details {

// Returns producer clustered ops for each operand of the operation, including the pure view ops between producer and
// user input.
SmallVector<OpWithViewInputs> getParentOpWithViewInputs(mlir::Operation* op);

// Follows view-like users of a VF block argument and returns the first non-view compute uses.
SmallVector<mlir::OpOperand*> getComputeUses(mlir::BlockArgument currInputArg);

// Maps an operation from the previous or current VF region to its cloned operation inside the merged VF region.
mlir::Operation* getMappedOpInMergedVF(mlir::Operation* op, VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currOp,
                                       VPU::VerticalFusionOp mergedOp);

// Returns a pending rollback strategy for the clustered op when available, otherwise returns the op's current strategy.
VPU::MultiClusterStrategy getMultiClusterStrategy(
        VPU::ClusteredOpInterface clusteredOp,
        const DenseMap<VPU::ClusteredOpInterface, VPU::MultiClusterStrategy>& rollbackStrategy);

}  // namespace vpux::VPU::VF::v2::details
