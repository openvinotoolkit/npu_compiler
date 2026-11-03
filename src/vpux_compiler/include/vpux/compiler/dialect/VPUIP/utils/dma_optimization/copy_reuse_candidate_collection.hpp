//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/edge_model.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <optional>

namespace vpux {
enum class DmaDirectionType;
}  // namespace vpux

namespace vpux::VPUIP::DmaOptimization {

/// One copy-result reuse opportunity discovered from a collected DMA edge.
/// Candidate collection only records data-flow facts; destination legality are checked later by
/// CopyReuseApprovalWindow.
struct CopyReuseCandidate final {
    /// CopyOp whose result may be reused or removed.
    mutable VPUIP::CopyOp copy;
    /// Compute operands reached from the copy result through a supported view chain.
    SmallVector<mlir::OpOperand*, 1> consumerOperands;
    /// Snapshot captured by EdgeCollector for type, memory-space, and input-view equivalence checks.
    const NodeDetails* nodeDetails = nullptr;
};

/// Returns the supported DMA direction for copy-result reuse candidates.
std::optional<DmaDirectionType> getCopyReuseDirection(VPUIP::CopyOp copy);

/// Extracts copy-result reuse candidates from FanOut-like DMA edges.
SmallVector<CopyReuseCandidate, 16> collectCopyReuseCandidates(const EdgeCollection& edges);

}  // namespace vpux::VPUIP::DmaOptimization
