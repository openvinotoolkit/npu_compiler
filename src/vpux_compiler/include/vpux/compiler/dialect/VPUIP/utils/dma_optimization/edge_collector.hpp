//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/// Summary:
/// Declares the read-only DMA edge collector. It scans a function region
/// and returns the EdgeCollection consumed by optimization planning.
///
/// DMA edge ops are grouped with a union-find over data-input connectivity and
/// shared external values. Each connected component is materialized as one
/// N-to-M Edge: edge ops become EdgeNode entries, external producers become
/// source EdgeBoundary entries, external consumers become target EdgeBoundary
/// entries, and node adjacency uses negative sentinels to refer to boundary
/// slots. The resulting EdgeType is inferred from the number of sources and
/// targets: Normal, FanIn, FanOut, Many2Many.
///

#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/edge_model.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPUIP {

/// Region cut points supplied by grouping layers.
/// Internal ops are skipped by the global collector. Values in `valueToPort`
/// are represented as LoopRegion boundaries instead of ordinary IR values.
struct LoopRegionBoundaryInfo {
    llvm::DenseSet<mlir::Value> inputCutPoints;
    llvm::DenseSet<mlir::Value> outputCutPoints;
    llvm::DenseSet<mlir::Operation*> internalOps;
    llvm::DenseMap<mlir::Value, size_t> cutPointToGroupIndex;

    llvm::DenseMap<mlir::Value, LoopRegionPortLocation> valueToPort;

    bool hasGroups() const {
        return !inputCutPoints.empty() || !outputCutPoints.empty();
    }
};

/// True for ops treated as compute boundaries by the DMA edge model.
bool isComputeOp(mlir::Operation* op);

/// True for ops carrying DMA edge data flow (Transparent or Junction roles).
bool isEdgeOp(mlir::Operation* op);

/// True for view-like ops (PermuteCast / ShapeCast / GenericReshape /
/// QuantizeCast / DistributedCast). SubView and ConcatView are excluded
/// because they serve as structural boundaries.
bool isViewLikeOp(mlir::Operation* op);

/// Build a `NodeDetails` snapshot for an edge op. Null input returns the default snapshot.
NodeDetails buildNodeDetails(mlir::Operation* op);

/// Collects typed DMA edges with shared DAG nodes.
/// Edge ops belong to exactly one collected edge. Legal boundaries may be
/// shared by multiple edges, but they are never stored as Edge nodes.
class EdgeCollector {
public:
    explicit EdgeCollector(Logger log);

    /// Collect edges without planning or mutating IR.
    EdgeCollection collect(mlir::func::FuncOp funcOp,
                           const LoopRegionBoundaryInfo& loopRegionBoundaries = LoopRegionBoundaryInfo());

private:
    // Connected-component tracking over collected edge ops.
    int32_t findComponentRoot(int32_t edgeIndex);
    void mergeEdgeComponents(int32_t lhsEdgeIndex, int32_t rhsEdgeIndex);
    int32_t getOrCreateEdgeIndex(mlir::Operation* edgeOp);

    /// True for ops owned by LoopRegion-internal collection.
    bool isInternal(mlir::Operation* op) const;

    Logger _log;
    const LoopRegionBoundaryInfo* _loopRegionBoundaries = nullptr;

    /// Union-find index assigned to each collected edge op.
    llvm::DenseMap<mlir::Operation*, int32_t> _edgeIndexByOp;
    /// Component parent index for each collected edge op index.
    SmallVector<int32_t> _componentParentByEdgeIndex;

    // Diagnostics
    size_t _diagEdgeOps = 0;
    size_t _diagBoundaryOps = 0;
    size_t _diagSkippedInternal = 0;
    size_t _diagDroppedIncompleteEdges = 0;
    size_t _diagDroppedNoSources = 0;
    size_t _diagDroppedNoTargets = 0;
};

}  // namespace VPUIP
}  // namespace vpux
