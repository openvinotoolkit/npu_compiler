//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/// Summary:
/// Defines the compact data model shared by DMA edge collection and
/// optimization: edge boundaries, per-op node snapshots, typed edges, and
/// edge collections.
///
/// The collector represents every collected data-flow region with the same
/// N-to-M `Edge` structure:
/// - `sources` are legal producer boundaries: compute ops, constants, allocs,
///   function arguments, or LoopRegion ports.
/// - `targets` are legal consumer boundaries: compute ops, function returns,
///   or LoopRegion ports.
/// - `nodes` are the Copy/SubView/ConcatView/view-like operations between the
///   boundaries. They form a directed acyclic graph, not a linear queue.
///
/// `EdgeType` describes the boundary topology of that shared structure:
/// - `Normal`: one source and one target. It usually models a single DMA/view
///   chain such as producer -> Copy/SubView/... -> consumer.
/// - `FanOut`: one source and multiple targets. The same produced value feeds
///   several consumer branches through shared or sibling edge nodes.
/// - `FanIn`: multiple sources and one target. Several producer branches
///   converge, typically through a ConcatView-like junction.
/// - `Many2Many`: multiple sources and multiple targets. The edge may contain
///   one or more junction nodes that both gather and redistribute data.
/// - `LoopRegion`: an edge whose boundary crosses or lives inside a LoopRegion.
///
/// EdgeNode adjacency uses one integer encoding for both in-edge nodes and
/// boundary links: non-negative values index `Edge::nodes`; negative values are
/// boundary sentinels, where `~idx` selects a source or target slot. For fast
/// stream extraction, `sourceEntryNodes` maps each source to its first touched
/// node, and `targetExitNodes` maps each target to its last touched node. A
/// value of -1 means the boundary connects directly without an in-edge node.

#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

#include <cstddef>

namespace vpux {
namespace VPUIP {

/// Memory space codes captured in NodeDetails.
constexpr int64_t kMemSpaceDDR = 0;
constexpr int64_t kMemSpaceCMX = 1;

struct LoopRegion;

/// Region-port role used when an edge boundary represents a LoopRegion port.
enum class LoopRegionPortKind : uint8_t {
    Input,       ///< logical input port; data flows into the region
    Output,      ///< logical output port; data flows out of the region
    DestBuffer,  ///< shared destination-buffer port used by tiled outputs
};

/// Address of a LoopRegion port boundary.
struct LoopRegionPortLocation {
    const VPUIP::LoopRegion* region = nullptr;
    uint32_t portIdx = 0;
    LoopRegionPortKind portKind{};
};

/// Typed boundary of a collected DMA edge.
///
/// Boundaries are the only legal sources and targets of an Edge. Copy,
/// SubView, ConcatView, and transparent view-like ops must appear as EdgeNode
/// entries inside an edge, not as boundaries.
struct EdgeBoundary {
    enum class Kind {
        None,         ///< unset; must not appear in finalised edges
        ComputeOp,    ///< op is an NCE / SwKernel / qualifying ConvertDMA
        FuncArg,      ///< arg is a function block argument
        Constant,     ///< op is a constant-like declaration (e.g. const.Declare, StorageElementTable)
        AllocBuffer,  ///< op is a buffer allocation (memref.alloc / VPURT.AllocDistributed)
        FuncReturn,   ///< edge feeds func.return
        LoopRegion,   ///< region/portIdx address a RegionPort of a LoopRegion
    };

    Kind kind = Kind::None;

    // Mutually exclusive payloads; only the field matching `kind` is meaningful.
    mlir::Operation* op = nullptr;    ///< ComputeOp / Constant / AllocBuffer / FuncReturn
    mlir::BlockArgument arg = {};     ///< FuncArg
    LoopRegionPortLocation loopPort;  ///< LoopRegion port location
    /// Result index for multi-result compute sources.
    uint32_t resultIdx = 0;
    /// Operand index for multi-operand compute and func.return targets.
    uint32_t operandIdx = 0;

    static EdgeBoundary none();
    static EdgeBoundary compute(mlir::Operation* op);
    static EdgeBoundary computeResult(mlir::Operation* op, uint32_t resultIdx);
    static EdgeBoundary computeOperand(mlir::Operation* op, uint32_t operandIdx);
    static EdgeBoundary funcArg(mlir::BlockArgument arg);
    static EdgeBoundary constant(mlir::Operation* op);
    static EdgeBoundary allocBuffer(mlir::Operation* op);
    static EdgeBoundary funcReturn(mlir::Operation* op, uint32_t operandIdx);
    static EdgeBoundary loopRegion(LoopRegionPortLocation location);
    static EdgeBoundary loopRegion(const VPUIP::LoopRegion* region, uint32_t portIdx, LoopRegionPortKind portKind);
};

/// Snapshot of one operation inside an Edge.
/// Optimizers use this cached type, memory-space, and rewrite metadata instead
/// of repeatedly querying the IR.
struct NodeDetails {
    enum class Kind { Copy, Slice, Concat, ViewOp, Other };

    Kind kind = Kind::Other;
    mlir::Operation* originalOp = nullptr;

    /// Input and output types captured at collection time.
    mlir::Type inputType;
    mlir::Type outputType;

    /// Set to true when this node has been optimized away.
    bool removed = false;

    /// Input/output memory spaces captured from the node types; -1 means unavailable.
    int64_t srcMemSpace = -1;
    int64_t dstMemSpace = -1;

    /// True when this node should be materialized as a DistributedCast instead
    /// of the original CopyOp.
    bool syntheticDistCast = false;

    /// Destination buffer for VPUIP Copy nodes.
    mlir::Value dstBuff;

    /// Rewrite hint for same-CMX strided copy elimination.
    bool eliminateByStridedRewrite = false;

    bool isCopy() const;
    bool isViewOp() const;
    bool isSlice() const;
};

/// Topology shape of a collected DMA edge.
enum class EdgeType : uint8_t {
    Normal,      ///< 1 source, 1 target, no junction
    FanOut,      ///< 1 source, N targets, no junction
    FanIn,       ///< N sources, 1 target, usually through a ConcatView node
    Many2Many,   ///< N sources, M targets, usually through junction nodes
    LoopRegion,  ///< region-internal or region-boundary edge
};

/// Infer the non-LoopRegion edge topology from its external boundary counts.
EdgeType inferEdgeTypeFromBoundaryCounts(size_t sourceCount, size_t targetCount);

/// One operation on the collected edge DAG.
/// `details` carries the per-op snapshot used by optimizers and rewriters.
struct EdgeNode {
    mlir::Operation* op = nullptr;
    /// Predecessors on the DAG.
    /// idx >= 0 => in-edge node (`Edge::nodes[idx]`).
    /// idx < 0 => source boundary; source index = `~idx` (= `-(idx + 1)`).
    SmallVector<int32_t, 2> preds;
    /// Successors on the DAG.
    /// idx >= 0 => in-edge node.
    /// idx < 0 => target boundary; target index = `~idx`.
    SmallVector<int32_t, 2> succs;
    NodeDetails details;
};

/// Collected data-flow subgraph between legal edge boundaries.
struct Edge {
    EdgeType type = EdgeType::Normal;
    SmallVector<EdgeBoundary, 1> sources;
    SmallVector<EdgeBoundary, 1> targets;

    /// DAG representation between sources and targets.
    /// Adjacency encoding inside EdgeNode::preds and EdgeNode::succs:
    ///   idx >= 0 => in-edge node index (i.e. nodes[idx])
    ///   idx < 0 => boundary; the sentinel `~idx` indexes sources or targets.
    /// Always test `idx >= 0` before dereferencing nodes[idx].
    SmallVector<EdgeNode, 0> nodes;

    /// One node-index per source. Same negative-sentinel rule applies:
    /// a value of -1 means the source feeds straight into a target (no node).
    SmallVector<int32_t, 1> sourceEntryNodes;

    /// One node-index per target. -1 means the target is fed directly by a source.
    SmallVector<int32_t, 1> targetExitNodes;

    bool isNormal() const;
    bool isFanIn() const;
    bool isFanOut() const;
    bool isMany2Many() const;
    bool isLoopRegion() const;

    /// True when at least one node represents a Concat-like junction.
    bool hasConcatNode() const;

    /// Checks boundary kinds, DAG sentinel ranges, entry/exit indices, and type consistency.
    bool validate(Logger log) const;
};

/// Typed N-to-M edges produced by `EdgeCollector`.
/// One edge per connected DMA/view subgraph.
struct EdgeCollection {
    SmallVector<Edge, 0> edges;
};

}  // namespace VPUIP
}  // namespace vpux
