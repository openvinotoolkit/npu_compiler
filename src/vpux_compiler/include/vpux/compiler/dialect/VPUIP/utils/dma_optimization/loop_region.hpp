//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/// Summary:
/// Declares the LoopRegion model used to cut unrolled tiled/VF compute groups
/// out of the global DMA edge collection.
///
/// A LoopRegion represents one logical compute region that was expanded into
/// several tile lanes. The region kind is derived from the loop marker on its
/// compute ops:
/// - Kind::Tiling uses `tiling_loop_index`; each tile usually holds one layer.
/// - Kind::VF uses `vf_loop_index`; each tile can hold a chain of VF layers.
///
/// The global EdgeCollector stops at RegionPort boundaries. A logical input
/// port describes one outside value feeding one or more per-tile input edges.
/// A logical output port describes per-tile output edges assembled back into
/// one outside value. Destination-buffer ports model shared output buffers that
/// are consumed by tiled output operands.
///
/// Edges inside the region are ordinary Edge objects stored on RegionTile:
/// inputEdges connect input ports to tile compute, outputEdges connect tile
/// compute to output ports, intraTileEdges connect adjacent VF layers inside a
/// tile, and destBufferEdges connect destination-buffer ports to tile compute.
/// RegionPort::connectedEdges records which tile-local edges belong to a
/// logical port.

#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/edge_model.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <cstdint>

namespace vpux {
namespace VPUIP {

/// Address of one per-tile edge connected to a RegionPort.
struct TileEdgeLocation {
    /// Index in LoopRegion::tiles.
    uint32_t tileIndex = 0;

    /// Index in the RegionTile edge vector selected by the port kind.
    uint32_t edgeIndexInTile = 0;
};

/// Outside-visible connection point of a LoopRegion.
/// It records the boundary value and the tile-local edges attached to it.
struct RegionPort {
    mlir::Value externalValue;
    EdgeBoundary externalBoundary;
    SmallVector<TileEdgeLocation> connectedEdges;
};

/// One physical tile lane inside a LoopRegion.
struct RegionTile {
    /// Compute ops in this tile. VF tiles are ordered by layer index.
    SmallVector<mlir::Operation*> ops;

    /// Per-tile input chains from input ports to ops.front().
    SmallVector<Edge, 0> inputEdges;

    /// Per-tile output chains from ops.back() to output ports.
    SmallVector<Edge, 0> outputEdges;

    /// VF-only chains between adjacent compute layers in the same tile.
    SmallVector<Edge, 0> intraTileEdges;

    /// Per-tile destination-buffer chains from dest-buffer ports to compute ops.
    SmallVector<Edge, 0> destBufferEdges;
};

/// Unrolled tiled/VF regions are split at RegionPorts.
/// The global EdgeCollection owns edges outside those ports. LoopRegion owns
/// the per-tile edge segments that run between ports and tile compute ops.
struct LoopRegion {
    enum class Kind {
        Tiling,  ///< marked by `tiling_loop_index`; one compute op per tile
        VF,      ///< marked by `vf_loop_index`; potentially multiple layers per tile
    };

    Kind kind = Kind::Tiling;

    /// `tiling_loop_index` (Kind::Tiling) or `vf_loop_index` (Kind::VF).
    int64_t loopIndex = -1;

    /// Physical tile lanes in execution order.
    SmallVector<RegionTile, 0> tiles;

    /// Outside-visible input ports (one per logical region input).
    SmallVector<RegionPort, 0> inputs;

    /// Outside-visible output ports (one per logical region output).
    SmallVector<RegionPort, 0> outputs;

    /// Shared destination buffers used by tile output operands.
    SmallVector<RegionPort, 0> destBuffers;
};

}  // namespace VPUIP
}  // namespace vpux
