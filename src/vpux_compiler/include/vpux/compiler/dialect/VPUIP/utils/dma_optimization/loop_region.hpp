//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/// Summary:
/// Declares the LoopRegion model used to describe tiled/VF compute operations
/// after their loops have been unrolled.
///
/// A LoopRegion groups the operations produced by unrolling one tiled or VF
/// loop. The region kind is derived from attributes preserved on those ops:
/// - Kind::Tiling uses `tiling_loop_index` and a structured tile coordinate.
/// - Kind::VF uses `vf_loop_index`; each tile can hold a chain of VF layers.
///
/// LoopRegionBuilder reconstructs the logical inputs, destination buffers,
/// outputs, and tile membership that are no longer explicit after unrolling.
/// ```mlir
///   %logicalOutputs = LoopRegion inputs(%logicalInputs)
///                                outputs(%destBuffers)
/// ```
/// EdgeCollector builds the Copy/View topology independently and attaches these
/// LoopRegionInfo metadata to its nodes and boundaries.

#include "vpux/compiler/core/loop_type.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <cstdint>
#include <string>

namespace vpux::VPUIP::DmaOptimization {

using LoopRegionKind = LoopType;

enum class LoopRegionPortKind : uint8_t {
    Input,       ///< logical input port; data flows into the region
    Output,      ///< logical output port; data flows out of the region
    DestBuffer,  ///< destination-buffer port used to hold the results of logical outputs.
};

struct LoopRegion {
    LoopRegionKind kind = LoopRegionKind::None;
    int64_t loopIndex = -1;
};

/// Describes where an operation, value, or operand participates in a
/// LoopRegion. `hasPort` distinguishes a logical region port from an internal
/// region/tile information.
struct LoopRegionInfo : LoopRegion {
    /// Structured tile identity for Tiling regions.
    TileCoordinateAttr tileCoordinate = nullptr;
    /// Scalar tile identity for VF regions.
    int64_t tileIndex = -1;
    LoopRegionPortKind portKind = LoopRegionPortKind::Input;
    uint32_t portIdx = 0;
    bool hasPort = false;

    bool isValid() const {
        return kind != LoopRegionKind::None && loopIndex >= 0;
    }

    bool hasTileIdentity() const {
        if (kind == LoopRegionKind::Tiling) {
            return tileCoordinate != nullptr;
        }
        if (kind == LoopRegionKind::VF) {
            return tileIndex >= 0;
        }
        return false;
    }

    bool hasSameRegionIdentity(const LoopRegionInfo& other) const {
        return kind == other.kind && loopIndex == other.loopIndex;
    }

    bool hasSameTileIdentity(const LoopRegionInfo& other) const {
        if (kind != other.kind) {
            return false;
        }
        if (kind == LoopRegionKind::Tiling) {
            return tileCoordinate == other.tileCoordinate;
        }
        if (kind == LoopRegionKind::VF) {
            return tileIndex == other.tileIndex;
        }
        return true;
    }

    bool operator==(const LoopRegionInfo& other) const {
        return hasSameRegionIdentity(other) && hasSameTileIdentity(other) && hasPort == other.hasPort &&
               (!hasPort || (portKind == other.portKind && portIdx == other.portIdx));
    }
};

LoopRegionInfo createLoopRegionInfo(const LoopRegion& region);
LoopRegionInfo createLoopRegionInfo(const LoopRegion& region, mlir::Operation* op);
LoopRegionInfo createLoopRegionPortInfo(const LoopRegionInfo& regionInfo, uint32_t portIdx,
                                        LoopRegionPortKind portKind);

/// Read-only LoopRegionInfo entries reconstructed for all LoopRegions in one function.
struct LoopRegionBoundaryInfo {
    /// Example: a compute op tagged with `vf_loop_index = 4` and
    /// `vf_loop_tile_index = 1` maps to VF(loop=4, tile=1).
    llvm::DenseMap<mlir::Operation*, LoopRegionInfo> opToLoopRegion;

    /// Records every LoopRegion role of an SSA value. For example,

    /// A shared `%input` can have
    /// `{VF, loop=4, Input port=0}` and
    /// `{VF, loop=5, Input port=0}` when it enters two different regions.
    llvm::DenseMap<mlir::Value, SmallVector<LoopRegionInfo, 1>> valueToLoopRegions;

    /// Records the exact LoopRegion role of each use of a value. For example:
    ///
    /// %tile0 = VPUIP.SW.Kernel inputs(%shared as %input0) ... // VF tile 0
    /// %tile1 = VPUIP.SW.Kernel inputs(%shared as %input1) ... // VF tile 1
    //
    /// Both operands reference `%shared`,
    /// `%input0` maps to `{VF, loop=4, tile=0, Input port=0}`
    /// `%input1` maps to `{VF, loop=4, tile=1, Input port=0}`.
    llvm::DenseMap<mlir::OpOperand*, LoopRegionInfo> operandToLoopRegion;
};

LoopRegionInfo getLoopRegionInfo(const LoopRegionInfo& info, bool includeTile = true);
StringRef stringifyLoopRegionPortKind(LoopRegionPortKind kind);
std::string stringifyLoopRegionInfo(const LoopRegionInfo& info);

/// Reconstructs logical LoopRegion inputs, outputs, destination buffers, and
/// per-tile operation/value/operand info from attributes on unrolled VPUIP
/// operations. The builder does not modify IR or build DMA Edge topology.
class LoopRegionBuilder {
public:
    explicit LoopRegionBuilder(Logger log);

    LoopRegionBoundaryInfo build(mlir::func::FuncOp funcOp);

    const SmallVector<LoopRegion, 0>& getRegions() const {
        return _regions;
    }

private:
    SmallVector<LoopRegion, 0> _regions;
    Logger _log;
};

}  // namespace vpux::VPUIP::DmaOptimization
