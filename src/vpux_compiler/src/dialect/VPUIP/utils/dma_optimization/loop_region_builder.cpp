//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/loop_region.hpp"

#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/edge_collector.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>

namespace vpux::VPUIP::DmaOptimization {

// LoopRegionBuilder reads the loop region attributes left after tiling or VF
// regions are unrolled. These attributes include `tiling_loop_index`,
// `vf_loop_index`, `vf_loop_tile_index`, and `loop_tile_coord`.
//
// The builder reconstructs the logical region signature from these attributes:
//
//   %logicalOutputs = LoopRegion inputs(%logicalInputs)
//                                outputs(%destBuffers)
//
// Logical inputs are read-side values entering the region, destination buffers
// are write-side operands of its tile-local compute operations, and logical
// outputs are values that leave the region. The builder records this abstraction
// as domain and port information; it does not recreate a LoopRegion operation, change
// IR, or construct Edge topology.
//
// For example, consider two VF tile lanes that read the same external input and
// assemble their results through one ConcatView:
//
//   input -> conv0(vf_loop=4, tile=0, dest=dest0) -> result0 --+
//   input -> conv1(vf_loop=4, tile=1, dest=dest1) -> result1 --+-> ConcatView -> output
//
// The builder identifies one VF LoopRegion `(kind=VF, loopIndex=4)`, one shared
// logical Input port, one logical Output port, and the exact tile membership of
// both compute operations, results, and consumer operand slots.

// The output of loop region builder is a read-only view of the loop region
// information through three complementary maps in LoopRegionBoundaryInfo:
//
// - opToLoopRegion:  records the internal region/tile membership of an operation.
//   In this example, conv0 maps to `{VF, loop=4, tile=0}` and conv1 maps to
//   `{VF, loop=4, tile=1}`. It also records tagged non-compute operations.
//
// - valueToLoopRegions: records every role of an SSA value. The shared `input`
//   has one region-level `{VF, loop=4, Input port=0}` entry and one tile-specific
//   Input entry for each conv. `result0`/`result1` have both internal tile entries
//   and Output-port entries, while `output` has the region-level Output port entry.
//   Each dest buffer similarly has a region-level and a tile-specific DestBuffer
//   port entry.
//
// - operandToLoopRegion: records the exact role of one consumer operand. This
//   removes the ambiguity in valueToLoopRegions: conv0's input operand maps to
//   tile 0 and conv1's input operand maps to tile 1, even though both consume
//   the same `input` value. Each compute output-buffer operand likewise maps to
//   its tile-specific DestBuffer port.
//

LoopRegionInfo createLoopRegionInfo(const LoopRegion& region) {
    LoopRegionInfo info;
    info.kind = region.kind;
    info.loopIndex = region.loopIndex;
    return info;
}

LoopRegionInfo createLoopRegionInfo(const LoopRegion& region, mlir::Operation* op) {
    auto info = createLoopRegionInfo(region);
    if (op == nullptr) {
        return info;
    }
    // Creates operation-specific LoopRegionInfo for either LoopRegion kind. Tiling uses
    // `loop_tile_coord`; VF additionally uses `vf_loop_tile_index` for its
    // physical lane identity.
    const auto loopAttributes = getLoopAttributes(op);
    info.tileCoordinate = loopAttributes.loopTileCoord;
    if (region.kind == LoopRegionKind::VF && loopAttributes.vfLoopTileIndex != nullptr) {
        info.tileIndex = mlir::cast<mlir::IntegerAttr>(loopAttributes.vfLoopTileIndex).getInt();
    }
    return info;
}

LoopRegionInfo createLoopRegionPortInfo(const LoopRegionInfo& regionInfo, uint32_t portIdx,
                                        LoopRegionPortKind portKind) {
    auto info = regionInfo;
    info.portKind = portKind;
    info.portIdx = portIdx;
    info.hasPort = true;
    return info;
}

LoopRegionInfo getLoopRegionInfo(const LoopRegionInfo& info, bool includeTile) {
    LoopRegionInfo result;
    if (!info.isValid()) {
        return result;
    }
    result.kind = info.kind;
    result.loopIndex = info.loopIndex;
    if (includeTile) {
        result.tileCoordinate = info.tileCoordinate;
        result.tileIndex = info.tileIndex;
    }
    return result;
}

LoopRegionInfo getLoopRegionInfo(const EdgeBoundary& boundary, bool includeTile) {
    return getLoopRegionInfo(boundary.loopRegionInfo, includeTile);
}

StringRef stringifyLoopRegionPortKind(LoopRegionPortKind kind) {
    switch (kind) {
    case LoopRegionPortKind::Input:
        return "Input";
    case LoopRegionPortKind::Output:
        return "Output";
    case LoopRegionPortKind::DestBuffer:
        return "DestBuffer";
    }
    return "Unknown";
}

std::string stringifyLoopRegionInfo(const LoopRegionInfo& info) {
    if (!info.isValid()) {
        return "none";
    }
    if (info.kind == LoopRegionKind::Tiling) {
        return llvm::formatv("{0}(loop={1},coord={2})", toString(info.kind), info.loopIndex, info.tileCoordinate).str();
    }
    return llvm::formatv("{0}(loop={1},tile={2})", toString(info.kind), info.loopIndex, info.tileIndex).str();
}

namespace {

struct LoopRegionCandidate {
    LoopRegion region;
    SmallVector<mlir::Operation*> internalOps;
};

//===----------------------------------------------------------------------===//
// Boundary helpers
//===----------------------------------------------------------------------===//

mlir::Value skipCopyAndViewForPortValue(mlir::Value val, const llvm::DenseSet<mlir::Operation*>& regionOpSet) {
    while (true) {
        auto* defOp = val.getDefiningOp();
        if (defOp == nullptr) {
            return val;
        }
        if (regionOpSet.contains(defOp)) {
            return val;
        }
        if (mlir::isa<VPUIP::CopyOp>(defOp) || isViewLikeOp(defOp)) {
            val = defOp->getOperand(0);
            continue;
        }
        return val;
    }
}

SmallVector<mlir::OpOperand*> discoverLogicalInputs(mlir::Operation* op) {
    if (mlir::isa_and_present<VPUIP::StubOp>(op)) {
        SmallVector<mlir::OpOperand*> inputs;
        inputs.reserve(op->getNumOperands());
        for (auto& operand : op->getOpOperands()) {
            inputs.push_back(&operand);
        }
        return inputs;
    }
    if (mlir::isa_and_present<VPUIP::LayerOpInterface>(op)) {
        SmallVector<mlir::OpOperand*> inputs;
        for (auto& operand : VPUIP::getLayerInOpOperands(op)) {
            inputs.push_back(&operand);
        }
        return inputs;
    }
    if (auto concatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(op)) {
        SmallVector<mlir::OpOperand*> inputs;
        inputs.reserve(concatOp.getInputs().size());
        for (unsigned operandIndex = 0; operandIndex < concatOp.getInputs().size(); ++operandIndex) {
            inputs.push_back(&op->getOpOperand(operandIndex));
        }
        return inputs;
    }
    if (isEdgeOp(op) && op->getNumOperands() > 0) {
        return {&op->getOpOperand(0)};
    }
    return {};
}

void validateLoopTileCoordinate(mlir::Operation* op, TileCoordinateAttr coordinate) {
    VPUX_THROW_UNLESS(op->hasAttr(LOOP_TILE_COORD), "Loop operation at '{0}' is missing required '{1}' attribute",
                      op->getLoc(), LOOP_TILE_COORD);
    VPUX_THROW_UNLESS(coordinate != nullptr,
                      "Loop operation at '{0}' has malformed '{1}' attribute: expected a dictionary", op->getLoc(),
                      LOOP_TILE_COORD);

    const auto axes = coordinate.getAs<mlir::DenseI64ArrayAttr>("axes");
    const auto indices = coordinate.getAs<mlir::DenseI64ArrayAttr>("indices");
    VPUX_THROW_UNLESS(axes != nullptr && indices != nullptr && coordinate.size() == 2,
                      "Loop operation at '{0}' has malformed '{1}' attribute: expected only dense i64 'axes' and "
                      "'indices' arrays",
                      op->getLoc(), LOOP_TILE_COORD);
    VPUX_THROW_UNLESS(axes.size() == indices.size(),
                      "Loop operation at '{0}' has malformed '{1}' attribute: axes and indices sizes differ",
                      op->getLoc(), LOOP_TILE_COORD);

    int64_t previousAxis = -1;
    for (const auto [axis, index] : llvm::zip_equal(axes.asArrayRef(), indices.asArrayRef())) {
        VPUX_THROW_UNLESS(axis >= 0 && index >= 0,
                          "Loop operation at '{0}' has malformed '{1}' attribute: axes and indices must be "
                          "non-negative",
                          op->getLoc(), LOOP_TILE_COORD);
        VPUX_THROW_UNLESS(axis > previousAxis,
                          "Loop operation at '{0}' has malformed '{1}' attribute: axes must be strictly increasing",
                          op->getLoc(), LOOP_TILE_COORD);
        previousAxis = axis;
    }
}

// Groups compute operations that belong to the same tile.
// Tiling region uses loop_tile_coord to group operations by their tile coordinates.
// VF uses vf_loop_tile_index to group operations by their VF tile indices.
//
SmallVector<SmallVector<mlir::Operation*>> partitionComputeOpsByTile(llvm::ArrayRef<mlir::Operation*> opsIn,
                                                                     LoopRegionKind kind) {
    SmallVector<SmallVector<mlir::Operation*>> result;
    if (kind == LoopRegionKind::Tiling) {
        llvm::DenseMap<TileCoordinateAttr, SmallVector<mlir::Operation*>> opsByTile;
        SmallVector<TileCoordinateAttr> tileOrder;
        for (auto* op : opsIn) {
            if (!isComputeOp(op)) {
                continue;
            }
            const auto coordinate = getLoopAttributes(op).loopTileCoord;
            auto [bucket, inserted] = opsByTile.try_emplace(coordinate);
            if (inserted) {
                tileOrder.push_back(coordinate);
            }
            bucket->second.push_back(op);
        }
        result.reserve(tileOrder.size());
        for (const auto coordinate : tileOrder) {
            result.push_back(std::move(opsByTile[coordinate]));
        }
    } else if (kind == LoopRegionKind::VF) {
        llvm::DenseMap<int64_t, SmallVector<mlir::Operation*>> opsByTile;
        SmallVector<int64_t> tileOrder;
        for (auto* op : opsIn) {
            if (!isComputeOp(op)) {
                continue;
            }
            const auto tileAttr = getLoopAttributes(op).vfLoopTileIndex;
            const auto tileIndex = tileAttr == nullptr ? -1 : mlir::cast<mlir::IntegerAttr>(tileAttr).getInt();
            auto& bucket = opsByTile[tileIndex];
            if (bucket.empty()) {
                tileOrder.push_back(tileIndex);
            }
            bucket.push_back(op);
        }
        llvm::sort(tileOrder);
        result.reserve(tileOrder.size());
        for (auto tileIndex : tileOrder) {
            result.push_back(std::move(opsByTile[tileIndex]));
        }
    }
    return result;
}

//===----------------------------------------------------------------------===//
// Candidate discovery
//===----------------------------------------------------------------------===//

void collectRegionCandidates(mlir::Operation* funcOp, SmallVector<LoopRegionCandidate, 0>& candidates) {
    llvm::DenseMap<int64_t, SmallVector<mlir::Operation*>> tilingIndexToOps;
    llvm::DenseMap<int64_t, SmallVector<mlir::Operation*>> vfIndexToOps;
    // Build a map of loop index to operations for both tiling and VF loops.
    funcOp->walk([&](mlir::Operation* op) {
        const auto loopAttributes = getLoopAttributes(op);
        const auto requiresTileCoordinate = isComputeOp(op);
        const auto hasTileCoordinate = op->hasAttr(LOOP_TILE_COORD);
        if (loopAttributes.vfLoopIndex != nullptr) {
            if (requiresTileCoordinate || hasTileCoordinate) {
                validateLoopTileCoordinate(op, loopAttributes.loopTileCoord);
            }
            const auto loopIndex = mlir::cast<mlir::IntegerAttr>(loopAttributes.vfLoopIndex).getInt();
            vfIndexToOps[loopIndex].push_back(op);
        } else if (loopAttributes.tilingLoopIndex != nullptr) {
            if (requiresTileCoordinate || hasTileCoordinate) {
                validateLoopTileCoordinate(op, loopAttributes.loopTileCoord);
            }
            const auto loopIndex = mlir::cast<mlir::IntegerAttr>(loopAttributes.tilingLoopIndex).getInt();
            tilingIndexToOps[loopIndex].push_back(op);
        }
    });
    // Record candidates for each loop index.
    for (auto& [idx, ops] : tilingIndexToOps) {
        LoopRegionCandidate candidate;
        candidate.region = {LoopRegionKind::Tiling, idx};
        candidate.internalOps = std::move(ops);
        candidates.push_back(std::move(candidate));
    }

    for (auto& [idx, ops] : vfIndexToOps) {
        LoopRegionCandidate candidate;
        candidate.region = {LoopRegionKind::VF, idx};
        candidate.internalOps = std::move(ops);
        candidates.push_back(std::move(candidate));
    }

    llvm::sort(candidates, [](const LoopRegionCandidate& a, const LoopRegionCandidate& b) {
        if (a.region.kind != b.region.kind) {
            return static_cast<int>(a.region.kind) < static_cast<int>(b.region.kind);
        }
        return a.region.loopIndex < b.region.loopIndex;
    });
}

/// Usually, logical outputs are either the outputs of ConcatView operations
/// that assemble tiled results or the results of internal operations that
/// escape the loop region.
SmallVector<mlir::Value> discoverLogicalOutputs(llvm::ArrayRef<mlir::Operation*> internalOps) {
    SmallVector<mlir::Value> logicalOutputs;
    llvm::DenseSet<mlir::Value> outputSet;
    SmallVector<mlir::Value> concatOutputs;
    llvm::DenseSet<mlir::Value> concatOutputSet;
    llvm::DenseMap<mlir::Value, SmallVector<mlir::Value>> concatOutputToResults;
    llvm::DenseSet<mlir::Value> resultsAssembledByConcat;
    llvm::DenseSet<mlir::Operation*> internalOpSet(internalOps.begin(), internalOps.end());

    const auto addLogicalOutput = [&](mlir::Value value) {
        if (outputSet.insert(value).second) {
            logicalOutputs.push_back(value);
        }
    };

    const auto isTransparentDataInput = [](mlir::Operation* user, unsigned operandNumber) {
        return operandNumber == 0 && (mlir::isa<VPUIP::CopyOp, VPUIP::SubViewOp>(user) || isViewLikeOp(user));
    };

    const auto feedsConcatInput = [&](mlir::Value rootValue) {
        SmallVector<mlir::Value> valuesToVisit = {rootValue};
        llvm::DenseSet<mlir::Value> visitedValues;
        while (!valuesToVisit.empty()) {
            const auto value = valuesToVisit.pop_back_val();
            if (!visitedValues.insert(value).second) {
                continue;
            }
            for (auto& use : value.getUses()) {
                auto* user = use.getOwner();
                if (user == nullptr || internalOpSet.contains(user)) {
                    continue;
                }
                if (auto concatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(user)) {
                    if (use.getOperandNumber() < concatOp.getInputs().size()) {
                        return true;
                    }
                    continue;
                }
                if (isTransparentDataInput(user, use.getOperandNumber())) {
                    valuesToVisit.append(user->getResults().begin(), user->getResults().end());
                }
            }
        }
        return false;
    };

    const auto reachesOutsideRegion = [&](mlir::Value rootValue) {
        SmallVector<mlir::Value> valuesToVisit = {rootValue};
        llvm::DenseSet<mlir::Value> visitedValues;
        while (!valuesToVisit.empty()) {
            const auto value = valuesToVisit.pop_back_val();
            if (!visitedValues.insert(value).second) {
                continue;
            }
            for (auto& use : value.getUses()) {
                auto* user = use.getOwner();
                if (user == nullptr || internalOpSet.contains(user)) {
                    continue;
                }
                if (auto concatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(user)) {
                    if (use.getOperandNumber() < concatOp.getInputs().size()) {
                        valuesToVisit.push_back(concatOp.getOutput());
                    }
                    continue;
                }
                if (isTransparentDataInput(user, use.getOperandNumber())) {
                    valuesToVisit.append(user->getResults().begin(), user->getResults().end());
                    continue;
                }
                return true;
            }
        }
        return false;
    };

    const auto discoverConcatLogicalOutputs = [&]() {
        for (auto* op : internalOps) {
            for (auto result : op->getResults()) {
                SmallVector<mlir::Value> valuesToVisit = {result};
                llvm::DenseSet<mlir::Value> visitedValues;
                while (!valuesToVisit.empty()) {
                    const auto value = valuesToVisit.pop_back_val();
                    if (!visitedValues.insert(value).second) {
                        continue;
                    }
                    for (auto& use : value.getUses()) {
                        auto* user = use.getOwner();
                        if (auto concatOp = mlir::dyn_cast_or_null<VPUIP::ConcatViewOp>(user)) {
                            if (use.getOperandNumber() < concatOp.getInputs().size()) {
                                const auto concatOutput = concatOp.getOutput();
                                if (concatOutputSet.insert(concatOutput).second) {
                                    concatOutputs.push_back(concatOutput);
                                }
                                concatOutputToResults[concatOutput].push_back(result);
                                valuesToVisit.push_back(concatOutput);
                            }
                            continue;
                        }
                        if (isTransparentDataInput(user, use.getOperandNumber())) {
                            valuesToVisit.append(user->getResults().begin(), user->getResults().end());
                        }
                    }
                }
            }
        }
        for (auto concatOutput : concatOutputs) {
            if (!feedsConcatInput(concatOutput) && reachesOutsideRegion(concatOutput)) {
                addLogicalOutput(concatOutput);
                for (auto result : concatOutputToResults[concatOutput]) {
                    resultsAssembledByConcat.insert(result);
                }
            }
        }
    };

    /// An escaping result is an internal operation result with a consumer
    /// outside loop region. These results are considered logical outputs.
    const auto discoverEscapingResultOutputs = [&]() {
        for (auto* op : internalOps) {
            for (auto result : op->getResults()) {
                if (resultsAssembledByConcat.contains(result)) {
                    continue;
                }
                if (reachesOutsideRegion(result)) {
                    addLogicalOutput(result);
                }
            }
        }
    };

    discoverConcatLogicalOutputs();
    discoverEscapingResultOutputs();
    return logicalOutputs;
}

void registerValueToLoopRegions(LoopRegionBoundaryInfo& boundaryInfo, mlir::Value value,
                                const LoopRegionInfo& regionInfo) {
    if (!value || !regionInfo.isValid()) {
        return;
    }
    auto& valueToLoopRegionsMap = boundaryInfo.valueToLoopRegions[value];
    if (!llvm::is_contained(valueToLoopRegionsMap, regionInfo)) {
        valueToLoopRegionsMap.push_back(regionInfo);
    }
}

void registerOpToLoopRegion(LoopRegionBoundaryInfo& boundaryInfo, mlir::Operation* op,
                            const LoopRegionInfo& regionInfo) {
    if (op != nullptr && regionInfo.isValid()) {
        boundaryInfo.opToLoopRegion[op] = regionInfo;
    }
}

void registerOperandToLoopRegions(LoopRegionBoundaryInfo& boundaryInfo, mlir::OpOperand* operand,
                                  const LoopRegionInfo& regionInfo) {
    if (operand != nullptr && regionInfo.isValid()) {
        boundaryInfo.operandToLoopRegion[operand] = regionInfo;
    }
}

//===----------------------------------------------------------------------===//
// BoundaryInfo population
//===----------------------------------------------------------------------===//

void populateRegionBoundaryInfo(const LoopRegionCandidate& candidate, const LoopRegion& region,
                                LoopRegionBoundaryInfo& boundaryInfo) {
    llvm::DenseSet<mlir::Operation*> regionOps(candidate.internalOps.begin(), candidate.internalOps.end());
    llvm::DenseMap<mlir::Value, uint32_t> inputPorts;
    llvm::DenseMap<mlir::Value, uint32_t> outputPorts;
    llvm::DenseMap<mlir::Value, uint32_t> destBufferPorts;

    auto getPortMap = [&](LoopRegionPortKind portKind) -> llvm::DenseMap<mlir::Value, uint32_t>& {
        switch (portKind) {
        case LoopRegionPortKind::Input:
            return inputPorts;
        case LoopRegionPortKind::Output:
            return outputPorts;
        case LoopRegionPortKind::DestBuffer:
            return destBufferPorts;
        }
        return inputPorts;
    };

    auto getOrCreatePort = [&](mlir::Value value, LoopRegionPortKind portKind) -> uint32_t {
        if (!value) {
            return static_cast<uint32_t>(-1);
        }

        auto& portMap = getPortMap(portKind);
        auto it = portMap.find(value);
        if (it != portMap.end()) {
            return it->second;
        }

        const auto portIdx = static_cast<uint32_t>(portMap.size());
        portMap[value] = portIdx;
        registerValueToLoopRegions(boundaryInfo, value,
                                   createLoopRegionPortInfo(createLoopRegionInfo(region), portIdx, portKind));
        return portIdx;
    };

    auto canonicalizePortBackingValue = [&](mlir::Value operand) {
        auto value = skipCopyAndViewForPortValue(operand, regionOps);
        while (auto* defOp = value.getDefiningOp()) {
            if (!mlir::isa<VPUIP::SubViewOp>(defOp)) {
                break;
            }
            value = skipCopyAndViewForPortValue(defOp->getOperand(0), regionOps);
        }
        return value;
    };

    auto resolveLogicalInputValue = [&](mlir::Value value) {
        value = canonicalizePortBackingValue(value);
        while (auto* defOp = value.getDefiningOp()) {
            if (!regionOps.contains(defOp) || !(mlir::isa<VPUIP::CopyOp>(defOp) || isViewLikeOp(defOp))) {
                break;
            }
            value = canonicalizePortBackingValue(defOp->getOperand(0));
        }
        return value;
    };

    auto isProducedInsideRegion = [&](mlir::Value operand) {
        auto value = resolveLogicalInputValue(operand);
        auto* defOp = value.getDefiningOp();
        return defOp != nullptr && regionOps.contains(defOp);
    };

    auto registerBodyValuePort = [&](mlir::Value bodyValue, LoopRegionPortKind portKind,
                                     const LoopRegionInfo& tileRegionInfo, mlir::Value portValue = {},
                                     mlir::OpOperand* consumerOperand = nullptr) {
        auto externalValue = portValue ? portValue : bodyValue;
        const auto portIdx = getOrCreatePort(externalValue, portKind);
        if (portIdx == static_cast<uint32_t>(-1)) {
            return;
        }
        const auto portInfo = createLoopRegionPortInfo(tileRegionInfo, portIdx, portKind);
        registerValueToLoopRegions(boundaryInfo, bodyValue, portInfo);
        registerOperandToLoopRegions(boundaryInfo, consumerOperand, portInfo);
    };

    // Record tagged non-compute operations before processing compute tiles. For
    // example, a VF-local ShapeCast tagged with (vf_loop_index=4,
    // vf_loop_tile_index=1) and its result both receive the domain
    // VF(loop=4, tile=1). These are internal-domain entries, not port entries.
    for (auto* op : candidate.internalOps) {
        if (isComputeOp(op)) {
            continue;
        }
        const auto regionInfo = createLoopRegionInfo(region, op);
        registerOpToLoopRegion(boundaryInfo, op, regionInfo);
        for (auto result : op->getResults()) {
            registerValueToLoopRegions(boundaryInfo, result, regionInfo);
        }
    }

    // The boundary information follows the logical LoopRegion assembly:
    //   %logicalOutputs = LoopRegion inputs(%logicalInputs)
    //                                outputs(%destBuffers)

    // Register all logical Output ports and their backing DestBuffer ports
    // before visiting compute operations.
    // For example:
    //   tile0_result --+
    //   tile1_result --+-> ConcatView(outputs=out_buf) -> region_output
    //
    // `region_output` is the logical Output port; `out_buf` is the only
    // DestBuffer port. Tile-local temporary output buffers are internal and
    // must not become region boundary ports.
    llvm::DenseMap<mlir::Value, mlir::Value> logicalOutputByResult;
    auto logicalOutputs = discoverLogicalOutputs(candidate.internalOps);
    for (auto logicalOutput : logicalOutputs) {
        getOrCreatePort(logicalOutput, LoopRegionPortKind::Output);

        if (auto concatOp = mlir::dyn_cast_or_null<VPUIP::ConcatViewOp>(logicalOutput.getDefiningOp())) {
            const auto destBuffer = canonicalizePortBackingValue(concatOp.getOutputBuff());
            const auto destPortIdx = getOrCreatePort(destBuffer, LoopRegionPortKind::DestBuffer);
            registerValueToLoopRegions(boundaryInfo, concatOp.getOutputBuff(),
                                       createLoopRegionPortInfo(createLoopRegionInfo(region), destPortIdx,
                                                                LoopRegionPortKind::DestBuffer));
            continue;
        }

        logicalOutputByResult[logicalOutput] = logicalOutput;
    }

    // Partition compute operations and populate their per-tile information.
    // For example:
    //   conv0_0 -> conv1_0 -> tile0_out --+
    //   conv0_1 -> conv1_1 -> tile1_out --+--+-> ConcatView -> region_output
    //
    // partitionComputeOpsByTile returns [[conv0_0, conv1_0], [conv0_1, conv1_1]] because
    // conv0_0 and conv1_0 belong to tile 0.
    // conv0_1 and conv1_1 belong to tile 1.
    const auto tileBuckets = partitionComputeOpsByTile(candidate.internalOps, region.kind);

    for (const auto& bucket : tileBuckets) {
        for (auto* layer : bucket) {
            const auto tileRegionInfo = createLoopRegionInfo(region, layer);

            registerOpToLoopRegion(boundaryInfo, layer, tileRegionInfo);

            // Step 1: outputs. Every result keeps the operation's tile information.
            // If the result is a logical output, or feeds a ConcatView logical
            // output, it also gets the matching Output-port information.
            for (auto result : layer->getResults()) {
                registerValueToLoopRegions(boundaryInfo, result, tileRegionInfo);

                auto logicalOutputIt = logicalOutputByResult.find(result);
                if (logicalOutputIt != logicalOutputByResult.end()) {
                    registerBodyValuePort(result, LoopRegionPortKind::Output, tileRegionInfo, logicalOutputIt->second);
                }
            }

            // Step 2: destination buffer. Direct logical outputs use the
            // output-buffer operand with the same result index. ConcatView
            // outputs keep only the region-level DestBuffer registered above.
            if (auto layerOp = mlir::dyn_cast<VPUIP::LayerOpInterface>(layer)) {
                auto outputOperands = layerOp.getOutOpOperands();
                for (auto result : layer->getResults()) {
                    auto logicalOutputIt = logicalOutputByResult.find(result);
                    if (logicalOutputIt == logicalOutputByResult.end() || logicalOutputIt->second != result) {
                        continue;
                    }
                    const auto resultIndex = result.getResultNumber();
                    if (resultIndex >= outputOperands.size()) {
                        continue;
                    }
                    auto& outputOperand = outputOperands[resultIndex];
                    const auto portValue = canonicalizePortBackingValue(outputOperand.get());
                    getOrCreatePort(portValue, LoopRegionPortKind::DestBuffer);
                    registerBodyValuePort(outputOperand.get(), LoopRegionPortKind::DestBuffer, tileRegionInfo,
                                          portValue, &outputOperand);
                }
            }

            // Step 3: inputs. External values become Input ports. Values
            // produced inside the region remain internal data flow.
            for (auto* operand : discoverLogicalInputs(layer)) {
                if (isProducedInsideRegion(operand->get())) {
                    registerOperandToLoopRegions(boundaryInfo, operand, tileRegionInfo);
                    continue;
                }
                const auto portValue = resolveLogicalInputValue(operand->get());
                registerBodyValuePort(operand->get(), LoopRegionPortKind::Input, tileRegionInfo, portValue, operand);
            }
        }
    }
}

}  // namespace

//
// LoopRegionBuilder
//

LoopRegionBuilder::LoopRegionBuilder(Logger log): _log(log) {
}

LoopRegionBoundaryInfo LoopRegionBuilder::build(mlir::func::FuncOp funcOp) {
    LoopRegionBoundaryInfo info;
    _regions.clear();

    if (funcOp == nullptr) {
        return info;
    }

    SmallVector<LoopRegionCandidate, 0> candidates;
    // Find all loop region candidates within the function operation
    // by analyzing loop region attributes on the operations.
    collectRegionCandidates(funcOp.getOperation(), candidates);
    if (candidates.empty()) {
        _log.trace("LoopRegion builder: no loop regions found!");
        return info;
    }

    _regions.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        _regions.push_back(candidate.region);
    }
    // Populate the boundary information for each loop region based on the collected candidates.
    // The boundary information follows the logical LoopRegion assembly:
    //   %logicalOutputs = LoopRegion inputs(%logicalInputs)
    //                                outputs(%destBuffers)
    // LogicalOutputs/ LogicalInputs/ DestinationBuffers are Region Boundary information.
    // This information is needed to correctly inform the edge collector about data flow between regions.
    for (size_t regionIdx = 0; regionIdx < candidates.size(); ++regionIdx) {
        populateRegionBoundaryInfo(candidates[regionIdx], _regions[regionIdx], info);
    }

    return info;
}

}  // namespace vpux::VPUIP::DmaOptimization
