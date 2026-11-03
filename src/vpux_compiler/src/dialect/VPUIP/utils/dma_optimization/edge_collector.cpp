//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Summary:
// - Edge collector reads one VPUIP function and builds small data-flow graphs for DMA
//   copy optimization. Edge collector does not change IR.
//
// - Boundaries: compute ops, function args, constants, allocs, and function
//   returns. These are the sources and targets of an Edge.
//
// - Nodes: Copy, SubView, ConcatView, and simple view-like ops. These are the
//   editable operations between boundaries.
//
// - Grouping algorithm: connected nodes are joined with union-find.
//   Nodes are connected when data flows from one to another, or
//   when sibling nodes read the same external source.
//   Each connected group becomes one Edge.
//
// - Edge type: 1 source + 1 target is Normal; 1 source + many targets is
//   FanOut; many sources + 1 target is FanIn; many sources + many targets is
//   Many2Many.
//
// - LoopRegion: loop regions do not split collection. LoopRegionBuilder gives
//   loop/tile/port/destination-buffer information, and EdgeCollector attaches
//   it to normal boundaries and nodes as metadata.

#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/edge_collector.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/scope_exit.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/OpDefinition.h>

#include <cstdint>
#include <utility>

namespace {

//===----------------------------------------------------------------------===//
// Boundary identity helpers
//===----------------------------------------------------------------------===//

// BoundaryKey is a temporary identity used while materializing one Edge.
//
// Edge::sources and Edge::targets store each external boundary once, while
// EdgeNode::preds/succs reference those boundary slots with negative indices.
// The maps below make repeated references reuse the same boundary slot instead
// of pushing duplicate EdgeBoundary objects.
//
// Example FanOut:
//   compute0 -> Copy -> {compute1, compute2}
// `compute0` is registered once in edge.sources and the Copy node points to
// that source slot. `compute1` and `compute2` are separate target slots because
// they are different consumer operands.
//
// Source keys identify the producer value: block argument number or
// defining-op/result-number. Target keys identify the consumer operand:
// user-op/operand-number.
struct BoundaryKey {
    mlir::Operation* op = nullptr;
    int32_t index = 0;

    bool operator==(const BoundaryKey& other) const {
        return op == other.op && index == other.index;
    }
};

// llvm::DenseMap requires custom key types to provide reserved empty and
// tombstone keys, plus hashing and equality.
struct BoundaryKeyInfo {
    static BoundaryKey getEmptyKey() {
        return {reinterpret_cast<mlir::Operation*>(uintptr_t(-1)), -2};
    }
    static BoundaryKey getTombstoneKey() {
        return {reinterpret_cast<mlir::Operation*>(uintptr_t(-2)), -3};
    }
    static unsigned getHashValue(const BoundaryKey& key) {
        return llvm::hash_combine(key.op, key.index);
    }
    static bool isEqual(const BoundaryKey& lhs, const BoundaryKey& rhs) {
        return lhs == rhs;
    }
};

struct ResolvedBoundary {
    BoundaryKey key;
    vpux::VPUIP::DmaOptimization::EdgeBoundary boundary;
};

using ResolvedSourceBoundary = ResolvedBoundary;
using ResolvedTargetBoundary = ResolvedBoundary;

BoundaryKey valueBoundaryKey(mlir::Value value) {
    if (!value) {
        return {};
    }
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
        return {nullptr, static_cast<int32_t>(blockArg.getArgNumber())};
    }

    auto opResult = mlir::dyn_cast<mlir::OpResult>(value);
    return {value.getDefiningOp(), opResult ? static_cast<int32_t>(opResult.getResultNumber()) : 0};
}

}  // namespace

namespace vpux::VPUIP::DmaOptimization {

namespace {

//===----------------------------------------------------------------------===//
// Memory-space helpers
//===----------------------------------------------------------------------===//

std::optional<VPU::MemoryKind> getMemoryKind(mlir::Type type) {
    auto ndType = mlir::dyn_cast_or_null<vpux::NDTypeInterface>(type);
    if (ndType == nullptr) {
        return std::nullopt;
    }
    return ndType.getMemoryKind();
}

//===----------------------------------------------------------------------===//
// Op classification and data-flow operands
//===----------------------------------------------------------------------===//

struct OpTrait {
    enum class Role : uint8_t { Unknown, Boundary, Transparent, Junction };

    Role role = Role::Unknown;
    EdgeBoundary::Kind boundaryKind = EdgeBoundary::Kind::None;
};

OpTrait classifyOp(mlir::Operation* op) {
    if (op == nullptr) {
        return {};
    }

    if (mlir::isa<VPUIP::NCEClusterTaskOp, VPUIP::SwKernelOp, VPUIP::StubOp, VPUIP::GatherDMAOp, VPUIP::ConvertDMAOp,
                  VPUIP::ExpandDMAOp, VPUIP::PerAxisTileDMAOp, VPUIP::UpsamplingDMAOp, VPUIP::SpaceToDepthDMAOp,
                  VPUIP::DepthToSpaceDMAOp, VPUIP::PermuteDMAOp>(op)) {
        return {OpTrait::Role::Boundary, EdgeBoundary::Kind::ComputeOp};
    }

    if (op->hasTrait<mlir::OpTrait::ConstantLike>() || mlir::isa<VPUIP::StorageElementTableOp>(op)) {
        return {OpTrait::Role::Boundary, EdgeBoundary::Kind::Constant};
    }
    if (vpux::isBufAllocOp(op)) {
        return {OpTrait::Role::Boundary, EdgeBoundary::Kind::AllocBuffer};
    }

    if (mlir::isa<VPUIP::ConcatViewOp>(op)) {
        return {OpTrait::Role::Junction, EdgeBoundary::Kind::None};
    }

    if (mlir::isa<VPUIP::CopyOp, VPUIP::SubViewOp, VPUIP::PermuteCastOp, VPUIP::ShapeCastOp, VPUIP::GenericReshapeOp,
                  VPUIP::QuantizeCastOp, VPUIP::DistributedCastOp>(op)) {
        return {OpTrait::Role::Transparent, EdgeBoundary::Kind::None};
    }

    return {};
}

bool isLayerDataInOperand(mlir::Operation* op, unsigned operandIndex) {
    for (auto& operand : VPUIP::getLayerInOpOperands(op)) {
        if (operand.getOperandNumber() == operandIndex) {
            return true;
        }
    }
    return false;
}
bool isNCEDataInOperand(VPUIP::NCEClusterTaskOp nceOp, unsigned operandIndex) {
    const auto containsOperandIndex = [operandIndex](mlir::MutableOperandRange operands) {
        return llvm::any_of(operands, [operandIndex](mlir::OpOperand& operand) {
            return operand.getOperandNumber() == operandIndex;
        });
    };

    // Start with the NCE operands already covered by the current optimization
    // model. Other auxiliary inputs can be enabled here after per-input
    // legality and performance validation, tracked by E#227976.
    return containsOperandIndex(nceOp.getInputMutable()) || containsOperandIndex(nceOp.getWeightsMutable()) ||
           containsOperandIndex(nceOp.getWeightTableMutable());
}

bool isDataInOperand(mlir::Operation* op, unsigned operandIndex) {
    if (op == nullptr || operandIndex >= op->getNumOperands()) {
        return false;
    }
    // Copy reads only its input operand. The output buffer is a write target,
    // so it must not create a producer edge into the copy.
    if (mlir::isa<VPUIP::CopyOp>(op)) {
        return operandIndex == 0;
    }
    // SubView reads from its source buffer. Dynamic offsets/sizes describe the
    // slice and are not tensor data flowing through the edge.
    if (mlir::isa<VPUIP::SubViewOp>(op)) {
        return operandIndex == 0;
    }
    // ConcatView operands include tile inputs followed by output_buff.
    // The inputs are the normal fan-in producers. Keep output_buff as a
    // connectivity operand too: it is the view source of the ConcatView result
    // and ties SubView/Copy writes into that destination buffer to the same
    // FanIn edge.
    if (mlir::isa<VPUIP::ConcatViewOp>(op)) {
        return true;
    }
    // StubOp has no destination-buffer operands; all operands are read-side
    // inputs of the compute placeholder.
    if (mlir::isa<VPUIP::StubOp>(op)) {
        return true;
    }
    if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
        return isNCEDataInOperand(nceOp, operandIndex);
    }
    if (mlir::isa_and_present<VPUIP::LayerOpInterface>(op)) {
        return isLayerDataInOperand(op, operandIndex);
    }
    // Pure view-like ops carry data through their first operand.
    return operandIndex == 0;
}

bool isDataFlowResultUse(mlir::Operation* user, unsigned operandIndex, const OpTrait& trait) {
    if (user == nullptr || operandIndex >= user->getNumOperands()) {
        return false;
    }
    if (mlir::isa<mlir::func::ReturnOp>(user)) {
        return true;
    }

    if (trait.role == OpTrait::Role::Transparent || trait.role == OpTrait::Role::Junction ||
        (trait.role == OpTrait::Role::Boundary && trait.boundaryKind == EdgeBoundary::Kind::ComputeOp)) {
        return isDataInOperand(user, operandIndex);
    }

    return false;
}

bool isLayerOutputOperand(mlir::Operation* op, unsigned operandIndex) {
    if (!mlir::isa_and_present<VPUIP::LayerOpInterface>(op)) {
        return false;
    }
    for (auto& operand : VPUIP::getLayerOutOpOperands(op)) {
        if (operand.getOperandNumber() == operandIndex) {
            return true;
        }
    }
    return false;
}

//===----------------------------------------------------------------------===//
// Edge boundary resolver
//===----------------------------------------------------------------------===//

/// Resolves the legal EdgeBoundary for values that cross an Edge component.
///
/// Edge nodes contain only DMA/view operations. Their external endpoints must
/// be represented by stable boundaries: compute ops, constants, alloc buffers,
/// or function args/returns. LoopRegionInfo is attached as metadata instead
/// of becoming hard topology boundaries. The resolver hides the IR
/// walking needed to find those endpoints:
/// - producer resolution walks backward through transparent ops and junctions
///   until it finds the source boundary for an edge node input;
/// - consumer resolution maps direct users to target boundaries;
/// - LoopRegion maps are consulted only to annotate boundaries/nodes.
///
/// Example:
///   compute0 -> Copy -> SubView -> compute1
/// When materializing the Copy input, producer resolution returns compute0 as
/// the source boundary. When materializing the SubView result use, consumer
/// resolution returns compute1 as the target boundary.
class EdgeBoundaryResolver {
public:
    explicit EdgeBoundaryResolver(const LoopRegionBoundaryInfo& boundaries);

    /// Resolves the source boundary for a data-in value of an EdgeNode.
    /// LoopRegion participation is attached to the ordinary producer boundary.
    SmallVector<ResolvedSourceBoundary, 2> resolveProducerBoundaries(mlir::Value producerValue) const;

    /// Resolves the target boundaries for one use of an EdgeNode result.
    /// LoopRegion participation is attached to the ordinary consumer boundary.
    SmallVector<ResolvedTargetBoundary, 2> resolveConsumerBoundaries(mlir::Operation* targetOp, unsigned operandIndex,
                                                                     OpTrait targetTrait) const;

    /// Resolves the LoopRegionInfo to attach to one EdgeNode.
    LoopRegionInfo resolveEdgeNodeLoopRegionInfo(mlir::Operation* op) const;

private:
    // "Lookup" means reading the direct operation LoopRegionInfo from the
    // read-only LoopRegionBoundaryInfo built by LoopRegionBuilder. "Resolve"
    // means selecting one LoopRegionInfo from value, operation, or operand entries.
    // "Attach" means copying one resolved LoopRegionInfo into EdgeBoundary::loopRegionInfo
    // or EdgeNode::loopRegionInfo, so later optimizers can read it from Edge.
    LoopRegionInfo resolveValueLoopRegionInfo(mlir::Value value) const;
    LoopRegionInfo lookupOperationLoopRegionInfo(mlir::Operation* op) const;
    EdgeBoundary attachLoopRegionInfoFromValue(EdgeBoundary boundary, mlir::Value value) const;
    EdgeBoundary attachLoopRegionInfoFromUse(EdgeBoundary boundary, mlir::Operation* targetOp,
                                             unsigned operandIndex) const;

    /// LoopRegionInfo entries supplied by LoopRegionBuilder identify the loop,
    /// tile, and optional port of a value, operation, or operand.
    /// EdgeCollector only copies this information to EdgeBoundary/EdgeNode;
    /// optimizers use them later.
    const LoopRegionBoundaryInfo& _loopRegionBoundaries;
};

//===----------------------------------------------------------------------===//
// EdgeBoundaryResolver: LoopRegion output assembly cache
//===----------------------------------------------------------------------===//

EdgeBoundaryResolver::EdgeBoundaryResolver(const LoopRegionBoundaryInfo& boundaries)
        : _loopRegionBoundaries(boundaries) {
}

// Pick the LoopRegion port information for one SSA value.
// A value can have several LoopRegionInfo entries: a region-level port and tile-level ports.
// Prefer one clear tile-level port. If several tiles share the same value,
// keep the region-level port and let the consumer operation provide the tile.
std::optional<LoopRegionInfo> chooseLoopRegionPortInfo(ArrayRef<LoopRegionInfo> infos,
                                                       const LoopRegionInfo& operationInfo) {
    const LoopRegionInfo* portIdentity = nullptr;
    const LoopRegionInfo* logicalPort = nullptr;
    const LoopRegionInfo* tilePort = nullptr;
    bool hasMultipleTilePorts = false;

    for (const auto& info : infos) {
        if (!info.hasPort || (operationInfo.isValid() && !info.hasSameRegionIdentity(operationInfo))) {
            continue;
        }

        if (portIdentity == nullptr) {
            portIdentity = &info;
        } else if (!portIdentity->hasSameRegionIdentity(info) || portIdentity->portKind != info.portKind ||
                   portIdentity->portIdx != info.portIdx) {
            return std::nullopt;
        }

        if (!info.hasTileIdentity()) {
            logicalPort = &info;
            continue;
        }
        if (tilePort != nullptr && !tilePort->hasSameTileIdentity(info)) {
            hasMultipleTilePorts = true;
        } else {
            tilePort = &info;
        }
    }

    if (tilePort != nullptr && !hasMultipleTilePorts) {
        return *tilePort;
    }
    if (logicalPort == nullptr) {
        return std::nullopt;
    }

    if (operationInfo.isValid() && logicalPort->hasSameRegionIdentity(operationInfo)) {
        auto resolved = operationInfo;
        resolved.hasPort = true;
        resolved.portKind = logicalPort->portKind;
        resolved.portIdx = logicalPort->portIdx;
        return resolved;
    }
    return *logicalPort;
}

LoopRegionInfo EdgeBoundaryResolver::resolveValueLoopRegionInfo(mlir::Value value) const {
    if (!value) {
        return {};
    }
    auto valueIt = _loopRegionBoundaries.valueToLoopRegions.find(value);
    auto operationInfo = lookupOperationLoopRegionInfo(value.getDefiningOp());
    if (valueIt != _loopRegionBoundaries.valueToLoopRegions.end()) {
        if (auto portInfo = chooseLoopRegionPortInfo(valueIt->second, operationInfo)) {
            return *portInfo;
        }
        if (valueIt->second.size() == 1) {
            return valueIt->second.front();
        }
    }
    if (operationInfo.isValid()) {
        return operationInfo;
    }
    return {};
}

LoopRegionInfo EdgeBoundaryResolver::lookupOperationLoopRegionInfo(mlir::Operation* op) const {
    if (op == nullptr) {
        return {};
    }
    auto opIt = _loopRegionBoundaries.opToLoopRegion.find(op);
    if (opIt != _loopRegionBoundaries.opToLoopRegion.end()) {
        return opIt->second;
    }
    return {};
}

LoopRegionInfo EdgeBoundaryResolver::resolveEdgeNodeLoopRegionInfo(mlir::Operation* op) const {
    if (op == nullptr) {
        return {};
    }

    // A result that is a LoopRegion port is the best match, so return it now.
    // Otherwise, save the first valid result information. Use it only when the
    // operation itself has no LoopRegionInfo.
    LoopRegionInfo resultInfo;
    for (auto result : op->getResults()) {
        auto info = resolveValueLoopRegionInfo(result);
        if (info.hasPort) {
            return info;
        }
        if (!resultInfo.isValid() && info.isValid()) {
            resultInfo = info;
        }
    }

    const auto operationInfo = lookupOperationLoopRegionInfo(op);
    if (operationInfo.isValid()) {
        return operationInfo;
    }
    if (resultInfo.isValid()) {
        return resultInfo;
    }

    // No result or operation LoopRegionInfo was found. Check inputs as the last option.
    for (auto operand : op->getOperands()) {
        auto info = resolveValueLoopRegionInfo(operand);
        if (info.isValid()) {
            return info;
        }
    }
    return {};
}

EdgeBoundary EdgeBoundaryResolver::attachLoopRegionInfoFromValue(EdgeBoundary boundary, mlir::Value value) const {
    // Source boundaries are created from a producer value. Copy the information found
    // for that value onto the boundary.
    if (!boundary.loopRegionInfo.isValid()) {
        boundary.loopRegionInfo = resolveValueLoopRegionInfo(value);
    }
    return boundary;
}

EdgeBoundary EdgeBoundaryResolver::attachLoopRegionInfoFromUse(EdgeBoundary boundary, mlir::Operation* targetOp,
                                                               unsigned operandIndex) const {
    // Use the exact operand LoopRegionInfo first. The same SSA value can feed several
    // LoopRegion tiles, so the consumer operand tells which tile this use means.
    if (!boundary.loopRegionInfo.isValid() && targetOp != nullptr && operandIndex < targetOp->getNumOperands()) {
        auto& operand = targetOp->getOpOperand(operandIndex);
        auto operandIt = _loopRegionBoundaries.operandToLoopRegion.find(&operand);
        if (operandIt != _loopRegionBoundaries.operandToLoopRegion.end()) {
            boundary.loopRegionInfo = operandIt->second;
        } else {
            boundary.loopRegionInfo = resolveValueLoopRegionInfo(operand.get());
        }
    }
    if (!boundary.loopRegionInfo.isValid()) {
        boundary.loopRegionInfo = lookupOperationLoopRegionInfo(targetOp);
    }
    return boundary;
}

//===----------------------------------------------------------------------===//
// EdgeBoundaryResolver: source-side traversal
//===----------------------------------------------------------------------===//

SmallVector<ResolvedSourceBoundary, 2> EdgeBoundaryResolver::resolveProducerBoundaries(
        mlir::Value producerValue) const {
    SmallVector<ResolvedSourceBoundary, 2> out;
    if (!producerValue) {
        return out;
    }

    auto* definingOp = producerValue.getDefiningOp();
    if (definingOp == nullptr) {
        if (auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(producerValue)) {
            out.push_back({valueBoundaryKey(producerValue),
                           attachLoopRegionInfoFromValue(EdgeBoundary::funcArg(blockArgument), producerValue)});
        }
        return out;
    }

    const auto trait = classifyOp(definingOp);
    switch (trait.role) {
    case OpTrait::Role::Boundary:
        switch (trait.boundaryKind) {
        case EdgeBoundary::Kind::ComputeOp: {
            // For multi-result compute ops, keep the exact result that feeds
            // the edge. Example: SwKernel#1 -> Copy.
            uint32_t resultIdx = 0;
            if (auto opResult = mlir::dyn_cast<mlir::OpResult>(producerValue)) {
                if (opResult.getOwner() == definingOp) {
                    resultIdx = opResult.getResultNumber();
                }
            }
            out.push_back(
                    {valueBoundaryKey(producerValue),
                     attachLoopRegionInfoFromValue(EdgeBoundary::computeResult(definingOp, resultIdx), producerValue)});
            return out;
        }
        case EdgeBoundary::Kind::Constant:
            out.push_back({valueBoundaryKey(producerValue),
                           attachLoopRegionInfoFromValue(EdgeBoundary::constant(definingOp), producerValue)});
            return out;
        case EdgeBoundary::Kind::AllocBuffer:
            out.push_back({valueBoundaryKey(producerValue),
                           attachLoopRegionInfoFromValue(EdgeBoundary::allocBuffer(definingOp), producerValue)});
            return out;
        default:
            return out;
        }
    case OpTrait::Role::Transparent:
    case OpTrait::Role::Junction:
        return out;
    case OpTrait::Role::Unknown:
        return out;
    }
    return out;
}

//===----------------------------------------------------------------------===//
// EdgeBoundaryResolver: target-side traversal
//===----------------------------------------------------------------------===//

SmallVector<ResolvedTargetBoundary, 2> EdgeBoundaryResolver::resolveConsumerBoundaries(mlir::Operation* targetOp,
                                                                                       unsigned operandIndex,
                                                                                       OpTrait targetTrait) const {
    SmallVector<ResolvedTargetBoundary, 2> out;
    if (targetOp == nullptr) {
        return out;
    }
    if (mlir::isa<mlir::func::ReturnOp>(targetOp)) {
        out.push_back({BoundaryKey{targetOp, static_cast<int32_t>(operandIndex)},
                       attachLoopRegionInfoFromUse(EdgeBoundary::funcReturn(targetOp, operandIndex), targetOp,
                                                   operandIndex)});
        return out;
    }
    if (operandIndex >= targetOp->getNumOperands()) {
        return out;
    }

    if (targetTrait.role == OpTrait::Role::Boundary && targetTrait.boundaryKind == EdgeBoundary::Kind::ComputeOp) {
        out.push_back({BoundaryKey{targetOp, static_cast<int32_t>(operandIndex)},
                       attachLoopRegionInfoFromUse(EdgeBoundary::computeOperand(targetOp, operandIndex), targetOp,
                                                   operandIndex)});
        return out;
    }

    if (targetTrait.role == OpTrait::Role::Transparent || targetTrait.role == OpTrait::Role::Junction) {
        return out;
    }

    return out;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Op-classification helpers
//===----------------------------------------------------------------------===//

bool isComputeOp(mlir::Operation* op) {
    const auto trait = classifyOp(op);
    return trait.role == OpTrait::Role::Boundary && trait.boundaryKind == EdgeBoundary::Kind::ComputeOp;
}

bool isEdgeOp(mlir::Operation* op) {
    const auto trait = classifyOp(op);
    return trait.role == OpTrait::Role::Transparent || trait.role == OpTrait::Role::Junction;
}

bool isViewLikeOp(mlir::Operation* op) {
    return mlir::isa_and_nonnull<VPUIP::PermuteCastOp, VPUIP::ShapeCastOp, VPUIP::GenericReshapeOp,
                                 VPUIP::QuantizeCastOp, VPUIP::DistributedCastOp>(op);
}

//===----------------------------------------------------------------------===//
// Node detail snapshots
//===----------------------------------------------------------------------===//

NodeDetails buildNodeDetails(mlir::Operation* op) {
    NodeDetails details;
    details.originalOp = op;

    if (op == nullptr) {
        return details;
    }

    if (op->getNumOperands() > 0) {
        details.inputType = op->getOperand(0).getType();
        details.srcMemSpace = getMemoryKind(details.inputType);
    }
    if (op->getNumResults() > 0) {
        details.outputType = op->getResult(0).getType();
        details.dstMemSpace = getMemoryKind(details.outputType);
    }

    if (auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(op)) {
        details.kind = NodeDetails::Kind::Copy;
        details.dstBuff = copyOp.getOutputBuff();
    } else if (mlir::isa<VPUIP::SubViewOp>(op)) {
        details.kind = NodeDetails::Kind::Slice;
    } else if (mlir::isa<VPUIP::ConcatViewOp>(op)) {
        details.kind = NodeDetails::Kind::Concat;
    } else if (isViewLikeOp(op)) {
        details.kind = NodeDetails::Kind::ViewOp;
    } else {
        details.kind = NodeDetails::Kind::Other;
    }

    return details;
}

EdgeCollector::EdgeCollector(Logger log): _log(log) {
}

//===----------------------------------------------------------------------===//
// Connected components
//===----------------------------------------------------------------------===//

int32_t EdgeCollector::findComponentRoot(int32_t edgeIndex) {
    while (_componentParentByEdgeIndex[edgeIndex] != edgeIndex) {
        _componentParentByEdgeIndex[edgeIndex] = _componentParentByEdgeIndex[_componentParentByEdgeIndex[edgeIndex]];
        edgeIndex = _componentParentByEdgeIndex[edgeIndex];
    }
    return edgeIndex;
}

void EdgeCollector::mergeEdgeComponents(int32_t lhsEdgeIndex, int32_t rhsEdgeIndex) {
    const int32_t lhsRoot = findComponentRoot(lhsEdgeIndex);
    const int32_t rhsRoot = findComponentRoot(rhsEdgeIndex);
    if (lhsRoot == rhsRoot) {
        return;
    }
    // Union into the smaller index for determinism (matches walk order).
    if (lhsRoot < rhsRoot) {
        _componentParentByEdgeIndex[rhsRoot] = lhsRoot;
    } else {
        _componentParentByEdgeIndex[lhsRoot] = rhsRoot;
    }
}

int32_t EdgeCollector::getOrCreateEdgeIndex(mlir::Operation* edgeOp) {
    auto edgeIndexIt = _edgeIndexByOp.find(edgeOp);
    if (edgeIndexIt != _edgeIndexByOp.end()) {
        return edgeIndexIt->second;
    }
    const int32_t edgeIndex = static_cast<int32_t>(_componentParentByEdgeIndex.size());
    _componentParentByEdgeIndex.push_back(edgeIndex);
    _edgeIndexByOp[edgeOp] = edgeIndex;
    return edgeIndex;
}

//===----------------------------------------------------------------------===//
// Edge materialization helpers
//
// EdgeBoundaryResolver resolves transparent chains, ConcatView junctions, LoopRegion
// ports, and region-internal output assembly. It also records direct
// result/operand indices when the resolved boundary is a compute op.
//
// Example:
//   compute0 -> Copy -> SubView -> compute1
// Copy/SubView become EdgeNode entries. Producer resolution walks from the
// Copy input back to compute0 and returns source EdgeBoundaries.
// Consumer resolution walks from the SubView result use to compute1 and returns
// target EdgeBoundaries.
//===----------------------------------------------------------------------===//

namespace {

using BoundaryMap = llvm::DenseMap<BoundaryKey, int32_t, BoundaryKeyInfo>;

int32_t registerBoundary(SmallVector<EdgeBoundary, 1>& boundaries, BoundaryMap& boundaryIndexByKey, BoundaryKey key,
                         EdgeBoundary boundary) {
    auto boundaryIt = boundaryIndexByKey.find(key);
    if (boundaryIt != boundaryIndexByKey.end()) {
        return boundaryIt->second;
    }

    const auto boundaryIndex = static_cast<int32_t>(boundaries.size());
    boundaries.push_back(boundary);
    boundaryIndexByKey[key] = boundaryIndex;
    return boundaryIndex;
}

void fillBoundaryEntryNodes(const Edge& edge, SmallVector<int32_t, 1>& entryNodes) {
    entryNodes.assign(edge.sources.size(), -1);
    for (size_t nodeIndex = 0; nodeIndex < edge.nodes.size(); ++nodeIndex) {
        for (int32_t link : edge.nodes[nodeIndex].preds) {
            if (link >= 0) {
                continue;
            }

            const int32_t sourceIndex = ~link;
            if (sourceIndex >= 0 && static_cast<size_t>(sourceIndex) < entryNodes.size() &&
                entryNodes[sourceIndex] == -1) {
                entryNodes[sourceIndex] = static_cast<int32_t>(nodeIndex);
            }
        }
    }
}

void fillBoundaryExitNodes(const Edge& edge, SmallVector<int32_t, 1>& exitNodes) {
    exitNodes.assign(edge.targets.size(), -1);
    for (size_t nodeIndex = 0; nodeIndex < edge.nodes.size(); ++nodeIndex) {
        for (int32_t link : edge.nodes[nodeIndex].succs) {
            if (link >= 0) {
                continue;
            }

            const int32_t targetIndex = ~link;
            if (targetIndex >= 0 && static_cast<size_t>(targetIndex) < exitNodes.size()) {
                exitNodes[targetIndex] = static_cast<int32_t>(nodeIndex);
            }
        }
    }
}

void appendUniqueLink(SmallVectorImpl<int32_t>& links, llvm::SmallDenseSet<int32_t, 4>& linkSet, int32_t link) {
    if (linkSet.insert(link).second) {
        links.push_back(link);
    }
}

void fillLoopRegionInfos(Edge& edge) {
    edge.loopRegionInfos.clear();

    auto appendUnique = [&](const LoopRegionInfo& info) {
        if (info.isValid() && !llvm::is_contained(edge.loopRegionInfos, info)) {
            edge.loopRegionInfos.push_back(info);
        }
    };

    for (const auto& source : edge.sources) {
        appendUnique(source.loopRegionInfo);
    }
    for (const auto& target : edge.targets) {
        appendUnique(target.loopRegionInfo);
    }
    for (const auto& node : edge.nodes) {
        appendUnique(node.loopRegionInfo);
    }
}

void fillInputViewChains(Edge& edge) {
    for (auto& node : edge.nodes) {
        if (node.op == nullptr || node.op->getNumOperands() == 0 || node.preds.size() != 1 ||
            (!node.details.isCopy() && !node.details.isSlice() && !node.details.isViewOp())) {
            continue;
        }

        InputViewChain chain;
        chain.source = node.op->getOperand(0);
        const auto predecessor = node.preds.front();
        if (predecessor >= 0) {
            const auto predecessorIndex = static_cast<size_t>(predecessor);
            if (predecessorIndex >= edge.nodes.size()) {
                continue;
            }

            const auto& predecessorNode = edge.nodes[predecessorIndex];
            auto* predecessorOp = predecessorNode.op;
            if (mlir::isa<VPUIP::SubViewOp>(predecessorOp) || isViewLikeOp(predecessorOp)) {
                if (predecessorOp->getNumOperands() != 1 || predecessorOp->getNumResults() != 1 ||
                    predecessorOp->getResult(0) != chain.source || !predecessorOp->getRegions().empty() ||
                    !predecessorNode.details.inputViewChain.has_value()) {
                    continue;
                }

                chain = *predecessorNode.details.inputViewChain;
                chain.views.push_back(predecessorOp);
            }
        }
        node.details.inputViewChain = std::move(chain);
    }
}

}  // namespace

//===----------------------------------------------------------------------===//
// collect: build connected Edge DAGs from function-level ops.
//
// The collector treats Copy/SubView/ConcatView/view-like ops as edge nodes and
// legal producer/consumer ops as boundaries. It does not mutate IR.
//
// Algorithm:
// 1. Walk the function and assign every visible edge op a temporary edge index.
//    LoopRegion-internal ops are ordinary visible edge ops; LoopRegionInfo is
//    attached as metadata and do not stop traversal.
// 2. Use connected-component tracking to merge edge indices when two edge ops
//    touch through data-in operands, or when sibling edge ops read the same
//    external source value.
// 3. Walk the function again to group each connected component in IR order.
// 4. Materialize one Edge per component. Edge nodes store local node-to-node
//    adjacency; external predecessors/successors are registered as EdgeBoundary
//    entries resolved through EdgeBoundaryResolver.
//
// Example:
//   compute0 -> Copy -> SubView -> compute1
// becomes one Normal edge with Copy/SubView nodes, compute0 as the source
// boundary, and compute1 as the target boundary.
//
//   compute0 -> Copy -> {compute1, compute2}
// becomes one FanOut edge: the shared Copy node belongs to one connected
// component with one source boundary and two target boundaries.
//===----------------------------------------------------------------------===//

EdgeCollection EdgeCollector::collect(mlir::func::FuncOp func, const LoopRegionBoundaryInfo& loopRegionBoundaries) {
    EdgeCollection out;
    _loopRegionBoundaries = &loopRegionBoundaries;
    VPUX_SCOPE_EXIT {
        _loopRegionBoundaries = nullptr;
    };
    _edgeIndexByOp.clear();
    _componentParentByEdgeIndex.clear();
    _diagEdgeOps = 0;
    _diagBoundaryOps = 0;
    _diagDroppedIncompleteEdges = 0;
    _diagDroppedNoSources = 0;
    _diagDroppedNoTargets = 0;

    if (func == nullptr) {
        return out;
    }
    mlir::Operation* funcOp = func.getOperation();

    // Resolve edge component endpoints from SSA values/users to EdgeBoundary objects.
    EdgeBoundaryResolver boundaryResolver(loopRegionBoundaries);

    const auto isResultUseForCollection = [&](mlir::Operation* user, unsigned operandIndex) {
        const auto userTrait = mlir::isa<mlir::func::ReturnOp>(user) ? OpTrait{} : classifyOp(user);
        if (isDataFlowResultUse(user, operandIndex, userTrait)) {
            return true;
        }
        return userTrait.role == OpTrait::Role::Boundary && userTrait.boundaryKind == EdgeBoundary::Kind::ComputeOp &&
               isLayerOutputOperand(user, operandIndex);
    };

    // Step 1: assign edge ids and union adjacent edge ops.
    funcOp->walk([&](mlir::Operation* op) {
        if (op == funcOp) {
            return;
        }
        if (!isEdgeOp(op)) {
            if (classifyOp(op).role == OpTrait::Role::Boundary) {
                ++_diagBoundaryOps;
            }
            return;
        }
        ++_diagEdgeOps;
        const int32_t edgeIndex = getOrCreateEdgeIndex(op);

        for (auto operandIndex = 0U; operandIndex < op->getNumOperands(); ++operandIndex) {
            if (!isDataInOperand(op, operandIndex)) {
                continue;
            }
            const auto operandValue = op->getOperand(operandIndex);
            if (auto* definingOp = operandValue.getDefiningOp()) {
                if (isEdgeOp(definingOp)) {
                    const int32_t producerEdgeIndex = getOrCreateEdgeIndex(definingOp);
                    mergeEdgeComponents(edgeIndex, producerEdgeIndex);
                }
            }
        }

        for (mlir::Value resultValue : op->getResults()) {
            for (mlir::OpOperand& useRef : resultValue.getUses()) {
                mlir::Operation* user = useRef.getOwner();
                if (user == nullptr) {
                    continue;
                }
                if (isEdgeOp(user) && isDataInOperand(user, useRef.getOperandNumber())) {
                    const int32_t userEdgeIndex = getOrCreateEdgeIndex(user);
                    mergeEdgeComponents(edgeIndex, userEdgeIndex);
                }
            }
        }
    });

    // Step 1b: merge sibling edge ops that read the same external source.
    llvm::DenseMap<mlir::Value, int32_t> firstEdgeIndexBySourceValue;
    for (auto& [op, edgeIndex] : _edgeIndexByOp) {
        for (auto operandIndex = 0U; operandIndex < op->getNumOperands(); ++operandIndex) {
            if (!isDataInOperand(op, operandIndex)) {
                continue;
            }
            const auto operandValue = op->getOperand(operandIndex);
            mlir::Operation* definingOp = operandValue.getDefiningOp();
            // Only merge via shared external (non-EdgeOp) sources.
            if (isEdgeOp(definingOp)) {
                continue;
            }
            auto [firstEdgeIt, inserted] =
                    firstEdgeIndexBySourceValue.try_emplace(operandValue, findComponentRoot(edgeIndex));
            if (!inserted) {
                mergeEdgeComponents(edgeIndex, firstEdgeIt->second);
            }
        }
    }

    if (_edgeIndexByOp.empty()) {
        _log.info("[EdgeCollector] funcOp has no EdgeOps (edgeOps={0}, boundaries={1})", _diagEdgeOps,
                  _diagBoundaryOps);
        return out;
    }

    // Step 2: group edge ops by connected-component root in IR walk order.
    llvm::DenseMap<int32_t, SmallVector<mlir::Operation*, 4>> edgeOpsByComponentRoot;
    SmallVector<int32_t, 16> componentRootOrder;
    funcOp->walk([&](mlir::Operation* op) {
        auto edgeIndexIt = _edgeIndexByOp.find(op);
        if (edgeIndexIt == _edgeIndexByOp.end()) {
            return;
        }
        const int32_t componentRoot = findComponentRoot(edgeIndexIt->second);
        auto& componentOps = edgeOpsByComponentRoot[componentRoot];
        if (componentOps.empty()) {
            componentRootOrder.push_back(componentRoot);
        }
        componentOps.push_back(op);
    });

    // Step 3: materialize one Edge per root.
    for (int32_t componentRoot : componentRootOrder) {
        const auto& edgeOps = edgeOpsByComponentRoot[componentRoot];
        Edge edge;
        edge.nodes.reserve(edgeOps.size());

        llvm::DenseMap<mlir::Operation*, int32_t> localNodeIndexByOp;
        for (size_t edgeOpIndex = 0; edgeOpIndex < edgeOps.size(); ++edgeOpIndex) {
            localNodeIndexByOp[edgeOps[edgeOpIndex]] = static_cast<int32_t>(edgeOpIndex);
        }

        BoundaryMap sourceBoundaryMap;
        BoundaryMap targetBoundaryMap;

        for (mlir::Operation* op : edgeOps) {
            EdgeNode node;
            node.op = op;
            node.details = buildNodeDetails(op);
            node.loopRegionInfo = boundaryResolver.resolveEdgeNodeLoopRegionInfo(op);
            edge.nodes.push_back(std::move(node));
        }

        for (size_t edgeOpIndex = 0; edgeOpIndex < edgeOps.size(); ++edgeOpIndex) {
            mlir::Operation* op = edgeOps[edgeOpIndex];
            EdgeNode& node = edge.nodes[edgeOpIndex];
            llvm::SmallDenseSet<int32_t, 4> predLinks;
            llvm::SmallDenseSet<int32_t, 4> succLinks;

            for (auto operandIndex = 0U; operandIndex < op->getNumOperands(); ++operandIndex) {
                if (!isDataInOperand(op, operandIndex)) {
                    continue;
                }
                const auto operandValue = op->getOperand(operandIndex);
                mlir::Operation* definingOp = operandValue.getDefiningOp();
                if (definingOp != nullptr) {
                    auto localNodeIndexIt = localNodeIndexByOp.find(definingOp);
                    if (localNodeIndexIt != localNodeIndexByOp.end()) {
                        appendUniqueLink(node.preds, predLinks, localNodeIndexIt->second);
                        continue;
                    }
                }

                for (const auto& resolvedBoundary : boundaryResolver.resolveProducerBoundaries(operandValue)) {
                    if (resolvedBoundary.boundary.kind == EdgeBoundary::Kind::None) {
                        continue;
                    }
                    const int32_t sourceBoundaryIndex = registerBoundary(
                            edge.sources, sourceBoundaryMap, resolvedBoundary.key, resolvedBoundary.boundary);
                    appendUniqueLink(node.preds, predLinks, ~sourceBoundaryIndex);
                }
            }

            for (mlir::Value resultValue : op->getResults()) {
                for (mlir::OpOperand& useRef : resultValue.getUses()) {
                    mlir::Operation* user = useRef.getOwner();
                    if (user == nullptr) {
                        continue;
                    }
                    const unsigned targetOperandIndex = useRef.getOperandNumber();
                    const auto userTrait = mlir::isa<mlir::func::ReturnOp>(user) ? OpTrait{} : classifyOp(user);
                    if (!isResultUseForCollection(user, targetOperandIndex)) {
                        continue;
                    }
                    auto localNodeIndexIt = localNodeIndexByOp.find(user);
                    if (localNodeIndexIt != localNodeIndexByOp.end()) {
                        appendUniqueLink(node.succs, succLinks, localNodeIndexIt->second);
                        continue;
                    }

                    for (const auto& resolvedBoundary :
                         boundaryResolver.resolveConsumerBoundaries(user, targetOperandIndex, userTrait)) {
                        if (resolvedBoundary.boundary.kind == EdgeBoundary::Kind::None) {
                            continue;
                        }
                        const int32_t targetBoundaryIndex = registerBoundary(
                                edge.targets, targetBoundaryMap, resolvedBoundary.key, resolvedBoundary.boundary);
                        appendUniqueLink(node.succs, succLinks, ~targetBoundaryIndex);
                    }
                }
            }
        }

        if (edge.sources.empty() || edge.targets.empty()) {
            ++_diagDroppedIncompleteEdges;
            if (edge.sources.empty()) {
                ++_diagDroppedNoSources;
            }
            if (edge.targets.empty()) {
                ++_diagDroppedNoTargets;
            }
            _log.trace("[EdgeCollector] drop incomplete component root={0}: nodes={1}, sources={2}, targets={3}",
                       componentRoot, edge.nodes.size(), edge.sources.size(), edge.targets.size());
            continue;
        }

        fillBoundaryEntryNodes(edge, edge.sourceEntryNodes);
        fillBoundaryExitNodes(edge, edge.targetExitNodes);
        fillInputViewChains(edge);
        edge.type = inferEdgeTypeFromBoundaryCounts(edge.sources.size(), edge.targets.size());
        fillLoopRegionInfos(edge);

        out.edges.push_back(std::move(edge));
    }

    _log.info("[EdgeCollector] edgeOps={0} boundaries={1} edges={2} droppedIncomplete={3} noSources={4} noTargets={5}",
              _diagEdgeOps, _diagBoundaryOps, out.edges.size(), _diagDroppedIncompleteEdges, _diagDroppedNoSources,
              _diagDroppedNoTargets);

    return out;
}

}  // namespace vpux::VPUIP::DmaOptimization
