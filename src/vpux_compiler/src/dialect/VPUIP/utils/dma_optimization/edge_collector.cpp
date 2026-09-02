//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    vpux::VPUIP::EdgeBoundary boundary;
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

namespace vpux {
namespace VPUIP {

namespace {

//===----------------------------------------------------------------------===//
// Memory-space helpers
//===----------------------------------------------------------------------===//

int64_t getMemorySpaceCode(VPU::MemoryKind memoryKind) {
    switch (memoryKind) {
    case VPU::MemoryKind::DDR:
        return kMemSpaceDDR;
    case VPU::MemoryKind::CMX_NN:
        return kMemSpaceCMX;
    default:
        return -1;
    }
}

int64_t getMemorySpaceCode(mlir::Type type) {
    auto ndType = mlir::dyn_cast_or_null<vpux::NDTypeInterface>(type);
    if (ndType == nullptr) {
        return -1;
    }
    return getMemorySpaceCode(ndType.getMemoryKind());
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
    if (mlir::isa_and_nonnull<VPUIP::LayerOpInterface>(op)) {
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

//===----------------------------------------------------------------------===//
// Edge boundary resolver
//===----------------------------------------------------------------------===//

/// Resolves the legal EdgeBoundary for values that cross an Edge component.
///
/// Edge nodes contain only DMA/view operations. Their external endpoints must
/// be represented by stable boundaries: compute ops, constants, alloc buffers,
/// function args/returns, or LoopRegion ports. The resolver hides the IR
/// walking needed to find those endpoints:
/// - producer resolution walks backward through transparent ops and junctions
///   until it finds the source boundary for an edge node input;
/// - consumer resolution maps direct users to target boundaries;
/// - LoopRegion port maps and LoopRegion output assembly ConcatView chains are
///   consulted so region crossings become LoopRegion boundaries instead of
///   ordinary ops.
///
/// Example:
///   compute0 -> Copy -> SubView -> compute1
/// When materializing the Copy input, producer resolution returns compute0 as
/// the source boundary. When materializing the SubView result use, consumer
/// resolution returns compute1 as the target boundary.
class EdgeBoundaryResolver {
public:
    /// Keeps a reference to LoopRegion cut-point information and caches
    /// LoopRegion output assembly ops that should resolve as output ports.
    explicit EdgeBoundaryResolver(const LoopRegionBoundaryInfo& boundaries);

    /// Resolves the source boundary for a data-in value of an EdgeNode.
    /// Region crossings must be represented explicitly through LoopRegion
    /// ports; an internal producer without a port is a builder invariant break.
    SmallVector<ResolvedSourceBoundary, 2> resolveProducerBoundaries(mlir::Value producerValue) const;

    /// Resolves the target boundaries for one use of an EdgeNode result.
    /// Region crossings must be represented explicitly through LoopRegion
    /// ports; an internal consumer without a port is a builder invariant break.
    SmallVector<ResolvedTargetBoundary, 2> resolveConsumerBoundaries(mlir::Operation* targetOp, unsigned operandIndex,
                                                                     OpTrait targetTrait) const;

private:
    /// Returns a LoopRegion boundary when `value` is registered as a region
    /// input, output, or destination-buffer port.
    EdgeBoundary probeLoopRegionPort(mlir::Value value) const;

    /// Returns a LoopRegion output boundary when `op` is part of a cached
    /// LoopRegion output assembly chain.
    EdgeBoundary probeLoopRegionOutputAssembly(mlir::Operation* op) const;

    bool isLoopRegionInternalOp(mlir::Operation* op) const;

    /// LoopRegion boundary metadata supplied by LoopRegionBuilder.
    const LoopRegionBoundaryInfo& _loopRegionBoundaries;

    /// ConcatView op that assembles a LoopRegion output -> output port location.
    llvm::DenseMap<mlir::Operation*, LoopRegionPortLocation> _loopRegionOutputAssemblyOpToPort;
};

//===----------------------------------------------------------------------===//
// EdgeBoundaryResolver: LoopRegion output assembly cache
//===----------------------------------------------------------------------===//

EdgeBoundaryResolver::EdgeBoundaryResolver(const LoopRegionBoundaryInfo& boundaries)
        : _loopRegionBoundaries(boundaries) {
    // valueToPort stores the LoopRegion input and output values as logical ports.
    // So LoopRegion is treated as normal compute op, %Outputs = LoopRegion(%Inputs).
    for (const auto& valueAndPort : _loopRegionBoundaries.valueToPort) {
        const auto& portLocation = valueAndPort.second;
        if (portLocation.portKind != LoopRegionPortKind::Output) {
            continue;
        }
        mlir::Value currentValue = valueAndPort.first;
        while (currentValue) {
            auto* definingOp = currentValue.getDefiningOp();
            if (definingOp == nullptr) {
                break;
            }
            if (mlir::isa<VPUIP::ConcatViewOp>(definingOp)) {
                _loopRegionOutputAssemblyOpToPort.try_emplace(definingOp, portLocation);
            }
            const auto trait = classifyOp(definingOp);
            if (trait.role == OpTrait::Role::Boundary) {
                break;
            }
            if (definingOp->getNumOperands() == 0) {
                break;
            }
            currentValue = definingOp->getOperand(0);
        }
    }
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
            out.push_back({valueBoundaryKey(producerValue), EdgeBoundary::funcArg(blockArgument)});
        }
        return out;
    }

    // LoopRegion ports are explicit collection boundaries. Prefer them over
    // looking through region-internal assembly.
    if (auto boundary = probeLoopRegionPort(producerValue); boundary.kind != EdgeBoundary::Kind::None) {
        out.push_back({valueBoundaryKey(producerValue), boundary});
        return out;
    }

    // Some LoopRegion outputs are assembled by ConcatView chains. Those
    // assembly ops are logical region ports, not ordinary edge nodes.
    if (auto boundary = probeLoopRegionOutputAssembly(definingOp); boundary.kind != EdgeBoundary::Kind::None) {
        out.push_back({valueBoundaryKey(producerValue), boundary});
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
            out.push_back({valueBoundaryKey(producerValue), EdgeBoundary::computeResult(definingOp, resultIdx)});
            return out;
        }
        case EdgeBoundary::Kind::Constant:
            out.push_back({valueBoundaryKey(producerValue), EdgeBoundary::constant(definingOp)});
            return out;
        case EdgeBoundary::Kind::AllocBuffer:
            out.push_back({valueBoundaryKey(producerValue), EdgeBoundary::allocBuffer(definingOp)});
            return out;
        default:
            return out;
        }
    case OpTrait::Role::Transparent:
    case OpTrait::Role::Junction:
        if (isLoopRegionInternalOp(definingOp)) {
            VPUX_THROW("LoopRegion internal producer at '{0}' reaches a global edge without a LoopRegion port",
                       definingOp->getLoc());
        }
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
                       EdgeBoundary::funcReturn(targetOp, operandIndex)});
        return out;
    }
    if (operandIndex >= targetOp->getNumOperands()) {
        return out;
    }

    if (targetTrait.role == OpTrait::Role::Boundary && targetTrait.boundaryKind == EdgeBoundary::Kind::ComputeOp) {
        out.push_back({BoundaryKey{targetOp, static_cast<int32_t>(operandIndex)},
                       EdgeBoundary::computeOperand(targetOp, operandIndex)});
        return out;
    }

    // A direct hit on a cached output assembly op means the edge exits through
    // a LoopRegion output port rather than a normal consumer op.
    if (auto boundary = probeLoopRegionOutputAssembly(targetOp); boundary.kind != EdgeBoundary::Kind::None) {
        out.push_back({BoundaryKey{targetOp, -1}, boundary});
        return out;
    }

    const auto targetOperand = targetOp->getOperand(operandIndex);
    if (auto boundary = probeLoopRegionPort(targetOperand); boundary.kind != EdgeBoundary::Kind::None) {
        out.push_back({BoundaryKey{targetOp, static_cast<int32_t>(operandIndex)}, boundary});
        return out;
    }

    if (targetTrait.role == OpTrait::Role::Transparent || targetTrait.role == OpTrait::Role::Junction) {
        if (isLoopRegionInternalOp(targetOp)) {
            VPUX_THROW("LoopRegion internal consumer at '{0}' is reached from a global edge without a LoopRegion port",
                       targetOp->getLoc());
        }
        return out;
    }

    return out;
}

//===----------------------------------------------------------------------===//
// EdgeBoundaryResolver: LoopRegion probes
//===----------------------------------------------------------------------===//

EdgeBoundary EdgeBoundaryResolver::probeLoopRegionPort(mlir::Value value) const {
    if (!value) {
        return EdgeBoundary::none();
    }
    auto portIt = _loopRegionBoundaries.valueToPort.find(value);
    if (portIt == _loopRegionBoundaries.valueToPort.end()) {
        return EdgeBoundary::none();
    }
    return EdgeBoundary::loopRegion(portIt->second);
}

bool EdgeBoundaryResolver::isLoopRegionInternalOp(mlir::Operation* op) const {
    return op != nullptr && _loopRegionBoundaries.internalOps.contains(op);
}

EdgeBoundary EdgeBoundaryResolver::probeLoopRegionOutputAssembly(mlir::Operation* op) const {
    if (op == nullptr) {
        return EdgeBoundary::none();
    }
    auto portIt = _loopRegionOutputAssemblyOpToPort.find(op);
    if (portIt == _loopRegionOutputAssemblyOpToPort.end()) {
        return EdgeBoundary::none();
    }
    return EdgeBoundary::loopRegion(portIt->second);
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
        details.srcMemSpace = getMemorySpaceCode(details.inputType);
    }
    if (op->getNumResults() > 0) {
        details.outputType = op->getResult(0).getType();
        details.dstMemSpace = getMemorySpaceCode(details.outputType);
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
// EdgeCollector state helpers
//===----------------------------------------------------------------------===//

bool EdgeCollector::isInternal(mlir::Operation* op) const {
    if (_loopRegionBoundaries == nullptr || op == nullptr) {
        return false;
    }
    return _loopRegionBoundaries->internalOps.contains(op);
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

}  // namespace

//===----------------------------------------------------------------------===//
// collect: build connected Edge DAGs from function-level ops.
//
// The collector treats Copy/SubView/ConcatView/view-like ops as edge nodes and
// legal producer/consumer ops as boundaries. It does not mutate IR.
//
// Algorithm:
// 1. Walk the function and assign every visible edge op a temporary edge index.
//    LoopRegion-internal ops are skipped because they are collected separately.
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
    _diagSkippedInternal = 0;
    _diagDroppedIncompleteEdges = 0;
    _diagDroppedNoSources = 0;
    _diagDroppedNoTargets = 0;

    if (func == nullptr) {
        return out;
    }
    mlir::Operation* funcOp = func.getOperation();

    // Resolve edge component endpoints from SSA values/users to EdgeBoundary objects.
    EdgeBoundaryResolver boundaryResolver(loopRegionBoundaries);

    // Step 1: assign edge ids and union adjacent edge ops.
    funcOp->walk([&](mlir::Operation* op) {
        if (op == funcOp) {
            return;
        }
        if (isInternal(op)) {
            ++_diagSkippedInternal;
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
                if (!isInternal(definingOp) && isEdgeOp(definingOp)) {
                    const int32_t producerEdgeIndex = getOrCreateEdgeIndex(definingOp);
                    mergeEdgeComponents(edgeIndex, producerEdgeIndex);
                }
            }
        }

        for (mlir::Value resultValue : op->getResults()) {
            for (mlir::OpOperand& useRef : resultValue.getUses()) {
                mlir::Operation* user = useRef.getOwner();
                if (user == nullptr || isInternal(user)) {
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
            if (definingOp != nullptr && !isInternal(definingOp) && isEdgeOp(definingOp)) {
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
        _log.info("[EdgeCollector] funcOp has no EdgeOps (edgeOps={0}, boundaries={1}, skippedInternal={2})",
                  _diagEdgeOps, _diagBoundaryOps, _diagSkippedInternal);
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
                if (definingOp != nullptr && !isInternal(definingOp)) {
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
                    if (!isDataFlowResultUse(user, targetOperandIndex, userTrait)) {
                        continue;
                    }
                    if (!isInternal(user)) {
                        auto localNodeIndexIt = localNodeIndexByOp.find(user);
                        if (localNodeIndexIt != localNodeIndexByOp.end()) {
                            appendUniqueLink(node.succs, succLinks, localNodeIndexIt->second);
                            continue;
                        }
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
        edge.type = inferEdgeTypeFromBoundaryCounts(edge.sources.size(), edge.targets.size());

        out.edges.push_back(std::move(edge));
    }

    _log.info("[EdgeCollector] edgeOps={0} boundaries={1} skippedInternal={2} edges={3} "
              "droppedIncomplete={4} noSources={5} noTargets={6}",
              _diagEdgeOps, _diagBoundaryOps, _diagSkippedInternal, out.edges.size(), _diagDroppedIncompleteEdges,
              _diagDroppedNoSources, _diagDroppedNoTargets);

    return out;
}

}  // namespace VPUIP
}  // namespace vpux
