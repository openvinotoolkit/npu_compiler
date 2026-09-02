//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/edge_model.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Operation.h>

namespace vpux {
namespace VPUIP {

namespace {

//
// Validation helpers
//
constexpr const char* kEdgeValidateLogPrefix = "[Edge::validate]";

template <typename... Args>
bool validationError([[maybe_unused]] Logger log, [[maybe_unused]] StringLiteral format,
                     [[maybe_unused]] Args&&... args) {
#ifdef VPUX_DEVELOPER_BUILD
    log.warning(format, std::forward<Args>(args)...);
#endif
    return false;
}

bool isProducerBoundaryKind(EdgeBoundary::Kind kind) {
    switch (kind) {
    case EdgeBoundary::Kind::ComputeOp:
    case EdgeBoundary::Kind::Constant:
    case EdgeBoundary::Kind::AllocBuffer:
    case EdgeBoundary::Kind::FuncArg:
    case EdgeBoundary::Kind::LoopRegion:
        return true;
    default:
        return false;
    }
}

bool isConsumerBoundaryKind(EdgeBoundary::Kind kind) {
    switch (kind) {
    case EdgeBoundary::Kind::ComputeOp:
    case EdgeBoundary::Kind::FuncReturn:
    case EdgeBoundary::Kind::LoopRegion:
        return true;
    default:
        return false;
    }
}

const char* stringifyBoundaryKind(EdgeBoundary::Kind kind) {
    switch (kind) {
    case EdgeBoundary::Kind::None:
        return "None";
    case EdgeBoundary::Kind::ComputeOp:
        return "ComputeOp";
    case EdgeBoundary::Kind::FuncArg:
        return "FuncArg";
    case EdgeBoundary::Kind::Constant:
        return "Constant";
    case EdgeBoundary::Kind::AllocBuffer:
        return "AllocBuffer";
    case EdgeBoundary::Kind::FuncReturn:
        return "FuncReturn";
    case EdgeBoundary::Kind::LoopRegion:
        return "LoopRegion";
    }
    return "?";
}

const char* stringifyEdgeType(EdgeType type) {
    switch (type) {
    case EdgeType::Normal:
        return "Normal";
    case EdgeType::FanOut:
        return "FanOut";
    case EdgeType::FanIn:
        return "FanIn";
    case EdgeType::Many2Many:
        return "Many2Many";
    case EdgeType::LoopRegion:
        return "LoopRegion";
    }
    return "?";
}

bool containsLoopRegionBoundary(llvm::ArrayRef<EdgeBoundary> boundaries) {
    for (const auto& boundary : boundaries) {
        if (boundary.kind == EdgeBoundary::Kind::LoopRegion) {
            return true;
        }
    }
    return false;
}

bool validateEdgeBoundaries(Logger log, const char* callerName, EdgeType type, llvm::ArrayRef<EdgeBoundary> sources,
                            llvm::ArrayRef<EdgeBoundary> targets) {
    if (sources.empty()) {
        return validationError(log, "{0} {1}: empty sources", callerName, stringifyEdgeType(type));
    }
    if (targets.empty()) {
        return validationError(log, "{0} {1}: empty targets", callerName, stringifyEdgeType(type));
    }

    for (size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
        const auto& source = sources[sourceIndex];
        if (!isProducerBoundaryKind(source.kind)) {
            return validationError(log, "{0} {1}: sources[{2}] has producer-illegal kind {3}", callerName,
                                   stringifyEdgeType(type), sourceIndex, stringifyBoundaryKind(source.kind));
        }
    }

    for (size_t targetIndex = 0; targetIndex < targets.size(); ++targetIndex) {
        const auto& target = targets[targetIndex];
        if (!isConsumerBoundaryKind(target.kind)) {
            return validationError(log, "{0} {1}: targets[{2}] has consumer-illegal kind {3}", callerName,
                                   stringifyEdgeType(type), targetIndex, stringifyBoundaryKind(target.kind));
        }
    }

    if (type == EdgeType::LoopRegion && !containsLoopRegionBoundary(sources) && !containsLoopRegionBoundary(targets)) {
        return validationError(log, "{0} LoopRegion: expected at least one LoopRegion boundary", callerName);
    }

    return true;
}

bool validateEdgeType(Logger log, const char* callerName, EdgeType type, size_t sourceCount, size_t targetCount) {
    if (type == EdgeType::LoopRegion) {
        return true;
    }

    if (sourceCount == 0 || targetCount == 0) {
        return false;
    }

    const auto expectedType = inferEdgeTypeFromBoundaryCounts(sourceCount, targetCount);
    if (type == expectedType) {
        return true;
    }

    return validationError(log, "{0} type={1} but {2}x{3} boundaries imply {4}", callerName, stringifyEdgeType(type),
                           sourceCount, targetCount, stringifyEdgeType(expectedType));
}

bool validateNodeLink(Logger log, size_t nodeIndex, const char* edgeNodeFieldName, size_t linkIndex, int32_t link,
                      int32_t nodeCount, int32_t boundaryCount, const char* boundaryName) {
    if (link >= 0) {
        if (link < nodeCount) {
            return true;
        }

        return validationError(log, "{0} node[{1}].{2}[{3}] points to node[{4}], but node count is {5}",
                               kEdgeValidateLogPrefix, nodeIndex, edgeNodeFieldName, linkIndex, link, nodeCount);
    }

    const int32_t boundaryIndex = ~link;
    if (boundaryIndex >= 0 && boundaryIndex < boundaryCount) {
        return true;
    }

    return validationError(log, "{0} node[{1}].{2}[{3}]={4} decodes to {5}[{6}], but {5} count is {7}",
                           kEdgeValidateLogPrefix, nodeIndex, edgeNodeFieldName, linkIndex, link, boundaryName,
                           boundaryIndex, boundaryCount);
}

bool validateBoundaryTouchNodes(Logger log, const char* listName, llvm::ArrayRef<int32_t> touchNodes,
                                size_t expectedSize, int32_t nodeCount) {
    if (!touchNodes.empty() && touchNodes.size() != expectedSize) {
        return validationError(log, "{0} {1}.size()={2} != expected size {3}", kEdgeValidateLogPrefix, listName,
                               touchNodes.size(), expectedSize);
    }

    for (size_t touchIndex = 0; touchIndex < touchNodes.size(); ++touchIndex) {
        const int32_t nodeIndex = touchNodes[touchIndex];
        if (nodeIndex < -1 || nodeIndex >= nodeCount) {
            return validationError(log, "{0} {1}[{2}]={3} out of [-1,{4})", kEdgeValidateLogPrefix, listName,
                                   touchIndex, nodeIndex, nodeCount);
        }
    }

    return true;
}

bool hasLink(llvm::ArrayRef<int32_t> links, int32_t link) {
    for (const auto candidate : links) {
        if (candidate == link) {
            return true;
        }
    }
    return false;
}

bool validateNodeLinkSymmetry(Logger log, llvm::ArrayRef<EdgeNode> nodes) {
    const auto nodeCount = static_cast<int32_t>(nodes.size());

    for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
        const auto& node = nodes[nodeIndex];
        const auto nodeIndexAsLink = static_cast<int32_t>(nodeIndex);

        for (const auto succ : node.succs) {
            if (succ < 0 || succ >= nodeCount) {
                continue;
            }
            if (!hasLink(nodes[succ].preds, nodeIndexAsLink)) {
                return validationError(log, "{0} node[{1}].succs points to node[{2}], but the reverse pred is missing",
                                       kEdgeValidateLogPrefix, nodeIndex, succ);
            }
        }

        for (const auto pred : node.preds) {
            if (pred < 0 || pred >= nodeCount) {
                continue;
            }
            if (!hasLink(nodes[pred].succs, nodeIndexAsLink)) {
                return validationError(log, "{0} node[{1}].preds points to node[{2}], but the reverse succ is missing",
                                       kEdgeValidateLogPrefix, nodeIndex, pred);
            }
        }
    }

    return true;
}

bool validateBoundaryTouchConsistency(Logger log, const char* listName, llvm::ArrayRef<int32_t> touchNodes,
                                      llvm::ArrayRef<EdgeNode> nodes, bool isSource) {
    if (touchNodes.empty()) {
        return true;
    }

    for (size_t boundaryIndex = 0; boundaryIndex < touchNodes.size(); ++boundaryIndex) {
        const auto boundaryLink = ~static_cast<int32_t>(boundaryIndex);
        const auto touchesBoundary = [boundaryLink, isSource](const EdgeNode& node) {
            return hasLink(isSource ? llvm::ArrayRef<int32_t>(node.preds) : llvm::ArrayRef<int32_t>(node.succs),
                           boundaryLink);
        };

        const auto touchNodeIndex = touchNodes[boundaryIndex];
        if (touchNodeIndex == -1) {
            for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
                if (touchesBoundary(nodes[nodeIndex])) {
                    return validationError(log, "{0} {1}[{2}] is -1, but node[{3}] touches the boundary",
                                           kEdgeValidateLogPrefix, listName, boundaryIndex, nodeIndex);
                }
            }
            continue;
        }

        if (!touchesBoundary(nodes[touchNodeIndex])) {
            return validationError(log, "{0} {1}[{2}] points to node[{3}], but that node does not touch the boundary",
                                   kEdgeValidateLogPrefix, listName, boundaryIndex, touchNodeIndex);
        }
    }

    return true;
}

}  // namespace

EdgeType inferEdgeTypeFromBoundaryCounts(size_t sourceCount, size_t targetCount) {
    if (sourceCount == 1 && targetCount == 1) {
        return EdgeType::Normal;
    }
    if (sourceCount == 1 && targetCount >= 2) {
        return EdgeType::FanOut;
    }
    if (sourceCount >= 2 && targetCount == 1) {
        return EdgeType::FanIn;
    }
    return EdgeType::Many2Many;
    // LoopRegion edges are inferred from the presence of LoopRegion boundaries, not counts.
}

//
// EdgeBoundary
//
EdgeBoundary EdgeBoundary::none() {
    return {};
}

EdgeBoundary EdgeBoundary::compute(mlir::Operation* op) {
    EdgeBoundary boundary;
    boundary.kind = Kind::ComputeOp;
    boundary.op = op;
    return boundary;
}

EdgeBoundary EdgeBoundary::computeResult(mlir::Operation* op, uint32_t resultIdx) {
    auto boundary = compute(op);
    boundary.resultIdx = resultIdx;
    return boundary;
}

EdgeBoundary EdgeBoundary::computeOperand(mlir::Operation* op, uint32_t operandIdx) {
    auto boundary = compute(op);
    boundary.operandIdx = operandIdx;
    return boundary;
}

EdgeBoundary EdgeBoundary::funcArg(mlir::BlockArgument arg) {
    EdgeBoundary boundary;
    boundary.kind = Kind::FuncArg;
    boundary.arg = arg;
    return boundary;
}

EdgeBoundary EdgeBoundary::constant(mlir::Operation* op) {
    EdgeBoundary boundary;
    boundary.kind = Kind::Constant;
    boundary.op = op;
    return boundary;
}

EdgeBoundary EdgeBoundary::allocBuffer(mlir::Operation* op) {
    EdgeBoundary boundary;
    boundary.kind = Kind::AllocBuffer;
    boundary.op = op;
    return boundary;
}

EdgeBoundary EdgeBoundary::funcReturn(mlir::Operation* op, uint32_t operandIdx) {
    EdgeBoundary boundary;
    boundary.kind = Kind::FuncReturn;
    boundary.op = op;
    boundary.operandIdx = operandIdx;
    return boundary;
}

EdgeBoundary EdgeBoundary::loopRegion(LoopRegionPortLocation location) {
    EdgeBoundary boundary;
    boundary.kind = Kind::LoopRegion;
    boundary.loopPort = location;
    return boundary;
}

EdgeBoundary EdgeBoundary::loopRegion(const VPUIP::LoopRegion* region, uint32_t portIdx, LoopRegionPortKind portKind) {
    return loopRegion({region, portIdx, portKind});
}

//
// NodeDetails
//
bool NodeDetails::isCopy() const {
    return kind == Kind::Copy;
}

bool NodeDetails::isViewOp() const {
    return kind == Kind::ViewOp;
}

bool NodeDetails::isSlice() const {
    return kind == Kind::Slice;
}

//
// Edge
//
bool Edge::isNormal() const {
    return type == EdgeType::Normal;
}

bool Edge::isFanIn() const {
    return type == EdgeType::FanIn;
}

bool Edge::isFanOut() const {
    return type == EdgeType::FanOut;
}

bool Edge::isMany2Many() const {
    return type == EdgeType::Many2Many;
}

bool Edge::isLoopRegion() const {
    return type == EdgeType::LoopRegion;
}

bool Edge::hasConcatNode() const {
    for (const auto& node : nodes) {
        if (node.details.kind == NodeDetails::Kind::Concat) {
            return true;
        }
    }
    return false;
}

bool Edge::validate(Logger log) const {
    const int32_t nodeCount = static_cast<int32_t>(nodes.size());
    const int32_t sourceCount = static_cast<int32_t>(sources.size());
    const int32_t targetCount = static_cast<int32_t>(targets.size());

    if (!validateEdgeBoundaries(log, kEdgeValidateLogPrefix, type, sources, targets)) {
        return false;
    }

    // In EdgeNode adjacency, non-negative links address edge.nodes. Negative
    // links are boundary sentinels: ~link gives the source or target index.
    for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
        const auto& node = nodes[nodeIndex];
        if (node.op == nullptr) {
            return validationError(log, "{0} node[{1}].op is null", kEdgeValidateLogPrefix, nodeIndex);
        }

        for (size_t predIndex = 0; predIndex < node.preds.size(); ++predIndex) {
            if (!validateNodeLink(log, nodeIndex, "preds", predIndex, node.preds[predIndex], nodeCount, sourceCount,
                                  "source")) {
                return false;
            }
        }
        for (size_t succIndex = 0; succIndex < node.succs.size(); ++succIndex) {
            if (!validateNodeLink(log, nodeIndex, "succs", succIndex, node.succs[succIndex], nodeCount, targetCount,
                                  "target")) {
                return false;
            }
        }
    }

    if (!validateNodeLinkSymmetry(log, nodes)) {
        return false;
    }

    // A value of -1 means the boundary connects directly without an in-edge
    // node. Any other value must be a valid index into edge.nodes.
    if (!validateBoundaryTouchNodes(log, "sourceEntryNodes", sourceEntryNodes, sources.size(), nodeCount)) {
        return false;
    }
    if (!validateBoundaryTouchNodes(log, "targetExitNodes", targetExitNodes, targets.size(), nodeCount)) {
        return false;
    }
    if (!validateBoundaryTouchConsistency(log, "sourceEntryNodes", sourceEntryNodes, nodes, /*isSource=*/true)) {
        return false;
    }
    if (!validateBoundaryTouchConsistency(log, "targetExitNodes", targetExitNodes, nodes, /*isSource=*/false)) {
        return false;
    }

    if (!validateEdgeType(log, kEdgeValidateLogPrefix, type, sources.size(), targets.size())) {
        return false;
    }

    return true;
}

}  // namespace VPUIP
}  // namespace vpux
