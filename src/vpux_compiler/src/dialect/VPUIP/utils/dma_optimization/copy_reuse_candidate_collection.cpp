//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/copy_reuse_candidate_collection.hpp"

#include "vpux/compiler/core/dma_cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/edge_collector.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

using namespace vpux;

namespace vpux::VPUIP::DmaOptimization {

std::optional<DmaDirectionType> getCopyReuseDirection(VPUIP::CopyOp copy) {
    const auto sourceType = mlir::dyn_cast<NDTypeInterface>(copy.getInput().getType());
    const auto destinationType = mlir::dyn_cast<NDTypeInterface>(copy.getOutput().getType());
    if (sourceType == nullptr || destinationType == nullptr ||
        sourceType.getMemoryKind() == destinationType.getMemoryKind()) {
        return std::nullopt;
    }
    if (sourceType.getMemoryKind() == VPU::MemoryKind::DDR &&
        destinationType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
        return DmaDirectionType::DDR_TO_CMX;
    }
    // E-172833: CMX_TO_DDR, DDR_TO_DDR, CMX_TO_CMX will be support soon.
    return std::nullopt;
}

namespace {

const EdgeBoundary* getTargetBoundary(const Edge& edge, int32_t successor) {
    if (successor >= 0) {
        return nullptr;
    }
    const auto targetIndex = static_cast<size_t>(~successor);
    return targetIndex < edge.targets.size() ? &edge.targets[targetIndex] : nullptr;
}

mlir::OpOperand* getTargetOperand(mlir::Value value, const EdgeBoundary* target) {
    if (target == nullptr || target->op == nullptr || target->operandIdx >= target->op->getNumOperands()) {
        return nullptr;
    }
    auto* targetOperand = &target->op->getOpOperand(target->operandIdx);
    return targetOperand->get() == value ? targetOperand : nullptr;
}

std::optional<CopyReuseCandidate> buildCopyReuseCandidate(VPUIP::CopyOp copy,
                                                          SmallVector<mlir::OpOperand*, 1> consumerOperands,
                                                          const EdgeNode& node) {
    if (!node.details.inputViewChain.has_value() || consumerOperands.empty()) {
        return std::nullopt;
    }
    return CopyReuseCandidate{copy, std::move(consumerOperands), &node.details};
}

std::optional<CopyReuseCandidate> buildDdrToCmxFanOutCandidate(VPUIP::CopyOp copy, const EdgeNode& node,
                                                               const Edge& edge) {
    SmallVector<mlir::OpOperand*, 1> consumerOperands;

    for (auto successor : node.succs) {
        auto value = copy.getResult();
        // Traverse the chain of view/slice operations starting from the copy's result.
        // Support:
        // Shared buffer -> DDR2CMX -> viewOps -> computeOp0
        //               -> DDR2CMX -> viewOps -> computeOp1
        while (successor >= 0) {
            const auto successorIndex = static_cast<size_t>(successor);
            if (successorIndex >= edge.nodes.size()) {
                return std::nullopt;
            }
            const auto& userNode = edge.nodes[successorIndex];
            auto* maybeViewOp = userNode.op;
            if (maybeViewOp == nullptr || (!userNode.details.isSlice() && !userNode.details.isViewOp()) ||
                maybeViewOp->getOperand(0) != value || userNode.succs.size() != 1) {
                return std::nullopt;
            }
            value = maybeViewOp->getResult(0);
            successor = userNode.succs.front();
        }
        // Find and check the target compute boundary
        // Make sure copy result is used by a compute operation
        const auto* target = getTargetBoundary(edge, successor);
        auto* targetOperand = getTargetOperand(value, target);
        if (targetOperand == nullptr || target->kind != EdgeBoundary::Kind::ComputeOp) {
            return std::nullopt;
        }
        consumerOperands.push_back(targetOperand);
    }
    return buildCopyReuseCandidate(copy, std::move(consumerOperands), node);
}

void appendFanOutLikeCopyCandidates(const Edge& edge, SmallVectorImpl<CopyReuseCandidate>& candidates) {
    if (!edge.isFanOut()) {
        return;
    }
    for (const auto& node : edge.nodes) {
        auto copy = mlir::dyn_cast_if_present<VPUIP::CopyOp>(node.op);
        if (copy == nullptr || node.succs.empty()) {
            continue;
        }
        const auto direction = getCopyReuseDirection(copy);
        if (!direction.has_value()) {
            continue;
        }
        // Only support DDR2CMX optimization for now
        auto candidate = buildDdrToCmxFanOutCandidate(copy, node, edge);
        if (candidate.has_value()) {
            candidates.push_back(std::move(*candidate));
        }
    }
}

}  // namespace

SmallVector<CopyReuseCandidate, 16> collectCopyReuseCandidates(const EdgeCollection& edgeCollection) {
    SmallVector<CopyReuseCandidate, 16> candidates;
    for (const auto& edge : edgeCollection.edges) {
        appendFanOutLikeCopyCandidates(edge, candidates);
    }
    return candidates;
}

}  // namespace vpux::VPUIP::DmaOptimization
