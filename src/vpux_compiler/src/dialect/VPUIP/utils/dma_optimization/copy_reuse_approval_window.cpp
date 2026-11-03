//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/copy_reuse_approval_window.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>

using namespace vpux;

namespace vpux::VPUIP::DmaOptimization {

namespace {

// Check if the kept copy is compatible with the sibling copy in terms of output buffer type and input view chain.
// Return true if the kept copy and the sibling copy is type compatible and have equivalent input view chains.
bool areCopyTransfersCompatible(const CopyReuseCandidate& keptCopy, const CopyReuseCandidate& sibling) {
    if (keptCopy.copy.getResult().getType() != sibling.copy.getResult().getType()) {
        return false;
    }
    return keptCopy.nodeDetails->hasEquivalentInputViewChain(*sibling.nodeDetails);
}

/// Check if every candidate is a single-use copy feeding a zero-offset weight table.
bool checkWeightTableGuards(llvm::ArrayRef<CopyReuseCandidate> candidates) {
    return llvm::all_of(candidates, [](const CopyReuseCandidate& candidate) {
        if (candidate.consumerOperands.size() != 1) {
            return false;
        }
        const auto& inputViewChain = *candidate.nodeDetails->inputViewChain;
        if (!mlir::isa_and_nonnull<Const::DeclareOp>(inputViewChain.source.getDefiningOp())) {
            return false;
        }
        auto* consumerOperand = candidate.consumerOperands.front();
        auto nce = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(consumerOperand->getOwner());
        return nce != nullptr && nce.getIsZeroOffsetWeightsTable() && candidate.copy->hasOneUse() &&
               consumerOperand->get() == candidate.copy.getResult() && consumerOperand->get() == nce.getWeightTable();
    });
}

//===----------------------------------------------------------------------===//
// Destination legality
//===----------------------------------------------------------------------===//

bool checkCopyReuseDestinationLegality(llvm::ArrayRef<CopyReuseCandidate> candidates) {
    return checkWeightTableGuards(candidates);
}

/// Get the output buffer owner of a sibling copy that can be safely erased.
/// Returns nullptr if the output buffer owner cannot be erased.
mlir::Operation* getErasableSiblingCopyOutputBufferOwner(const AliasesInfo& aliasesInfo,
                                                         const CopyReuseCandidate& sibling) {
    const auto siblingOutputBuffer = sibling.copy.getOutputBuff();
    if (mlir::isa<mlir::BlockArgument>(siblingOutputBuffer) || !siblingOutputBuffer.hasOneUse()) {
        return nullptr;
    }
    auto& onlyUse = *siblingOutputBuffer.use_begin();
    // CopyOp operand 1 is the output buffer.
    if (onlyUse.getOwner() != sibling.copy.getOperation() || onlyUse.getOperandNumber() != 1) {
        return nullptr;
    }
    auto* outputBufferDefiningOp = siblingOutputBuffer.getDefiningOp();
    if (outputBufferDefiningOp == nullptr || outputBufferDefiningOp->getNumResults() != 1) {
        return nullptr;
    }
    // Output buffer cannot be erased if it has multiple roots or aliases.
    const auto roots = aliasesInfo.getRoots(siblingOutputBuffer);
    if (roots.size() != 1 || roots.front() != siblingOutputBuffer) {
        return nullptr;
    }
    const auto aliases = aliasesInfo.getAllAliases(siblingOutputBuffer);
    if (llvm::any_of(aliases, [&](mlir::Value alias) {
            return alias != siblingOutputBuffer && alias != sibling.copy.getResult();
        })) {
        return nullptr;
    }
    return outputBufferDefiningOp;
}

}  // namespace

CopyReuseApprovalWindow::CopyReuseApprovalWindow(const AliasesInfo& aliasesInfo, Logger log)
        : _aliasesInfo(aliasesInfo), _log(log) {
}

SmallVector<ApprovedCopyReuse, 4> CopyReuseApprovalWindow::approve(
        llvm::ArrayRef<CopyReuseCandidate> candidates) const {
    SmallVector<ApprovedCopyReuse, 4> approvedReuses;
    llvm::DenseSet<mlir::Operation*> approvedCopyOps;

    for (size_t keptCopyIndex = 0; keptCopyIndex < candidates.size(); ++keptCopyIndex) {
        const auto& keptCandidate = candidates[keptCopyIndex];
        if (approvedCopyOps.contains(keptCandidate.copy.getOperation())) {
            continue;
        }
        const auto direction = getCopyReuseDirection(keptCandidate.copy);
        if (!direction.has_value()) {
            continue;
        }
        // Check copy direction same
        // Check copy and input view chain compatibility
        SmallVector<CopyReuseCandidate, 4> compatibleCopies{keptCandidate};
        for (const auto& candidate : llvm::drop_begin(candidates, keptCopyIndex + 1)) {
            if (approvedCopyOps.contains(candidate.copy.getOperation()) ||
                getCopyReuseDirection(candidate.copy) != direction ||
                !areCopyTransfersCompatible(keptCandidate, candidate)) {
                continue;
            }
            compatibleCopies.push_back(candidate);
        }
        if (compatibleCopies.size() < 2) {
            continue;
        }
        _log.trace("[CopyReuse] Step 1 transfer compatibility approved");

        // Check dest buffer legality
        //   a. WeightTable buffer should be single-consumer and zero-offset
        //   b. Weight buffer should be constant, no sparsity map and no compression
        //   c. Activation buffer should not have input_sparsity_map
        //   d. etc.
        // Now only check weight table
        if (!checkCopyReuseDestinationLegality(compatibleCopies)) {
            continue;
        }

        SmallVector<ApprovedCopyReuse, 4> approvedCopyReuse;
        const auto& keptCopy = compatibleCopies.front();
        bool allSiblingOutputBuffersCanBeRemoved = true;
        for (const auto& sibling : llvm::drop_begin(compatibleCopies)) {
            auto* deadOutputBufferDefiningOp = getErasableSiblingCopyOutputBufferOwner(_aliasesInfo, sibling);
            if (deadOutputBufferDefiningOp == nullptr) {
                allSiblingOutputBuffersCanBeRemoved = false;
                break;
            }

            approvedCopyReuse.push_back({sibling.copy, sibling.copy.getResult(), keptCopy.copy.getResult(),
                                         deadOutputBufferDefiningOp, sibling.nodeDetails->inputViewChain->views});
        }
        if (!allSiblingOutputBuffersCanBeRemoved) {
            continue;
        }

        approvedReuses.append(approvedCopyReuse);
        for (const auto& candidate : compatibleCopies) {
            approvedCopyOps.insert(candidate.copy.getOperation());
        }
        _log.trace("[CopyReuse] Step 2 destination legality approved");
    }

    return approvedReuses;
}

}  // namespace vpux::VPUIP::DmaOptimization
