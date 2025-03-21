//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_OPTIMIZEPARALLELCOPIES
#define GEN_PASS_DEF_OPTIMIZEPARALLELCOPIES
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
namespace {

// E130855: Fuse copy only its ComputeOp is less than 3 steps to previous ComputeOp
/*
    1% = NCE (buffer producer)      1% = NCE (buffer producer)

    2% = DMA %1                     2% = DMA %1
    3% = NCE %2                     3% = NCE %2

    4% = DMA %1
    5% = NCE %4                     5% = NCE %2
                    -->
    6% = DMA %1
    7% = NCE %6                     7% = NCE %2

    8% = DMA %1                     8% = DMA %1
    9% = NCE %8                     9% = NCE %8

    10% = DMA %1
    11% = NCE %10                   11% = NCE %8
*/
constexpr int32_t COMPUTE_OP_DISTANCE_COST = 3;
constexpr int32_t TWO_AXIS_DISTANCE_COST = 2;

//
// ParallelCopiesRewriter
//

class ParallelCopiesRewriter final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    ParallelCopiesRewriter(mlir::MLIRContext* ctx, Logger log, const bool enableOptimizeConstCopy,
                           const DenseMap<mlir::Operation*, uint32_t> computeOpPosition)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx),
              _log(log),
              _enableOptimizeConstCopy(enableOptimizeConstCopy),
              _computeOpPosition(computeOpPosition) {
        setDebugName("ParallelCopiesRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isLegalAndBenifitParallelCopiesRewriter(VPUIP::CopyOp origOp, NDTypeInterface inputType,
                                                 NDTypeInterface outputType, VPU::NCEInterpolateModeAttr modeAttr,
                                                 IE::InterpolateCoordModeAttr coordModeAttr) const;
    std::optional<uint32_t> getComputeOpPosition(mlir::Operation* op) const;
    bool isCopyFusable(VPUIP::CopyOp copyOp, Logger& log) const;
    void insertUserPosition(VPUIP::NCEClusterTaskOp nceConvUserOp, std::set<uint32_t>& positions) const;

    Logger _log;
    bool _enableOptimizeConstCopy;
    DenseMap<mlir::Operation*, uint32_t> _computeOpPosition;
    mutable mlir::DenseSet<VPUIP::CopyOp> _twoAxisTilingCache;
};

// Get compute operation user position of copy operation, currently only check in-place NCEEltwise & NCEConv.
// Return the position of compute operation, if not found, return std::nullopt;
std::optional<uint32_t> ParallelCopiesRewriter::getComputeOpPosition(mlir::Operation* op) const {
    uint32_t mininumPos = std::numeric_limits<uint32_t>::max();
    auto users = op->getUsers();
    for (const auto& user : users) {
        auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(user);
        if (nceOp != nullptr &&
            ((nceOp.getTaskType() == VPUIP::NCETaskType::ELTWISE && nceOp.getIsInplace().value_or(false)) ||
             nceOp.getTaskType() == VPUIP::NCETaskType::CONV)) {
            auto it = _computeOpPosition.find(nceOp);
            VPUX_THROW_WHEN(it == _computeOpPosition.end(), "Expected nceOp was not in _computeOpPosition map");
            // Get the closest ComputeOp to CopyOp
            if (it->second < mininumPos) {
                mininumPos = it->second;
            }
        }
    }
    if (mininumPos < std::numeric_limits<uint32_t>::max()) {
        return mininumPos;
    }
    return std::nullopt;
}

bool hasSiblingCopyFusable(VPUIP::SubViewOp subViewOp, VPUIP::CopyOp copyOp, mlir::Value buffer, Logger log) {
    auto parentOp = buffer.getDefiningOp();
    if (parentOp != nullptr && parentOp->getNumResults() <= 0) {
        log.trace("Is not fusable because parent does not have any consumers");
        return false;
    }
    for (auto siblingOp : buffer.getUsers()) {
        log.trace("Processing siblingOp {0}", siblingOp->getLoc());
        if (!mlir::isa<VPUIP::CopyOp>(*siblingOp)) {
            if (!mlir::isa<VPUIP::SubViewOp>(*siblingOp)) {
                continue;
            } else {
                // TODO: E#116963
                auto childOfSiblingOp = to_vector(siblingOp->getResult(0).getUsers()).back();
                if (!mlir::isa<VPUIP::CopyOp>(childOfSiblingOp)) {
                    continue;
                }
                // match SubView->Copy
                if (subViewOp == nullptr) {
                    continue;
                }
                auto siblingSubViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(siblingOp);
                if (parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets()) !=
                            parseIntArrayAttr<int64_t>(siblingSubViewOp.getStaticOffsets()) ||
                    parseIntArrayAttr<int64_t>(subViewOp.getStaticSizes()) !=
                            parseIntArrayAttr<int64_t>(siblingSubViewOp.getStaticSizes())) {
                    continue;
                }
                siblingOp = childOfSiblingOp;
            }
        }

        // Check 3: current op's consumers are copied to DDR immediately after execution
        for (const auto childOfSiblingOp : siblingOp->getResult(0).getUsers()) {
            log.trace("Processing childOfSiblingOp {0}", childOfSiblingOp->getLoc());
            if (childOfSiblingOp->use_empty()) {
                log.trace("Is not fusable because childOfSiblingOp haven't consumers");
                continue;
            }
            for (const auto grandChildOfSiblingOp : childOfSiblingOp->getResult(0).getUsers()) {
                auto concatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(grandChildOfSiblingOp);
                // If the ChildOfSiblingOp is a multi-shaveOp there will be a ConcatViewOp after ChildOfSiblingOp, skip
                // this ConcatViewOp and continue the optimization.
                auto childCopyOfSiblingOp =
                        (concatOp != nullptr) ? mlir::dyn_cast<VPUIP::CopyOp>(*(concatOp.getOutput().user_begin()))
                                              : mlir::dyn_cast<VPUIP::CopyOp>(grandChildOfSiblingOp);
                if (childCopyOfSiblingOp == nullptr) {
                    log.trace("Is not fusable because childOfSiblingOp is not CopyOp");
                    continue;
                }
                const auto input = childCopyOfSiblingOp.getInput().getType().cast<vpux::NDTypeInterface>();
                const auto output = childCopyOfSiblingOp.getOutput().getType().cast<vpux::NDTypeInterface>();
                if (input.getMemoryKind() != VPU::MemoryKind::CMX_NN ||
                    output.getMemoryKind() != VPU::MemoryKind::DDR) {
                    log.trace("Is not fusable because childCopyOfSiblingOp is not CMX->DDR copy");
                    return false;
                }
            }
        }

        if (siblingOp != copyOp) {
            return true;
        }
    }
    return false;
}

bool ParallelCopiesRewriter::isCopyFusable(VPUIP::CopyOp copyOp, Logger& log) const {
    // Check 1: copy DDR->CMX
    const auto srcMemory = copyOp.getInput().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    const auto dstMemory = copyOp.getOutput().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    if (srcMemory == dstMemory || srcMemory == VPU::MemoryKind::CMX_NN) {
        log.trace("Is not fusable because not DDR->CMX copy: {0}->{1}", srcMemory, dstMemory);
        return false;
    }

    auto copyUsers = copyOp->getUsers();
    for (auto* user : copyUsers) {
        while (VPUIP::isPureViewOp(user)) {
            if (mlir::isa<VPUIP::ConcatViewOp>(user)) {
                // If usage is through concat operation then optimization cannot be performed because
                // concat with different inputs requires different output buffers and each needs to be handled
                // by dedicated copy, which will refer to different output buffer
                log.trace("Is not fusable because user is concat op");
                return false;
            } else {
                if (user->getUsers().empty()) {
                    break;
                }
                user = *user->getUsers().begin();
            }
        }
    }

    // Optimize copies for weights. If serveral convolutions share same weights, the weight copies can be optimized with
    // single copy e.g. cases when the NCEOps that shares the same weights
    // Note that weight table and compressed convolution cannot apply this optimization. This is because
    // 1. for weight table, contents of weigthTable need to be adjusted with proper pointer value
    // 2. for compressed convolution, const data like weight also will be adjusted in ConvWeightsCompression pass,
    // will prevent the copy optimization.
    if (mlir::isa_and_nonnull<Const::DeclareOp>(copyOp.getInput().getDefiningOp())) {
        if (!_enableOptimizeConstCopy) {
            log.trace("Is not fusable because enableOptimizeConstCopy is not enabled");
            return false;
        }
        if (copyOp.getInput().getDefiningOp()->hasOneUse()) {
            log.trace("Is not fusable because has one use");
            return false;
        }

        auto copyOutput = copyOp.getOutput();
        for (const auto& user : copyUsers) {
            if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(user)) {
                if (VPUIP::hasBoundedBuffers(swKernelOp) || swKernelOp.getDynamicInputShapesMap().has_value()) {
                    log.trace("Is not fusable because swKernelOp is a dynamic operation");
                    return false;
                }
            }

            // TODO: to adjust codes for the two branchs with same condition.
            if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(user)) {
                if (nceOp.getWeights() != copyOutput || VPUIP::canWeightsBeCompressed(nceOp) ||
                    nceOp.getWeightsSparsityMap() != nullptr) {
                    log.trace("Is not fusable because copyOutput is not weights or weights can be compressed");
                    return false;
                }
            } else if (auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(user)) {
                // check for NCE multicluster task, no need for Shave multicluster task
                if (!vpux::VPUIP::hasDistributedOperand(nceTask)) {
                    continue;
                }
                auto weights = nceTask.getWeights();
                if (copyOutput != weights) {
                    log.trace("Is not fusable because copyOutput is not weights");
                    return false;
                }
                if (VPUIP::canTilingWeightsBeCompressed(nceTask)) {
                    log.trace("Is not fusable because tiling weights can be compressed");
                    return false;
                }
            }
        }
        return true;
    }

    auto subViewFusable = false;
    if (auto subViewOp = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(copyOp.getInput().getDefiningOp())) {
        subViewFusable = hasSiblingCopyFusable(subViewOp, copyOp, subViewOp.getSource(), log);
    }
    // We have 2 calls here, one to check if we have SubViewOp 1..n SubviewOp
    // Other for TilingCopy 1..n TilingCopy
    if (!subViewFusable && !hasSiblingCopyFusable(nullptr, copyOp, copyOp.getInput(), log)) {
        log.trace("Is not fusable because doesn't have fusable sibling");
        return false;
    }

    if (!_twoAxisTilingCache.empty()) {
        if (_twoAxisTilingCache.contains(copyOp)) {
            return false;
        }
    }

    return true;
}

bool arePositionsConsecutive(const std::set<uint32_t>& positions) {
    if (positions.size() < 2) {
        return false;
    }

    auto it = positions.begin();
    auto prev = *it;
    ++it;
    for (; it != positions.end(); ++it) {
        if (*it != prev + 1) {
            return false;
        }
        prev = *it;
    }
    return true;
}

void ParallelCopiesRewriter::insertUserPosition(VPUIP::NCEClusterTaskOp nceConvUserOp,
                                                std::set<uint32_t>& positions) const {
    uint32_t userOpPos = std::numeric_limits<uint32_t>::max();
    userOpPos = _computeOpPosition.find(nceConvUserOp)->second;
    positions.insert(userOpPos);
}

mlir::LogicalResult ParallelCopiesRewriter::matchAndRewrite(VPUIP::CopyOp originCopyOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), originCopyOp->getName(), originCopyOp->getLoc());

    if (originCopyOp->use_empty()) {
        return mlir::failure();
    }

    auto nestedLogger = _log.nest();
    if (!isCopyFusable(originCopyOp, nestedLogger)) {
        return mlir::failure();
    }

    bool isClusterCopy = vpux::VPUIP::hasDistributedOperand(originCopyOp);

    const auto isSubViewSameFunc = [](VPUIP::SubViewOp srcSubView, VPUIP::SubViewOp siblingSubView) {
        if (srcSubView == siblingSubView) {
            return false;
        }

        return (srcSubView.getStaticOffsets() == siblingSubView.getStaticOffsets()) &&
               (srcSubView.getStaticSizes() == siblingSubView.getStaticSizes()) &&
               (srcSubView.getStaticStrides() == siblingSubView.getStaticStrides());
    };

    const auto isCopySameFunc = [&](VPUIP::CopyOp srcCopyOp, mlir::Operation* op) {
        if (vpux::VPUIP::hasDistributedOperand(srcCopyOp) != vpux::VPUIP::hasDistributedOperand(op)) {
            return false;
        }

        auto siblingCopy = mlir::dyn_cast<VPUIP::CopyOp>(op);
        if (siblingCopy == nullptr) {
            return false;
        }

        if (srcCopyOp.getResult().getType() != op->getResult(0).getType()) {
            return false;
        }

        auto srcSubView = srcCopyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
        auto siblingSubView = siblingCopy.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
        if (isClusterCopy && vpux::VPUIP::hasDistributedOperand(op)) {
            srcSubView = srcCopyOp.getOutput().getDefiningOp<VPUIP::SubViewOp>();
            siblingSubView = siblingCopy.getOutputs()[0].getDefiningOp<VPUIP::SubViewOp>();
        }

        if (srcSubView != nullptr && siblingSubView != nullptr && isSubViewSameFunc(srcSubView, siblingSubView)) {
            return true;
        }

        if (srcSubView == nullptr && siblingSubView == nullptr) {
            return true;
        }

        return false;
    };

    const auto isSameCopyFunc = [&](VPUIP::CopyOp srcCopyOp, mlir::Operation* op) {
        VPUX_THROW_WHEN(srcCopyOp == nullptr, "Expected CopyOp and op to be valid");
        auto siblingCopy = mlir::dyn_cast<VPUIP::CopyOp>(op);
        return siblingCopy != nullptr && siblingCopy == srcCopyOp;
    };

    VPUIP::CopyOp prevCopyOp = nullptr;
    VPUIP::CopyOp newRootCopyOp = originCopyOp;
    uint32_t invalidPostion = std::numeric_limits<uint32_t>::max();
    uint32_t prevComputePostion = invalidPostion;
    uint32_t prevComputePostion4TwoAxis = invalidPostion;

    auto getSiblingEltwise = [&](mlir::Operation* rootCopyOp) -> bool {
        if (!getComputeOpPosition(rootCopyOp).has_value()) {
            return false;
        }

        auto parentOp = rootCopyOp->getOperand(0).getDefiningOp();
        if (parentOp == nullptr) {
            return false;
        }
        auto curParentOp = mlir::isa<VPUIP::SubViewOp>(parentOp) ? parentOp->getOperand(0).getDefiningOp() : parentOp;
        if (curParentOp == nullptr) {
            return false;
        }

        auto parentOpUsers = to_small_vector(curParentOp->getResult(0).getUsers());
        for (auto* user : llvm::make_early_inc_range(parentOpUsers | reversed)) {
            SmallVector<mlir::Operation*> siblingComputeOps;
            if (mlir::isa<VPUIP::SubViewOp>(user)) {
                for (auto siblingCopy : user->getUsers()) {
                    for (auto item : siblingCopy->getUsers()) {
                        siblingComputeOps.push_back(item);
                    }
                }
            } else {
                for (auto item : user->getUsers()) {
                    siblingComputeOps.push_back(item);
                }
            }

            for (const auto& mayComputeOp : siblingComputeOps) {
                if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(mayComputeOp)) {
                    if (nceOp.getTaskType() == VPUIP::NCETaskType::ELTWISE && nceOp.getIsInplace().value_or(false)) {
                        return true;
                    }
                }
            }
        }
        return false;
    };

    auto checkSiblingCopies = [&](mlir::Operation* targetOp) -> bool {
        auto rootCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(targetOp);
        if (rootCopyOp == nullptr) {
            return false;
        }
        const auto srcMemory = rootCopyOp.getInput().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
        const auto dstMemory = rootCopyOp.getOutput().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
        if (srcMemory == dstMemory || srcMemory == VPU::MemoryKind::CMX_NN) {
            return false;
        }

        auto users = rootCopyOp->getUsers();
        std::set<uint32_t> positions;

        // 1. Check if the users of rootCopyOp are consecutive CONVs
        for (auto* userOp : users) {
            auto nceConvUserOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(userOp);
            if (nceConvUserOp == nullptr || nceConvUserOp.getTaskType() != VPUIP::NCETaskType::CONV) {
                return false;
            }
            insertUserPosition(nceConvUserOp, positions);
        }

        if (positions.size() >= 2) {
            return arePositionsConsecutive(positions);
        }

        positions.clear();
        auto parentOp = rootCopyOp.getInput().getDefiningOp();
        // 2. check if the weight/input is from subview, and the subview parent has sibling subview copies,
        //    which are used by consecutive CONVs
        if (auto subViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(parentOp)) {
            if (auto subViewParentOp = subViewOp.getSource().getDefiningOp()) {
                auto parentOpUsers = to_small_vector(subViewParentOp->getResult(0).getUsers());

                for (auto* siblingOp : llvm::make_early_inc_range(parentOpUsers | reversed)) {
                    auto siblingSubViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(siblingOp);
                    if (siblingSubViewOp == nullptr ||
                        (!isSubViewSameFunc(subViewOp, siblingSubViewOp) && subViewOp != siblingSubViewOp)) {
                        continue;
                    }
                    auto siblingCopyOp = *siblingSubViewOp.getResult().getUsers().begin();
                    if (isCopySameFunc(rootCopyOp, siblingCopyOp)) {
                        auto user = *siblingCopyOp->getUsers().begin();
                        auto nceConvUserOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(user);
                        if (nceConvUserOp == nullptr || nceConvUserOp.getTaskType() != VPUIP::NCETaskType::CONV) {
                            return false;
                        }
                        insertUserPosition(nceConvUserOp, positions);
                    }
                }

                return arePositionsConsecutive(positions);
            }
        }

        // 3. check if the weight/input parent has sibling copies, which are used by consecutive CONVs
        auto parentOpUsers = to_small_vector(parentOp->getResult(0).getUsers());
        positions.clear();
        for (auto* siblingOp : llvm::make_early_inc_range(parentOpUsers | reversed)) {
            if (isSameCopyFunc(rootCopyOp, siblingOp)) {
                auto user = *siblingOp->getUsers().begin();
                auto nceConvUserOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(user);
                if (nceConvUserOp == nullptr || nceConvUserOp.getTaskType() != VPUIP::NCETaskType::CONV) {
                    return false;
                }
                insertUserPosition(nceConvUserOp, positions);
            }
        }

        return arePositionsConsecutive(positions);
    };

    auto isTilingOnTwoAxis = [&](mlir::Operation* rootCopyOp) -> bool {
        auto rootCopyOpAsCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(rootCopyOp);
        if (rootCopyOpAsCopyOp == nullptr) {
            return false;
        }

        if (!_twoAxisTilingCache.empty()) {
            if (_twoAxisTilingCache.contains(rootCopyOpAsCopyOp)) {
                return true;
            }
        }

        auto user = *rootCopyOp->getUsers().begin();
        auto nceConvUserOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(user);
        if (nceConvUserOp == nullptr || nceConvUserOp.getTaskType() != VPUIP::NCETaskType::CONV) {
            return false;
        }

        auto nceConvUserOpWeight = nceConvUserOp.getWeights().getDefiningOp();
        auto nceConvUserOpInput = nceConvUserOp.getInput().getDefiningOp();

        if (nceConvUserOpWeight == nullptr || nceConvUserOpInput == nullptr) {
            return false;
        }

        if (isSameCopyFunc(rootCopyOpAsCopyOp, nceConvUserOpWeight)) {
            return checkSiblingCopies(nceConvUserOpInput);
        } else if (isSameCopyFunc(rootCopyOpAsCopyOp, nceConvUserOpInput)) {
            return checkSiblingCopies(nceConvUserOpWeight);
        }

        return false;
    };

    // To confirm there is NCE::ELTWISE in target pattern
    auto hasEltwiseUser = getSiblingEltwise(originCopyOp);

    const auto isWithinCostDistance = [&](mlir::Operation* op, bool isTilingOnTwoAxis) {
        // Cost-based optimize copy strategy: Check if the user of the CopyOp is a computeOp,
        // currently only supporting NCE::ELTWISE & NCE::CONV. If it is a computeOp, examine the distance
        // between adjacent computeOps. If the distance is less than or equal to 3, the CopyOp
        // between computeOps can be optimized; otherwise, it will be retained.
        // The rationale behind this is to prevent the optimization of all copies from resulting
        // in an excessively long buffer memory live range, which would lead to continuous
        // occupation of CMX without the possibility of release.
        // There are still some regressions when target pattern are conv compute ops only, so ELTWISE only or
        // CONV-ELTWISE mixed patterns are support now.
        if (hasEltwiseUser) {
            auto currComputePosition = getComputeOpPosition(op);
            if (currComputePosition.has_value()) {
                bool closeToPrev = prevComputePostion != invalidPostion &&
                                   std::abs(static_cast<int>(currComputePosition.value() - prevComputePostion)) <
                                           COMPUTE_OP_DISTANCE_COST;
                if (!closeToPrev || isSameCopyFunc(originCopyOp, op)) {
                    prevComputePostion = currComputePosition.value();
                    prevCopyOp = mlir::cast<VPUIP::CopyOp>(op);
                    return false;
                }
                // update newRootCopyOp with prevCopyOp
                newRootCopyOp = mlir::cast<VPUIP::CopyOp>(prevCopyOp);
            }
        }

        // If two-dimensional tiling undergoes copy fusion, it can easily cause the weight/input
        // to occupy CMX for an extended period, leading to significant spilling.
        if (isTilingOnTwoAxis) {
            auto currComputePosition = getComputeOpPosition(op);
            if (currComputePosition.has_value()) {
                bool closeToPrev =
                        prevComputePostion4TwoAxis != invalidPostion &&
                        std::abs(static_cast<int>(currComputePosition.value() - prevComputePostion4TwoAxis)) <
                                TWO_AXIS_DISTANCE_COST;
                if (prevComputePostion4TwoAxis == invalidPostion) {
                    prevComputePostion4TwoAxis = currComputePosition.value();
                }

                if (!closeToPrev) {
                    auto copyop = mlir::dyn_cast<VPUIP::CopyOp>(op);
                    this->_twoAxisTilingCache.insert(copyop);
                    return false;
                }

                prevComputePostion4TwoAxis = currComputePosition.value();
            }
        }
        // If op is not targeted ComputeOps, skip by return true
        // If the computeOp is within the cost step, return true
        return true;
    };

    const auto updateSiblingCopyOutputBuff = [&](VPUIP::CopyOp srcCopyOp, mlir::Operation* op) {
        // Get the sibling copy
        auto siblingCopy = mlir::dyn_cast<VPUIP::CopyOp>(op);
        if (siblingCopy == nullptr) {
            nestedLogger.trace("Sibling op is not copy at {0}", op->getLoc());
            return;
        }
        // Get the buffer linked to copy output
        auto copyOpOutputBuff = srcCopyOp.getOutputBuff();
        // Get the buffer linked to sibling copy output that will be fused
        auto siblingCopyOutputBuff = siblingCopy.getOutputBuff();
        // Replace the usage of sibling copy output buffer with copy output buffer
        rewriter.replaceAllUsesWith(siblingCopyOutputBuff, copyOpOutputBuff);
    };

    auto sharedBuffer = newRootCopyOp.getInput();
    bool hasReplaceParallelCopies = false;
    bool isRootCopyTilingOnTwoAxis = isTilingOnTwoAxis(newRootCopyOp);
    if (isRootCopyTilingOnTwoAxis) {
        auto originCopyOpComputePosition = getComputeOpPosition(originCopyOp.getOperation());
        if (originCopyOpComputePosition.has_value()) {
            prevComputePostion4TwoAxis = originCopyOpComputePosition.value();
        }
    }
    // Optimize pattern: SharedBuffer -> ParentOp(SubView) -> CopyOp
    if (auto subViewOp = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(sharedBuffer.getDefiningOp())) {
        auto parentOpusers = to_small_vector(subViewOp.getSource().getUsers());
        for (auto* siblingOp : llvm::make_early_inc_range(parentOpusers | reversed)) {
            auto siblingSubViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(siblingOp);
            if (siblingSubViewOp == nullptr || !isSubViewSameFunc(subViewOp, siblingSubViewOp)) {
                continue;
            }

            auto siblingCopyOp = *siblingSubViewOp.getResult().getUsers().begin();
            if (!isCopySameFunc(newRootCopyOp, siblingCopyOp) ||
                !isWithinCostDistance(siblingCopyOp, isRootCopyTilingOnTwoAxis) ||
                isSameCopyFunc(newRootCopyOp, siblingCopyOp)) {
                continue;
            }

            nestedLogger.trace("Fuse SubView op {0} to {1}", siblingSubViewOp->getLoc(), subViewOp->getLoc());
            updateSiblingCopyOutputBuff(newRootCopyOp, siblingCopyOp);
            rewriter.replaceAllUsesWith(siblingSubViewOp->getResult(0), subViewOp->getResult(0));
            rewriter.replaceAllUsesWith(siblingCopyOp->getResult(0), newRootCopyOp->getResult(0));
            rewriter.eraseOp(siblingCopyOp);
            rewriter.eraseOp(siblingSubViewOp);
            hasReplaceParallelCopies = true;

            for (auto user : newRootCopyOp->getResult(0).getUsers()) {
                if (user->isBeforeInBlock(newRootCopyOp)) {
                    newRootCopyOp->moveBefore(user);
                }
            }
            for (auto user : subViewOp->getResult(0).getUsers()) {
                if (user->isBeforeInBlock(subViewOp)) {
                    subViewOp->moveBefore(user);
                }
            }
            auto copyOpOutputBuff = newRootCopyOp.getOutputBuff();
            for (auto user : copyOpOutputBuff.getUsers()) {
                if (user->isBeforeInBlock(copyOpOutputBuff.getDefiningOp())) {
                    copyOpOutputBuff.getDefiningOp()->moveBefore(user);
                }
            }
        }
    }

    // Optimize pattern: ParentOp -> CopyOp
    prevCopyOp = nullptr;
    prevComputePostion = std::numeric_limits<uint32_t>::max();
    auto parentOpUsers = to_small_vector(sharedBuffer.getUsers());
    isRootCopyTilingOnTwoAxis = isTilingOnTwoAxis(newRootCopyOp);
    if (isRootCopyTilingOnTwoAxis) {
        auto originCopyOpComputePosition = getComputeOpPosition(newRootCopyOp.getOperation());
        if (originCopyOpComputePosition.has_value()) {
            prevComputePostion4TwoAxis = originCopyOpComputePosition.value();
        }
    }
    for (auto* siblingOp : llvm::make_early_inc_range(parentOpUsers | reversed)) {
        if (!isCopySameFunc(newRootCopyOp, siblingOp) || !isWithinCostDistance(siblingOp, isRootCopyTilingOnTwoAxis) ||
            isSameCopyFunc(newRootCopyOp, siblingOp)) {
            continue;
        }

        updateSiblingCopyOutputBuff(newRootCopyOp, siblingOp);
        if (!isClusterCopy) {
            auto siblingCopy = mlir::dyn_cast<VPUIP::CopyOp>(siblingOp);
            nestedLogger.trace("Fuse Copy op {0} to {1}", siblingCopy->getLoc(), newRootCopyOp->getLoc());

            rewriter.replaceAllUsesWith(siblingCopy->getResult(0), newRootCopyOp->getResult(0));
        } else if (vpux::VPUIP::hasDistributedOperand(siblingOp)) {
            nestedLogger.trace("Fuse distributed Copy op {0} to {1}", siblingOp->getLoc(), newRootCopyOp->getLoc());

            rewriter.replaceAllUsesWith(siblingOp->getResult(0), newRootCopyOp->getResult(0));
        }
        rewriter.eraseOp(siblingOp);
        hasReplaceParallelCopies = true;

        for (auto user : newRootCopyOp->getResult(0).getUsers()) {
            if (user->isBeforeInBlock(newRootCopyOp)) {
                newRootCopyOp->moveBefore(user);
            }
        }
        auto copyOpOutputBuff = newRootCopyOp.getOutputBuff();
        for (auto user : copyOpOutputBuff.getUsers()) {
            if (user->isBeforeInBlock(copyOpOutputBuff.getDefiningOp())) {
                copyOpOutputBuff.getDefiningOp()->moveBefore(user);
            }
        }
    }

    return mlir::success(hasReplaceParallelCopies);
}

//
// OptimizeParallelCopiesPass
//

class OptimizeParallelCopiesPass final : public VPUIP::impl::OptimizeParallelCopiesBase<OptimizeParallelCopiesPass> {
public:
    explicit OptimizeParallelCopiesPass(bool enableOptimizeConstCopy, Logger log)
            : _enableOptimizeConstCopy(enableOptimizeConstCopy) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    bool _enableOptimizeConstCopy;

    auto getDistanceMap();
    void safeRunOnFunc() final;
};

auto OptimizeParallelCopiesPass::getDistanceMap() {
    // E131418: Current solution scans computeOp following IR order, which is temporary solution
    // In real case, the operation is a tree structure which may contain multiple opreations
    // in same level, for example
    /*
    //                   DMA
    //                    |             -- level 0
    //                   NCE
    //                /   |   \
    //             DMA   DMA  DMA
    //              |     |    |
    //             NCE   NCE  NCE       -- level 1
    // Extense the tree structure when storing the position of computeOp in map
    */

    DenseMap<mlir::Operation*, uint32_t> computeOpPosition;
    auto func = getOperation();
    uint32_t pos = 0;
    func->walk([&](mlir::Operation* op) {
        if (mlir::isa<VPUIP::NCEClusterTaskOp, VPUIP::SwKernelOp>(op)) {
            computeOpPosition.insert({op, pos++});
        };
    });

    return computeOpPosition;
}

void OptimizeParallelCopiesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto computeOpPosition = getDistanceMap();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ParallelCopiesRewriter>(&ctx, _log, _enableOptimizeConstCopy, computeOpPosition);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createOptimizeParallelCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeParallelCopiesPass(bool enableOptimizeConstCopy, Logger log) {
    return std::make_unique<OptimizeParallelCopiesPass>(enableOptimizeConstCopy, log);
}
