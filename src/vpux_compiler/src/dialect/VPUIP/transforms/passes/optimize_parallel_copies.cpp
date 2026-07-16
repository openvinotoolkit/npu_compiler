//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/sub_byte.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/SetVector.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_OPTIMIZEPARALLELCOPIES
#define GEN_PASS_DEF_OPTIMIZEPARALLELCOPIES
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
namespace {

// E130855: Fuse copy only when its ComputeOp is within 3 steps of the previous ComputeOp.
// The compute-distance heuristic is applied when at least one sibling compute op is in-place ELTWISE.
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
    ParallelCopiesRewriter(mlir::MLIRContext* ctx, const Logger& log, const bool enableOptimizeConstCopy,
                           const DenseMap<mlir::Operation*, uint32_t>& computeOpPosition,
                           const DenseMap<uint32_t, std::optional<uint32_t>>& tiledOpNearestDistances)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx),
              _log(log),
              _enableOptimizeConstCopy(enableOptimizeConstCopy),
              _computeOpPosition(computeOpPosition),
              _tiledOpNearestDistances(tiledOpNearestDistances) {
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
    DenseMap<uint32_t, std::optional<uint32_t>> _tiledOpNearestDistances;
    mutable mlir::DenseSet<VPUIP::CopyOp> _twoAxisTilingCache;
    mutable DenseMap<mlir::Value, bool> _fullBufferReloadCopyCache;
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

// Detect local repeated Subview -> Copy groups that represent a full input buffer.
// If the first repeated group already needs at least the available CMX size, reloading the buffer for the second group
// is beneficial to avoid spilling.
// e.g. for 2 repeated groups, we have:
//                 Input buffer
//     |        |               |        |
//  SubViewA SubViewB       SubViewA' SubViewB'
//     |        |               |        |
//   CopyA    CopyB          CopyA'    CopyB'
// If CopyA and CopyB together need a large CMX size, CopyA' and CopyB' are required to reload the buffers.
bool isBeneficialToKeepCopy(ArrayRef<mlir::Operation*> parentOpUsers, VPUIP::CopyOp originCopyOp) {
    if (parentOpUsers.empty()) {
        return false;
    }

    auto sortedParentOpUsers = to_small_vector(parentOpUsers);
    llvm::sort(sortedParentOpUsers, [](mlir::Operation* lhs, mlir::Operation* rhs) {
        return lhs->isBeforeInBlock(rhs);
    });

    const auto isSameSubViewRange = [](VPUIP::SubViewOp lhs, VPUIP::SubViewOp rhs) {
        return (lhs.getStaticOffsets() == rhs.getStaticOffsets()) && (lhs.getStaticSizes() == rhs.getStaticSizes()) &&
               (lhs.getStaticStrides() == rhs.getStaticStrides());
    };

    const auto getCopyUser = [](VPUIP::SubViewOp subViewOp) -> VPUIP::CopyOp {
        for (auto* user : subViewOp.getResult().getUsers()) {
            if (auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(user)) {
                return copyOp;
            }
        }
        return nullptr;
    };

    // Keep only SubView -> Copy entries before detecting full-buffer reload cycles.
    SmallVector<std::pair<VPUIP::SubViewOp, VPUIP::CopyOp>> subViewCopies;
    for (auto* parentUser : sortedParentOpUsers) {
        auto subViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(parentUser);
        // Reject if the SubView has multiple uses since it's beneficial to share one buffer.
        //    SubViewA       SubViewB       SubViewA  SubViewB
        //    |      |       |      |  =>      |        |
        //   Copy   Copy   Copy   Copy       Copy      Copy
        if (subViewOp == nullptr || !parentUser->hasOneUse()) {
            return false;
        }

        auto copyOp = getCopyUser(subViewOp);
        if (copyOp == nullptr) {
            continue;
        }

        subViewCopies.push_back({subViewOp, copyOp});
    }
    if (subViewCopies.size() < 2) {
        return false;
    }

    const auto arch = config::getArch(originCopyOp);
    const auto availableCMXPerCluster = VPU::getTotalCMXSize(originCopyOp);

    const auto repeatedRangeExceedsCmx =
            [&](ArrayRef<std::pair<VPUIP::SubViewOp, VPUIP::CopyOp>> repeatedRange) -> bool {
        SmallVector<Byte> firstCycleBufferSizes;
        for (auto subViewCopy : repeatedRange) {
            auto copyOp = subViewCopy.second;

            const auto copyOutputType = mlir::cast<vpux::NDTypeInterface>(copyOp.getOutput().getType());
            if (copyOutputType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
                return false;
            }

            firstCycleBufferSizes.push_back(copyOutputType.getTotalAllocSize());
        }

        const auto requiredCMX = VPU::calculateAlignedBuffersMemoryRequirement(arch, firstCycleBufferSizes);
        return requiredCMX >= availableCMXPerCluster;
    };

    // Adjacent duplicate ranges such as A A B B is expected to fuse Copy.
    for (size_t index = 1; index < subViewCopies.size(); ++index) {
        if (isSameSubViewRange(subViewCopies[index - 1].first, subViewCopies[index].first)) {
            return false;
        }
    }

    const auto subViewCopyCount = subViewCopies.size();
    SmallVector<SmallVector<size_t>> commonRangeLengths(subViewCopyCount + 1);
    for (auto& row : commonRangeLengths) {
        row.resize(subViewCopyCount + 1, 0);
    }

    for (size_t lhsPos = subViewCopyCount; lhsPos > 0; --lhsPos) {
        const auto lhsIndex = lhsPos - 1;
        for (size_t rhsPos = subViewCopyCount; rhsPos > 0; --rhsPos) {
            const auto rhsIndex = rhsPos - 1;
            if (isSameSubViewRange(subViewCopies[lhsIndex].first, subViewCopies[rhsIndex].first)) {
                commonRangeLengths[lhsIndex][rhsIndex] = commonRangeLengths[lhsIndex + 1][rhsIndex + 1] + 1;
            }
        }
    }

    const auto isSameRangeAt = [&](size_t lhsIndex, size_t rhsIndex, size_t rangeSize) -> bool {
        return commonRangeLengths[lhsIndex][rhsIndex] >= rangeSize;
    };

    // Detect repeated cycles and check the first cycle:
    // A B A B: cycle is A B.
    // A B A B C D C D: cycles are A B and C D.
    for (size_t startIndex = 0; startIndex < subViewCopyCount;) {
        std::optional<size_t> repeatedCycleSize = std::nullopt;
        const auto remainingSize = subViewCopyCount - startIndex;
        for (size_t cycleSize = 1; cycleSize <= remainingSize / 2; ++cycleSize) {
            if (isSameRangeAt(startIndex, startIndex + cycleSize, cycleSize)) {
                repeatedCycleSize = cycleSize;
                break;
            }
        }

        if (!repeatedCycleSize.has_value()) {
            ++startIndex;
            continue;
        }

        const auto cycleSize = repeatedCycleSize.value();
        const auto cycleStart = startIndex;
        if (repeatedRangeExceedsCmx(ArrayRef(subViewCopies).slice(cycleStart, cycleSize))) {
            return true;
        }
        startIndex += 2 * cycleSize;
        while (startIndex + cycleSize <= subViewCopyCount && isSameRangeAt(cycleStart, startIndex, cycleSize)) {
            startIndex += cycleSize;
        }
    }

    return false;
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
                // If the ChildOfSiblingOp is a multi-shaveOp there will be a ConcatViewOp after ChildOfSiblingOp,
                // skip this ConcatViewOp and continue the optimization.
                auto childCopyOfSiblingOp =
                        (concatOp != nullptr) ? mlir::dyn_cast<VPUIP::CopyOp>(*(concatOp.getOutput().user_begin()))
                                              : mlir::dyn_cast<VPUIP::CopyOp>(grandChildOfSiblingOp);
                if (childCopyOfSiblingOp == nullptr) {
                    log.trace("Is not fusable because childOfSiblingOp is not CopyOp");
                    continue;
                }
                const auto input = mlir::cast<vpux::NDTypeInterface>(childCopyOfSiblingOp.getInput().getType());
                const auto output = mlir::cast<vpux::NDTypeInterface>(childCopyOfSiblingOp.getOutput().getType());
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
    const auto srcMemory = mlir::cast<vpux::NDTypeInterface>(copyOp.getInput().getType()).getMemoryKind();
    const auto dstMemory = mlir::cast<vpux::NDTypeInterface>(copyOp.getOutput().getType()).getMemoryKind();
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

    // Optimize copies for weights. If several convolutions share same weights, the weight copies can be optimized
    // with single copy e.g. cases when the NCEOps that share the same weights Note that weight table and
    // compressed convolution cannot apply this optimization. This is because
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

            // Due to some regression, we limited the optimization for low bit weights, and only for the inputs like
            // scale/bias/pallet table
            if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(user)) {
                if (nceOp.getWeights() != nullptr) {
                    auto weightType = mlir::cast<vpux::NDTypeInterface>(nceOp.getWeights().getType());
                    auto elementSize = vpux::getElemTypeSize(weightType.getElementType());
                    if ((copyOutput == nceOp.getWeightTableBias() || copyOutput == nceOp.getWeightTableScale() ||
                         copyOutput == nceOp.getPalletLookupTable())) {
                        if (vpux::Const::isSubByte(elementSize.count())) {
                            continue;
                        } else {
                            return false;
                        }
                    }
                }

                if (nceOp.getWeights() != copyOutput || VPUIP::canWeightsBeCompressed(nceOp) ||
                    nceOp.getWeightsSparsityMap() != nullptr) {
                    log.trace("Is not fusable because copyOutput is not weights or weights can be compressed");
                    return false;
                }
            }
        }
        return true;
    }

    auto subViewFusable = false;
    if (auto subViewOp = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(copyOp.getInput().getDefiningOp())) {
        auto subViewSource = subViewOp.getSource();
        // Check pattern SubViewOp 1..n SubViewOp.
        subViewFusable = hasSiblingCopyFusable(subViewOp, copyOp, subViewSource, log);

        if (subViewFusable) {
            auto keepCopyIt = _fullBufferReloadCopyCache.find(subViewSource);
            if (keepCopyIt == _fullBufferReloadCopyCache.end()) {
                auto parentOpusers = to_small_vector(subViewSource.getUsers());
                keepCopyIt = _fullBufferReloadCopyCache
                                     .try_emplace(subViewSource, isBeneficialToKeepCopy(parentOpusers, copyOp))
                                     .first;
            }
            if (keepCopyIt->second) {
                return false;
            }
        }
    }
    // Check pattern CopyOp 1..n CopyOp. This also covers multiple CopyOps from the same SubView result.
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

    const auto areEquivalentCopies = [&](VPUIP::CopyOp srcCopyOp, mlir::Operation* op) {
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

    const auto isIdenticalCopyOp = [&](VPUIP::CopyOp srcCopyOp, mlir::Operation* op) {
        VPUX_THROW_WHEN(srcCopyOp == nullptr, "Expected CopyOp and op to be valid");
        auto siblingCopy = mlir::dyn_cast<VPUIP::CopyOp>(op);
        return siblingCopy != nullptr && siblingCopy == srcCopyOp;
    };

    VPUIP::CopyOp prevCopyOp = nullptr;
    VPUIP::CopyOp newRootCopyOp = originCopyOp;
    uint32_t invalidPostion = std::numeric_limits<uint32_t>::max();
    uint32_t prevComputePostion = invalidPostion;
    uint32_t prevComputePostion4TwoAxis = invalidPostion;

    auto arch = config::getArch(originCopyOp);

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
        llvm::SmallSetVector<mlir::Operation*, 8> allSiblingComputeOps;
        for (auto* user : llvm::make_early_inc_range(parentOpUsers | reversed)) {
            if (mlir::isa<VPUIP::SubViewOp>(user)) {
                for (auto siblingCopy : user->getUsers()) {
                    for (auto item : siblingCopy->getUsers()) {
                        allSiblingComputeOps.insert(item);
                    }
                }
            } else {
                for (auto item : user->getUsers()) {
                    allSiblingComputeOps.insert(item);
                }
            }
        }

        for (const auto& mayComputeOp : allSiblingComputeOps) {
            if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(mayComputeOp)) {
                if (nceOp.getTaskType() == VPUIP::NCETaskType::ELTWISE && nceOp.getIsInplace().value_or(false)) {
                    return true;
                }
            }
        }
        return false;
    };
    auto arePositionsConsecutive = [&](const std::set<uint32_t>& positions) -> bool {
        if (positions.size() < 2) {
            return false;
        }

        auto nearestTiledOpDistance = _tiledOpNearestDistances.at(*positions.begin()).value_or(0);
        // multi dim tiling VF need at least 4 tiled ops to be considered for consecutive check
        auto isCandidateForVF = positions.size() >= 4;
        size_t countConsecutive = 0;
        auto it = positions.begin();
        auto prev = *it;

        for (++it; it != positions.end(); ++it) {
            auto distance = *it - prev;
            if (*it <= prev + 1 || (isCandidateForVF && distance == nearestTiledOpDistance)) {
                ++countConsecutive;
            }
            prev = *it;
        }

        double consecutiveRatio = static_cast<double>(countConsecutive) / (positions.size() - 1);
        // If at least 90% of the positions are consecutive, we consider it as a valid case for optimization
        // It's a workaround for E#172473, where we have a case with multi-dim tiling
        // and we want to avoid unnecessary spillings in the case.
        // This workaround will be removed by another solution on E#172578
        //
        // For VPUX3XXX, we need to be more strict and require 100% consecutive positions.
        // Otherwise, we will have regressions due to increased dpu cost. More details in E#174330.
        return isArchVPUX3XXX(arch) ? consecutiveRatio >= 1.0 : consecutiveRatio >= 0.9;
    };

    auto checkSiblingCopies = [&](mlir::Operation* targetOp) -> bool {
        auto rootCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(targetOp);
        if (rootCopyOp == nullptr) {
            return false;
        }
        const auto srcMemory = mlir::cast<vpux::NDTypeInterface>(rootCopyOp.getInput().getType()).getMemoryKind();
        const auto dstMemory = mlir::cast<vpux::NDTypeInterface>(rootCopyOp.getOutput().getType()).getMemoryKind();
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
        if (parentOp == nullptr) {
            return false;
        }

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
                    if (areEquivalentCopies(rootCopyOp, siblingCopyOp)) {
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
            if (areEquivalentCopies(rootCopyOp, siblingOp)) {
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

        if (isIdenticalCopyOp(rootCopyOpAsCopyOp, nceConvUserOpWeight)) {
            return checkSiblingCopies(nceConvUserOpInput);
        } else if (isIdenticalCopyOp(rootCopyOpAsCopyOp, nceConvUserOpInput)) {
            return checkSiblingCopies(nceConvUserOpWeight);
        }

        return false;
    };

    // To confirm there is NCE::ELTWISE in target pattern
    auto hasEltwiseUser = getSiblingEltwise(originCopyOp);

    const auto isWithinCostDistance = [&](mlir::Operation* op, bool isTilingOnTwoAxis,
                                          std::optional<uint32_t>& nearestDistanceForTilingOnTwoAxis) -> bool {
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
                if (!closeToPrev || isIdenticalCopyOp(originCopyOp, op)) {
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
                uint32_t distanceCost = 0;
                if (nearestDistanceForTilingOnTwoAxis.has_value()) {
                    if (nearestDistanceForTilingOnTwoAxis.value() == 1) {
                        // Two axis tiling only without VF
                        distanceCost = TWO_AXIS_DISTANCE_COST;
                    } else {
                        // Two axis tiling with VF, make the threshold a bit larger
                        distanceCost = COMPUTE_OP_DISTANCE_COST + nearestDistanceForTilingOnTwoAxis.value();
                    }
                }
                bool closeToPrev =
                        prevComputePostion4TwoAxis != invalidPostion &&
                        std::abs(static_cast<int>(currComputePosition.value() - prevComputePostion4TwoAxis)) <
                                static_cast<int>(distanceCost);
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
    std::optional<uint32_t> nearestDistanceForTilingOnTwoAxis = std::nullopt;
    if (isRootCopyTilingOnTwoAxis) {
        auto originCopyOpComputePosition = getComputeOpPosition(originCopyOp.getOperation());
        if (originCopyOpComputePosition.has_value()) {
            prevComputePostion4TwoAxis = originCopyOpComputePosition.value();
            nearestDistanceForTilingOnTwoAxis = _tiledOpNearestDistances.at(originCopyOpComputePosition.value());
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
            if (!areEquivalentCopies(newRootCopyOp, siblingCopyOp) ||
                !isWithinCostDistance(siblingCopyOp, isRootCopyTilingOnTwoAxis, nearestDistanceForTilingOnTwoAxis) ||
                isIdenticalCopyOp(newRootCopyOp, siblingCopyOp)) {
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
        if (!areEquivalentCopies(newRootCopyOp, siblingOp) ||
            !isWithinCostDistance(siblingOp, isRootCopyTilingOnTwoAxis, nearestDistanceForTilingOnTwoAxis) ||
            isIdenticalCopyOp(newRootCopyOp, siblingOp)) {
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

    DenseMap<mlir::Operation*, uint32_t> getDistanceMap();
    std::set<uint32_t> getTiledSiblingOps(mlir::Operation* op,
                                          const DenseMap<mlir::Operation*, uint32_t>& computeOpPosition);
    DenseMap<uint32_t, std::optional<uint32_t>> getNearestDistanceForTiledOps(
            const DenseMap<mlir::Operation*, uint32_t>& computeOpPosition);
    void safeRunOnFunc() final;
};

DenseMap<mlir::Operation*, uint32_t> OptimizeParallelCopiesPass::getDistanceMap() {
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

// For compute op, if it comes from tiling of one op, return all the tiled sibling ops
std::set<uint32_t> OptimizeParallelCopiesPass::getTiledSiblingOps(
        mlir::Operation* op, const DenseMap<mlir::Operation*, uint32_t>& computeOpPosition) {
    VPUX_THROW_UNLESS(computeOpPosition.find(op) != computeOpPosition.end(),
                      "Operation not found in computeOpPosition map");

    auto isSameTaskType = [](mlir::Operation* op1, mlir::Operation* op2) -> bool {
        auto nceOp1 = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op1);
        auto nceOp2 = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op2);
        if (nceOp1 != nullptr && nceOp2 != nullptr) {
            return nceOp1.getTaskType() == nceOp2.getTaskType();
        }

        auto swOp1 = mlir::dyn_cast<VPUIP::SwKernelOp>(op1);
        auto swOp2 = mlir::dyn_cast<VPUIP::SwKernelOp>(op2);
        if (swOp1 != nullptr && swOp2 != nullptr) {
            return getSwKernelEntryName(swOp1) == getSwKernelEntryName(swOp2);
        }
        return false;
    };

    std::set<uint32_t> siblingComputeOpSet;
    siblingComputeOpSet.insert(computeOpPosition.at(op));
    for (auto& arg : op->getOpOperands()) {
        auto operand = arg.get();
        auto operandIdx = arg.getOperandNumber();

        auto copyOp = mlir::dyn_cast_or_null<VPUIP::CopyOp>(operand.getDefiningOp());
        if (copyOp == nullptr) {
            continue;
        }

        auto input = copyOp.getInput();
        if (auto subViewOp = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(input.getDefiningOp())) {
            /* check tiled op from pattern:
                                   Source
                              /               \
                          Subview           SubView
                             |                 |
                            Copy              Copy
                             |                 |
                            Op              SiblingOp
            */
            auto siblingOps = subViewOp.getSource().getUsers();
            for (auto* siblingOp : siblingOps) {
                if (siblingOp == subViewOp || !mlir::isa<VPUIP::SubViewOp>(siblingOp) || siblingOp->use_empty() ||
                    VPU::hasMultiBranches(siblingOp)) {
                    continue;
                }
                auto siblingCopy = mlir::dyn_cast_or_null<VPUIP::CopyOp>(*(siblingOp->user_begin()));
                if (siblingCopy == nullptr || siblingCopy->use_empty() || VPU::hasMultiBranches(siblingCopy)) {
                    continue;
                }
                auto use = siblingCopy->use_begin();
                auto siblingComputeOp = use->getOwner();
                if (use->getOperandNumber() != operandIdx || !isSameTaskType(siblingComputeOp, op)) {
                    continue;
                }
                auto isExpectedComputeOp = computeOpPosition.find(siblingComputeOp) != computeOpPosition.end();
                if (isExpectedComputeOp) {
                    siblingComputeOpSet.insert(computeOpPosition.at(siblingComputeOp));
                }
            }
        } else {
            /* check tiled op from pattern:
                                     Source
                                /               \
                              Copy              Copy
                               |                 |
                              Op              SiblingOp
            */

            auto parentOpusers = input.getUsers();
            for (auto* siblingOp : parentOpusers) {
                if (siblingOp == copyOp || !mlir::isa<VPUIP::CopyOp>(siblingOp) || siblingOp->use_empty() ||
                    VPU::hasMultiBranches(siblingOp)) {
                    continue;
                }
                auto use = siblingOp->use_begin();
                auto siblingComputeOp = use->getOwner();
                if (use->getOperandNumber() != operandIdx || siblingComputeOp->getName() != op->getName()) {
                    continue;
                }
                auto isExpectedComputeOp = computeOpPosition.find(siblingComputeOp) != computeOpPosition.end();
                if (isExpectedComputeOp) {
                    siblingComputeOpSet.insert(computeOpPosition.at(siblingComputeOp));
                }
            }
        }
    }
    return siblingComputeOpSet;
}

DenseMap<uint32_t, std::optional<uint32_t>> OptimizeParallelCopiesPass::getNearestDistanceForTiledOps(
        const DenseMap<mlir::Operation*, uint32_t>& computeOpPosition) {
    DenseMap<uint32_t, std::optional<uint32_t>> nearestDistanceForTiledOp;
    nearestDistanceForTiledOp.reserve(computeOpPosition.size());

    auto findNearestDistance = [](const std::set<uint32_t>& siblingOpSet) -> std::optional<uint32_t> {
        if (siblingOpSet.size() <= 1) {
            return std::nullopt;
        }
        uint32_t nearestDistance = std::numeric_limits<uint32_t>::max();
        auto posList = to_small_vector(siblingOpSet);
        for (size_t i : irange(posList.size() - 1)) {
            auto dist = posList[i + 1] > posList[i] ? posList[i + 1] - posList[i] : posList[i] - posList[i + 1];
            if (dist < nearestDistance) {
                nearestDistance = dist;
            }
        }
        return nearestDistance;
    };

    // Sort by IR position before iterating to ensure deterministic results.
    // DenseMap<mlir::Operation*, uint32_t> iterates in pointer-hash order, which varies across runs
    // due to ASLR/heap layout differences. When sibling sets overlap (op A claims op C as sibling and
    // op B also claims op C), the final nearestDistance for C depends on which op is processed last.
    // Sorting by IR position (the map's value) makes this deterministic.
    SmallVector<std::pair<mlir::Operation*, uint32_t>> sortedOps(computeOpPosition.begin(), computeOpPosition.end());
    llvm::sort(sortedOps, [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    llvm::DenseSet<uint32_t> processedOps;
    for (auto& [op, pos] : sortedOps) {
        if (processedOps.contains(pos)) {
            continue;
        }
        auto siblingOpSet = getTiledSiblingOps(op, computeOpPosition);
        auto nearestDistance = findNearestDistance(siblingOpSet);
        for (auto& siblingOp : siblingOpSet) {
            nearestDistanceForTiledOp[siblingOp] = nearestDistance;
            processedOps.insert(siblingOp);
        }
    }
    VPUX_THROW_UNLESS(
            nearestDistanceForTiledOp.size() == computeOpPosition.size(),
            "Missing some compute ops in nearestDistanceForTiledOp map: actual op size {0}, expected op size {1}",
            nearestDistanceForTiledOp.size(), computeOpPosition.size());
    return nearestDistanceForTiledOp;
}

void OptimizeParallelCopiesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto computeOpPosition = getDistanceMap();
    auto tiledOpNearestDistances = getNearestDistanceForTiledOps(computeOpPosition);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ParallelCopiesRewriter>(&ctx, _log, _enableOptimizeConstCopy, computeOpPosition,
                                         tiledOpNearestDistances);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
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
