//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/range.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

// To explicitly control the patterns exec order to assure dependency
// benefitLevels[0] is highest benefit level and represent the relative pattern is the first one to run
const uint32_t levelCount = 4;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

//
// CopyOpSequence
//

class CopyOpSequence final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CopyOpSequence(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CopyOpSequence::matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const {
    /*
     Remove redundant Copy-to-Copy sequence:
         ParentCopyOp
              |
           CopyOp
     */
    _log.trace("CopyOpSequence: Copy at {0}", copyOp->getLoc());
    auto nestedLogger = _log.nest();
    auto parentCopyOp = copyOp.input().getDefiningOp<VPUIP::CopyOp>();
    if (parentCopyOp == nullptr) {
        StringRef parentOpName = "None";
        if (auto parentOp = copyOp.input().getDefiningOp()) {
            parentOpName = parentOp->getName().getStringRef();
        } else if (copyOp.input().isa<mlir::BlockArgument>()) {
            parentOpName = "BlockArgument";
        }
        nestedLogger.trace("Cannot match because parent isn't CopyOp, but '{0}'", parentOpName);
        return mlir::failure();
    }

    if (parentCopyOp.output_buff().isa<mlir::BlockArgument>() ||
        !(isBufAllocOp(parentCopyOp.output_buff().getDefiningOp()) ||
          VPUIP::getRootAlloc<mlir::memref::AllocOp>(parentCopyOp.output_buff()))) {
        nestedLogger.trace("Cannot match because parent's output buffer is not produced by allocation");
        return mlir::failure();
    }

    for (auto user : parentCopyOp.output().getUsers()) {
        if (mlir::isa<VPUIP::SubViewOp>(user)) {
            // if intermediate SubViewOp users, skip due to accuracy loss
            // TODO E#35612: implement support for intermediate SubViewOp users
            nestedLogger.trace("Cannot match because intermediate SubViewOp users, skip due to accuracy loss");
            return mlir::failure();
        }
    }

    rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(copyOp, parentCopyOp.input(), copyOp.output_buff());

    // CopyOp can have MemoryEffect so "hanging" unused parentCopyOp might not be erased by MLIR automatically
    if (parentCopyOp->use_empty()) {
        rewriter.eraseOp(parentCopyOp);
    }

    nestedLogger.trace("Successfully fused sequence of copies into one op");
    return mlir::success();
}

//
// NCEClusterCopyOpSequence
//

class NCEClusterCopyOpSequence final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    NCEClusterCopyOpSequence(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult NCEClusterCopyOpSequence::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                              mlir::PatternRewriter& rewriter) const {
    // Eliminate copy pairs - spills to DDR
    _log.trace("NCEClusterCopyOpSequence: Copy at {0}", copyOp->getLoc());
    auto nestedLogger = _log.nest();

    auto clusterTiling = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    if (clusterTiling == nullptr) {
        nestedLogger.trace("Cannot match because copy operation isn't wrapped by NCEClusterTilingOp");
        return mlir::failure();
    }

    auto parentClusterTiling = clusterTiling->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
    if (parentClusterTiling == nullptr) {
        nestedLogger.trace("Cannot match because source producer isn't wrapped by NCEClusterTilingOp");
        return mlir::failure();
    }

    auto parentCopy = parentClusterTiling.getInnerTaskOpOfType<VPUIP::CopyOp>();
    if (parentCopy == nullptr) {
        nestedLogger.trace("Cannot match because precessor isn't CopyOp");
        return mlir::failure();
    }

    // The I/O types of this CopyOp-chain should be similar
    auto producerInput = parentClusterTiling.getOperand(0);
    auto output = clusterTiling.getResult(0);

    if (producerInput.getType() != output.getType()) {
        auto inDistributedType = VPUIP::extractDataType(producerInput).dyn_cast<VPUIP::DistributedBufferType>();
        auto outDistributedType = VPUIP::extractDataType(output).dyn_cast<VPUIP::DistributedBufferType>();

        if (inDistributedType == nullptr || outDistributedType == nullptr) {
            nestedLogger.trace("Cannot match because types aren't distributed");
            return mlir::failure();
        }

        if (VPU::isDistributedCastCompatible(inDistributedType, outDistributedType).failed()) {
            nestedLogger.trace("Cannot match because of types incompatibility: '{0}' != '{1}'", inDistributedType,
                               outDistributedType);
            return mlir::failure();
        }

        rewriter.setInsertionPointAfter(parentClusterTiling);
        rewriter.replaceOpWithNewOp<VPUIP::DistributedCastOp>(clusterTiling, output.getType(), producerInput);

        if (parentClusterTiling->use_empty()) {
            rewriter.eraseOp(parentClusterTiling);
        }
        nestedLogger.trace("Successfully fused sequence of NCEClusterTiled copies into one op");
        return mlir::success();
    }

    rewriter.replaceOp(clusterTiling, producerInput);
    if (parentClusterTiling->use_empty()) {
        rewriter.eraseOp(parentClusterTiling);
    }
    nestedLogger.trace("Successfully fused sequence of NCEClusterTiled copies into one op");
    return mlir::success();
}

//
// CMXToCMXCopy
//

/*  Sparse case is more complex since GroupSparseBufferOp takes place:

        (alloc data)-> GroupOp -> (subview 1)   -> (single grouped buffer)
                    /           \ (subview ...) -> (single grouped buffer)
         (alloc SM)             \ (subview N)   -> (single grouped buffer)

    ClusterTask can write into 2 buffers therefore group ungroup pair of ops is added.
    Then ClusterTask or Tiling op can write into intermediate individual buffers without copy:

        (alloc data)-> GroupOp -> (subview 1)   -> UnGroupOp -> (individual buffers) -> GroupOp
                    /           \ (subview ...) -> UnGroupOp -> (individual buffers) -> GroupOp
         (alloc SM)             \ (subview N)   -> UnGroupOp -> (individual buffers) -> GroupOp
*/
VPUIP::GroupSparseBufferOp createGroupUnGroupPair(vpux::VPUIP::SubViewOp copySubView, mlir::PatternRewriter& rewriter) {
    auto sparseType = copySubView.result().getType().cast<VPUIP::SparseBufferType>();

    auto unGroupOp = rewriter.create<VPUIP::UngroupSparseBufferOp>(copySubView.getLoc(), copySubView.getResult());
    auto dataBuffer = unGroupOp.data();
    auto sparsityMap = unGroupOp.sparsityMap();
    auto seTable = unGroupOp.storageElementTable();

    auto groupOp =
            rewriter.create<VPUIP::GroupSparseBufferOp>(copySubView.getLoc(), dataBuffer, sparsityMap, seTable,
                                                        sparseType.getIsWeights(), sparseType.getCompressionScheme());

    copySubView.getResult().replaceAllUsesExcept(groupOp.getResult(),
                                                 llvm::SmallPtrSet<mlir::Operation*, 1>{unGroupOp});

    return groupOp;
}

bool isCMX2CMXCopy(vpux::VPU::MemoryKind srcMemory, vpux::VPU::MemoryKind dstMemory) {
    return srcMemory == dstMemory && srcMemory == VPU::MemoryKind::CMX_NN;
}

template <class ConcreteType>
ConcreteType getParentNceOp(mlir::Operation* copyOp) {
    auto parentOp = copyOp->getOperand(0).getDefiningOp<ConcreteType>();
    if (parentOp == nullptr) {
        if (auto parentGroupOp = copyOp->getOperand(0).getDefiningOp<VPUIP::GroupSparseBufferOp>()) {
            return parentGroupOp->getOperand(0).getDefiningOp<ConcreteType>();
        }
    }
    return parentOp;
}

mlir::LogicalResult removeClusterTilingCMXToCMXCopy(VPUIP::NCEClusterTilingOp copyClusterOp,
                                                    mlir::PatternRewriter& rewriter, Logger log) {
    log.trace("removeClusterTilingCMXToCMXCopy: Copy at {0}", copyClusterOp->getLoc());
    auto nestedLogger = log.nest();

    auto innerCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(copyClusterOp.getInnerTaskOp());
    if (innerCopyOp == nullptr) {
        nestedLogger.trace("Cannot match because tiling op does not contain Copy");
        return mlir::failure();
    }

    auto inputType = copyClusterOp->getOperand(0).getType().dyn_cast<vpux::NDTypeInterface>();
    auto outputType = copyClusterOp->getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();
    // Only remove redundant CMX2CMX CopyOps
    if (!isCMX2CMXCopy(inputType.getMemoryKind(), outputType.getMemoryKind())) {
        nestedLogger.trace("Cannot match because the transfer is not CMX->CMX");
        return mlir::failure();
    }

    // CMX Concat case with subView, update the buffers used
    if (auto copySubView = copyClusterOp.output_buffs()[0].getDefiningOp<VPUIP::SubViewOp>()) {
        // case with subView - retrieve operations to be re-linked
        auto masterBuffer = VPUIP::getRootAlloc<VPURT::AllocDistributed>(copySubView.source());
        if (masterBuffer == nullptr) {
            nestedLogger.trace("Cannot match because source isn't master buffer");
            return mlir::failure();
        }

        auto parentNCEClusterOp = getParentNceOp<VPUIP::NCEClusterTilingOp>(copyClusterOp);
        if (parentNCEClusterOp == nullptr) {
            nestedLogger.trace("Cannot match because copy is not a successor of NCEClusterTiling");
            return mlir::failure();
        }

        const auto updateParentNCEOp = [&](size_t argIdx, mlir::Value value,
                                           VPUIP::GroupSparseBufferOp newGroupOp = nullptr) {
            // Update resut types of NCEClusterTiling
            parentNCEClusterOp->getResult(argIdx).setType(value.getType());
            // Update output buffers of NCEClusterTiling
            parentNCEClusterOp.output_buffs()[argIdx].replaceAllUsesWith(value);
            // Update inner NCEClusterTask
            const auto newInnerType = value.getType().dyn_cast<VPUIP::DistributedBufferType>().getCompactType();
            // Update block arguments
            size_t totalArgNum = 1;
            if (newGroupOp != nullptr) {
                totalArgNum = newGroupOp->getNumOperands();
            }
            // Output operands are placed in the end
            parentNCEClusterOp.body()
                    .getArgument(parentNCEClusterOp.getNumOperands() - (totalArgNum) + argIdx)
                    .setType(newInnerType);
            // Update result types
            parentNCEClusterOp.getInnerTaskOp()->getResult(argIdx).setType(newInnerType);
            // Update new group op to use results of parentNCEClusterOp
            if (newGroupOp != nullptr) {
                newGroupOp->setOperand(argIdx, parentNCEClusterOp.results()[argIdx]);
            }
        };

        copySubView->moveBefore(parentNCEClusterOp);

        if (auto subviewParentGroupOp = mlir::dyn_cast<VPUIP::GroupSparseBufferOp>(masterBuffer)) {
            rewriter.setInsertionPoint(parentNCEClusterOp);
            auto newGroupOp = createGroupUnGroupPair(copySubView, rewriter);
            if (newGroupOp == nullptr) {
                return mlir::failure();
            }
            auto newBuffers = SmallVector<mlir::Value>({newGroupOp.data(), newGroupOp.sparsityMap()});
            // Go through individual buffers and update corresponding operands.
            // Fist run is data and the second run is sparsity map.
            for (auto newBuffer : newBuffers | indexed) {
                updateParentNCEOp(newBuffer.index(), newBuffer.value(), newGroupOp);
            }
            // Replace all uses of copy op with GoupOp
            newGroupOp->moveAfter(parentNCEClusterOp);
            copyClusterOp->replaceAllUsesWith(newGroupOp);
        } else {
            // replace the copy with the subView
            updateParentNCEOp(0 /*result index*/, copySubView, nullptr);
            copyClusterOp->replaceAllUsesWith(parentNCEClusterOp);
        }
        // update IR location of the master buffer
        if (copySubView->isBeforeInBlock(masterBuffer)) {
            VPUIP::moveRootAllocBefore(masterBuffer, copySubView);
        }
    } else if (inputType == outputType) {
        // case with no subView
        copyClusterOp->replaceAllUsesWith(copyClusterOp.getOperand(0).getDefiningOp());
    } else {
        log.trace("Copy not optimized {0}", copyClusterOp->getLoc());
        return mlir::failure();
    }
    rewriter.eraseOp(copyClusterOp);

    nestedLogger.trace("Successfully removed sequence");
    return mlir::success();
}

mlir::LogicalResult removeCMXToCMXCopy(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter, Logger log) {
    // Check current CopyOp source and destination
    log.trace("removeCMXToCMXCopy: Copy at {0}", copyOp->getLoc());
    auto nestedLogger = log.nest();

    auto inputType = copyOp.input().getType().cast<vpux::NDTypeInterface>();
    auto outputType = copyOp.output().getType().cast<vpux::NDTypeInterface>();

    // Only remove redundant CMX2CMX CopyOps
    if (!isCMX2CMXCopy(inputType.getMemoryKind(), outputType.getMemoryKind())) {
        nestedLogger.trace("Cannot match because the transfer is not CMX->CMX");
        return mlir::failure();
    }
    // CMX Concat case with SubView, update the buffers used
    if (auto copySubView = mlir::dyn_cast<VPUIP::SubViewOp>(copyOp.output_buff().getDefiningOp())) {
        // case with SubView - retrieve operations to be re-linked
        auto parentNCE = getParentNceOp<VPUIP::NCEClusterTaskOp>(copyOp);

        if (parentNCE == nullptr) {
            nestedLogger.trace("Cannot match because copy operation is not a successor of NCEClusterTask");
            return mlir::failure();
        }

        auto masterBuffer = VPUIP::getRootAlloc<mlir::memref::AllocOp>(copySubView->getOperand(0));
        if (masterBuffer == nullptr) {
            nestedLogger.trace("Cannot match because source isn't master buffer");
            return mlir::failure();
        }

        VPUIP::moveRootAllocBefore(copySubView, parentNCE);
        if (auto subviewParentGroupOp = mlir::dyn_cast<VPUIP::GroupSparseBufferOp>(masterBuffer)) {
            rewriter.setInsertionPoint(parentNCE);
            auto newGroupOp = createGroupUnGroupPair(copySubView, rewriter);
            if (newGroupOp == nullptr) {
                return mlir::failure();
            }
            // Go through individual buffers and update corresponding operands
            parentNCE->getResult(0).setType(newGroupOp.data().getType());
            parentNCE->getResult(1).setType(newGroupOp.sparsityMap().getType());
            // Update NCEClusterTask output buffers
            parentNCE.output_buff().replaceAllUsesWith(newGroupOp.data());
            parentNCE.output_sparsity_map_buff().replaceAllUsesWith(newGroupOp.sparsityMap());
            // Update GroupOp to use results of NCEClusterTask, but not new UngroupOp
            newGroupOp->setOperand(0, parentNCE.output());
            newGroupOp->setOperand(1, parentNCE.output_sparsity_map());
            // Replace all uses of copy op with new GroupOp
            newGroupOp->moveAfter(parentNCE);
            copyOp->replaceAllUsesWith(newGroupOp);
        } else {
            // replace the copy with the subView
            parentNCE->getResult(0).setType(copySubView->getResult(0).getType());
            parentNCE.output_buff().replaceAllUsesWith(copySubView->getResult(0));
            copyOp.output().replaceAllUsesWith(copyOp.input());
        }
        // update IR location of the master buffer
        if (copySubView->isBeforeInBlock(masterBuffer)) {
            VPUIP::moveRootAllocBefore(masterBuffer, copySubView);
        }
    } else if (inputType == outputType) {
        // case with no subView
        copyOp.output().replaceAllUsesWith(copyOp.input());
    } else {
        log.trace("Copy not optimized {0}", copyOp->getLoc());
        return mlir::failure();
    }
    rewriter.eraseOp(copyOp);
    nestedLogger.trace("Successfully removed sequence");
    return mlir::success();
}

class CMXToCMXCopy final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CMXToCMXCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CMXToCMXCopy::matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const {
    /*
     Remove CMX2CMX Copy without SubView:
         Copy(DDR2CMX)                    Copy(DDR2CMX)
              |                                |
            NCEOp           =>               NCEOp
              |
         Copy(CMX2CMX)

     Remove CMX2CMX Copy with SubView:
        Copy(DDR2CMX)                Copy(DDR2CMX)  SubView
              |                                \     /
            NCEOp       SubView   =>            NCEOp
               \         /
              Copy(CMX2CMX)
     */
    if (auto clusterTilingCopy = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
        return removeClusterTilingCMXToCMXCopy(clusterTilingCopy, rewriter, _log);
    } else {
        return removeCMXToCMXCopy(copyOp, rewriter, _log);
    }
}

//
// DDRToDDRCopyOfNCECluster
//

class DDRToDDRCopyOfNCECluster final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    DDRToDDRCopyOfNCECluster(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isCopyToDDR(VPUIP::CopyOp copyOp) {
    auto origOp = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>() == nullptr ? copyOp.getOperation()
                                                                                  : copyOp->getParentOp();
    return origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getMemoryKind() == VPU::MemoryKind::DDR;
}

bool isCopyFromDDR(VPUIP::CopyOp copyOp) {
    auto origOp = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>() == nullptr ? copyOp.getOperation()
                                                                                  : copyOp->getParentOp();
    return origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().getMemoryKind() == VPU::MemoryKind::DDR;
}

bool isDDR2DDROfNCEClusterInput(VPUIP::CopyOp copyOp) {
    // ParentOp should be a Subview
    // ChildOp should be a copy op wrapped in ClusterTilingOp
    auto parentOp = copyOp.input().getDefiningOp<VPUIP::SubViewOp>();
    if (parentOp == nullptr) {
        return false;
    }
    if (copyOp.output().getUsers().empty()) {
        return false;
    }
    auto childOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*copyOp.output().getUsers().begin());
    return (childOp != nullptr) && (childOp.getInnerTaskOpOfType<VPUIP::CopyOp>() != nullptr);
}

bool hasValidParallelCopyBranchWithSubView(VPUIP::CopyOp copyOp, VPUIP::NCEClusterTilingOp parentOp) {
    if (parentOp->hasOneUse()) {
        return false;
    }

    auto subview = copyOp.output_buff().getDefiningOp<VPUIP::SubViewOp>();
    if (subview == nullptr) {
        return false;
    }

    // If a CMX to DDR copy's input is a subview of SOH's output, the CMX2DDR copy's input tensor will have a SEGMENTED
    // or OVERLAPPED distribution. But the output data of the tensor's subview may be distributed on one cluster or
    // multiple clusters.In the current compiler logic, when calculating DMA cost and unroll DMA, it is assumed that the
    // data of the Tensor with SEGMENTED or OVERLAPPED distribution is distributed on multiple clusters. Therefore, SOH
    // optimization is temporarily turned off and turned on after subsequent compiler support.E60342
    if (auto distType = VPUIP::extractDataType(parentOp.inputs()[0]).dyn_cast<VPUIP::DistributedBufferType>()) {
        if (distType.getDistribution().mode().getValue() == VPU::DistributionMode::SEGMENTED ||
            distType.getDistribution().mode().getValue() == VPU::DistributionMode::OVERLAPPED) {
            auto subviewshape = subview.result().getType().cast<vpux::NDTypeInterface>().getShape().raw();
            auto numTiles = parseIntArrayAttr<int64_t>(distType.getDistribution().num_tiles());
            if (subviewshape.size() == 4 && subviewshape[Dims4D::Act::H.ind()] % numTiles[Dims4D::Act::H.ind()] != 0) {
                return false;
            }
        }
    }

    // check other parallel branch if it's a valid copy branch or not
    for (auto siblingOp : parentOp.results().getUsers()) {
        // Considering padding/slice case: Tiling_copy -> subview -> copy
        if (auto siblingSubview = mlir::dyn_cast<VPUIP::SubViewOp>(*siblingOp)) {
            if (siblingSubview.getResult().hasOneUse()) {
                auto childOp = siblingSubview.getResult().getUsers().begin();
                if (auto childCopy = mlir::dyn_cast<VPUIP::CopyOp>(*childOp)) {
                    // Case is okay : pass
                    if (!childCopy.output_buff().getDefiningOp<VPUIP::SubViewOp>()) {
                        return false;
                    }
                }
            } else {
                return false;
            }
        } else if (auto siblingCopy = mlir::dyn_cast<VPUIP::CopyOp>(*siblingOp)) {
            if (siblingCopy != copyOp) {
                if (!siblingCopy.output_buff().getDefiningOp<VPUIP::SubViewOp>()) {
                    return false;
                }
            }
        } else {
            return false;
        }
    }
    // check all branches and okay
    return true;
}

// For the case: parent of copyOp only have one output branch
// Parallel case should be processed by isParallelDDR2DDROfNCEClusterOutput()
// for clear logic
bool isDDR2DDROfNCEClusterOutput(VPUIP::CopyOp copyOp) {
    // ParentOp should be a copy op wrapped in ClusterTilingOp
    // ChildOp should be a concat
    auto parentOp = copyOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
    if (parentOp == nullptr || parentOp.getInnerTaskOpOfType<VPUIP::CopyOp>() == nullptr) {
        return false;
    }
    if (copyOp.output().getUsers().empty()) {
        return false;
    }
    for (auto user : copyOp.output().getUsers()) {
        if (!mlir::isa<VPUIP::ConcatViewOp>(*user)) {
            return false;
        }
    }

    return parentOp->hasOneUse();
}

bool isParallelDDR2DDROfNCEClusterOutput(VPUIP::CopyOp copyOp) {
    // ParentOp should be a copy op wrapped in ClusterTilingOp
    // ChildOp should be a concat
    auto parentOp = copyOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
    if (parentOp == nullptr || parentOp.getInnerTaskOpOfType<VPUIP::CopyOp>() == nullptr) {
        return false;
    }

    if (copyOp.output().getUsers().empty()) {
        return false;
    }
    for (auto user : copyOp.output().getUsers()) {
        if (!mlir::isa<VPUIP::ConcatViewOp>(*user)) {
            return false;
        }
    }

    /*
     Optimize the parallel DDR2DDR copies as CMX2DDR copies:
                 ClusterTiling_Copy(CMX2DDR)
                      /        \
            Copy(DDR2DDR)   (SubViews ->) Copy(DDR2DDR)
            /        \                 /       \
        SubView              |               SubView
                             |
                          Concat
    */
    return hasValidParallelCopyBranchWithSubView(copyOp, parentOp);
}

bool isStridedCopy(VPUIP::CopyOp copyOp) {
    // Here we check two options at the same time:
    // 1. Copy op is not strided, in the sense that step for copying dimension is 1
    // 2. Copy can handle full plane without offsets

    const auto outType = copyOp.output_buff().getType().cast<vpux::NDTypeInterface>();
    const auto order = outType.getDimsOrder();
    const auto memStrides = StrideReqs::compact(order.numDims()).calcStrides(order, outType);
    auto compactStrides = order.toLogicalOrder(memStrides);

    auto actStrides = outType.getStrides();
    VPUX_THROW_UNLESS(compactStrides.size() == actStrides.size(),
                      "Compact ({0}) and actual ({1}) strides size mismatch", compactStrides.size(), actStrides.size());

    for (size_t i = 1; i < compactStrides.size(); i++) {
        if (compactStrides[Dim(i)] != actStrides[Dim(i)]) {
            return true;
        }
    }

    return false;
}

bool isDDR2DDROfConcatInput(VPUIP::CopyOp copyOp) {
    // ParentOp should be a concatView op
    // ChildOp should be a concatView too
    auto parentConcatOp = copyOp.input().getDefiningOp<VPUIP::ConcatViewOp>();
    if (parentConcatOp == nullptr) {
        return false;
    }
    if (!copyOp.output().hasOneUse()) {
        return false;
    }

    auto childConcatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(*copyOp.output().getUsers().begin());
    if (childConcatOp == nullptr) {
        return false;
    }

    // Exclude strided dma case
    size_t constCopyCnt = 0;
    auto predicteChildConcatInput = [&](mlir::Value op) {
        auto copy = op.getDefiningOp<VPUIP::CopyOp>();
        if (copy == nullptr || isStridedCopy(copy)) {
            return false;
        }

        auto concat = copy.input().getDefiningOp<VPUIP::ConcatViewOp>();
        if (concat == nullptr) {
            auto subView = copy.input().getDefiningOp<VPUIP::SubViewOp>();
            if (subView == nullptr) {
                auto parentCopyInputConst = VPUIP::getRootConst(copy.input());
                if (parentCopyInputConst) {
                    constCopyCnt++;
                    return true;
                }
                return false;
            } else if (!subView.result().hasOneUse()) {
                return false;
            }
            concat = subView.source().getDefiningOp<VPUIP::ConcatViewOp>();
        }

        return concat == parentConcatOp;
    };

    /*
     E.g., Optimize the left DDR2DDR copy in below case:
     case 1:
                      ConcatView
                      /         \
             Copy(DDR2DDR)      SubView
                     \            \
                      \        Copy(DDR2DDR)
                       \        /
                           |
                           |
                       ConcatView
    case 2:
                ConcatView
                    |
             Copy(DDR2DDR)      const.Declare
                     \            |
                      \        Copy(DDR2DDR)
                       \        /
                           |
                           |
                       ConcatView
    */
    if (!llvm::all_of(childConcatOp.inputs(), predicteChildConcatInput)) {
        return false;
    }

    const auto childConcatInputsNum = childConcatOp.inputs().size();

    const auto parentConcatUsers = parentConcatOp.output().getUsers();
    const auto parentConcatUsersNum = std::distance(parentConcatUsers.begin(), parentConcatUsers.end());

    return (childConcatInputsNum - constCopyCnt) == static_cast<size_t>(parentConcatUsersNum);
}

mlir::LogicalResult removeDDR2DDRForNCEClusterInput(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter, Logger log) {
    copyOp.output().replaceAllUsesWith(copyOp.input());
    log.trace("Successfully removed DDRToDDR input copy {0} at {1}", copyOp->getName(), copyOp->getLoc());
    rewriter.eraseOp(copyOp);
    return mlir::success();
}

mlir::LogicalResult removeDDR2DDRForNCEClusterOutput(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter,
                                                     Logger log) {
    // CMX Concat case with subView, update the buffers used
    if (auto subViewOp = copyOp.output_buff().getDefiningOp<VPUIP::SubViewOp>()) {
        // case with subView - retrieve operations to be re-linked
        auto masterBuffer = VPUIP::getRootAlloc<mlir::memref::AllocOp>(subViewOp->getOperand(0));
        if (masterBuffer == nullptr) {
            log.trace("Cannot match because source isn't master buffer");
            return mlir::failure();
        }
        auto parentOp = copyOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
        // replace the copy with VPUIP subView
        rewriter.setInsertionPoint(parentOp);
        auto newSubViewOp = rewriter.create<VPUIP::SubViewOp>(
                subViewOp->getLoc(), subViewOp.source(), subViewOp.static_offsetsAttr(), subViewOp.static_sizesAttr(),
                subViewOp.static_stridesAttr());
        parentOp.output_buffs()[0].replaceAllUsesWith(newSubViewOp->getResult(0));
        parentOp->getResult(0).setType(newSubViewOp->getResult(0).getType());

        // update IR location of the master buffer
        if (newSubViewOp->isBeforeInBlock(masterBuffer)) {
            VPUIP::moveRootAllocBefore(masterBuffer, newSubViewOp);
        }
    } else {
        auto parentOp = copyOp.input().getDefiningOp<VPUIP::NCEClusterTilingOp>();
        auto allocOp = VPUIP::getRootAlloc<mlir::memref::AllocOp>(parentOp.output_buffs()[0]);
        if (allocOp == nullptr) {
            log.trace("Cannot match because source isn't master buffer");
            return mlir::failure();
        }

        for (auto user : copyOp.output().getUsers()) {
            auto concatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(user);
            concatOp.output_buff().replaceAllUsesWith(allocOp->getResult(0));
        }
    }

    copyOp.output().replaceAllUsesWith(copyOp.input());
    log.trace("Successfully removed Clustered DDRToDDR output copy {0} at {1}", copyOp->getName(), copyOp->getLoc());
    rewriter.eraseOp(copyOp);
    return mlir::success();
}

mlir::LogicalResult removeParallelDDR2DDRForNCEClusterOutput(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter,
                                                             Logger log) {
    auto parentOp = copyOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();

    for (auto user : llvm::make_early_inc_range(parentOp.getResult(0).getUsers())) {
        if (auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*user)) {
            auto subview = copyOp.output_buff().getDefiningOp<VPUIP::SubViewOp>();

            rewriter.setInsertionPointAfter(subview);
            const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                mlir::ValueRange newOperands) {
                builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
            };
            SmallVector<mlir::Value> inputsOutputOperands = {parentOp->getOperand(0), subview.result()};
            auto newCopyInCluster = rewriter.create<VPUIP::NCEClusterTilingOp>(
                    parentOp->getLoc(), subview->getResult(0).getType(), inputsOutputOperands, copyOutBodyBuilder);

            copyOp.output().replaceAllUsesWith(newCopyInCluster->getResult(0));

            log.trace("Successfully removed Parallel DDRToDDR output copy {0} at {1}", copyOp->getName(),
                      copyOp->getLoc());
            rewriter.eraseOp(copyOp);
        }
    }

    for (auto user : llvm::make_early_inc_range(parentOp.getResult(0).getUsers())) {
        if (auto subview = mlir::dyn_cast<VPUIP::SubViewOp>(*user)) {
            auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*subview.getResult().getUsers().begin());
            if (copyOp == nullptr) {
                log.trace("CopyOp is null");
                return mlir::failure();
            }
            auto outputSubview = copyOp.output_buff().getDefiningOp<VPUIP::SubViewOp>();
            if (outputSubview == nullptr) {
                log.trace("Output subview is null");
                return mlir::failure();
            }

            rewriter.setInsertionPointAfter(copyOp);
            // New a new subview for copy output
            auto newSubView = rewriter.create<VPUIP::SubViewOp>(
                    subview->getLoc(), parentOp->getOperand(0), subview.static_offsetsAttr(),
                    subview.static_sizesAttr(), subview.static_stridesAttr());

            const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                mlir::ValueRange newOperands) {
                builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
            };
            SmallVector<mlir::Value> inputsOutputOperands = {newSubView.result(), outputSubview.result()};
            auto newCopyInCluster = rewriter.create<VPUIP::NCEClusterTilingOp>(
                    parentOp->getLoc(), outputSubview.result().getType(), inputsOutputOperands, copyOutBodyBuilder);

            copyOp.output().replaceAllUsesWith(newCopyInCluster->getResult(0));
            log.trace("Successfully removed Parallel DDRToDDR output copy (with input subview) {0} at {1}",
                      copyOp->getName(), copyOp->getLoc());
            rewriter.eraseOp(copyOp);
            rewriter.eraseOp(subview);
        }
    }

    rewriter.eraseOp(parentOp);
    return mlir::success();
}

static inline void inferOpsTypeBetween(mlir::Operation* startOp, mlir::Operation* endOp,
                                       mlir::PatternRewriter& rewriter) {
    auto currentOp = startOp;

    while (currentOp != endOp) {
        mlir::SmallVector<mlir::Type> inferredTypes;
        auto op = mlir::cast<mlir::InferTypeOpInterface>(currentOp);
        VPUX_THROW_UNLESS(op.inferReturnTypes(rewriter.getContext(), op->getLoc(), currentOp->getOperands(),
                                              op->getAttrDictionary(), op->getRegions(), inferredTypes)
                                  .succeeded(),
                          "New type inference failed for '{0}'", op);
        for (auto result : currentOp->getResults()) {
            result.setType(inferredTypes[0]);
        }
        currentOp = currentOp->getNextNode();
    }
}

mlir::LogicalResult removeDDR2DDRForConcatInput(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter, Logger log) {
    auto parentConcatOp = copyOp.input().getDefiningOp<VPUIP::ConcatViewOp>();
    auto parentMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(parentConcatOp.output_buff());
    if (parentMemAlloc == nullptr) {
        log.trace("Cannot match because parentConcatOp output isn't master buffer");
        return mlir::failure();
    }

    auto childConcatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(*copyOp.output().getUsers().begin());
    auto childMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(childConcatOp.output_buff());
    if (childMemAlloc == nullptr) {
        log.trace("Cannot match because childConcatOp output isn't master buffer");
        return mlir::failure();
    }

    auto childMemSize = vpux::getTotalSize(childMemAlloc->getResult(0));
    auto parentMemSize = vpux::getTotalSize(parentMemAlloc->getResult(0));
    if (childMemSize <= parentMemSize) {
        log.error("There is no redundant Copy operation since the child size ({0}) <= parent size ({1})", childMemSize,
                  parentMemSize);
        return mlir::failure();
    }
    log.trace("Successfully removed DDRToDDR output copy {0} at {1} for Concat", copyOp->getName(), copyOp->getLoc());
    auto childCopySubview = copyOp.output_buff().getDefiningOp<VPUIP::SubViewOp>();

    auto newSubViewOp = rewriter.create<VPUIP::SubViewOp>(parentMemAlloc->getLoc(), childCopySubview.source(),
                                                          childCopySubview.static_offsetsAttr(),
                                                          childCopySubview.static_sizesAttr());

    // update IR location of the master buffer
    if (parentMemAlloc->isBeforeInBlock(newSubViewOp)) {
        VPUIP::moveRootAllocBefore(newSubViewOp, parentMemAlloc);
    }
    // update IR location of the master buffer
    if (newSubViewOp->isBeforeInBlock(childMemAlloc)) {
        VPUIP::moveRootAllocBefore(childMemAlloc, newSubViewOp);
    }

    parentMemAlloc->getResult(0).replaceAllUsesWith(newSubViewOp.result());
    rewriter.eraseOp(parentMemAlloc);
    // Re-Infer the Type of the Ops
    inferOpsTypeBetween(newSubViewOp, childConcatOp, rewriter);

    copyOp.output().replaceAllUsesWith(copyOp.input());
    rewriter.eraseOp(copyOp);
    return mlir::success();
}

mlir::LogicalResult DDRToDDRCopyOfNCECluster::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                              mlir::PatternRewriter& rewriter) const {
    /*
     Remove redundant DDR2DDR Copy of the NCECluster's input:
ClusterTiling_Copy                    ...        SubView
   (CMX2DDR)        SubView             \         /
          \         /              ClusterTiling_Copy(CMX2DDR)
          Copy(DDR2DDR)        =>            |
               |                           Concat
            Concat

     Remove redundant DDR2DDR Copy of the NCECluster's output:
             SubView                    SubView
              (DDR)                      (DDR)
                |                          |
          Copy(DDR2DDR)            ClusterTiling_Copy
                |                      (DDR2CMX)
        ClusterTiling_Copy    =>           |
            (DDR2CMX)              ClusterTiling_NCE
                |                          |
        ClusterTiling_NCE
                |

     Optimize the parallel DDR2DDR copies as CMX2DDR copies:
                ClusterTiling_Copy(CMX2DDR)
                      /        \
            Copy(DDR2DDR)       Copy(DDR2DDR)       =>
            /        \          /       \
        SubView           |            SubView
                          |
                        Concat

                         ...
                     /          \
ClusterTiling_Copy(CMX2DDR)   ClusterTiling_Copy(CMX2DDR)
            /        \          /       \
        SubView           |            SubView
                          |
                        Concat
     */
    _log.trace("DDRToDDRCopyOfNCECluster: Copy at {0}", copyOp->getLoc());
    auto nestedLogger = _log.nest();
    if (!isCopyFromDDR(copyOp) || !isCopyToDDR(copyOp)) {
        nestedLogger.trace("Cannot match because isn't DDR->DDR copy");
        return mlir::failure();
    }

    if (isDDR2DDROfNCEClusterInput(copyOp)) {
        return removeDDR2DDRForNCEClusterInput(copyOp, rewriter, nestedLogger);
    } else if (isDDR2DDROfNCEClusterOutput(copyOp)) {
        return removeDDR2DDRForNCEClusterOutput(copyOp, rewriter, nestedLogger);
    } else if (isParallelDDR2DDROfNCEClusterOutput(copyOp)) {
        // TODO: Add this optimization in single cluster case
        return removeParallelDDR2DDRForNCEClusterOutput(copyOp, rewriter, nestedLogger);
    } else if (isDDR2DDROfConcatInput(copyOp)) {
        return removeDDR2DDRForConcatInput(copyOp, rewriter, nestedLogger);
    }
    std::string possibleReason;
    if (copyOp.input().getDefiningOp<Const::DeclareOp>()) {
        possibleReason = " Copy from Constant isn't optimizable";
    }
    nestedLogger.trace("Unsupported pattern.{0}", possibleReason);
    return mlir::failure();
}

//
// ConcatViewWithCopyBase
//

class ConcatViewWithCopyBase : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    ConcatViewWithCopyBase(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp origOp, mlir::PatternRewriter& rewriter) const final;
    bool isLegalConcatViewPattern(VPUIP::ConcatViewOp origOp, vpux::Logger log) const;

    virtual bool hasLegalCopyUser(VPUIP::ConcatViewOp sourceOp) const = 0;
    virtual mlir::Value getOutputBuffer(mlir::Operation* sourceOp) const = 0;
    virtual VPUIP::LayerOpInterface createNewCopyOp(VPUIP::CopyOp copyInput, VPUIP::SubViewOp subViewOp,
                                                    mlir::PatternRewriter& rewriter) const = 0;

private:
    bool hasDuplicatedCopyOutput(VPUIP::ConcatViewOp origOp) const;

    Logger _log;
};

mlir::LogicalResult ConcatViewWithCopyBase::matchAndRewrite(VPUIP::ConcatViewOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Got ConcatView Op at '{0}'", origOp.getLoc());

    if (!isLegalConcatViewPattern(origOp, _log)) {
        _log.nest().trace("Cannot fuse this ConcatView Op");
        return mlir::failure();
    }

    auto* firstCopyOp = getFirstUser(origOp.getResult());
    VPUX_THROW_UNLESS(firstCopyOp != nullptr, "Cannot get the first user Op");

    SmallVector<mlir::Value> concatInputs;
    auto outBuffer = getOutputBuffer(firstCopyOp);
    auto outBufferDefiningOp = outBuffer.getDefiningOp();
    VPUX_THROW_WHEN(outBufferDefiningOp == nullptr, "Cannot get defining op for {0}", outBuffer);
    rewriter.setInsertionPointAfter(outBufferDefiningOp);
    for (auto input : origOp.inputs()) {
        auto copyOp = input.getDefiningOp<VPUIP::CopyOp>();
        auto subViewOp = copyOp.output_buff().getDefiningOp<VPUIP::SubViewOp>();

        auto newSubView =
                rewriter.create<VPUIP::SubViewOp>(copyOp.getLoc(), outBuffer, subViewOp.static_offsetsAttr(),
                                                  subViewOp.static_sizesAttr(), subViewOp.static_stridesAttr());
        auto newCopyOp = createNewCopyOp(copyOp, newSubView, rewriter);

        concatInputs.push_back(newCopyOp->getResult(0));
    }

    rewriter.setInsertionPointAfter(firstCopyOp);
    auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(firstCopyOp->getLoc(), concatInputs, outBuffer);
    SmallVector<mlir::Operation*> userCopyOps(origOp.output().user_begin(), origOp.output().user_end());
    for (auto userCopyOp : userCopyOps) {
        rewriter.replaceOp(userCopyOp, newConcatOp.output());
    }
    _log.nest().trace("Successfully simplified ConcatView");
    return mlir::success();
}

/*
  Check pattern:
  Copy (DDR2DDR)  ...  Copy (DDR2DDR)
       \               /
        Concat View (DDR)
             |
        Copy(DDR2CMX)
*/
bool ConcatViewWithCopyBase::isLegalConcatViewPattern(VPUIP::ConcatViewOp origOp, vpux::Logger log) const {
    if (!origOp.output().hasOneUse() && !hasDuplicatedCopyOutput(origOp)) {
        log.nest().trace("Cannot find user copy op at '{0}'", origOp);
        return false;
    }
    for (auto input : origOp.inputs()) {
        auto op = mlir::dyn_cast_or_null<VPUIP::CopyOp>(input.getDefiningOp());
        if (op == nullptr || !isCopyToDDR(op) || !isCopyFromDDR(op)) {
            return false;
        }
    }

    return hasLegalCopyUser(origOp);
}

bool ConcatViewWithCopyBase::hasDuplicatedCopyOutput(VPUIP::ConcatViewOp origOp) const {
    if (origOp.use_empty()) {
        return false;
    }
    auto isSameCopyType = [](mlir::Operation* preOp, mlir::Operation* nextOp) {
        auto preCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(preOp);
        auto nextCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(nextOp);
        if (preCopyOp != nullptr && nextCopyOp != nullptr) {
            auto preOutType = preCopyOp.output().getType().dyn_cast<vpux::NDTypeInterface>();
            auto nextOutType = preCopyOp.output().getType().dyn_cast<vpux::NDTypeInterface>();
            return preOutType == nextOutType;
        }

        auto preClusterCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(preOp);
        auto nextClusterCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(nextOp);
        if (preClusterCopyOp == nullptr || nextClusterCopyOp == nullptr) {
            return false;
        }
        auto preInnerCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(preClusterCopyOp.getInnerTaskOp());
        auto nextInnerCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(nextClusterCopyOp.getInnerTaskOp());
        if (preInnerCopyOp == nullptr || nextInnerCopyOp == nullptr) {
            return false;
        }
        auto preOutputType = preClusterCopyOp.getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();
        auto nextOutputType = nextClusterCopyOp.getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();
        return preOutputType == nextOutputType;
    };

    auto firstUser = *origOp.output().getUsers().begin();
    return llvm::all_of(origOp.output().getUsers(), [&](auto user) {
        return isSameCopyType(firstUser, user);
    });
}

//
// ConcatViewWithCopy
//

/*
  Copy (DDR -> DDR)  ...  Copy (DDR -> DDR)
                \               /
                Concat View (DDR)             =>           Copy (DDR -> CMX) ... Copy (DDR -> CMX)
                        |                                           \               /
                Copy (DDR -> CMX)                                   Concat View (CMX)
*/

class ConcatViewWithCopy final : public ConcatViewWithCopyBase {
public:
    ConcatViewWithCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : ConcatViewWithCopyBase(ctx, benefit, log) {
    }

public:
    virtual bool hasLegalCopyUser(VPUIP::ConcatViewOp sourceOp) const override;
    virtual mlir::Value getOutputBuffer(mlir::Operation* sourceOp) const override;
    virtual VPUIP::LayerOpInterface createNewCopyOp(VPUIP::CopyOp copyInput, VPUIP::SubViewOp subViewOp,
                                                    mlir::PatternRewriter& rewriter) const override;
};

bool ConcatViewWithCopy::hasLegalCopyUser(VPUIP::ConcatViewOp sourceOp) const {
    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*sourceOp->getUsers().begin());
    return copyOp != nullptr && isCopyFromDDR(copyOp) && !isCopyToDDR(copyOp);
}

mlir::Value ConcatViewWithCopy::getOutputBuffer(mlir::Operation* sourceOp) const {
    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(sourceOp);
    VPUX_THROW_WHEN(copyOp == nullptr, "Unexpected op type at '{0}'", sourceOp->getLoc());
    return copyOp.output_buff();
}

VPUIP::LayerOpInterface ConcatViewWithCopy::createNewCopyOp(VPUIP::CopyOp copyInput, VPUIP::SubViewOp subViewOp,
                                                            mlir::PatternRewriter& rewriter) const {
    return rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(copyInput, copyInput.input(), subViewOp.result());
}

//
// ConcatViewWithTilingCopy
//

/*
 Copy (DDR -> DDR)  ...  Copy (DDR -> DDR)
                \               /
                Concat View (DDR)             =>  Cluster Copy (DDR -> CMX) ... Cluster Copy (DDR -> CMX)
                        |                                           \               /
              Cluster Copy (DDR -> CMX)                             Concat View (CMX)
*/

class ConcatViewWithTilingCopy final : public ConcatViewWithCopyBase {
public:
    ConcatViewWithTilingCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : ConcatViewWithCopyBase(ctx, benefit, log) {
    }

public:
    virtual bool hasLegalCopyUser(VPUIP::ConcatViewOp sourceOp) const override;
    virtual mlir::Value getOutputBuffer(mlir::Operation* sourceOp) const override;
    virtual VPUIP::LayerOpInterface createNewCopyOp(VPUIP::CopyOp copyInput, VPUIP::SubViewOp subViewOp,
                                                    mlir::PatternRewriter& rewriter) const override;
};

bool ConcatViewWithTilingCopy::hasLegalCopyUser(VPUIP::ConcatViewOp sourceOp) const {
    auto clusterOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*sourceOp->getUsers().begin());
    if (clusterOp == nullptr) {
        return false;
    }

    auto copyOp = clusterOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
    if (copyOp == nullptr || isStridedCopy(copyOp)) {
        return false;
    }

    // Get the concat dims
    const auto inputType = sourceOp.inputs()[0].getType().cast<vpux::NDTypeInterface>();
    const auto outputType = sourceOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inputType.getShape();
    const auto outShape = outputType.getShape();
    VPUX_THROW_UNLESS(inShape.size() == outShape.size(), "Input shape size {0} is not equal to output shape size {1}",
                      inShape.size(), outShape.size());
    SmallVector<Dim> concatDims;
    for (auto idx : irange(inShape.size())) {
        if (inShape[Dim(idx)] != outShape[Dim(idx)]) {
            concatDims.push_back(Dim(idx));
        }
    }
    VPUX_THROW_WHEN(concatDims.empty(), "ConcatView inShape '{0}' same with the outShape '{1}'", inputType.getShape(),
                    outputType.getShape());

    const auto distributedType =
            VPUIP::extractDataType(clusterOp.output_buffs()[0]).dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Cannot get distributedType");
    const auto distribution = distributedType.getDistribution();

    if (distribution.num_tiles() != nullptr) {
        const auto isValidTile = [](auto dim) {
            return dim > 1;
        };

        const auto tilingScheme = parseIntArrayAttr<int64_t>(distribution.num_tiles());
        const auto tileAxis = std::distance(tilingScheme.begin(), llvm::find_if(tilingScheme, isValidTile));

        if (llvm::find(concatDims, Dim(tileAxis)) != concatDims.end()) {
            return false;
        }
    }

    return isCopyFromDDR(copyOp) && !isCopyToDDR(copyOp);
}

mlir::Value ConcatViewWithTilingCopy::getOutputBuffer(mlir::Operation* sourceOp) const {
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(sourceOp);
    VPUX_THROW_WHEN(clusterTilingOp == nullptr, "Unexpected op type at '{0}'", sourceOp);
    return clusterTilingOp.output_buffs()[0];
}

VPUIP::LayerOpInterface ConcatViewWithTilingCopy::createNewCopyOp(VPUIP::CopyOp copyInput, VPUIP::SubViewOp subViewOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(copyInput->getLoc(), newOperands[0], newOperands[1]);
    };
    auto inputsOutputOperands = {copyInput.input(), subViewOp.result()};
    auto newClusterTilingOutType = subViewOp.result().getType().cast<vpux::NDTypeInterface>();
    return rewriter.replaceOpWithNewOp<VPUIP::NCEClusterTilingOp>(copyInput, newClusterTilingOutType,
                                                                  inputsOutputOperands, bodyBuilder);
}

//
// FuseCopyToTheFrontOfTillingCopy
//
/*
 Fuse copy into the front Tilling copy
          |                |
  TillingCopy    =>  TillingCopy
          |                |
         Copy
          |
*/

class FuseCopyToTheFrontOfTillingCopy final : public mlir::OpRewritePattern<VPUIP::NCEClusterTilingOp> {
public:
    FuseCopyToTheFrontOfTillingCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::NCEClusterTilingOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::NCEClusterTilingOp clusterTilingCopyOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseCopyToTheFrontOfTillingCopy::matchAndRewrite(VPUIP::NCEClusterTilingOp clusterTilingCopy,
                                                                     mlir::PatternRewriter& rewriter) const {
    /*
    case 1:
              |                          |
      TillingCopy(CMX2DDR)    =>     TillingCopy(CMX2CMX)
              |                          |
           Copy(DDR2CMX)
              |

    case 2:
              |                          |
      TillingCopy(CMX2DDR)    =>     TillingCopy(CMX2DDR)
              |                          |
           Copy(DDR2DDR)
              |
    */

    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(clusterTilingCopy.getInnerTaskOp());
    if (copyOp == nullptr || isCopyFromDDR(copyOp) || !isCopyToDDR(copyOp)) {
        return mlir::failure();
    }

    if (!clusterTilingCopy->hasOneUse()) {
        return mlir::failure();
    }

    auto tilingCopyOutput = clusterTilingCopy.getResult(0);
    auto outType = tilingCopyOutput.getType().dyn_cast<vpux::NDTypeInterface>();
    auto userCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*(tilingCopyOutput.getUsers().begin()));
    if (!userCopyOp) {
        return mlir::failure();
    }

    auto userOutType = userCopyOp.output_buff().getType().dyn_cast<vpux::NDTypeInterface>();
    if (userOutType.changeMemSpace(VPU::MemoryKind::DDR) != outType) {
        return mlir::failure();
    }
    auto userOutputMemKind = userOutType.getMemoryKind();
    if (userOutputMemKind == VPU::MemoryKind::CMX_NN) {
        auto inputType = clusterTilingCopy.getOperand(0).getType().dyn_cast<vpux::NDTypeInterface>();
        Byte requiredCMX(0);
        requiredCMX += inputType.getTotalAllocSize();
        requiredCMX += userOutType.getTotalAllocSize();
        if (requiredCMX > VPU::getTotalCMXSize(userCopyOp)) {
            _log.trace("Available CMX size is {0}, but need {1}", VPU::getTotalCMXSize(userCopyOp), requiredCMX);
            return mlir::failure();
        }
    }

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(clusterTilingCopy->getLoc(), newOperands[0], newOperands[1]);
    };

    SmallVector<mlir::Value> inputsOutputOperands = {clusterTilingCopy->getOperand(0), userCopyOp.output_buff()};

    rewriter.setInsertionPointAfter(userCopyOp);
    auto newClusterTilingCopyOp = rewriter.create<VPUIP::NCEClusterTilingOp>(clusterTilingCopy->getLoc(), userOutType,
                                                                             inputsOutputOperands, bodyBuilder);
    userCopyOp.replaceAllUsesWith(newClusterTilingCopyOp);
    rewriter.eraseOp(userCopyOp);
    rewriter.eraseOp(clusterTilingCopy);
    return mlir::success();
}

//
// fuseLastCopy
//

void fuseLastCopy(VPUIP::CopyOp copyOp, const AliasesInfo& aliasesInfo, Logger log) {
    log.trace("fuseLastCopy: Copy at {0}", copyOp->getLoc());
    auto nestedLogger = log.nest();

    auto inSourceMemory = copyOp.input().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    auto outSourceMemory = copyOp.output().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    if (inSourceMemory != outSourceMemory) {
        nestedLogger.trace("Cannot match because the copy is not within the same memory space");
        return;
    }

    auto sourceOp = copyOp.input().getDefiningOp();
    if (sourceOp == nullptr) {
        nestedLogger.trace("Cannot match because copy input is a block argument");
        return;
    }

    const auto sourceRoots = aliasesInfo.getRoots(copyOp.input());
    mlir::Value sourceRoot = nullptr;
    if (sourceRoots.size() == 1) {
        sourceRoot = *sourceRoots.begin();
    } else if (sourceRoots.size() > 1) {
        mlir::Operation* sourceRootCandidate = nullptr;
        for (auto root : sourceRoots) {
            if (!root.hasOneUse()) {
                nestedLogger.trace("Cannot match because one of the roots has multiple uses : '{0}'", root);
                return;
            }
            auto groupedViewOp = mlir::dyn_cast<vpux::GroupedViewOpInterface>(*root.getUsers().begin());
            if (sourceRootCandidate == nullptr) {
                sourceRootCandidate = groupedViewOp;
            }
            if (sourceRootCandidate != groupedViewOp) {
                nestedLogger.trace(
                        "Cannot match because roots doesnot share common GroupedViewOp. Expected '{0}', but got '{1}'",
                        sourceRootCandidate, groupedViewOp);
                return;
            }
        }
        sourceRoot = sourceRootCandidate->getResult(0);
    }
    if (sourceRoot == nullptr) {
        nestedLogger.trace("Cannot match because of unexpected root of pattern");
        return;
    }

    if (sourceRoot == nullptr || sourceRoot.isa<mlir::BlockArgument>()) {
        nestedLogger.trace("Cannot match because input also is block argument");
        return;
    }

    auto sourceRootOp = sourceRoot.getDefiningOp();
    if (!isBufAllocOp(sourceRootOp) && !mlir::isa<VPUIP::GroupSparseBufferOp>(sourceRootOp)) {
        nestedLogger.trace("Cannot match because input isn't allocate op: '{0}'", sourceRootOp->getName());
        return;
    }

    VPUIP::ConcatViewOp concatViewOp;
    auto newBuffer = copyOp.output_buff();
    auto newOutput = copyOp.input();

    if (sourceRoot.getType() != copyOp.output_buff().getType()) {
        // check what operation changes the type
        auto typeCastOp = copyOp.input().getDefiningOp();

        if (auto quantizeCastOp = mlir::dyn_cast<VPUIP::QuantizeCastOp>(typeCastOp)) {
            // we will make a QuantizeCast over the output buffer and we will copy from CMX directly to output
            // buffer, and we will return the output buffer. After ConcatView and QuantizeCast will be redundant.
            // from CMX -> CopyOp[DDR] -> ConcatViewOp -> QuantizeCastOp -> CopyOp[block-arg] -> return CopyOp
            // Output of this step:
            //                        CMX -> CopyOp[QuantizeCastOp] -> return block-arg
            //   block-arg -> QuantizeCastOp /

            concatViewOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(quantizeCastOp.input().getDefiningOp());
            if (!concatViewOp) {
                nestedLogger.trace("Cannot match because of missed concat in QuantizeCast case");
                return;
            }

            mlir::OpBuilder builder(quantizeCastOp);
            builder.setInsertionPoint(sourceRoot.getDefiningOp());

            auto newQuantizeCast = builder.create<VPUIP::QuantizeCastOp>(concatViewOp.getLoc(), sourceRoot.getType(),
                                                                         copyOp.output_buff());

            quantizeCastOp.replaceAllUsesWith(quantizeCastOp.input());
            quantizeCastOp->erase();

            newBuffer = newQuantizeCast.output();
            newOutput = copyOp.output_buff();
        } else if (auto permuteCastOp = mlir::dyn_cast<VPUIP::PermuteCastOp>(typeCastOp)) {
            // do the permute in output

            mlir::OpBuilder builder(permuteCastOp);
            builder.setInsertionPoint(sourceRoot.getDefiningOp());

            auto newPermuteCast = builder.create<VPUIP::PermuteCastOp>(
                    permuteCastOp.getLoc(), sourceRoot.getType(), copyOp.output_buff(), permuteCastOp.dst_orderAttr(),
                    permuteCastOp.mem_permAttr());

            permuteCastOp.replaceAllUsesWith(permuteCastOp.source());
            permuteCastOp->erase();

            newBuffer = newPermuteCast.result();
            newOutput = copyOp.output_buff();
        } else {
            nestedLogger.trace("Cannot match because of missed concat in generic branch");
            return;
        }
    }

    // Function outputs have to be an alias of the output buffer
    log.trace("Root of the copy operation input {0}", sourceRoot);
    log.trace("Reassign outputs from {0} to {1}", sourceRoot, newBuffer);

    for (auto& use : llvm::make_early_inc_range(sourceRoot.getUses())) {
        log.nest().trace("Got user {0}", use.getOwner()->getName());
        log.nest().trace("Reassign {0} to {1}", use.get(), newBuffer);
        use.set(newBuffer);
    }

    copyOp.replaceAllUsesWith(newOutput);
    copyOp->erase();
    if (concatViewOp) {
        concatViewOp->erase();
    }

    if (sourceRootOp->use_empty()) {
        sourceRootOp->erase();
    }
}

//
// OptimizeCopiesPass
//

class OptimizeCopiesPass final : public VPUIP::OptimizeCopiesBase<OptimizeCopiesPass> {
public:
    explicit OptimizeCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void OptimizeCopiesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    // Note the below patterns exec order is defined by "benefitLevels" at the head
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<CopyOpSequence>(&ctx, benefitLevels[0], _log);
    patterns.add<NCEClusterCopyOpSequence>(&ctx, benefitLevels[0], _log);
    patterns.add<CMXToCMXCopy>(&ctx, benefitLevels[1], _log);
    patterns.add<DDRToDDRCopyOfNCECluster>(&ctx, benefitLevels[2], _log);
    patterns.add<ConcatViewWithCopy>(&ctx, benefitLevels[3], _log);
    patterns.add<ConcatViewWithTilingCopy>(&ctx, benefitLevels[3], _log);
    patterns.add<FuseCopyToTheFrontOfTillingCopy>(&ctx, benefitLevels[3], _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    func->walk([&](VPUIP::CopyOp op) {
        if (!op.output_buff().isa<mlir::BlockArgument>()) {
            return;
        }

        auto& aliasInfo = getAnalysis<AliasesInfo>();
        fuseLastCopy(op, aliasInfo, _log);
    });
}

}  // namespace

//
// createOptimizeCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeCopiesPass(Logger log) {
    return std::make_unique<OptimizeCopiesPass>(log);
}
