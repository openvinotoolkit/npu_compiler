//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <stdexcept>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_OPTIMIZECOPIES
#define GEN_PASS_DEF_OPTIMIZECOPIES
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

bool isCMX2CMXCopy(vpux::VPU::MemoryKind srcMemory, vpux::VPU::MemoryKind dstMemory) {
    return srcMemory == dstMemory && srcMemory == VPU::MemoryKind::CMX_NN;
}

bool isNonDistributedCastCompatible(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType) {
    auto inDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(inType);
    if (inDistributedType == nullptr || mlir::isa<VPUIP::DistributedBufferType>(outType)) {
        return false;
    }
    const auto mode = inDistributedType.getDistribution().getMode().getValue();
    if (!VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) &&
        mode != (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED)) {
        return false;
    }
    return inDistributedType.getShape() == outType.getShape() &&
           inDistributedType.getElementType() == outType.getElementType() &&
           inDistributedType.getMemoryKind() == outType.getMemoryKind() &&
           inDistributedType.getStrides() == outType.getStrides() &&
           inDistributedType.getDimsOrder() == outType.getDimsOrder();
}

// To explicitly control the patterns exec order to assure dependency
// benefitLevels[0] is highest benefit level and represent the relative pattern is the first one to run
const uint32_t levelCount = 4;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

// Check the user of the copyOp is an EltwiseOp with is_inplace
VPUIP::NCEClusterTaskOp getEltwiseInplaceUser(VPUIP::CopyOp copyOp) {
    mlir::Operation* op = copyOp.getOperation();

    auto opUsers = op->getResult(0).getUsers();
    if (opUsers.empty()) {
        return nullptr;
    }

    if (!op->hasOneUse()) {
        auto firstUserOp = *opUsers.begin();
        for (auto userOp : llvm::make_early_inc_range(opUsers)) {
            if (firstUserOp != userOp) {
                return nullptr;
            }
        }
    }

    auto copyUser = *opUsers.begin();
    auto userClusterTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(copyUser);
    if (userClusterTaskOp != nullptr && userClusterTaskOp.getTaskType() == VPUIP::NCETaskType::ELTWISE &&
        userClusterTaskOp.getIsInplace()) {
        return userClusterTaskOp;
    }

    return nullptr;
}

// Helper function to assert operation dominance in a graph.
// It helps to avoid the following situations:
// %COPY = VPUIP.Copy(%ALLOC)   // operand does not dominate this use
// %ALLOC = memref.alloc
// It is always safe to put allocations in the beginning of the block because they don't have any producers.
// First check if an operation is an allocation. If it is, move it to the beginning of the block.
// If a buffer is not produced by an allocation, check that it comes before its consumer.
// mlir::Operation::isBeforeInBlock is an expensive check, it should be avoided whenever possible.
// useMoveAfter flag determines the method of ordering.
// useMoveAfter = true: place copy operation after the buffer:
// %COPY = VPUIP.Copy(%ALLOC)
// %op1 = ...
// %ALLOC = memref.alloc
// becomes
// %op1 = ...
// %ALLOC = memref.alloc
// %COPY = VPUIP.Copy(%ALLOC)
// useMoveAfter = false: place buffer before the copy operation:
// %COPY = VPUIP.Copy(%ALLOC)
// %op1 = ...
// %ALLOC = memref.alloc
// becomes
// %ALLOC = memref.alloc
// %COPY = VPUIP.Copy(%ALLOC)
// %op1 = ...
void rearrangeOperations(mlir::Operation* buffer, mlir::Operation* copy, const bool useMoveAfter) {
    if (mlir::isa<mlir::memref::AllocOp, VPURT::AllocDistributed>(buffer)) {
        auto& block = buffer->getParentOfType<mlir::func::FuncOp>().getBody().front();
        mlir::Operation& firstOp = block.front();
        VPUIP::moveRootAllocBefore(buffer, &firstOp);
    } else if (copy->isBeforeInBlock(buffer)) {
        if (useMoveAfter) {
            copy->moveAfter(buffer);
        } else {
            VPUIP::moveRootAllocBefore(buffer, copy);
        }
    }
}

bool areDistributedBufferTypesCompatible(vpux::VPUIP::DistributedBufferType inDistributedType,
                                         vpux::VPUIP::DistributedBufferType outDistributedType,
                                         bool allowDifferentPerClusterMemoryView) {
    if (mlir::succeeded(VPU::isDistributedCastCompatible(inDistributedType, outDistributedType))) {
        return true;
    }

    if (!allowDifferentPerClusterMemoryView) {
        return false;
    }

    auto inDistribution = VPU::DistributionInfo::getClassFromAttr(inDistributedType.getDistribution());
    auto outDistribution = VPU::DistributionInfo::getClassFromAttr(outDistributedType.getDistribution());
    auto inType = mlir::cast<vpux::NDTypeInterface>(inDistributedType);
    auto outType = mlir::cast<vpux::NDTypeInterface>(outDistributedType);
    if (mlir::failed(areDistributionsCompatible(inType, inDistribution, outType, outDistribution,
                                                allowDifferentPerClusterMemoryView))) {
        return false;
    }

    SmallVector<Shape> sourceMemoryOffsets = inDistributedType.getPerClusterMemoryShapeOffsets();
    SmallVector<Shape> targetMemoryOffsets = outDistributedType.getPerClusterMemoryShapeOffsets();
    SmallVector<Shape> sourceMemoryShapes = inDistributedType.getPerClusterMemoryShapes();
    SmallVector<Shape> targetMemoryShapes = outDistributedType.getPerClusterMemoryShapes();

    // Check if the data range in target memory is included in the source memory on each cluster.
    for (size_t i = 0; i < sourceMemoryOffsets.size(); i++) {
        auto sourceMemoryOffset = sourceMemoryOffsets[i].raw();
        auto targetMemoryOffset = targetMemoryOffsets[i].raw();
        auto sourceMemoryShape = sourceMemoryShapes[i].raw();
        auto targetMemoryShape = targetMemoryShapes[i].raw();
        for (size_t j = 0; j < sourceMemoryOffset.size(); j++) {
            if ((sourceMemoryOffset[j] > targetMemoryOffset[j]) ||
                ((sourceMemoryOffset[j] + sourceMemoryShape[j]) < (targetMemoryOffset[j] + targetMemoryShape[j]))) {
                return false;
            }
        }
    }

    return true;
}

bool isLegalAndBenefitCreateCopyFromCMXToCMX(mlir::Operation* grandparentOp, VPUIP::CopyOp parentCopyOp,
                                             VPUIP::CopyOp copyOp, vpux::Logger& nestedLogger) {
    auto availableCMXSize = VPU::getTotalCMXSize(copyOp);
    auto srcDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(parentCopyOp.getInput().getType());
    auto dstDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(copyOp.getOutput().getType());

    if (srcDistributedType == nullptr || dstDistributedType == nullptr) {
        return false;
    }

    auto isDistributedTypeCompatibleWithCMXCopy =
            areDistributedBufferTypesCompatible(srcDistributedType, dstDistributedType, true);
    if (!isDistributedTypeCompatibleWithCMXCopy) {
        return false;
    }

    // check CMX
    if ((srcDistributedType.getTotalAllocSize() + dstDistributedType.getTotalAllocSize()) > availableCMXSize) {
        nestedLogger.trace("Cannot create Copy from CMX to CMX because memory is not enough");
        return false;
    }

    // avoid long spilling
    auto module = copyOp->getParentOfType<mlir::ModuleOp>();
    auto thresholdCMXSize = VPU::getTotalCMXFragmentationAwareSize(module);
    auto hasLongSpilling = [&](mlir::Value value) -> bool {
        // If the output buffer has a stride, it indicates that the output buffer is a part of a larger root buffer, and
        // other operations must be filling the remaining parts of the root buffer. Avoid applying the optimization in
        // this case, as the pre-reserved CMX memory could lead to performance issues with other operations, such as
        // causing dynamic spilling.
        const auto strideReqs = StrideReqs::compact(dstDistributedType.getShape().size());
        return !strideReqs.checkStrides(dstDistributedType) || llvm::any_of(value.getUsers(), [&](auto user) {
            auto userClusterTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(user);
            if (userClusterTaskOp != nullptr && userClusterTaskOp.getTaskType() == VPUIP::NCETaskType::ELTWISE) {
                SmallVector<mlir::Operation*> producerChain;
                auto producer = userClusterTaskOp.getInput() == value ? userClusterTaskOp.getWeights().getDefiningOp()
                                                                      : userClusterTaskOp.getInput().getDefiningOp();
                while (producer != nullptr && !producer->isBeforeInBlock(grandparentOp)) {
                    if (auto clusterTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(producer)) {
                        Byte requiredCMX = VPUIP::getRequiredCMXSize(clusterTaskOp);
                        if (requiredCMX + srcDistributedType.getTotalAllocSize() > thresholdCMXSize) {
                            return true;
                        }
                        producer = clusterTaskOp.getInput().getDefiningOp();
                    } else if (auto viewOp = mlir::dyn_cast<mlir::ViewLikeOpInterface>(producer)) {
                        producer = viewOp->getOperand(0).getDefiningOp();
                    } else {
                        break;
                    }
                }
            }

            return false;
        });
    };

    if (hasLongSpilling(copyOp.getOutput())) {
        nestedLogger.trace("Cannot create Copy from CMX to CMX because the copyOp output has long spilling");
        return false;
    }

    if (!parentCopyOp->hasOneUse()) {
        auto allCopyUsersAreCompatible = [&](mlir::Value value) -> bool {
            return llvm::all_of(value.getUsers(), [&](auto user) {
                if (auto userCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(user)) {
                    auto dstDistributedType = mlir::dyn_cast_or_null<VPUIP::DistributedBufferType>(
                            VPUIP::extractDataType(userCopyOp.getResult()));

                    return (dstDistributedType != nullptr) &&
                           areDistributedBufferTypesCompatible(srcDistributedType, dstDistributedType, true);
                }

                if (auto userSubViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(user)) {
                    if (!userSubViewOp->hasOneUse()) {
                        return false;
                    }

                    auto userCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*userSubViewOp.getResult().getUsers().begin());
                    if (userCopyOp == nullptr) {
                        return false;
                    }

                    auto dstDistributedType = mlir::dyn_cast_or_null<VPUIP::DistributedBufferType>(
                            VPUIP::extractDataType(userCopyOp.getResult()));

                    return (dstDistributedType != nullptr) &&
                           areDistributedBufferTypesCompatible(srcDistributedType, dstDistributedType, true);
                }

                return false;
            });
        };

        if (!allCopyUsersAreCompatible(parentCopyOp.getOutput())) {
            nestedLogger.trace("Cannot create Copy from CMX to CMX because parent CopyOp has multiple uses and not all "
                               "of them are compatible");
            return false;
        }
    }

    return true;
}

//
// CopyOpSequence
//

class CopyOpSequence final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CopyOpSequence(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, WorkloadManagementMode workloadManagementMode,
                   Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit),
              _workloadManagementMode(workloadManagementMode),
              log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    WorkloadManagementMode _workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA;
    Logger log;
};

mlir::LogicalResult CopyOpSequence::matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const {
    log.trace("CopyOpSequence: Copy at {0}", copyOp->getLoc());

    auto isCopyOpDistributed = vpux::VPUIP::hasDistributedOperand(copyOp);

    auto nestedLogger = log.nest();
    if (mlir::isa<mlir::BlockArgument>(copyOp.getInput())) {
        nestedLogger.trace("CopyOpSequence: cannot match because the parent is not a CopyOp, but a BlockArgument");
        return mlir::failure();
    }

    auto distributedCastOp = copyOp.getInput().getDefiningOp<VPUIP::DistributedCastOp>();
    auto parentCopyOp = distributedCastOp != nullptr ? distributedCastOp.getInput().getDefiningOp<VPUIP::CopyOp>()
                                                     : copyOp.getInput().getDefiningOp<VPUIP::CopyOp>();
    if (parentCopyOp == nullptr) {
        if (auto parentOp = copyOp.getInput().getDefiningOp()) {
            nestedLogger.trace("CopyOpSequence: cannot match because parent CopyOp is {0}",
                               parentOp->getName().getStringRef());
            return mlir::failure();
        }
        nestedLogger.trace("CopyOpSequence: cannot match because parent CopyOp is nullptr");
        return mlir::failure();
    }
    if (!isCopyOpDistributed && vpux::VPUIP::hasDistributedOperand(parentCopyOp)) {
        nestedLogger.trace("CopyOpSequence: cannot match non-distributed copyOp with a distributed CopyOp parent");
        return mlir::failure();
    }
    if (isCopyOpDistributed && !vpux::VPUIP::hasDistributedOperand(parentCopyOp)) {
        nestedLogger.trace("CopyOpSequence: cannot match distributed copyOp with a non-distributed CopyOp parent");
        return mlir::failure();
    }

    if (mlir::isa<mlir::BlockArgument>(parentCopyOp.getOutputBuff()) ||
        !(isBufAllocOp(parentCopyOp.getOutputBuff().getDefiningOp()) ||
          VPUIP::getRootAlloc<mlir::memref::AllocOp>(parentCopyOp.getOutputBuff()) ||
          VPUIP::getRootAlloc<VPURT::AllocDistributed>(parentCopyOp.getOutputBuff()))) {
        nestedLogger.trace("CopyOpSequence: cannot match because parent's output buffer is not produced by allocation");
        return mlir::failure();
    }

    if (!vpux::VPUIP::hasDistributedOperand(parentCopyOp)) {
        for (auto user : parentCopyOp.getOutput().getUsers()) {
            if (mlir::isa<VPUIP::SubViewOp>(user)) {
                // if intermediate SubViewOp users, skip due to accuracy loss
                // TODO E#35612: implement support for intermediate SubViewOp users
                nestedLogger.trace(
                        "CopyOpSequence: cannot match because intermediate SubViewOp users, skip due to accuracy loss");
                return mlir::failure();
            }
        }
    }
    auto reduce = [&]() {
        if (!isCopyOpDistributed) {
            rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(copyOp, parentCopyOp.getInput(), copyOp.getOutputBuff());

            // CopyOp can have MemoryEffect so "hanging" unused parentCopyOp might not be erased by MLIR automatically
            if (parentCopyOp->use_empty()) {
                rewriter.eraseOp(parentCopyOp);
            }

            nestedLogger.trace("CopyOpSequence: successfully fused sequence of copies into one op");
        } else {
            rewriter.replaceOp(copyOp, parentCopyOp.getInput());
            if (distributedCastOp && distributedCastOp->use_empty()) {
                rewriter.eraseOp(distributedCastOp);
            }
            if (parentCopyOp->use_empty()) {
                rewriter.eraseOp(parentCopyOp);
            }
            nestedLogger.trace("CopyOpSequence: successfully fused sequence of distributed copies into one op");
        }

        return mlir::success();
    };

    // In case the new copyOp will be eliminated after copyOp sequence optimization, and the user of copyOp is
    // an EltwiseOp with is_inplace, then the inplace buffer for EltwiseOp should be updated.
    auto parentCopyOpInputType = mlir::cast<vpux::NDTypeInterface>(parentCopyOp.getInput().getType());
    auto copyOutput = copyOp.getOutputBuff();
    auto copyOutputType = mlir::cast<vpux::NDTypeInterface>(copyOutput.getType());

    auto areValuesCompatible = [&](mlir::Value input, mlir::Value output, vpux::Logger& nestedLogger,
                                   bool allowDifferentPerClusterMemoryView) -> bool {
        VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(input.getType()) ||
                                  mlir::isa<vpux::VPUIP::DistributedBufferType>(input.getType()),
                          "Unsupported buffer type");

        if (mlir::isa<mlir::MemRefType>(input.getType())) {
            return input.getType() == output.getType();
        }

        auto inDistributedType = mlir::dyn_cast_or_null<VPUIP::DistributedBufferType>(VPUIP::extractDataType(input));
        auto outDistributedType = mlir::dyn_cast_or_null<VPUIP::DistributedBufferType>(VPUIP::extractDataType(output));
        if (inDistributedType == nullptr || outDistributedType == nullptr) {
            nestedLogger.trace("CopyOpSequence: types are not distributed");
            return false;
        }

        return areDistributedBufferTypesCompatible(inDistributedType, outDistributedType,
                                                   allowDifferentPerClusterMemoryView);
    };

    if (!isCopyOpDistributed && !isCMX2CMXCopy(parentCopyOpInputType.getMemoryKind(), copyOutputType.getMemoryKind())) {
        nestedLogger.trace("CopyOpSequence: optimizing non-CMX2CMX copy");
        return reduce();
    }
    if (!isCopyOpDistributed && parentCopyOpInputType != copyOutputType) {
        nestedLogger.trace("CopyOpSequence: optimizing two CMX2DDR/DDR2DDR/DDR2CMX copies");
        return reduce();
    }

    auto grandparentOp = parentCopyOp.getInput().getDefiningOp();
    if (grandparentOp == nullptr) {
        nestedLogger.trace("CopyOpSequence: cannot match because grandparent of current CopyOp is not an operation");
        return mlir::failure();
    }
    auto isGrandparentCompatible =
            areValuesCompatible(grandparentOp->getResult(0), copyOp.getResult(), nestedLogger, false);

    if (auto eltwiseInPlaceUser = getEltwiseInplaceUser(copyOp)) {
        nestedLogger.trace("CopyOpSequence: CopyOp is an eltwise in_place NCEClusterTask user");
        auto isUserClusterTaskCompatible =
                areValuesCompatible(eltwiseInPlaceUser->getResult(0), copyOp.getResult(), nestedLogger, false);

        if (isCopyOpDistributed && !vpux::VPUIP::hasDistributedOperand(eltwiseInPlaceUser)) {
            nestedLogger.trace(
                    "CopyOpSequence: cannot fuse a distributed CopyOp with non-distributed user NCEClusterTask");
            return mlir::failure();
        }

        // Found the inplace buffer of nceOp and replace use
        auto nceOutputBuff = vpux::VPUIP::getRootBuffer(eltwiseInPlaceUser.getOutputBuff());
        auto copyOpOutBuff = vpux::VPUIP::getRootBuffer(copyOp.getOutputBuff());

        if (nceOutputBuff == copyOpOutBuff) {
            if (isCopyOpDistributed) {
                if (!vpux::VPUIP::hasDistributedOperand(grandparentOp)) {
                    nestedLogger.trace("CopyOpSequence: cannot match because current CopyOp is distributed and its "
                                       "grandparent is non-distributed");
                    return mlir::failure();
                }
                if (mlir::dyn_cast_or_null<VPUIP::NCEClusterTaskOp>(grandparentOp) == nullptr) {
                    nestedLogger.trace("CopyOpSequence: cannot match because current CopyOp is distributed and its "
                                       "grandparent isn't an NCEClusterTask");
                    return mlir::failure();
                }
            }

            if (!isGrandparentCompatible) {
                nestedLogger.trace("CopyOpSequence: cannot match because types aren't compatible");
                return mlir::failure();
            }

            auto nceParentOutput = VPUIP::getLayerOutputs(grandparentOp)[0];
            if (isCopyOpDistributed) {
                rewriter.replaceAllUsesWith(nceOutputBuff, nceParentOutput);
            }

            // Need to insert a DistributedCast if the types aren't exactly the same but still compatible
            if (isCopyOpDistributed && isUserClusterTaskCompatible) {
                nestedLogger.trace("CopyOpSequence: current CopyOp is distributed");
                eltwiseInPlaceUser->getResult(0).setType(grandparentOp->getResult(0).getType());
                rewriter.setInsertionPointAfter(eltwiseInPlaceUser);
                auto resultDistributedCast = rewriter.create<VPUIP::DistributedCastOp>(
                        eltwiseInPlaceUser.getLoc(), copyOp.getResult().getType(), eltwiseInPlaceUser.getOutput());
                eltwiseInPlaceUser->replaceUsesWithIf(resultDistributedCast, [&](mlir::OpOperand& use) {
                    return use.getOwner() != resultDistributedCast;
                });
            } else if (!isCopyOpDistributed) {
                nestedLogger.trace("CopyOpSequence: current CopyOp is non-distributed");
                // Check ViewLikeOp without output_buff
                const auto isViewLikeOpWithoutOutputBuff = [&](mlir::Operation* op) -> bool {
                    return mlir::isa<VPUIP::DistributedCastOp, VPUIP::NonDistributedCastOp, VPUIP::SubViewOp,
                                     VPUIP::PermuteCastOp, VPUIP::QuantizeCastOp, VPUIP::GenericReshapeOp,
                                     VPUIP::ShapeCastOp, VPUIP::StubOp, VPUIP::ViewOp, VPUIP::ExtractFlatSliceOp>(op);
                };

                mlir::Value parentCopyOpInputBuff;
                // For view-like ops getLayerOutputs will incorrectly return
                // value that is an input to it rather than returning its own
                // output. Instead just call getResult directly.
                if (isViewLikeOpWithoutOutputBuff(grandparentOp)) {
                    parentCopyOpInputBuff = grandparentOp->getResult(0);
                } else {
                    parentCopyOpInputBuff = VPUIP::getLayerOutputs(grandparentOp)[0];
                }

                //
                // In case there is a ViewOp make ElemType consistent for Inplace buffer
                // After optimize copyOp sequence, need to get the root buffer of Input Buf 1 as new input buffer of
                // ViewOp
                //             |
                //     CMX: Input Buf 1
                //             |
                //     CopyOp (CMX2DDR)
                //             |
                //     CopyOp (DDR2CMX)
                //     (CMX: Input Buf 2)           (CMX: Input Buf 3)
                //                     \                 /
                //             EltwiseOp with Inplace (Input Buf 2)
                //                     /                 |
                //     ViewOp (Input Buf 2)
                //
                mlir::Operation* viewOp = nullptr;
                for (auto user : nceOutputBuff.getUsers()) {
                    if (mlir::isa<VPUIP::ViewOp>(user->getResult(0).getDefiningOp())) {
                        viewOp = mlir::cast<VPUIP::ViewOp>(user->getResult(0).getDefiningOp());
                    }
                }

                if (viewOp == nullptr) {
                    rewriter.replaceAllUsesWith(nceOutputBuff, parentCopyOpInputBuff);
                } else {
                    rewriter.replaceAllUsesExcept(nceOutputBuff, parentCopyOpInputBuff, viewOp);
                    rewriter.replaceOpWithNewOp<VPUIP::ViewOp>(viewOp, viewOp->getResult(0).getType(),
                                                               vpux::VPUIP::getRootBuffer(parentCopyOpInputBuff));
                }
            }
        }
    }

    if (isCopyOpDistributed && grandparentOp->getResult(0).getType() != copyOp.getResult().getType()) {
        if (!isGrandparentCompatible) {
            // it may cause WLM failure in V0 PWLM mode
            if ((_workloadManagementMode > WorkloadManagementMode::PWLM_V0_LCA) &&
                isLegalAndBenefitCreateCopyFromCMXToCMX(grandparentOp, parentCopyOp, copyOp, nestedLogger)) {
                // rewrite with a new CMX2CMX copy
                rewriter.setInsertionPointAfter(copyOp);
                rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(copyOp, parentCopyOp.getInput(), copyOp.getOutputBuff());

                if (distributedCastOp && distributedCastOp->use_empty()) {
                    rewriter.eraseOp(distributedCastOp);
                }

                if (parentCopyOp->use_empty()) {
                    rewriter.eraseOp(parentCopyOp);
                }

                nestedLogger.trace(
                        "CopyOpSequence: successfully fused sequence of distributed copies into one CMX2CMX "
                        "Copy when the input type and output type have different memory views on each cluster");

                return mlir::success();
            } else {
                nestedLogger.trace("CopyOpSequence: cannot match because types aren't compatible when current CopyOp "
                                   "is distributed");
                return mlir::failure();
            }
        }

        // check CMX
        auto availableCMXSize = VPU::getTotalCMXSize(copyOp);
        auto rootBuffer = vpux::VPUIP::getRootBuffer(parentCopyOp.getInput());
        auto rootBufferType = mlir::dyn_cast<vpux::NDTypeInterface>(rootBuffer.getType());
        auto rootBufferSize = rootBufferType.getTotalAllocSize();
        auto copyOutputType = mlir::dyn_cast<vpux::NDTypeInterface>(copyOp.getOutput().getType());
        auto copyOutputSize = copyOutputType.getTotalAllocSize();

        if (rootBufferSize > copyOutputSize) {
            auto increasedBufferSize = rootBufferSize - copyOutputSize;
            for (auto user : copyOp.getOutput().getUsers()) {
                if (mlir::isa<VPUIP::NCEClusterTaskOp>(user)) {
                    Byte requiredCMX = VPUIP::getRequiredCMXSize(user);
                    if ((requiredCMX + increasedBufferSize) > availableCMXSize) {
                        nestedLogger.trace("CopyOpSequence: cannot match because CMX size is not enough for the "
                                           "increased size from root buffer");
                        return mlir::failure();
                    }
                }
            }
        }

        rewriter.setInsertionPointAfter(parentCopyOp);
        rewriter.replaceOpWithNewOp<VPUIP::DistributedCastOp>(copyOp, copyOp.getResult().getType(),
                                                              parentCopyOp.getInput());

        if (parentCopyOp->use_empty()) {
            rewriter.eraseOp(parentCopyOp);
        }
        nestedLogger.trace(
                "CopyOpSequence: successfully fused sequence of distributed copies into one op when producer type does "
                "not match the output type.");
        return mlir::success();
    }
    nestedLogger.trace("CopyOpSequence: didn't find NCEClusterTask eltwise in_place user. Fallback optimization.");

    return reduce();
}

//
// CMXToCMXCopy
//

bool isHighDimInputStrideCopy(VPUIP::CopyOp distributedCopyOp) {
    if (!mlir::isa_and_nonnull<VPUIP::SubViewOp>(distributedCopyOp.getOperand(0).getDefiningOp())) {
        return false;
    }
    // Copy cannot be eliminated for nested SubViewOps
    auto isNestedSubviewUser = llvm::any_of(distributedCopyOp->getUsers(), [](mlir::Operation* user) {
        return mlir::isa<VPUIP::SubViewOp>(user);
    });
    if (isNestedSubviewUser) {
        return false;
    }
    auto inputType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedCopyOp->getOperand(0).getType());
    auto outputType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedCopyOp->getResult(0).getType());
    const auto inputElemSize = inputType.getElemTypeSize();
    const auto inputShape = inputType.getShape();
    const auto inputLayout = inputType.getDimsOrder();
    const auto outputElemSize = outputType.getElemTypeSize();
    const auto outputShape = outputType.getShape();
    const auto outputLayout = outputType.getDimsOrder();
    if (inputElemSize != outputElemSize || inputShape != outputShape || inputLayout != outputLayout) {
        return false;
    }
    auto inputMemShape = inputType.getMemShape().raw();
    auto inputMemStrides = inputType.getMemStrides().raw();
    auto getStrideDim = [&]() -> Dim {
        for (auto ind : irange(inputMemShape.size()) | reversed) {
            auto dim = Dim(ind);
            if (ind == inputMemShape.size() - 1 && inputMemStrides[ind] != inputElemSize) {
                return dim;
            } else if (ind != inputMemShape.size() - 1) {
                const auto prevMemDim = ind + 1;
                if (inputMemStrides[ind] != inputMemStrides[prevMemDim] * inputMemShape[prevMemDim]) {
                    return dim;
                }
            }
        }
        return Dim(0);
    };
    auto strideDim = getStrideDim();
    return strideDim == Dims4D::Act::N;
}

bool isDistributedInOutCompatible(VPUIP::CopyOp distributedCopyOp) {
    const auto distributedCopyInput = distributedCopyOp->getOperand(0);
    const auto distributedCopyOutput = distributedCopyOp->getResult(0);
    const auto inDistributedType =
            mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(VPUIP::extractDataType(distributedCopyInput));
    const auto outDistributedType =
            mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(VPUIP::extractDataType(distributedCopyOutput));
    if (inDistributedType != outDistributedType) {
        if (inDistributedType == nullptr || outDistributedType == nullptr) {
            return false;
        }

        if (VPU::areDistributionAttrsCompatible(inDistributedType, outDistributedType).failed()) {
            return false;
        }
    }

    return true;
}

bool isExcludedUser(mlir::Operation* op) {
    // For normal case, NCE or groupOp conncet to ConcatView directly
    if (mlir::isa<VPUIP::ConcatViewOp>(op)) {
        return true;
    }

    // For sparse with distributedCast case, NCE or groupOp conncet to distributedCastOp
    if (auto castOp = mlir::dyn_cast<VPUIP::DistributedCastOp>(op)) {
        if (castOp->hasOneUse() && mlir::isa<VPUIP::ConcatViewOp>(*castOp.getResult().getUsers().begin())) {
            return true;
        }
    }
    return false;
}

bool needInsertCopies(mlir::Operation* op, size_t resultIndex) {
    if (op->use_empty()) {
        return false;
    }

    for (auto user : op->getResult(resultIndex).getUsers()) {
        if (VPUIP::isPureViewOp(user)) {
            if (isExcludedUser(user)) {
                continue;
            }

            // currently we can only propagate stride through quantizeCast, but could not for other view like. For
            // example: 1x16x32x64 genericReshape to 1x16x16x128(NCHW), if input stride is in H [33280,2080,65,1], don't
            // know how to set output stride. Some special case may work, like input stride in C [34816,2048,64, 1],
            // but haven't been handled.
            if (mlir::isa<VPUIP::QuantizeCastOp>(user)) {
                VPUX_THROW_UNLESS(user->getNumResults() == 1, "QuantizeCastOp must have single output");
                if (needInsertCopies(user, 0)) {
                    return true;
                }
                continue;
            }

            // Insert copies for other view like operation
            return true;
        }

        if (!mlir::isa<VPUIP::CopyOp>(user)) {
            return true;
        }
    }
    return false;
}

void propagateStrideInfo(mlir::Operation* parent, size_t resultIndex, mlir::PatternRewriter& rewriter) {
    if (parent->use_empty()) {
        return;
    }

    auto origOutType = mlir::cast<vpux::NDTypeInterface>(parent->getResult(resultIndex).getType());
    const auto inReqs = StrideReqs::compact(origOutType.getRank());
    if (inReqs.checkStrides(origOutType)) {
        return;
    }
    auto parentStrides = getStrides(parent->getResult(resultIndex));

    for (auto user : llvm::make_early_inc_range(parent->getResult(resultIndex).getUsers())) {
        if (isExcludedUser(user)) {
            continue;
        }

        if (mlir::isa<VPUIP::CopyOp>(user)) {
            continue;
        }

        if (mlir::isa<VPUIP::QuantizeCastOp>(user)) {
            VPUX_THROW_UNLESS(user->getNumResults() == 1, "QuantizeCastOp must have single output");
            auto origType = mlir::cast<vpux::NDTypeInterface>(user->getResult(0).getType());
            auto newType = origType.changeStrides(parentStrides);
            user->getResult(0).setType(newType);
            propagateStrideInfo(user, 0, rewriter);
            continue;
        }

        if (vpux::VPUIP::hasDistributedOperand(user)) {
            // DistrCopy need to re-create to make sure stride info propagated.
            auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(user);
            rewriter.setInsertionPointAfter(parent);
            auto newCopyOp =
                    rewriter.create<VPUIP::CopyOp>(copyOp->getLoc(), copyOp->getOperand(0), copyOp.getOutputBuff());
            auto allocOp = copyOp.getOutputBuff().getDefiningOp();
            rearrangeOperations(allocOp, newCopyOp, true);
            rewriter.replaceOp(copyOp, newCopyOp->getResult(0));
            continue;
        }

        VPUX_THROW("Unsupported operation type {0} to propagate stride info", user->getName());
    }
}

void insertCopiesAfterNCETask(VPUIP::NCEClusterTaskOp parentNCE, size_t resultIndex, mlir::Type origType,
                              mlir::PatternRewriter& rewriter) {
    auto nceOutType = mlir::dyn_cast<vpux::NDTypeInterface>(origType);
    rewriter.setInsertionPointAfter(parentNCE);
    // To DDR
    auto newDDRType = nceOutType.changeMemSpace(VPU::MemoryKind::DDR);
    auto newAllocDDROp = rewriter.create<mlir::memref::AllocOp>(appendLoc(parentNCE->getLoc(), "_new_DDR_buffer"),
                                                                mlir::cast<mlir::MemRefType>(newDDRType));
    auto newCopyToDDR = rewriter.create<VPUIP::CopyOp>(appendLoc(parentNCE->getLoc(), "_stride_to_compact"),
                                                       parentNCE->getResult(resultIndex), newAllocDDROp);

    // To CMX
    auto newAllocCMXOp = rewriter.create<mlir::memref::AllocOp>(appendLoc(parentNCE->getLoc(), "_new_CMX_buffer"),
                                                                mlir::cast<mlir::MemRefType>(nceOutType));
    auto newCopyToCMX = rewriter.create<VPUIP::CopyOp>(parentNCE->getLoc(), newCopyToDDR.getResult(), newAllocCMXOp);

    rewriter.replaceUsesWithIf(parentNCE->getResult(resultIndex), newCopyToCMX.getResult(),
                               [&](mlir::OpOperand& opOperand) {
                                   return opOperand.getOwner() != newCopyToDDR && !isExcludedUser(opOperand.getOwner());
                               });
}

void insertCopiesAfterNCETaskDistributedBuffer(VPUIP::NCEClusterTaskOp nceTask, size_t resultIndex, mlir::Type origType,
                                               mlir::PatternRewriter& rewriter) {
    auto nceOutDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(origType);
    auto nceOutType = mlir::dyn_cast<vpux::NDTypeInterface>(nceOutDistributedType.getCompactType());
    rewriter.setInsertionPointAfter(nceTask);
    // To DDR
    auto newDDRType = nceOutType.changeMemSpace(VPU::MemoryKind::DDR);
    auto newAllocDDROp = rewriter.create<mlir::memref::AllocOp>(appendLoc(nceTask->getLoc(), "_new_DDR_buffer"),
                                                                mlir::cast<mlir::MemRefType>(newDDRType));

    auto newTillingCopyToDDR =
            rewriter.create<VPUIP::CopyOp>(appendLoc(nceTask->getLoc(), "_stride_to_compact"),
                                           nceTask->getResult(resultIndex), static_cast<mlir::Value>(newAllocDDROp));
    // To CMX
    auto newDistributeBuff = rewriter.create<VPURT::AllocDistributed>(appendLoc(nceTask->getLoc(), "_new_CMX_buffer"),
                                                                      nceOutDistributedType, nullptr, nullptr);
    auto newTillingCopyToCMX = rewriter.create<VPUIP::CopyOp>(nceTask->getLoc(), newTillingCopyToDDR->getResult(0),
                                                              static_cast<mlir::Value>(newDistributeBuff.getBuffer()));

    rewriter.replaceUsesWithIf(
            nceTask->getResult(resultIndex), newTillingCopyToCMX->getResult(0), [&](mlir::OpOperand& opOperand) {
                return opOperand.getOwner() != newTillingCopyToDDR && !isExcludedUser(opOperand.getOwner());
            });
}

void handleStrideForOtherUsers(mlir::Operation* parent, size_t resultIndex, mlir::Type origType,
                               mlir::PatternRewriter& rewriter, Logger log) {
    if (needInsertCopies(parent, resultIndex)) {
        if (auto nceTask = mlir::dyn_cast_or_null<VPUIP::NCEClusterTaskOp>(parent)) {
            if (!vpux::VPUIP::hasDistributedOperand(nceTask)) {
                insertCopiesAfterNCETask(nceTask, resultIndex, origType, rewriter);
            } else {
                insertCopiesAfterNCETaskDistributedBuffer(nceTask, resultIndex, origType, rewriter);
            }
        } else {
            VPUX_THROW("Incorrect parent type {0}", parent->getName());
        }
        log.trace("Insert a pair of copy to handle stride");
    } else {
        propagateStrideInfo(parent, resultIndex, rewriter);
        log.trace("Propagate stride info to child");
    }
}

/// Finds an op->getResult(x) that is equal to value. Returns result index.
size_t findMatchingResultIndex(mlir::Value value, mlir::Operation* op) {
    const auto& results = op->getResults();
    auto it = llvm::find(results, value);
    VPUX_THROW_WHEN(it == results.end(), "Failed to find {0} in op's results: {1}", value, op);
    return std::distance(op->getResults().begin(), it);
}

class CMXToCMXCopy final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CMXToCMXCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger log;
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

     For Cluster-ed scenario, it is possible to have:
        Copy(DDR2CMX)                Copy(DDR2CMX)  SubView
              |                            |           |
            NCEOp        =>                |    (DistributedCast)
              |                            \        / (output_buff)
    (DistributedCast)  SubView                NCEOp
               \         / (output_buff)
              Copy(CMX2CMX)

    For Cluster-ed scenario with sparsity map, final subgraph should be:
               Alloc Data    Alloc SparsityMap
                    |                |
                 SubView          SubView    -> (if original SubView output had multiple
                    |                |           consumers, this is their new producer)
            (DistributedCast) (DistributedCast)
                    \                /
  (output_data_buff) \              / (output_sparsity_map_buff)
                      \            /
                           NCEOp


     */
    // Check current CopyOp source and destination
    log.trace("CMX2CMXCopy: Copy at {0}", copyOp->getLoc());
    auto nestedLogger = log.nest();

    auto inputType = mlir::cast<vpux::NDTypeInterface>(copyOp.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(copyOp.getOutput().getType());

    // Only remove redundant CMX2CMX CopyOps
    if (!isCMX2CMXCopy(inputType.getMemoryKind(), outputType.getMemoryKind())) {
        nestedLogger.trace("Cannot match because the transfer is not CMX->CMX");
        return mlir::failure();
    }

    auto distributedCast = copyOp.getInput().getDefiningOp<VPUIP::DistributedCastOp>();

    // if detect the subview before input, remove the copies may cause CMX OOM
    // so check whether removing the CMX2CMX copy will exceed CMX limitation
    auto isInputCompatible = true;
    if (!vpux::VPUIP::hasDistributedOperand(copyOp)) {
        if (auto definingOp = copyOp.getInput().getDefiningOp()) {
            if (auto subViewOpBefore = mlir::dyn_cast<VPUIP::SubViewOp>(definingOp)) {
                // find the pattern: source data -> subview -> copy -> users
                // check for all users, if any user's CMX size exceeds the limitation, then cannot remove the copy
                auto newInputCmxSize =
                        mlir::cast<vpux::NDTypeInterface>(subViewOpBefore.getSource().getType()).getTotalAllocSize();
                for (auto copyOpUser : copyOp.getOutput().getUsers()) {
                    Byte requiredCMX = VPUIP::getRequiredCMXSize(copyOpUser);
                    requiredCMX -= inputType.getTotalAllocSize();
                    requiredCMX += newInputCmxSize;
                    nestedLogger.trace("CMX Demand: {0} -> {1} for {2}", VPUIP::getRequiredCMXSize(copyOpUser),
                                       requiredCMX, copyOpUser->getLoc());
                    if (requiredCMX > VPU::getTotalCMXSize(copyOp)) {
                        isInputCompatible = false;
                        break;
                    }
                }
            }
        }

        // TileOp's required CMX is always bigger than CopyOp.
        // For Dynamic_Tile, its boundary of input and output is same when compiling to looks like needing same CMX.
        // But for runtime, it gets actual shapes. So removing the copies may cause runtime issue.
        for (auto copyOpUser : copyOp->getUsers()) {
            if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(copyOpUser)) {
                if (getSwKernelEntryName(swKernelOp) == "dynamic_tile") {
                    isInputCompatible = false;
                    break;
                }
            }
        }
    } else {
        isInputCompatible = isDistributedInOutCompatible(copyOp);
    }

    nestedLogger.trace("isInputCompatible={0}", isInputCompatible);

    // CMX Concat case with SubView, update the buffers used
    if (auto copySubView = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>()) {
        nestedLogger.trace("Found copySubView: {0}", copySubView);
        // case with SubView - retrieve operations to be re-linked
        auto parentNCE = distributedCast == nullptr ? copyOp.getInput().getDefiningOp()
                                                    : distributedCast.getInput().getDefiningOp();

        if (!vpux::VPUIP::hasDistributedOperand(copyOp) && !mlir::isa<VPUIP::NCEClusterTaskOp>(parentNCE)) {
            nestedLogger.trace("Cannot match because copy operation is non-distributed so it must be a successor of "
                               "NCEClusterTask");
            return mlir::failure();
        }

        if (vpux::VPUIP::hasDistributedOperand(copyOp) && mlir::isa<VPUIP::SubViewOp>(parentNCE)) {
            nestedLogger.trace("Cannot remove Copy because of SubViewOp");
            return mlir::failure();
        }

        auto copySubViewInput = copySubView->getOperand(0);
        auto masterBuffer = llvm::TypeSwitch<mlir::Type, mlir::Operation*>(copySubViewInput.getType())
                                    .Case<mlir::MemRefType>([&copySubViewInput](mlir::Type) {
                                        return VPUIP::getRootAlloc<mlir::memref::AllocOp>(copySubViewInput);
                                    })
                                    .Case<VPUIP::DistributedBufferType>([&copySubViewInput](mlir::Type) {
                                        return VPUIP::getRootAlloc<VPURT::AllocDistributed>(copySubViewInput);
                                    })
                                    .Default([&nestedLogger](mlir::Type type) {
                                        nestedLogger.trace("Unknown buffer type: {0}", type);
                                        return nullptr;
                                    });

        if (masterBuffer == nullptr) {
            nestedLogger.trace("Cannot match because source isn't master buffer");
            return mlir::failure();
        }

        if (!vpux::VPUIP::hasDistributedOperand(copyOp)) {
            VPUIP::moveRootAllocBefore(copySubView, parentNCE);
        }

        const auto updateParentNCEOp = [&parentNCE](size_t argIdx, mlir::Value value) {
            // Update result types of NCEClusterTask
            parentNCE->getResult(checked_cast<unsigned int>(argIdx)).setType(value.getType());
            // Update output buffers of NCEClusterTask
            mlir::OperandRange layerOutputs = VPUIP::getLayerOutputs(parentNCE);
            layerOutputs[argIdx].replaceAllUsesWith(value);
            layerOutputs[argIdx].setType(value.getType());
        };

        const auto inputValue = (distributedCast == nullptr) ? copyOp.getInputs()[0] : distributedCast.getInput();
        const auto outputBuffIndex = findMatchingResultIndex(inputValue, parentNCE);
        const auto origType = mlir::dyn_cast<vpux::NDTypeInterface>(parentNCE->getResult(outputBuffIndex).getType());

        if (vpux::VPUIP::hasDistributedOperand(copyOp) &&
            VPUIP::getLayerOutputs(parentNCE)[outputBuffIndex].getDefiningOp<VPUIP::SubViewOp>()) {
            nestedLogger.trace("NCE output is already the subview of Concat");
            return mlir::failure();
        }

        auto nceClusterOutput = copySubView.getResult();
        if (vpux::VPUIP::hasDistributedOperand(copyOp)) {
            copySubView->moveBefore(parentNCE);

            // replace the copy with the subView
            if (distributedCast != nullptr) {
                rewriter.setInsertionPointAfter(copySubView);
                auto ndTypeIfValue = mlir::cast<vpux::NDTypeInterface>(copySubView.getType());
                const auto strides = ndTypeIfValue.getStrides();
                auto distributedCastType = mlir::cast<vpux::NDTypeInterface>(distributedCast->getOperand(0).getType())
                                                   .changeStrides(strides);

                nestedLogger.trace("Creating DistributedCastOp with input = {0} and output type = {1}.", copySubView,
                                   distributedCastType);

                auto newDistrCast = rewriter.create<VPUIP::DistributedCastOp>(parentNCE->getLoc(), distributedCastType,
                                                                              copySubView);
                nceClusterOutput = newDistrCast.getResult();
            }
        }

        // replace the copy with the subView
        updateParentNCEOp(outputBuffIndex, nceClusterOutput);
        rewriter.replaceAllUsesWith(copyOp->getResult(0), parentNCE->getResult(outputBuffIndex));

        // update IR location of the master buffer
        rearrangeOperations(masterBuffer, copySubView, false);

        rewriter.eraseOp(copyOp);
        if (distributedCast != nullptr) {
            rewriter.eraseOp(distributedCast);
        }

        // now we need to propagate stride info to other users
        handleStrideForOtherUsers(parentNCE, outputBuffIndex, origType, rewriter, log);
        nestedLogger.trace("Successfully optimized ConcatView with SubView");
    } else if (inputType == outputType && isInputCompatible &&
               ((!vpux::VPUIP::hasDistributedOperand(copyOp)) ||
                (vpux::VPUIP::hasDistributedOperand(copyOp) && isHighDimInputStrideCopy(copyOp)))) {
        // case with no subView after output
        rewriter.replaceAllUsesWith(copyOp.getOutput(), copyOp.getInput());
        rewriter.eraseOp(copyOp);
        if (distributedCast != nullptr) {
            rewriter.eraseOp(distributedCast);
        }
        nestedLogger.trace("Successfully optimized ConcatView without SubView");
    } else {
        log.trace("Copy not optimized {0}", copyOp->getLoc());
        return mlir::failure();
    }

    nestedLogger.trace("Successfully removed sequence");
    return mlir::success();
}

//
// DDRToDDRCopy
//

class DDRToDDRCopy final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    DDRToDDRCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger log;
};

bool isDDR2DDRCopyInput(VPUIP::CopyOp copyOp) {
    // ChildOp should be a distributed copy op
    if (copyOp.getOutput().getUsers().empty()) {
        return false;
    }

    auto isDistributedCopyOp = [](mlir::Operation* user) {
        if (auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(user)) {
            return VPUIP::hasDistributedOperand(copyOp);
        }
        return false;
    };

    auto isLegalUpdateViewLikeInType = [](mlir::Operation* op, mlir::Value newInput) {
        auto iface = mlir::dyn_cast<mlir::InferTypeOpInterface>(*op);
        SmallVector<mlir::Type> newTypes;
        const auto isLegal =
                iface.inferReturnTypes(op->getContext(), op->getLoc(), mlir::ValueRange{newInput},
                                       op->getAttrDictionary(), op->getPropertiesStorage(), op->getRegions(), newTypes)
                        .succeeded();
        return isLegal;
    };

    for (auto copyOpUser : copyOp.getOutput().getUsers()) {
        // TODO: Extend for other ViewLike ops E#74293
        if (mlir::isa<VPUIP::ShapeCastOp, VPUIP::SubViewOp>(copyOpUser) &&
            mlir::isa<mlir::InferTypeOpInterface>(copyOpUser)) {
            if (!isLegalUpdateViewLikeInType(copyOpUser, copyOp.getInput())) {
                return false;
            }

            for (auto pureViewOpUser : copyOpUser->getResult(0).getUsers()) {
                if (!isDistributedCopyOp(pureViewOpUser)) {
                    return false;
                }
            }
        } else if (!isDistributedCopyOp(copyOpUser)) {
            return false;
        }
    }

    return true;
}

bool hasValidParallelCopyBranchWithSubView(VPUIP::CopyOp copyOp, VPUIP::CopyOp parentOp) {
    if (parentOp->hasOneUse()) {
        return false;
    }

    auto subview = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
    if (subview == nullptr) {
        return false;
    }

    // If a CMX to DDR copy's input is a subview of SOH's output, the CMX2DDR copy's input tensor will have a SEGMENTED
    // or OVERLAPPED distribution. But the output data of the tensor's subview may be distributed on one cluster or
    // multiple clusters.In the current compiler logic, when calculating DMA cost and unroll DMA, it is assumed that the
    // data of the Tensor with SEGMENTED or OVERLAPPED distribution is distributed on multiple clusters. Therefore, SOH
    // optimization is temporarily turned off and turned on after subsequent compiler support.E60342
    if (auto distType =
                mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(VPUIP::extractDataType(parentOp.getInput()))) {
        if (distType.getDistribution().getMode().getValue() == VPU::DistributionMode::SEGMENTED ||
            distType.getDistribution().getMode().getValue() == VPU::DistributionMode::OVERLAPPED) {
            auto subviewshape = mlir::cast<vpux::NDTypeInterface>(subview.getResult().getType()).getShape().raw();
            auto numTiles = parseIntArrayAttr<int64_t>(distType.getDistribution().getNumTiles());
            if (subviewshape.size() == 4 && subviewshape[Dims4D::Act::H.ind()] % numTiles[Dims4D::Act::H.ind()] != 0) {
                return false;
            }

            // In case of the CMX2DDR copy's input tensor has a SEGMENTED or OVERLAPPED distribution and the tile Axis
            // is H, and if the output data of the tensor's subview has tile offsets including H, then the tile result
            // may be incorrect after SOH optimization (When the offset is a non first cluster / the offset is only
            // used in a single cluster or a few clusters / the offset exists across two consecutive clusters), as
            // current compiler logic not support this case in calculating DMA cost and unroll DMA
            // TODO: Add optimization for this case, #E80157
            for (auto user : llvm::make_early_inc_range(parentOp.getResult().getUsers())) {
                if (auto subview = mlir::dyn_cast<VPUIP::SubViewOp>(*user)) {
                    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*subview.getResult().getUsers().begin());
                    if (vpux::VPUIP::hasDistributedOperand(copyOp)) {
                        continue;
                    }
                    auto offsetAttr = subview.getStaticOffsetsAttr();
                    const auto offsetsArray = parseIntArrayAttr<int64_t>(offsetAttr);
                    const auto tilingScheme = parseIntArrayAttr<int64_t>(distType.getDistribution().getNumTiles());
                    const auto tileAxis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
                    if (copyOp && offsetsArray[tileAxis]) {
                        return false;
                    }
                }
            }
        }
    }

    // check other parallel branch if it's a valid copy branch or not
    for (auto siblingOp : parentOp.getResult().getUsers()) {
        // Considering padding/slice case: distributed copy -> subview -> copy

        if (auto siblingSubview = mlir::dyn_cast<VPUIP::SubViewOp>(*siblingOp)) {
            if (!siblingSubview.getResult().hasOneUse()) {
                return false;
            }

            auto childOp = siblingSubview.getResult().getUsers().begin();
            auto childCopy = mlir::dyn_cast<VPUIP::CopyOp>(*childOp);
            // If childCopy is nullptr or its output buffer is not defined by a SubViewOp, return false.
            if (!childCopy || vpux::VPUIP::hasDistributedOperand(childCopy) ||
                !childCopy.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>()) {
                return false;
            }

        } else if (auto siblingCopy = mlir::dyn_cast<VPUIP::CopyOp>(*siblingOp)) {
            if (vpux::VPUIP::hasDistributedOperand(siblingCopy)) {
                return false;
            }
            // If siblingCopy is not the same as copyOp and its output buffer is not defined by a SubViewOp, return
            // false.
            if (siblingCopy != copyOp && !siblingCopy.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>()) {
                return false;
            }

        } else {
            return false;
        }
    }
    // check all branches and okay
    return true;
}

// For the case: parent of copyOp only have one output branch
// Parallel case should be processed by isParallelDDR2DDROfOutput()
// for clear logic
bool isDDR2DDROutput(VPUIP::CopyOp copyOp) {
    // ParentOp should be a distributed copy op
    // ChildOp should be a concat
    auto parentOp = copyOp->getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    if (parentOp == nullptr || !vpux::VPUIP::hasDistributedOperand(parentOp)) {
        return false;
    }
    if (copyOp.getOutput().getUsers().empty()) {
        return false;
    }
    for (auto user : copyOp.getOutput().getUsers()) {
        if (!mlir::isa<VPUIP::ConcatViewOp>(*user)) {
            return false;
        }
    }

    return parentOp->hasOneUse();
}

bool isParallelDDR2DDROutput(VPUIP::CopyOp copyOp) {
    // ParentOp should be a distributed copy
    // ChildOp should be a concat
    auto parentOp = copyOp->getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    if (parentOp == nullptr || !vpux::VPUIP::hasDistributedOperand(parentOp)) {
        return false;
    }

    if (copyOp.getOutput().getUsers().empty()) {
        return false;
    }
    for (auto user : copyOp.getOutput().getUsers()) {
        if (!mlir::isa<VPUIP::ConcatViewOp>(*user)) {
            return false;
        }
    }

    /*
     Optimize the parallel DDR2DDR copies as CMX2DDR copies:
                 DistributedCopy(CMX2DDR)
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

    const auto outType = mlir::cast<vpux::NDTypeInterface>(copyOp.getOutputBuff().getType());
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

bool isDDR2DDRConcatInput(VPUIP::CopyOp copyOp) {
    // ParentOp should be a concatView op
    // ChildOp should be a concatView too
    auto parentConcatOp = copyOp.getInput().getDefiningOp<VPUIP::ConcatViewOp>();
    if (parentConcatOp == nullptr) {
        return false;
    }
    if (!copyOp.getOutput().hasOneUse()) {
        return false;
    }

    auto childConcatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(*copyOp.getOutput().getUsers().begin());
    if (childConcatOp == nullptr) {
        return false;
    }

    // Exclude strided dma case
    size_t constCopyCnt = 0;
    auto predicteChildConcatInput = [&](mlir::Value op) {
        auto copy = op.getDefiningOp<VPUIP::CopyOp>();
        if (copy == nullptr || isStridedCopy(copy) || vpux::VPUIP::hasDistributedOperand(copy)) {
            return false;
        }

        auto concat = copy.getInput().getDefiningOp<VPUIP::ConcatViewOp>();
        if (concat == nullptr) {
            auto subView = copy.getInput().getDefiningOp<VPUIP::SubViewOp>();
            if (subView == nullptr) {
                auto parentCopyInputConst = VPUIP::getRootConst(copy.getInput());
                if (parentCopyInputConst) {
                    constCopyCnt++;
                    return true;
                }
                return false;
            } else if (!subView.getResult().hasOneUse()) {
                return false;
            }
            concat = subView.getSource().getDefiningOp<VPUIP::ConcatViewOp>();
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
    if (!llvm::all_of(childConcatOp.getInputs(), predicteChildConcatInput)) {
        return false;
    }

    const auto childConcatInputsNum = childConcatOp.getInputs().size();

    const auto parentConcatUsers = parentConcatOp.getOutput().getUsers();
    const auto parentConcatUsersNum = std::distance(parentConcatUsers.begin(), parentConcatUsers.end());

    return (childConcatInputsNum - constCopyCnt) == static_cast<size_t>(parentConcatUsersNum);
}

mlir::LogicalResult removeDDR2DDRCopyInput(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter, Logger log) {
    rewriter.replaceAllUsesWith(copyOp.getOutput(), copyOp.getInput());

    // Update ViewLike Op Output Type
    SmallVector<mlir::Operation*> viewLikeOps;
    for (auto copyOpUser : copyOp.getInput().getUsers()) {
        if (mlir::isa<VPUIP::ShapeCastOp, VPUIP::SubViewOp>(copyOpUser)) {
            viewLikeOps.push_back(copyOpUser);
        }
    }

    for (auto viewLikeOp : viewLikeOps) {
        vpux::inferReturnTypes(viewLikeOp, vpux::InferShapedTypeMode::ALL);
    }

    log.trace("Successfully removed DDRToDDR input copy {0} at {1}", copyOp->getName(), copyOp->getLoc());
    rewriter.eraseOp(copyOp);
    return mlir::success();
}

mlir::LogicalResult removeDDR2DDROutput(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter, Logger log) {
    // CMX Concat case with subView, update the buffers used
    if (auto subViewOp = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>()) {
        // case with subView - retrieve operations to be re-linked
        auto masterBuffer = VPUIP::getRootAlloc<mlir::memref::AllocOp>(subViewOp->getOperand(0));
        if (masterBuffer == nullptr) {
            log.trace("Cannot match because source isn't master buffer");
            return mlir::failure();
        }
        auto parentOp = copyOp.getInput().getDefiningOp();
        // replace the copy with VPUIP subView
        rewriter.setInsertionPoint(parentOp);
        auto newSubViewOp = rewriter.create<VPUIP::SubViewOp>(
                subViewOp->getLoc(), subViewOp.getSource(), subViewOp.getStaticOffsetsAttr(),
                subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());
        rewriter.replaceAllUsesWith(VPUIP::getLayerOutputs(parentOp)[0], newSubViewOp->getResult(0));
        parentOp->getResult(0).setType(newSubViewOp->getResult(0).getType());

        // update IR location of the master buffer
        rearrangeOperations(masterBuffer, newSubViewOp, false);
    } else {
        auto parentOp = copyOp.getInput().getDefiningOp();
        VPUX_THROW_WHEN(!vpux::VPUIP::hasDistributedOperand(parentOp), "Expected a distributed op.");
        auto allocOp = VPUIP::getRootAlloc<mlir::memref::AllocOp>(VPUIP::getLayerOutputs(parentOp)[0]);
        if (allocOp == nullptr) {
            log.trace("Cannot match because source isn't master buffer");
            return mlir::failure();
        }

        for (auto user : copyOp.getOutput().getUsers()) {
            auto concatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(user);
            concatOp.getOutputBuff().replaceAllUsesWith(allocOp->getResult(0));
        }
    }

    rewriter.replaceAllUsesWith(copyOp.getOutput(), copyOp.getInput());
    log.trace("Successfully removed DDRToDDR output copy {0} at {1}", copyOp->getName(), copyOp->getLoc());
    rewriter.eraseOp(copyOp);
    return mlir::success();
}

mlir::LogicalResult removeParallelDDR2DDROutput(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter, Logger log) {
    auto parentOp = copyOp->getOperand(0).getDefiningOp();

    for (auto user : llvm::make_early_inc_range(parentOp->getResult(0).getUsers())) {
        if (auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*user)) {
            auto subview = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();

            rewriter.setInsertionPointAfter(subview);
            auto newCopyInCluster =
                    rewriter.create<VPUIP::CopyOp>(parentOp->getLoc(), parentOp->getOperand(0), subview.getResult());

            rewriter.replaceAllUsesWith(copyOp.getOutput(), newCopyInCluster->getResult(0));

            log.trace("Successfully removed Parallel DDRToDDR output copy {0} at {1}", copyOp->getName(),
                      copyOp->getLoc());
            rewriter.eraseOp(copyOp);
        }
    }

    for (auto user : llvm::make_early_inc_range(parentOp->getResult(0).getUsers())) {
        if (auto subview = mlir::dyn_cast<VPUIP::SubViewOp>(*user)) {
            auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*subview.getResult().getUsers().begin());
            if (copyOp == nullptr) {
                log.trace("CopyOp is null");
                continue;
            }
            auto outputSubview = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
            if (outputSubview == nullptr) {
                log.trace("Output subview is null");
                continue;
            }

            rewriter.setInsertionPointAfter(copyOp);
            // New a new subview for copy output
            auto newSubView = rewriter.create<VPUIP::SubViewOp>(
                    subview->getLoc(), parentOp->getOperand(0), subview.getStaticOffsetsAttr(),
                    subview.getStaticSizesAttr(), subview.getStaticStridesAttr());

            auto newCopyInCluster = rewriter.create<VPUIP::CopyOp>(parentOp->getLoc(), newSubView.getResult(),
                                                                   outputSubview.getResult());

            rewriter.replaceAllUsesWith(copyOp.getOutput(), newCopyInCluster->getResult(0));
            log.trace("Successfully removed Parallel DDRToDDR output copy (with input subview) {0} at {1}",
                      copyOp->getName(), copyOp->getLoc());
            rewriter.eraseOp(copyOp);
            rewriter.eraseOp(subview);
        }
    }

    if (parentOp->use_empty()) {
        rewriter.eraseOp(parentOp);
    }
    return mlir::success();
}

static inline bool checkOpsSupportInferType(mlir::Operation* startOp, mlir::Operation* endOp, Logger log) {
    auto currentOp = startOp;

    while (currentOp != endOp) {
        if (!mlir::isa<mlir::InferTypeOpInterface, mlir::memref::AllocOp>(currentOp)) {
            log.trace("Unexpected op {0} at {1}", currentOp->getName(), currentOp->getLoc());
            return false;
        }
        currentOp = currentOp->getNextNode();
    }
    return true;
}

static inline void inferOpsTypeBetween(mlir::Operation* startOp, mlir::Operation* endOp) {
    auto currentOp = startOp;

    while (currentOp != endOp) {
        // In case the currentOp doesn't support mlir::InferTypeOpInterface,
        // then will setType based on the SubViewOp of this CopyOp,
        // no adapt to set the inner type as only the strides changed.

        // Only AllocOp and CopyOp will call this if func after checkOpsSupportInferType func
        // no need to infer AllocOp's type
        if (!mlir::isa<mlir::InferTypeOpInterface>(currentOp)) {
            auto distributedCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(currentOp);
            if (distributedCopyOp != nullptr && vpux::VPUIP::hasDistributedOperand(distributedCopyOp)) {
                for (auto result : currentOp->getResults() | indexed) {
                    result.value().setType(distributedCopyOp.getOutputBuff().getType());
                }
                currentOp = currentOp->getNextNode();
            } else if (mlir::isa<mlir::memref::AllocOp>(currentOp)) {
                currentOp = currentOp->getNextNode();
                continue;
            } else {
                VPUX_THROW("Unexpected op type '{0}' at '{1}'", currentOp->getName(), currentOp->getLoc());
            }
        } else {
            vpux::inferReturnTypes(currentOp, vpux::InferShapedTypeMode::ALL);
            currentOp = currentOp->getNextNode();
        }
    }
}

mlir::LogicalResult removeDDR2DDRConcatInput(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter, Logger log) {
    auto parentConcatOp = copyOp.getInput().getDefiningOp<VPUIP::ConcatViewOp>();
    auto parentMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(parentConcatOp.getOutputBuff());
    if (parentMemAlloc == nullptr) {
        log.trace("Cannot match because parentConcatOp output isn't master buffer");
        return mlir::failure();
    }

    auto childConcatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(*copyOp.getOutput().getUsers().begin());
    auto childMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(childConcatOp.getOutputBuff());
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

    if (!checkOpsSupportInferType(parentMemAlloc, childConcatOp, log)) {
        log.trace("Cannot match because some Ops doesn't support InferTypeOpInterface");
        return mlir::failure();
    }

    log.trace("Successfully removed DDRToDDR output copy {0} at {1} for Concat", copyOp->getName(), copyOp->getLoc());
    auto childCopySubview = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();

    auto newSubViewOp = rewriter.create<VPUIP::SubViewOp>(parentMemAlloc->getLoc(), childCopySubview.getSource(),
                                                          childCopySubview.getStaticOffsetsAttr(),
                                                          childCopySubview.getStaticSizesAttr());

    // update IR location of the master buffer
    if (parentMemAlloc->isBeforeInBlock(newSubViewOp)) {
        VPUIP::moveRootAllocBefore(newSubViewOp, parentMemAlloc);
    }
    // update IR location of the master buffer
    if (newSubViewOp->isBeforeInBlock(childMemAlloc)) {
        VPUIP::moveRootAllocBefore(childMemAlloc, newSubViewOp);
    }

    rewriter.replaceAllUsesWith(parentMemAlloc->getResult(0), newSubViewOp.getResult());
    rewriter.eraseOp(parentMemAlloc);
    // Re-Infer the Type of the Ops
    inferOpsTypeBetween(newSubViewOp, childConcatOp);

    rewriter.replaceAllUsesWith(copyOp.getOutput(), copyOp.getInput());
    rewriter.eraseOp(copyOp);
    return mlir::success();
}

mlir::LogicalResult DDRToDDRCopy::matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const {
    if (vpux::VPUIP::hasDistributedOperand(copyOp)) {
        return mlir::failure();
    }
    /*
     Remove redundant DDR2DDR Copy of the NCECluster's input:
DistributedCopy                    ...        SubView
   (CMX2DDR)        SubView             \         /
          \         /              DistributedCopy(CMX2DDR)
          Copy(DDR2DDR)        =>            |
               |                           Concat
            Concat

     Remove redundant DDR2DDR Copy of the NCECluster's output:
          Copy(DDR2DDR)                                PureViewOp(Optional)
                |                                             |
        PureViewOp(Optional: ShapeCast, SubView)       DistributedCopy
                |                                         (DDR2CMX)
        DistributedCopy              =>                    |
            (DDR2CMX)                                  DistributedNCE
                |                                             |
        DistributedNCE
                |

     Optimize the parallel DDR2DDR copies as CMX2DDR copies:
                DistributedCopy(CMX2DDR)
                      /        \
            Copy(DDR2DDR)       Copy(DDR2DDR)       =>
            /        \          /       \
        SubView           |            SubView
                          |
                        Concat

                         ...
                     /          \
DistributedCopy(CMX2DDR)   DistributedCopy(CMX2DDR)
            /        \          /       \
        SubView           |            SubView
                          |
                        Concat
     */
    log.trace("DDRToDDRCopy: Copy at {0}", copyOp->getLoc());
    auto nestedLogger = log.nest();
    if (!VPUIP::isCopyFromDDR(copyOp) || !VPUIP::isCopyToDDR(copyOp)) {
        nestedLogger.trace("Cannot match because isn't DDR->DDR copy");
        return mlir::failure();
    }

    if (isDDR2DDRCopyInput(copyOp)) {
        return removeDDR2DDRCopyInput(copyOp, rewriter, nestedLogger);
    } else if (isDDR2DDROutput(copyOp)) {
        return removeDDR2DDROutput(copyOp, rewriter, nestedLogger);
    } else if (isParallelDDR2DDROutput(copyOp)) {
        return removeParallelDDR2DDROutput(copyOp, rewriter, nestedLogger);
    } else if (isDDR2DDRConcatInput(copyOp)) {
        return removeDDR2DDRConcatInput(copyOp, rewriter, nestedLogger);
    }
    std::string possibleReason;
    if (copyOp.getInput().getDefiningOp<Const::DeclareOp>()) {
        possibleReason = " Copy from Constant isn't optimizable";
    }
    nestedLogger.trace("Unsupported pattern.{0}", possibleReason);
    return mlir::failure();
}

//
// ConcatViewWithCopy
//

class ConcatViewWithCopy : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    ConcatViewWithCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx, benefit), log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp origOp, mlir::PatternRewriter& rewriter) const final;

protected:
    Logger log;

private:
    bool hasDuplicatedCopyOutput(VPUIP::ConcatViewOp origOp) const;
};

mlir::FailureOr<VPU::DistributionInfoAttr> deducePermuteCastInputDistributionInfoAttr(
        VPUIP::PermuteCastOp permuteCast, VPUIP::DistributedBufferType outputDistributedType) {
    auto perm = permuteCast.getMemPerm();
    auto inversePerm = mlir::inversePermutation(perm);

    auto inPermuteType = mlir::cast<vpux::NDTypeInterface>(permuteCast->getOperand(0).getType());
    auto outPermuteType = mlir::cast<vpux::NDTypeInterface>(permuteCast->getResult(0).getType());

    return applyPermutationOnDistributionInfoAttr(outputDistributedType, inversePerm, outPermuteType.getDimsOrder(),
                                                  inPermuteType.getDimsOrder(), outPermuteType.getShape(),
                                                  inPermuteType.getShape());
}

mlir::LogicalResult adaptBufferTypeToPemuteCastInput(mlir::Value buffer, VPUIP::PermuteCastOp permuteCastOp,
                                                     Logger log) {
    const auto bufferType = buffer.getType();
    return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(bufferType)
            .Case<mlir::MemRefType>([&](mlir::Type) {
                log.trace("Adapting memref type");
                if (VPUIP::getRootAlloc<mlir::memref::AllocOp>(buffer) == nullptr) {
                    log.trace("Cannot match because buffer isn't master buffer");
                    return mlir::failure();
                }

                const auto permuteCastInputType =
                        mlir::cast<vpux::NDTypeInterface>(permuteCastOp.getSource().getType());
                const auto permuteCastInputOrder = permuteCastInputType.getDimsOrder();
                const auto permuteCastInputShape = permuteCastInputType.getShape();
                const auto outputType = mlir::cast<vpux::NDTypeInterface>(buffer.getType());
                buffer.setType(outputType.changeDimsOrder(permuteCastInputOrder).changeShape(permuteCastInputShape));
                return mlir::success();
            })
            .Case<VPUIP::DistributedBufferType>([&](mlir::Type) {
                log.trace("Adapting DistributedBufferType");
                if (VPUIP::getRootAlloc<VPURT::AllocDistributed>(buffer) == nullptr) {
                    log.trace("Cannot match because buffer isn't master buffer");
                    return mlir::failure();
                }

                const auto getNewDistributedType = [&](VPUIP::DistributedBufferType origType, ShapeRef newShape,
                                                       DimsOrder newOrder) -> VPUIP::DistributedBufferType {
                    const auto ctx = permuteCastOp->getContext();
                    const auto newDistribution =
                            deducePermuteCastInputDistributionInfoAttr(permuteCastOp, origType).value();
                    const auto newOrderMap = mlir::AffineMapAttr::get(newOrder.toAffineMap(ctx));
                    return VPUIP::DistributedBufferType::get(ctx, newShape.raw(), origType.getElementType(),
                                                             newOrderMap, origType.getMemSpace(), newDistribution);
                };

                const auto origDistributedBufferType = mlir::cast<VPUIP::DistributedBufferType>(bufferType);
                const auto permuteCastInputType =
                        mlir::cast<vpux::NDTypeInterface>(permuteCastOp.getSource().getType());
                const auto permuteCastInputShape = permuteCastInputType.getShape();
                const auto permuteCastInputOrder = permuteCastInputType.getDimsOrder();
                const auto newBufferType =
                        getNewDistributedType(origDistributedBufferType, permuteCastInputShape, permuteCastInputOrder);
                buffer.setType(newBufferType);
                return mlir::success();
            })
            .Default([&](mlir::Type) {
                log.trace("Unknown buffer type {0}", bufferType);
                return mlir::failure();
            });
}

/*
  Copy (DDR -> DDR)  ...  Copy (DDR -> DDR)
                \               /
                Concat View (DDR)             =>           Copy (DDR -> CMX) ... Copy (DDR -> CMX)
                        |                                           \               /
                  [PermuteCast]                                     Concat View (CMX)
                        |                                                   |
                Copy (DDR -> CMX)                                     [PermuteCast]
*/

/*
 Copy (DDR -> DDR)  ...  Copy (DDR -> DDR)
                \               /
                Concat View (DDR)             =>  Cluster Copy (DDR -> CMX) ... Cluster Copy (DDR -> CMX)
                        |                                           \               /
                  [PermuteCast]                                     Concat View (CMX)
                        |                                                   |
              Cluster Copy (DDR -> CMX)                             [PermuteCast]
*/

bool hasLegalCopyUser(VPUIP::ConcatViewOp sourceOp, vpux::Logger log) {
    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*sourceOp->getUsers().begin());
    VPUIP::PermuteCastOp maybePermuteCast = nullptr;
    if (copyOp == nullptr) {
        maybePermuteCast = mlir::dyn_cast<VPUIP::PermuteCastOp>(*sourceOp->getUsers().begin());
        if (maybePermuteCast == nullptr || !maybePermuteCast->hasOneUse()) {
            return false;
        }

        copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*maybePermuteCast->getUsers().begin());
    }
    if (copyOp == nullptr || !vpux::VPUIP::hasDistributedOperand(copyOp)) {
        return copyOp != nullptr && VPUIP::isCopyFromDDR(copyOp) && !VPUIP::isCopyToDDR(copyOp) &&
               !VPUIP::isCopyWithStaticStrides(copyOp);
    }
    VPUX_THROW_UNLESS(vpux::VPUIP::hasDistributedOperand(copyOp), "The VPUIP.CopyOp must be distributed");
    if (isStridedCopy(copyOp)) {
        return false;
    }

    // Get the concat dims
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(sourceOp.getInputs()[0].getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(sourceOp.getOutput().getType());
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
            mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(VPUIP::extractDataType(copyOp.getOutputBuff()));
    if (distributedType == nullptr) {
        log.trace("Cannot get distributedType");
        return false;
    }

    auto distribution = distributedType.getDistribution();
    if (maybePermuteCast != nullptr) {
        const auto result = deducePermuteCastInputDistributionInfoAttr(maybePermuteCast, distributedType);
        if (mlir::failed(result)) {
            return false;
        }

        distribution = result.value();
    }

    // For Overlapped mode, use compute_shape and compute_offset to unroll the DMA copy in unroll cluster copy
    // Then we will lost the stride info of the input. It will cause result incorrect
    //     DistrCopy (1x16x8x8)      DistrCopy(1x16x8x8)
    //                      \           /
    //                    Concat(1x32x8x8) (shape[1,32,5,8][1,32,5,8], offset[0,0,0,0][0,0,3,0])
    // TODO: E#78122 remove the checking after the jira fixed
    if (distribution.getMode().getValue() == VPU::DistributionMode::OVERLAPPED) {
        return false;
    }
    if (distribution.getNumTiles() != nullptr) {
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
        const auto tileAxis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
        auto outputLayout = outputType.getDimsOrder();
        auto tileAxisIndex = outputLayout.dimPos(Dim(tileAxis));
        auto isOutmostDimension = [&]() {
            for (auto i : concatDims) {
                if (tileAxisIndex > outputLayout.dimPos(i)) {
                    return false;
                }
            }
            return true;
        };
        if (llvm::find(concatDims, Dim(tileAxis)) != concatDims.end() ||
            (outShape[Dim(tileAxis)] % tilingScheme[tileAxis] != 0 && !isOutmostDimension())) {
            // If the output buffer on tile dim can not be divided evenly on each tile, the buffer will be discontinous
            // after concat, so need to avoid such tranform.
            // E.g.:
            // VPUIP.SubView %source [0, 0, 0, 0] [1, 512, 35, 36] ->SEGMENTED with numTiles = [1, 1, 4, 1]
            // VPUIP.SubView %source [0, 128, 0, 0] [1, 512, 35, 36] -> SEGMENTED with numTiles = [1, 1, 4, 1]
            // The distribution in memory for this example would be:
            //             Cluster 0        Cluster 1        Cluster 2        Cluster 3
            // offset0  x_______________________________________________________________
            //          |  9 lines of   |  9 lines of   |  9 lines of   |  8 lines of   |
            //          | actual data   | actual data   | actual data   | actual data   |
            //          |               |               |               |---------------|
            // offset1  x---------------|---------------|---------------|---------------|
            //          |  9 lines of   |  9 lines of   |  9 lines of   |  8 lines of   |
            //          | actual data   | actual data   | actual data   | actual data   |
            //          |_______________|_______________|_______________|_______________|
            // Unexpected concat on cluster3
            //
            // Particularly case be accept concats if the concat dim is more inner compared to the clustering dim
            // E.g.:
            // VPUIP.SubView %source [0, 0, 0, 0] [1, 35, 36, 512] ->SEGMENTED with numTiles = [1, 4, 1, 1]
            // VPUIP.SubView %source [0, 0, 0, 128] [1, 35, 36, 512] -> SEGMENTED with numTiles = [1, 4, 1, 1]
            // The distribution in memory for this example would be((data arranged on vertical axis)):
            //             Cluster 0        Cluster 1        Cluster 2        Cluster 3
            // offset0  x_______________________________________________________________
            //          |  9 lines of   |  9 lines of   |  9 lines of   |  8 lines of   |
            //          | actual data   | actual data   | actual data   | actual data   |
            //          |               |               |               |---------------|
            // offset1  x---------------|---------------|---------------|---------------|
            //          |  9 lines of   |  9 lines of   |  9 lines of   |  8 lines of   |
            //          | actual data   | actual data   | actual data   | actual data   |
            //          |_______________|_______________|_______________|_______________|
            //  the data in the last cluster is indeed contiguous
            return false;
        }
    }

    return VPUIP::isCopyFromDDR(copyOp) && !VPUIP::isCopyToDDR(copyOp);
}

bool hasDuplicatedCopyOutput(VPUIP::ConcatViewOp origOp) {
    if (origOp.use_empty()) {
        return false;
    }
    auto isSameCopyType = [](mlir::Operation* preOp, mlir::Operation* nextOp) {
        auto preCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(preOp);
        auto nextCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(nextOp);
        if (preCopyOp == nullptr || nextCopyOp == nullptr) {
            return false;
        }
        auto preOutType = mlir::dyn_cast<vpux::NDTypeInterface>(preCopyOp.getOutput().getType());
        auto nextOutType = mlir::dyn_cast<vpux::NDTypeInterface>(nextCopyOp.getOutput().getType());
        return preOutType == nextOutType;
    };

    auto firstUser = *origOp.getOutput().getUsers().begin();
    return llvm::all_of(origOp.getOutput().getUsers(), [&](auto user) {
        return isSameCopyType(firstUser, user);
    });
}

/*
  Check pattern:
  Copy (DDR2DDR)  ...  Copy (DDR2DDR)
       \               /
        Concat View (DDR)
             |
        [PermuteCast]
             |
        Copy(DDR2CMX)
*/
bool isLegalConcatViewPattern(VPUIP::ConcatViewOp origOp, vpux::Logger nestedLogger) {
    if (!origOp.getOutput().hasOneUse() && !hasDuplicatedCopyOutput(origOp)) {
        nestedLogger.trace("Cannot find user copy op at '{0}'", origOp);
        return false;
    }
    for (auto input : origOp.getInputs()) {
        auto op = mlir::dyn_cast_or_null<VPUIP::CopyOp>(input.getDefiningOp());
        if (op == nullptr || !VPUIP::isCopyToDDR(op) || !VPUIP::isCopyFromDDR(op)) {
            return false;
        }
    }

    return hasLegalCopyUser(origOp, nestedLogger);
}

mlir::LogicalResult ConcatViewWithCopy::matchAndRewrite(VPUIP::ConcatViewOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    log.trace("ConcatViewWithCopy: got VPUIP.ConcatViewOp at {0}", origOp.getLoc());
    auto nestedLogger = log.nest();
    if (!isLegalConcatViewPattern(origOp, nestedLogger)) {
        nestedLogger.trace("Cannot fuse this ConcatView Op {0}", origOp.getLoc());
        return mlir::failure();
    }

    mlir::Operation* firstCopyOp;
    auto* childOp = getFirstUser(origOp.getResult());
    auto permuteCastOp = mlir::dyn_cast<VPUIP::PermuteCastOp>(childOp);
    if (permuteCastOp != nullptr) {
        firstCopyOp = getFirstUser(permuteCastOp.getResult());
    } else {
        firstCopyOp = childOp;
    }
    VPUX_THROW_UNLESS(firstCopyOp != nullptr, "Cannot get the first user Op");

    log.trace("Got ConcatView Op at '{0}'", origOp.getLoc());

    SmallVector<mlir::Value> concatInputs;
    auto outBuffer = mlir::dyn_cast<VPUIP::CopyOp>(firstCopyOp).getOutputBuff();

    // record original buffer type before adaptBufferTypeToPemuteCastInput is called as it may be adjusted in it
    auto origBufferType = outBuffer.getType();
    // update buffer type if there is PermuteCastOp after ConcatViewOp
    if (permuteCastOp != nullptr) {
        if (mlir::failed(adaptBufferTypeToPemuteCastInput(outBuffer, permuteCastOp, log))) {
            log.nest().trace("Failed to adapt buffer type to PermuteCast input at '{0}'", origOp.getLoc());
            return mlir::failure();
        }
    }

    auto outBufferDefiningOp = outBuffer.getDefiningOp();
    VPUX_THROW_WHEN(outBufferDefiningOp == nullptr, "Cannot get defining op for {0}", outBuffer);
    rewriter.setInsertionPointAfter(outBufferDefiningOp);
    for (auto input : origOp.getInputs()) {
        auto copyOp = input.getDefiningOp<VPUIP::CopyOp>();
        auto subViewOp = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();

        auto newSubView =
                rewriter.create<VPUIP::SubViewOp>(copyOp.getLoc(), outBuffer, subViewOp.getStaticOffsetsAttr(),
                                                  subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());

        auto newCopyOp = rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(copyOp, copyOp.getInput(), newSubView.getResult());

        concatInputs.push_back(newCopyOp->getResult(0));
    }

    rewriter.setInsertionPointAfter(firstCopyOp);
    auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(firstCopyOp->getLoc(), concatInputs, outBuffer);
    if (permuteCastOp != nullptr) {
        auto newPermuteCastOutputType = origBufferType;
        if (auto inDistributedType =
                    mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(newConcatOp.getOutput().getType())) {
            auto perm = permuteCastOp.getMemPerm();
            auto inPermuteType = mlir::cast<vpux::NDTypeInterface>(permuteCastOp->getOperand(0).getType());
            auto outPermuteType = mlir::cast<vpux::NDTypeInterface>(permuteCastOp->getResult(0).getType());

            auto outDistribution = applyPermutationOnDistributionInfoAttr(
                    inDistributedType, perm, inPermuteType.getDimsOrder(), outPermuteType.getDimsOrder(),
                    inPermuteType.getShape(), outPermuteType.getShape());
            VPUX_THROW_WHEN(mlir::failed(outDistribution), "Failed to infer output distribution");
            const auto orderMap =
                    mlir::AffineMapAttr::get(outPermuteType.getDimsOrder().toAffineMap(rewriter.getContext()));
            newPermuteCastOutputType = VPUIP::DistributedBufferType::get(
                    rewriter.getContext(), outPermuteType.getShape().raw(), outPermuteType.getElementType(), orderMap,
                    inDistributedType.getMemSpace(), outDistribution.value());
        }

        auto newPermuteCastOp =
                rewriter.create<VPUIP::PermuteCastOp>(permuteCastOp->getLoc(), newPermuteCastOutputType, newConcatOp,
                                                      permuteCastOp.getDstOrderAttr(), permuteCastOp.getMemPermAttr());
        auto distributedCast = rewriter.createOrFold<VPUIP::DistributedCastOp>(permuteCastOp->getLoc(), origBufferType,
                                                                               newPermuteCastOp.getResult());

        for (auto userCopyOp : llvm::make_early_inc_range(permuteCastOp.getResult().getUsers())) {
            rewriter.replaceOp(userCopyOp, distributedCast);
        }
    } else {
        for (auto userCopyOp : llvm::make_early_inc_range(origOp.getOutput().getUsers())) {
            rewriter.replaceOp(userCopyOp, newConcatOp.getOutput());
        }
    }

    log.nest().trace("Successfully simplified ConcatView {0}", origOp->getLoc());
    return mlir::success();
}

//
// FuseCopyToTheFrontOfDistributedCopy
//
/*
 Fuse copy into the front of distributed copy
          |                |
  DistributedCopy    =>  DistributedCopy
          |                |
         Copy
          |
*/

class FuseCopyToTheFrontOfDistributedCopy final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    FuseCopyToTheFrontOfDistributedCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp distributedCopyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger log;
};

mlir::LogicalResult FuseCopyToTheFrontOfDistributedCopy::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    /*
    case 1:
              |                          |
      DistributedCopy(CMX2DDR)    =>     DistributedCopy(CMX2CMX)
              |                          |
           Copy(DDR2CMX)
              |

    case 2:
              |                          |
      DistributedCopy(CMX2DDR)    =>     DistributedCopy(CMX2DDR)
              |                          |
           Copy(DDR2DDR)
              |
    */

    if (copyOp == nullptr || VPUIP::isCopyFromDDR(copyOp) || !VPUIP::isCopyToDDR(copyOp) ||
        !vpux::VPUIP::hasDistributedOperand(copyOp)) {
        return mlir::failure();
    }

    if (!copyOp->hasOneUse()) {
        return mlir::failure();
    }

    auto distributedCopyOutput = copyOp.getResult();
    auto outType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedCopyOutput.getType());
    auto userCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*(distributedCopyOutput.getUsers().begin()));
    if (userCopyOp == nullptr || vpux::VPUIP::hasDistributedOperand(userCopyOp)) {
        return mlir::failure();
    }

    auto userOutType = mlir::dyn_cast<vpux::NDTypeInterface>(userCopyOp.getOutputBuff().getType());
    if (userOutType.changeMemSpace(VPU::MemoryKind::DDR) != outType) {
        return mlir::failure();
    }

    auto distributedCopyInput = copyOp.getInput();
    if (isNonDistributedCastCompatible(VPUIP::extractDataType(distributedCopyInput), userOutType)) {
        // In this case the pattern will be optimized as a NonDistributedCast op
        return mlir::failure();
    }

    auto userOutputMemKind = userOutType.getMemoryKind();
    if (userOutputMemKind == VPU::MemoryKind::CMX_NN) {
        auto inputType = mlir::dyn_cast<vpux::NDTypeInterface>(copyOp.getInput().getType());
        if (auto subviewOp = copyOp.getInput().getDefiningOp<VPUIP::SubViewOp>()) {
            inputType = mlir::cast<vpux::NDTypeInterface>(subviewOp.getViewSource().getType());
        }

        Byte requiredCMX(0);
        requiredCMX += inputType.getTotalAllocSize();
        requiredCMX += userOutType.getTotalAllocSize();
        if (requiredCMX > VPU::getTotalCMXSize(userCopyOp)) {
            log.trace("Available CMX size is {0}, but need {1}", VPU::getTotalCMXSize(userCopyOp), requiredCMX);
            return mlir::failure();
        }
    }

    rewriter.setInsertionPointAfter(userCopyOp);
    auto newDistributedCopyOp =
            rewriter.create<VPUIP::CopyOp>(copyOp->getLoc(), copyOp.getInput(), userCopyOp.getOutputBuff());
    rewriter.replaceAllUsesWith(userCopyOp->getResults(), newDistributedCopyOp->getResults());
    rewriter.eraseOp(userCopyOp);
    rewriter.eraseOp(copyOp);
    return mlir::success();
}

// FuseCopyToTheBackOfDistributedCopy
//
/*
 Fuse copy into the back Tilling copy
          |
         Copy
          |
    DistributedCopy   =>   DistributedCopy
          |
*/

class FuseCopyToTheBackOfDistributedCopy final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    FuseCopyToTheBackOfDistributedCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger log;
};

mlir::LogicalResult FuseCopyToTheBackOfDistributedCopy::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    /*
              |
        Copy(CMX2DDR)
              |                          |
      DistributedCopy(DDR2CMX)    =>     DistributedCopy(CMX2CMX)
              |                          |
    */

    if (vpux::VPUIP::hasDistributedOperand(copyOp)) {
        return mlir::failure();
    }

    if (!copyOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    auto userOfDistributedCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*copyOp.getOutput().getUsers().begin());
    if (userOfDistributedCopyOp == nullptr || !vpux::VPUIP::hasDistributedOperand(userOfDistributedCopyOp)) {
        return mlir::failure();
    }

    if (isCopyFromDDR(copyOp) || !isCopyToDDR(copyOp)) {
        return mlir::failure();
    }
    auto inType = mlir::dyn_cast<vpux::NDTypeInterface>(copyOp.getInput().getType());
    auto userInType = mlir::dyn_cast<vpux::NDTypeInterface>(userOfDistributedCopyOp.getOperand(0).getType());
    auto userOutType = mlir::dyn_cast<vpux::NDTypeInterface>(userOfDistributedCopyOp.getResult().getType());
    if (inType.changeMemSpace(VPU::MemoryKind::DDR) != userInType) {
        return mlir::failure();
    }

    auto userOutputMemKind = userOutType.getMemoryKind();
    if (userOutputMemKind == VPU::MemoryKind::CMX_NN) {
        auto inputType = mlir::dyn_cast<vpux::NDTypeInterface>(copyOp.getOperand(0).getType());
        Byte requiredCMX(0);
        requiredCMX += inputType.getTotalAllocSize();
        requiredCMX += userOutType.getTotalAllocSize();
        if (requiredCMX > VPU::getTotalCMXSize(userOfDistributedCopyOp)) {
            log.trace("Available CMX size is {0}, but need {1}", VPU::getTotalCMXSize(userOfDistributedCopyOp),
                      requiredCMX);
            return mlir::failure();
        }
    }

    rewriter.setInsertionPointAfter(userOfDistributedCopyOp);
    auto newDistributedCopyOp =
            rewriter.create<VPUIP::CopyOp>(copyOp->getLoc(), copyOp.getInput(), userOfDistributedCopyOp.getOperand(1));
    copyOp->dropAllUses();
    rewriter.eraseOp(copyOp);
    userOfDistributedCopyOp->replaceAllUsesWith(newDistributedCopyOp);
    rewriter.eraseOp(userOfDistributedCopyOp);

    return mlir::success();
}

//
// SubViewWithDistributedCopy
//
/*
 Move SubView after DistributedCopy, the assumption is to reduce copy op numbers if subview have multi distributed copy
users buffer
            /                            \
      subview(Tile on N)               subview(Tile on N)
           |                               |
DistrCopy(Segmented on N)       DistrCopy(Segmented on N)
           |                               |
         MatMul                         MatMul

                           =>

                       buffer
                         |
                   DistrCopy(Duplicated)
               /                            \
      subview(Tile on N)                 subview(Tile on N)
              |                              |
DistributedCast(Duplicated|Segmented)    DistributedCast(Duplicated|Segmented)
              |                              |
           MatMul                          MatMul

*/

class SubViewWithDistributedCopy : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    SubViewWithDistributedCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;
    mlir::Value getSuitableSubViewPattern(VPUIP::CopyOp origOp, vpux::Logger log) const;
    bool checkCMXFit(Byte cmxSize, mlir::Value topBuffer) const;

private:
    // Cache top buffers whose user matmul does not fit cmx
    //  to avoid duplicated calculation on other subview branches
    mutable llvm::SetVector<mlir::Value> _failedTopBuffers{};
    Logger log;
};

bool SubViewWithDistributedCopy::checkCMXFit(Byte cmxSize, mlir::Value topBuffer) const {
    auto type = mlir::dyn_cast<vpux::NDTypeInterface>(topBuffer.getType());
    // buffer will keep duplicated in cmx after distributed copy, so need to check the required cmx
    Byte requiredSize = type.getTotalAllocSize();
    if (type.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
        requiredSize += type.getTotalAllocSize();
    }
    return cmxSize >= requiredSize;
}

mlir::LogicalResult SubViewWithDistributedCopy::matchAndRewrite(VPUIP::CopyOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    auto nestedLogger = log.nest();
    auto topBuffer = getSuitableSubViewPattern(origOp, nestedLogger);
    if (topBuffer == nullptr) {
        return mlir::failure();
    }

    log.trace("SubViewWithCopyBase");
    auto ctx = origOp->getContext();
    const auto topBufferType = mlir::cast<vpux::NDTypeInterface>(topBuffer.getType());
    const auto copyOutputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
    const auto layout = mlir::AffineMapAttr::get(topBufferType.getDimsOrder().toAffineMap(origOp->getContext()));

    auto distributedType = mlir::cast<vpux::VPUIP::DistributedBufferType>(origOp.getResult().getType());
    auto distribution = distributedType.getDistribution();

    // create duplicated type
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
    const auto distributedAttr =
            VPU::DistributionInfoAttr::get(ctx, distributionModeAttr, distribution.getNumTiles(), nullptr, nullptr,
                                           nullptr, distribution.getNumClusters(), distribution.getAlignment(), nullptr,
                                           nullptr, nullptr, nullptr, nullptr, nullptr);

    auto distributedBufferType = VPUIP::DistributedBufferType::get(origOp->getContext(), topBufferType.getShape().raw(),
                                                                   topBufferType.getElementType(), layout,
                                                                   copyOutputType.getMemSpace(), distributedAttr);

    rewriter.setInsertionPointAfterValue(topBuffer);
    auto newBuffer = rewriter.create<VPURT::AllocDistributed>(appendLoc(origOp->getLoc(), "_extract"),
                                                              distributedBufferType, nullptr, nullptr);
    nestedLogger.trace("create new buff {0}", newBuffer);

    auto newCopy = rewriter.create<VPUIP::CopyOp>(appendLoc(origOp->getLoc(), "_extract"), topBuffer, newBuffer);
    nestedLogger.trace("Created ops '{0}'", newCopy);

    for (auto siblingOp : llvm::make_early_inc_range(topBuffer.getUsers())) {
        auto siblingSubViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(siblingOp);
        if (siblingSubViewOp == nullptr) {
            continue;
        }
        VPUX_THROW_UNLESS(siblingSubViewOp.getResult().hasOneUse(), "subview should has one use");
        auto siblingCopyOp = *siblingSubViewOp.getResult().getUsers().begin();

        rewriter.setInsertionPoint(siblingSubViewOp);
        nestedLogger.trace("Creating VPUIP.SubView '{0}' at '{1}'", siblingSubViewOp->getName(),
                           siblingSubViewOp->getLoc());
        auto newSliceOp = rewriter.create<VPUIP::SubViewOp>(
                appendLoc(siblingSubViewOp->getLoc(), "_CMX"), newCopy->getResult(0),
                siblingSubViewOp.getStaticOffsetsAttr(), siblingSubViewOp.getStaticSizesAttr());

        auto siblingCopyOutType =
                mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(siblingCopyOp->getResult(0).getType());
        auto siblingDistribution = siblingCopyOutType.getDistribution();

        const auto outDistributionModeAttr = VPU::DistributionModeAttr::get(
                ctx, VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED);
        VPU::DistributionInfoAttr targetDistributedAttr = nullptr;
        // If siblingDistribution has shapes and offsets set then call getNonOverlappedDistributedAttr to recompute them
        // else set them to nullptr
        if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(siblingDistribution)) {
            targetDistributedAttr = VPU::getNonOverlappedDistributedAttr(
                    siblingCopyOutType.getShape(), outDistributionModeAttr, siblingDistribution.getNumTiles(),
                    siblingDistribution.getNumClusters(), siblingDistribution.getAlignment(),
                    siblingDistribution.getUniformDistributedSegments(), ctx);
        } else {
            targetDistributedAttr = VPU::DistributionInfoAttr::get(
                    ctx, outDistributionModeAttr, siblingDistribution.getNumTiles(), siblingDistribution.getKernel(),
                    siblingDistribution.getPads(), siblingDistribution.getStrides(),
                    siblingDistribution.getNumClusters(), siblingDistribution.getAlignment(),
                    siblingDistribution.getUniformDistributedSegments(), nullptr, nullptr, nullptr, nullptr,
                    siblingDistribution.getEqualMemoryAndComputeView());
        }

        auto targetDistributedBufferType = VPUIP::DistributedBufferType::get(
                ctx, siblingCopyOutType.getShape().raw(), siblingCopyOutType.getElementType(),
                siblingCopyOutType.getLayout(), siblingCopyOutType.getMemSpace(), targetDistributedAttr);

        nestedLogger.trace("create new subview {0}", newSliceOp);
        auto distributedCastOp = rewriter.create<VPUIP::DistributedCastOp>(
                newSliceOp->getLoc(), targetDistributedBufferType, newSliceOp.getResult());
        nestedLogger.trace("create new cast {0}", distributedCastOp);

        rewriter.replaceAllUsesWith(siblingCopyOp->getResult(0), distributedCastOp.getResult());

        rewriter.eraseOp(siblingCopyOp);
        rewriter.eraseOp(siblingSubViewOp);
    }

    return mlir::success();
}

/*
  Check pattern:
          TopBuffer
             |
          SubView
             |
    DistrCopy(Segmented on dim N)
*/

mlir::Value SubViewWithDistributedCopy::getSuitableSubViewPattern(VPUIP::CopyOp origOp, vpux::Logger log) const {
    auto isDistributedCopyOpSegmentedOnN = [&log](VPUIP::CopyOp copyOp) {
        if (copyOp == nullptr) {
            log.trace("Not distributed Copy");
            return false;
        }
        auto outType = mlir::cast<NDTypeInterface>(copyOp.getResult().getType());
        const auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outType);
        if (distributedType == nullptr) {
            return false;
        }

        return VPU::isSegmentedOverN(distributedType.getDistribution());
    };

    auto doesDistributedCopyOpHasStridedOuput = [](VPUIP::CopyOp distributedCopy) {
        if (distributedCopy == nullptr || !vpux::VPUIP::hasDistributedOperand(distributedCopy)) {
            return false;
        }
        auto distributedCopyOutput = distributedCopy.getOutputs()[0];
        auto distributedCopyOutputType =
                mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(distributedCopyOutput));

        const auto outReqs = StrideReqs::compact(distributedCopyOutputType.getRank());
        return !outReqs.checkStrides(distributedCopyOutputType);
    };

    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    auto cmxSize = VPU::getTotalCMXSize(module);
    auto parentSubViewOp = origOp.getInput().getDefiningOp<VPUIP::SubViewOp>();
    if (parentSubViewOp == nullptr) {
        return nullptr;
    }
    mlir::Value topBuffer = parentSubViewOp.getSource();
    if (_failedTopBuffers.contains(topBuffer)) {
        return nullptr;
    }

    if (!checkCMXFit(cmxSize, topBuffer)) {
        return nullptr;
    }

    if (topBuffer.hasOneUse()) {
        return nullptr;
    }

    // Calculate the new required cmx size for user op on each subview branch, since the new input will be
    // changed into SEG|DUP instead of SEG
    auto doesBranchHaveProperUsersAndFitCMX = [&](VPUIP::CopyOp subviewCopy) {
        return llvm::all_of(subviewCopy->getUsers(), [&](auto user) {
            Byte requiredCMX = VPUIP::getRequiredCMXSize(user);
            // replace the original operand's required cmx size with new one
            requiredCMX -= getTotalSize(subviewCopy->getResult(0));
            requiredCMX += getTotalSize(topBuffer);
            if (requiredCMX > cmxSize) {
                return false;
            }

            // Because of SEG|DUP instead of SEG, a SubView user with explicit-output-shape would become illegal.
            // So stop the optimization once hit this case.
            // Track #E125638
            if (auto userSubView = mlir::dyn_cast<VPUIP::SubViewOp>(user)) {
                return !userSubView.getExplicitOutputShapes().has_value();
            }
            return true;
        });
    };

    auto topBufferUsers = topBuffer.getUsers();
    for (auto user : topBufferUsers) {
        if (!user->hasOneUse()) {
            return nullptr;
        }
        auto anotherSubView = mlir::dyn_cast<VPUIP::SubViewOp>(user);
        if (anotherSubView == nullptr || !VPUIP::isOpOnlySplitOnDim(anotherSubView, Dims4D::Act::N)) {
            return nullptr;
        }
        auto distributedCopy = mlir::dyn_cast<VPUIP::CopyOp>(*anotherSubView.getResult().getUsers().begin());
        if (distributedCopy == nullptr || !vpux::VPUIP::hasDistributedOperand(distributedCopy) ||
            !isDistributedCopyOpSegmentedOnN(distributedCopy) ||
            doesDistributedCopyOpHasStridedOuput(distributedCopy)) {
            return nullptr;
        }
        if (!doesBranchHaveProperUsersAndFitCMX(distributedCopy)) {
            _failedTopBuffers.insert(topBuffer);
            return nullptr;
        }
    }

    return topBuffer;
}

//
// DuplicatedCopyWithCMXCopy
//

/*
Remove copy op by change duplicated buffer into non-distributed buffer

       Distributed Buffer(Duplicated)                   Distributed Buffer(Duplicated)
             |                                                 |
       DistrCopy(CMX2DDR)                               NonDistributedCast
              |                       ==>                      |
         [PureViewLikeOps]                                [PureViewLikeOps]
              |
         Copy(DDR2CMX)
              |

*/

class DuplicatedCopyWithCMXCopy : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    DuplicatedCopyWithCMXCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp distributedCopy, mlir::PatternRewriter& rewriter) const final;

private:
    Logger log;
};

mlir::LogicalResult DuplicatedCopyWithCMXCopy::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                               mlir::PatternRewriter& rewriter) const {
    if (!copyOp->hasOneUse()) {
        return mlir::failure();
    }
    auto distributedCopyInput = copyOp.getInput();
    auto inType = mlir::dyn_cast<VPUIP::DistributedBufferType>(VPUIP::extractDataType(distributedCopyInput));
    if (inType == nullptr) {
        return mlir::failure();
    }
    if (inType.getDistribution().getNumTiles() == nullptr) {
        return mlir::failure();
    }
    auto mode = inType.getDistribution().getMode().getValue();
    if (!VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED)) {
        return mlir::failure();
    }

    auto inStrides = inType.getStrides();
    auto outStrides = mlir::dyn_cast<vpux::NDTypeInterface>(VPUIP::extractDataType(copyOp.getOutput())).getStrides();
    if (inStrides != outStrides) {
        return mlir::failure();
    }

    SmallVector<mlir::Operation*> viewLikeOps;
    auto userOp = *copyOp->getUsers().begin();
    while (mlir::isa<VPUIP::GenericReshapeOp, VPUIP::ShapeCastOp, VPUIP::PermuteCastOp>(userOp) &&
           userOp->hasOneUse()) {
        viewLikeOps.push_back(userOp);
        userOp = *userOp->getUsers().begin();
    }
    auto userCopy = mlir::dyn_cast<VPUIP::CopyOp>(userOp);
    if (userCopy == nullptr || VPUIP::isCopyToDDR(userCopy)) {
        return mlir::failure();
    }

    auto userInType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(userCopy.getInput()));
    auto userOutType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(userCopy.getOutput()));
    if (userInType.changeMemSpace(userOutType.getMemSpace()) != userOutType) {
        return mlir::failure();
    }
    auto symbolAttr = userOutType.getMemSpace();
    auto innerOutputType = mlir::dyn_cast<vpux::NDTypeInterface>(VPUIP::extractDataType(copyOp.getOutput()));
    auto newOutType = innerOutputType.changeMemSpace(symbolAttr);
    rewriter.setInsertionPointAfter(copyOp);
    auto castOp = rewriter.create<VPUIP::NonDistributedCastOp>(copyOp->getLoc(), newOutType, copyOp.getInput());

    log.trace("Create NonDistributedCast op at '{0}'", copyOp->getLoc());
    auto newOutput = castOp.getOutput();
    for (auto viewLikeOp : viewLikeOps) {
        mlir::IRMapping mapper;
        mapper.map(viewLikeOp->getOperands(), ArrayRef({newOutput}));
        auto* newViewLikeOp = rewriter.clone(*viewLikeOp, mapper);
        auto viewLikeOutType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(viewLikeOp->getResult(0)));
        auto newViewLikeOutType = viewLikeOutType.changeMemSpace(newOutType.getMemSpace());
        newViewLikeOp->getResult(0).setType(newViewLikeOutType);
        newOutput = newViewLikeOp->getResult(0);
    }
    for (auto viewLikeOp : viewLikeOps) {
        viewLikeOp->dropAllUses();
        rewriter.eraseOp(viewLikeOp);
    }
    copyOp->dropAllUses();
    rewriter.eraseOp(copyOp);
    rewriter.replaceOp(userCopy, {newOutput});

    return mlir::success();
}

//
// FuseCopiesThroughReshape
//

/*
  Fuse copy(with strided input) with distributed copy through reshape

    SubView(Strided input)                            SubView(Strided input)
              |                                                |
        Copy(DDR2DDR)                               DistributedCopy(DDR2CMX)
              |                       ==>                      |
         GenericReshape                                GenericReshape
              |
     DistributedCopy(DDR2CMX)

*/

class FuseCopiesThroughReshape : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    FuseCopiesThroughReshape(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger log;
};

mlir::LogicalResult FuseCopiesThroughReshape::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                              mlir::PatternRewriter& rewriter) const {
    if (vpux::VPUIP::hasDistributedOperand(copyOp)) {
        return mlir::failure();
    }

    auto copyOpInput = copyOp.getInput();
    auto copyOpInputType = mlir::cast<vpux::NDTypeInterface>(copyOpInput.getType());
    const auto inReqs = StrideReqs::compact(copyOpInputType.getRank());
    if (inReqs.checkStrides(copyOpInputType)) {
        log.trace("The input has no strides");
        return mlir::failure();
    }
    if (!copyOp->hasOneUse()) {
        return mlir::failure();
    }

    auto reshapeOp = mlir::dyn_cast<VPUIP::GenericReshapeOp>(*copyOp.getOutput().getUsers().begin());
    if (reshapeOp == nullptr) {
        return mlir::failure();
    }
    if (!reshapeOp->hasOneUse()) {
        return mlir::failure();
    }

    auto userClusterCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*reshapeOp->getResult(0).getUsers().begin());
    if (userClusterCopyOp == nullptr || !vpux::VPUIP::hasDistributedOperand(userClusterCopyOp)) {
        return mlir::failure();
    }

    auto origReshapeOutType = mlir::cast<vpux::NDTypeInterface>(reshapeOp.getOutput().getType());
    auto origReshapeInType = mlir::cast<vpux::NDTypeInterface>(reshapeOp.getInput().getType());
    auto outBuffer = userClusterCopyOp.getOutputBuff();
    auto outBufAlloc = VPUIP::getRootAlloc<VPURT::AllocDistributed>(outBuffer);
    if (outBufAlloc == nullptr) {
        // The case with pure view-like ops chain is not supported yet.
        // E#122314: support the pure view-like ops chain
        return mlir::failure();
    }

    auto origDistrType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outBuffer.getType());
    auto origDistrAttr = origDistrType.getDistribution();
    auto getDistributedAxesMapping = vpux::VPUIP::getDistributedAxesMappingAfterShapeChanged(
            origReshapeOutType, origReshapeInType, origDistrAttr, log);
    if (mlir::failed(getDistributedAxesMapping)) {
        return mlir::failure();
    }

    // Distributed Overlapped mode only support rank 4D/5D
    const int rank4D = 4;
    const int rank5D = 5;
    if (origDistrAttr.getMode().getValue() == VPU::DistributionMode::OVERLAPPED &&
        origReshapeInType.getShape().size() != rank4D && origReshapeInType.getShape().size() != rank5D) {
        return mlir::failure();
    }

    auto axesMapping = getDistributedAxesMapping.value();
    if (axesMapping.first == -1 || axesMapping.second == -1) {
        return mlir::failure();
    }
    auto newDistributedBeforeShapeChange = vpux::VPUIP::changeDistributedAxisOnDistributionInfoAttr(
            origDistrAttr, axesMapping.first, axesMapping.second, origReshapeInType.getShape());
    auto ctx = copyOp->getContext();
    const auto newOutputElemType = origReshapeInType.getElementType();
    const auto order = mlir::AffineMapAttr::get(origReshapeInType.getDimsOrder().toAffineMap(ctx));
    auto newDistributedBufferType =
            VPUIP::DistributedBufferType::get(ctx, origReshapeInType.getShape().raw(), newOutputElemType, order,
                                              origDistrType.getMemSpace(), newDistributedBeforeShapeChange);
    if (!VPUIP::isDistributedCompatibleAfterShapeChangeForViewOps<VPUIP::DistributedBufferType>(
                origDistrType, newDistributedBufferType)) {
        return mlir::failure();
    }
    outBuffer.setType(newDistributedBufferType);

    auto newDistributedOp = rewriter.create<VPUIP::CopyOp>(copyOp->getLoc(), copyOpInput, outBuffer);
    VPUIP::moveRootAllocBefore(outBufAlloc, newDistributedOp);
    rewriter.replaceOpWithNewOp<VPUIP::GenericReshapeOp>(userClusterCopyOp, origDistrType,
                                                         newDistributedOp->getResult(0));
    auto origAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(copyOp.getOutputBuff());
    copyOp.getOutput().replaceAllUsesWith(newDistributedOp.getResult());
    rewriter.eraseOp(copyOp);
    rewriter.eraseOp(origAlloc);
    log.trace("Successfully fused copies through reshape");
    return mlir::success();
}

class SubViewWithCopy : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    SubViewWithCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool hasTrivialStrides(vpux::NDTypeInterface ndType) const;
    SmallVector<int64_t> trimTrivialDims(vpux::NDTypeInterface ndType) const;
    bool isTrivialCopy(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType) const;
    mlir::Value getSuitableSubViewPattern(VPUIP::CopyOp origOp, vpux::Logger log) const;

    Logger log;
};

bool SubViewWithCopy::hasTrivialStrides(vpux::NDTypeInterface ndType) const {
    const auto elemTypeBitWidth = ndType.getElemTypeSize();
    const auto actStrides = ndType.getStrides();
    for (size_t i = 1; i < actStrides.size(); i++) {
        if (actStrides[Dim(i)] != elemTypeBitWidth) {
            return false;
        }
    }

    return true;
}

SmallVector<int64_t> SubViewWithCopy::trimTrivialDims(vpux::NDTypeInterface ndType) const {
    const auto order = ndType.getDimsOrder();
    const auto shape = order.toMemoryOrder(ndType.getShape());
    const auto isTrivialDim = [](const int64_t dim) -> bool {
        return dim != 1;
    };
    const auto firstNonTrivialDim = std::find_if(shape.begin(), shape.end(), isTrivialDim);
    return SmallVector<int64_t>(firstNonTrivialDim, shape.end());
}

bool SubViewWithCopy::isTrivialCopy(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType) const {
    const auto inMemShape = trimTrivialDims(inType);
    const auto outMemShape = trimTrivialDims(outType);

    if (inMemShape.size() != outMemShape.size()) {
        return false;
    }
    for (size_t idx = 1; idx < inMemShape.size(); idx++) {
        if (inMemShape[idx] != outMemShape[idx]) {
            return false;
        }
    }

    return hasTrivialStrides(inType) && hasTrivialStrides(outType);
}

mlir::Value SubViewWithCopy::getSuitableSubViewPattern(VPUIP::CopyOp copyOp, vpux::Logger log) const {
    auto maybeSubView = copyOp.getInput().getDefiningOp<VPUIP::SubViewOp>();
    if (maybeSubView == nullptr) {
        log.trace("SubViewWithCopy::getSuitableSubViewPattern: input producer is not a SubView.");
        return nullptr;
    }

    auto inType = mlir::cast<vpux::NDTypeInterface>(maybeSubView.getSource().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(maybeSubView.getResult().getType());
    // This check must be less strict.
    // The rewriter must be able to process copies of 3-d compact (non-strided) tensors.
    // However, the measurements show that the performance is even worse in that case.
    // The root cause is unclear.
    // [Track number: E#139988]
    if (!isTrivialCopy(inType, outType)) {
        log.trace("SubViewWithCopy::getSuitableSubViewPattern: strided copies cannot be replaced with a "
                  "ViewOp.");
        return nullptr;
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(copyOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(copyOp.getOutput().getType());
    if (inputType.getMemSpace() != outputType.getMemSpace()) {
        log.trace("SubViewWithCopy::getSuitableSubViewPattern: CMX <-> DRAM transfers cannot be replaced "
                  "with a ViewOp.");
        return nullptr;
    }

    return maybeSubView.getResult();
}

mlir::LogicalResult SubViewWithCopy::matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    auto nestedLogger = log.nest();
    auto value = getSuitableSubViewPattern(origOp, nestedLogger);
    if (value == nullptr) {
        return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<VPUIP::ViewOp>(origOp, origOp.getType(), value);

    return mlir::success();
}

//
// CopyOpSequenceWithSubview
//
/*
Convert DistributedCopy(CMX2DDR) -> SubView -> DistributedCopy(DDR2CMX) to SubView -> DistributedCopy(CMX2CMX)

DistributedCopy(CMX2DDR)
           |
        subview
           |
DistributedCopy(DDR2CMX)
           |
         Conv

            =>

        subview
           |
DistributedCopy(CMX2CMX)
           |
         Conv

*/

class CopyOpSequenceWithSubview final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CopyOpSequenceWithSubview(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit,
                              WorkloadManagementMode workloadManagementMode, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit),
              _workloadManagementMode(workloadManagementMode),
              log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    WorkloadManagementMode _workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA;
    Logger log;
};

mlir::LogicalResult CopyOpSequenceWithSubview::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                               mlir::PatternRewriter& rewriter) const {
    log.trace("CopyOpSequenceWithSubview: Copy at {0}", copyOp->getLoc());
    auto nestedLogger = log.nest();

    if (_workloadManagementMode <= WorkloadManagementMode::PWLM_V0_LCA) {
        return mlir::failure();
    }

    auto outputDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(copyOp.getOutput().getType());
    if (outputDistributedType == nullptr) {
        nestedLogger.trace("CopyOpSequenceWithSubview: Copy has no distributed output, skipping");
        return mlir::failure();
    }

    auto subViewOp = copyOp.getInput().getDefiningOp<VPUIP::SubViewOp>();
    if (subViewOp == nullptr) {
        return mlir::failure();
    }

    auto parentCopyOp = subViewOp.getSource().getDefiningOp<VPUIP::CopyOp>();
    if (parentCopyOp == nullptr) {
        return mlir::failure();
    }

    auto inputDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(parentCopyOp.getInput().getType());
    if (inputDistributedType == nullptr) {
        nestedLogger.trace("CopyOpSequenceWithSubview: Parent copy has no distributed input, skipping");
        return mlir::failure();
    }

    auto grandparentOp = parentCopyOp.getInput().getDefiningOp();
    if (grandparentOp == nullptr) {
        nestedLogger.trace(
                "CopyOpSequenceWithSubview: cannot match because grandparent of current CopyOp is not an operation");
        return mlir::failure();
    }

    // perform this optimization only when distributed buffer is compatible with subview
    // otherwise an accuracy degradation may occur
    if (!isSubViewCompatibleWithDistributedBuffer(subViewOp, inputDistributedType)) {
        return mlir::failure();
    }

    if (isLegalAndBenefitCreateCopyFromCMXToCMX(grandparentOp, parentCopyOp, copyOp, nestedLogger)) {
        rewriter.setInsertionPointAfter(copyOp);
        // create and insert a new subview
        auto newSubViewOp = rewriter.create<VPUIP::SubViewOp>(
                subViewOp->getLoc(), parentCopyOp.getInput(), subViewOp.getStaticOffsetsAttr(),
                subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());
        nestedLogger.trace("CopyOpSequenceWithSubview: created new subview at '{0}' with type {1}",
                           newSubViewOp->getLoc(), newSubViewOp->getResult(0).getType());

        // rewrite with a new CMX2CMX copy
        rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(copyOp, newSubViewOp->getResult(0), copyOp.getOutputBuff());

        if (subViewOp->use_empty()) {
            rewriter.eraseOp(subViewOp);
        }

        if (parentCopyOp->use_empty()) {
            rewriter.eraseOp(parentCopyOp);
        }

        nestedLogger.trace(
                "CopyOpSequenceWithSubview: successfully fused sequence of distributed copies into one CMX2CMX "
                "Copy when the input type and output type have different memory views on each cluster");
        return mlir::success();
    }

    nestedLogger.trace("CopyOpSequenceWithSubview: cannot match because distributed types are incompatible with "
                       "CMX2CMX Copy: '{0}' != '{1}'",
                       inputDistributedType, outputDistributedType);
    return mlir::failure();
}

//
// OptimizeCopiesPass
//

class OptimizeCopiesPass final : public VPUIP::impl::OptimizeCopiesBase<OptimizeCopiesPass> {
public:
    explicit OptimizeCopiesPass(const WorkloadManagementMode workloadManagementMode, Logger log)
            : _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    WorkloadManagementMode _workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA;
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void OptimizeCopiesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    if (workloadManagementModeOpt.hasValue()) {
        _workloadManagementMode = workloadManagementModeOpt;
    }

    // Note the below patterns exec order is defined by "benefitLevels" at the head
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<CopyOpSequence>(&ctx, benefitLevels[0], _workloadManagementMode, _log);
    patterns.add<CMXToCMXCopy>(&ctx, benefitLevels[1], _log);
    patterns.add<DDRToDDRCopy>(&ctx, benefitLevels[2], _log);
    patterns.add<ConcatViewWithCopy>(&ctx, benefitLevels[3], _log);
    patterns.add<FuseCopyToTheFrontOfDistributedCopy>(&ctx, benefitLevels[3], _log);
    patterns.add<FuseCopyToTheBackOfDistributedCopy>(&ctx, benefitLevels[3], _log);
    patterns.add<SubViewWithDistributedCopy>(&ctx, benefitLevels[3], _log);
    patterns.add<DuplicatedCopyWithCMXCopy>(&ctx, benefitLevels[3], _log);
    patterns.add<FuseCopiesThroughReshape>(&ctx, benefitLevels[3], _log);
    patterns.add<SubViewWithCopy>(&ctx, benefitLevels[3], _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    // Please note that the following optimization must be applied after the above patterns, as some condition checks
    // depend on the results of the previous optimizations. For example, determining if a buffer has a stride.
    mlir::RewritePatternSet patternsCopyOpSequenceWithSubview(&ctx);
    patternsCopyOpSequenceWithSubview.add<CopyOpSequenceWithSubview>(&ctx, benefitLevels[0], _workloadManagementMode,
                                                                     _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patternsCopyOpSequenceWithSubview),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

void vpux::VPUIP::registerOptimizeCopiesRewriters(vpux::RewriterRegistry& registry,
                                                  WorkloadManagementMode workloadManagementMode, Logger log) {
    registry.registerRewriter<CopyOpSequence>("copy-sequence", benefitLevels[0], workloadManagementMode, log);
    registry.registerRewriter<CMXToCMXCopy>("cmx-to-cmx", benefitLevels[1], log);
    registry.registerRewriter<DDRToDDRCopy>("ddr-to-ddr", benefitLevels[2], log);
    registry.registerRewriter<ConcatViewWithCopy>("concat-view-copy", benefitLevels[3], log);
    registry.registerRewriter<FuseCopyToTheFrontOfDistributedCopy>("fuse-copy-front-distributed", benefitLevels[3],
                                                                   log);
    registry.registerRewriter<FuseCopyToTheBackOfDistributedCopy>("fuse-copy-back-distributed", benefitLevels[3], log);
    registry.registerRewriter<SubViewWithDistributedCopy>("subview-distributed-copy", benefitLevels[3], log);
    registry.registerRewriter<DuplicatedCopyWithCMXCopy>("duplicated-copy-cmx-copy", benefitLevels[3], log);
    registry.registerRewriter<FuseCopiesThroughReshape>("fuse-copies-through-reshape", benefitLevels[3], log);
    registry.registerRewriter<SubViewWithCopy>("subview-copy", benefitLevels[3], log);
    registry.registerRewriter<CopyOpSequenceWithSubview>("copy-op-sequence-with-subview", benefitLevels[0],
                                                         workloadManagementMode, log);
}

void vpux::VPUIP::registerOptimizeCopiesSection(vpux::RewriterRegistry& registry) {
    registry.registerRewriterSet(
            "optimize-copies-set",
            [&](WorkloadManagementMode workloadManagementMode, Logger log) {
                registerOptimizeCopiesRewriters(registry, workloadManagementMode, log);
            },
            // E-184017: Support testing different arguments in cmd line
            WorkloadManagementMode::PWLM_V0_LCA, Logger("OptimizeCopies", LogLevel::Trace));
}

//
// createOptimizeCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeCopiesPass(WorkloadManagementMode workloadManagementMode,
                                                                  Logger log) {
    return std::make_unique<OptimizeCopiesPass>(workloadManagementMode, log);
}
