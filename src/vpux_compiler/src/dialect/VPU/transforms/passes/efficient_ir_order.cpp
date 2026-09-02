//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/reorder_ir_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

#include <utility>

namespace vpux::VPU {
#define GEN_PASS_DECL_EFFICIENTIRORDER
#define GEN_PASS_DEF_EFFICIENTIRORDER
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// EfficientIROrderPass
//

class EfficientIROrderPass final : public VPU::impl::EfficientIROrderBase<EfficientIROrderPass> {
public:
    explicit EfficientIROrderPass(bool enableReorderConcatBranches, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        _enableReorderConcatBranches = enableReorderConcatBranches;
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableReorderConcatBranches = false;
};

mlir::LogicalResult EfficientIROrderPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (enableReorderConcatBranches.hasValue()) {
        _enableReorderConcatBranches = enableReorderConcatBranches.getValue();
    }

    return mlir::success();
}

//
// safeRunOnModule
//

void reorderOperationsInVFBlock(VPU::VerticalFusionOp vfOp) {
    SmallVector<mlir::Operation*, 4> computeOpsInBlock;

    for (auto& op : vfOp.getBody()->without_terminator()) {
        if (mlir::isa<VPU::NCEOpInterface, VPU::SWOpInterface>(&op)) {
            computeOpsInBlock.push_back(&op);
        }
    }

    const auto hasMultipleComputeOpsInputs = [](mlir::Operation* op) {
        int computeOpCount = 0;

        for (mlir::Value operand : op->getOperands()) {
            auto inputOp = operand.getDefiningOp();
            if (mlir::isa_and_nonnull<VPU::NCEOpInterface, VPU::SWOpInterface>(inputOp)) {
                computeOpCount++;
            }
        }

        return computeOpCount > 1;
    };
    for (auto origOp : computeOpsInBlock | reversed) {
        // For operation has multiple computeOp inputs, place it right after it's last parent
        if (hasMultipleComputeOpsInputs(origOp)) {
            SmallVector<mlir::Operation*> parents;
            for (auto operand : origOp->getOperands()) {
                if (auto parentOp = operand.getDefiningOp()) {
                    parents.push_back(parentOp);
                }
            }
            if (!parents.empty()) {
                llvm::sort(parents, [](auto* lhs, auto* rhs) {
                    return lhs->isBeforeInBlock(rhs);
                });
                origOp->moveAfter(parents.back());
            }

            continue;
        }

        // For operation has single computeOp input, place it right before it's first user.
        // Examine all results to preserve dominance for multi-result ops (e.g. NCE ops
        // with a fused reduce output).
        mlir::Operation* firstUser = nullptr;
        for (auto result : origOp->getResults()) {
            auto* candidate = getFirstUser(result);
            if (candidate == nullptr) {
                continue;
            }
            if (firstUser == nullptr || candidate->isBeforeInBlock(firstUser)) {
                firstUser = candidate;
            }
        }
        if (firstUser != nullptr) {
            origOp->moveBefore(firstUser);
        }
    }
}

/**
 * Collect SDPA-related consumer vf ops to move after their common producer VF to reduce interference with SDPA
 * pattern scheduling. For example, op1 through op6 need to be executed before all SDPA branches are executed.
 *             producerOp
 *       /  |  \   ...  /  | \
 *     op1 op2 op3    op4 op5 op6
 *       \  |    \      \  |  \
 *      MatMul    \   MatMul   \
 *         |       |     |      |
 *      SoftMax    |  SoftMax   |
 *           \     |      \     |
 *            MatMul       MatMul
 */
using VFOpMoveAfter = std::pair<mlir::Operation*, VPU::VerticalFusionOp>;

void collectVfOpsToMoveBeforeSDPA(SmallVector<VFOpMoveAfter>& sdpaMoveOps, VPU::VerticalFusionOp vfOp) {
    if (vfOp->hasOneUse()) {
        return;
    }

    // Detect SDPA pattern which is aligned with PatternBasedVF
    auto detectSDPAPattern = [](VPU::VerticalFusionOp vfOp) {
        auto isSupportedOp = [](mlir::Operation* op) {
            if (auto concatOp = mlir::dyn_cast_if_present<VPU::ConcatOp>(op)) {
                auto concatInputs = concatOp.getInputs();
                if (concatInputs.size() != 2) {
                    return false;
                }

                return mlir::isa_and_present<Const::DeclareOp>(concatInputs[1].getDefiningOp());
            }

            return mlir::isa_and_present<VPU::ViewLikeOpInterface, VPU::ExpandOp>(op);
        };

        // `canBeMoved` confirms that vfOp only has one use
        auto* userOp = *vfOp->getUsers().begin();
        // Walk through supported ops before the SDPA VF block.
        while (isSupportedOp(userOp) && userOp->hasOneUse()) {
            userOp = *userOp->getUsers().begin();
        }
        auto userVfOp = mlir::dyn_cast_if_present<VPU::VerticalFusionOp>(userOp);
        if (userVfOp == nullptr || !userVfOp->hasOneUse()) {
            return false;
        }

        auto* body = userVfOp.getBody();
        if (body == nullptr) {
            return false;
        }

        SmallVector<mlir::Operation*> innerOps;
        for (auto& op : body->without_terminator()) {
            innerOps.push_back(&op);
        }

        // Currently, SDPA supports the following patterns:
        // 1. Conv -> SoftMax -> Conv
        // 2. Conv -> SoftMax -> Eltwise -> Conv
        // 3. Conv -> SoftMax -> Reduce -> Conv -> Eltwise (SoftMax decomposition)
        // So the number of operations in the VF block should be at least 3
        if (innerOps.size() < 3) {
            return false;
        }

        bool hasQKMatMul = false;
        bool hasSoftMax = false;
        bool hasVMatMul = false;
        for (auto* op : innerOps) {
            if (mlir::isa<VPU::NCEConvolutionOp>(op) && hasQKMatMul == false) {
                hasQKMatMul = true;
            } else if (mlir::isa<VPU::SoftMaxOp>(op)) {
                hasSoftMax = true;
            } else if (mlir::isa<VPU::NCEConvolutionOp>(op) && hasQKMatMul == true) {
                hasVMatMul = true;
            }
        }

        return hasQKMatMul && hasSoftMax && hasVMatMul;
    };

    auto canBeMoved = [](VPU::VerticalFusionOp vfOp, VPU::VerticalFusionOp rootVfOp) {
        if (!vfOp->hasOneUse()) {
            return false;
        }

        if (vfOp->getBlock() != rootVfOp->getBlock()) {
            return false;
        }

        // Make sure all operands are defined by either blockArg, rootVfOp, or an op before rootVfOp,
        // otherwise moving vfOp after rootVfOp may cause SSA dominance issues.
        for (auto operand : vfOp.getOperands()) {
            auto parentOp = operand.getDefiningOp();
            if (parentOp == nullptr || parentOp == rootVfOp) {
                continue;
            }

            if (parentOp->getBlock() != rootVfOp->getBlock() || !parentOp->isBeforeInBlock(rootVfOp)) {
                return false;
            }
        }

        return true;
    };

    SmallVector<mlir::Operation*> consumers = to_small_vector(vfOp->getUsers());
    if (llvm::all_of(consumers, [&](mlir::Operation* op) {
            auto vfBlock = mlir::dyn_cast_if_present<VPU::VerticalFusionOp>(op);
            if (vfBlock == nullptr || !canBeMoved(vfBlock, vfOp)) {
                return false;
            }
            return detectSDPAPattern(vfBlock);
        })) {
        llvm::sort(consumers, [](mlir::Operation* lhs, mlir::Operation* rhs) {
            return lhs->isBeforeInBlock(rhs);
        });
        // Record consumers in reverse move order.
        for (auto consumer : consumers | reversed) {
            sdpaMoveOps.push_back({consumer, vfOp});
        }
    }
}

bool hasVFBlock(mlir::func::FuncOp& func) {
    auto hasVFBlock = false;
    func->walk([&](VPU::VerticalFusionOp) {
        hasVFBlock = true;
        return;
    });

    return hasVFBlock;
}

// Make sure we execute branches in order
// This order is only beneficial when there is no NCEOp in parallel with SoftMax.
// Otherwise, 1-2-4-5-3-6 may be better than 1-2-3-4-5-6, because 2 and 4 can be parallel
//     NCEOp(1)      NCEOp(4)
//       |             |
//    SoftMax(2)     SoftMax(5)
//       |             |
//     NCEOp(3)      NCEOp(6)
//       |             |
//  ViewLikeOps   ViewLikeOps
//        \           /
//         \   ...   /
//            Concat
void reorderConcatBranches(VPU::ConcatOp concatOp) {
    SmallVector<mlir::Operation*> parents;
    for (auto operand : concatOp.getOperands()) {
        if (auto parentOp = operand.getDefiningOp()) {
            parents.push_back(parentOp);
        }
    }

    auto totalAvailableCMXSize = getTotalCMXSize(concatOp).count();
    SmallVector<SmallVector<mlir::Operation*>> patternOps(parents.size());
    for (auto parentIt : parents | indexed) {
        auto index = parentIt.index();
        auto currentOp = parentIt.value();
        while (mlir::isa_and_nonnull<VPU::ViewLikeOpInterface, VPU::SoftMaxOp, VPU::NCEOpInterface>(currentOp)) {
            if (!currentOp->hasOneUse()) {
                break;
            }

            if (auto softMaxOp = mlir::dyn_cast<VPU::SoftMaxOp>(currentOp)) {
                auto softMaxInType = mlir::cast<vpux::NDTypeInterface>(softMaxOp.getInput().getType());
                auto softMaxOutType = mlir::cast<vpux::NDTypeInterface>(softMaxOp.getOutput().getType());
                auto requiredCMXSize = VPU::getRequiredCMXSize(softMaxInType) + VPU::getRequiredCMXSize(softMaxOutType);
                // It is not beneficial to reorder if CMX space is sufficient, because NCEOp(4) can be parallelized with
                // SoftMax(2).
                // requiredCMXSize * 1.5: The minimum CMX size required for SoftMax(2) and NCEOp(4).
                if (requiredCMXSize.count() * 1.5 < totalAvailableCMXSize) {
                    break;
                }
            }

            patternOps[index].push_back(currentOp);
            auto operand = currentOp->getOperand(0);
            if (mlir::isa<mlir::BlockArgument>(operand)) {
                break;
            }

            currentOp = operand.getDefiningOp();
        }
    }

    // Only consider whether the SW task and the DPU task are symmetrical.
    SmallVector<SmallVector<mlir::Operation*>> filteredPatternOps;
    for (const auto& branch : patternOps) {
        const auto filteredBranch = to_small_vector(branch | filtered([](auto* op) {
                                                        return !mlir::isa<VPU::ViewLikeOpInterface>(op);
                                                    }));
        filteredPatternOps.push_back(filteredBranch);
    }

    auto isSymmetricConcat = [&filteredPatternOps] {
        if (filteredPatternOps.empty()) {
            return false;
        }

        const auto& firstBranch = filteredPatternOps.front();
        if (firstBranch.empty()) {
            return false;
        }
        for (const auto& branch : filteredPatternOps) {
            if (branch.size() != firstBranch.size()) {
                return false;
            }
            for (size_t i = 0; i < branch.size(); ++i) {
                if (branch[i]->getName() != firstBranch[i]->getName()) {
                    return false;
                }
            }
        }
        return true;
    }();

    if (!isSymmetricConcat) {
        return;
    }

    auto isBeneficialToReorder = [&patternOps] {
        const auto& firstBranch = patternOps.front();

        // If SoftMax exists, we can generally gain benefits by parallelizing it with DPU tasks.
        auto hasSoftMaxOp = llvm::any_of(firstBranch, [&](auto op) {
            return mlir::isa<VPU::SoftMaxOp>(op);
        });
        if (hasSoftMaxOp) {
            return true;
        }

        // If SoftMax does not exist and the DPU Task has another branch, there may be risks.
        auto hasDynamicWeightsFromOtherBranch = llvm::any_of(firstBranch, [&](auto op) {
            if (auto nceOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(op)) {
                auto weights = nceOp.getWeightsOperand();
                if (weights == nullptr || mlir::isa<mlir::BlockArgument>(weights)) {
                    return false;
                }

                return !mlir::isa_and_nonnull<Const::DeclareOp>(weights.getDefiningOp());
            }
            return false;
        });
        if (hasDynamicWeightsFromOtherBranch) {
            return false;
        }

        return true;
    }();

    if (!isBeneficialToReorder) {
        return;
    }

    llvm::sort(patternOps, [](const SmallVector<mlir::Operation*>& lhs, const SmallVector<mlir::Operation*>& rhs) {
        return lhs.back()->isBeforeInBlock(rhs.back());
    });

    for (const auto& branch : patternOps) {
        auto postOp = concatOp.getOperation();
        for (const auto& op : branch) {
            op->moveBefore(postOp);
            postOp = op;
        }
    }
}

void reordeNCEOpAsWeightScaleTable(VPU::NCEOpInterface nceOp) {
    auto weightTableScale = nceOp.getWeightTableScaleOperand();
    if (weightTableScale == nullptr || mlir::isa<mlir::BlockArgument>(weightTableScale)) {
        return;
    }

    auto consumerOp = nceOp.getOperation();
    auto currentOp = weightTableScale.getDefiningOp();
    SmallVector<mlir::Operation*> viewLikeOps;

    while (mlir::isa_and_present<VPU::ViewLikeOpInterface>(currentOp)) {
        if (!currentOp->hasOneUse()) {
            return;
        }

        viewLikeOps.push_back(currentOp);
        auto producerValue = currentOp->getOperand(0);
        if (mlir::isa<mlir::BlockArgument>(producerValue)) {
            return;
        }

        currentOp = producerValue.getDefiningOp();
    }

    if (!currentOp->hasOneUse()) {
        return;
    }

    if (!mlir::isa_and_present<VPU::NCEMaxPoolOp, VPU::NCEEltwiseOp>(currentOp)) {
        return;
    }

    if (!currentOp->isBeforeInBlock(consumerOp)) {
        return;
    }

    auto postOp = consumerOp;
    for (auto op : viewLikeOps) {
        op->moveBefore(postOp);
        postOp = op;
    }

    currentOp->moveBefore(postOp);
}

void EfficientIROrderPass::safeRunOnFunc() {
    auto func = getOperation();

    if (hasVFBlock(func)) {
        SmallVector<VFOpMoveAfter> sdpaMoveOps;

        // Reorder operations in every VF block for efficient execution
        func->walk([&](VPU::VerticalFusionOp vfOp) {
            reorderOperationsInVFBlock(vfOp);
            collectVfOpsToMoveBeforeSDPA(sdpaMoveOps, vfOp);
        });

        // Reorder operations before SDPA pattern for better scheduling
        for (auto& [consumer, vfOp] : sdpaMoveOps) {
            consumer->moveAfter(vfOp);
        }

        return;
    }

    if (_enableReorderConcatBranches) {
        func->walk([&](VPU::ConcatOp concatOp) {
            reorderConcatBranches(concatOp);
        });
    }

    auto operationsInBlock =
            to_small_vector(func.getOps<VPU::NCEOpInterface>() | transformed([](VPU::NCEOpInterface op) {
                                return op.getOperation();
                            }));
    VPU::reorderOperations(operationsInBlock);

    // Reorder MaxPool as scale table
    for (auto nceOp : func.getOps<VPU::NCEOpInterface>()) {
        reordeNCEOpAsWeightScaleTable(nceOp);
    }
}

}  // namespace

//
// createEfficientIROrderPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createEfficientIROrderPass(bool enableReorderConcatBranches, Logger log) {
    return std::make_unique<EfficientIROrderPass>(enableReorderConcatBranches, log);
}
