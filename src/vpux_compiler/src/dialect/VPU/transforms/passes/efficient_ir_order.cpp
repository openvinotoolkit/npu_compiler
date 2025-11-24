//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/reorder_ir_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

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

        // For operation has single computeOp input, place it right before it's first user
        auto* firstUser = getFirstUser(origOp->getResult(0));
        if (firstUser != nullptr) {
            origOp->moveBefore(firstUser);
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

    SmallVector<SmallVector<mlir::Operation*>> patternOps(parents.size());
    for (auto parentIt : parents | indexed) {
        auto index = parentIt.index();
        auto currentOp = parentIt.value();
        while (mlir::isa<VPU::ViewLikeOpInterface>(currentOp)) {
            if (!currentOp->hasOneUse()) {
                return;
            }
            patternOps[index].push_back(currentOp);
            auto operand = currentOp->getOperand(0);
            if (mlir::isa<mlir::BlockArgument>(operand)) {
                return;
            }

            currentOp = operand.getDefiningOp();
        }

        if (mlir::isa<VPU::NCEOpInterface>(currentOp)) {
            if (!currentOp->hasOneUse()) {
                return;
            }

            patternOps[index].push_back(currentOp);

            currentOp = currentOp->getOperand(0).getDefiningOp();
            if (auto softMaxOp = mlir::dyn_cast_or_null<VPU::SoftMaxOp>(currentOp)) {
                if (!softMaxOp.getOutput().hasOneUse()) {
                    return;
                }

                patternOps[index].push_back(softMaxOp);

                currentOp = softMaxOp.getInput().getDefiningOp();
                if (mlir::isa_and_nonnull<VPU::NCEOpInterface>(currentOp)) {
                    if (!currentOp->getResult(0).hasOneUse()) {
                        return;
                    }

                    // Consider CMX consumption so as not to prevent task parallelism
                    auto module = concatOp.getOperation()->getParentOfType<mlir::ModuleOp>();
                    const auto numClusters = config::getTileExecutor(module).getCount();
                    const auto availableCMXSizePerCluster = vpux::VPU::getTotalCMXSize(currentOp).count();
                    const auto totalAvailableCMXSize = availableCMXSizePerCluster * numClusters;

                    auto softMaxInType = mlir::cast<vpux::NDTypeInterface>(softMaxOp.getInput().getType());
                    auto softMaxOutType = mlir::cast<vpux::NDTypeInterface>(softMaxOp.getOutput().getType());
                    auto nceOpInType0 = mlir::cast<vpux::NDTypeInterface>(currentOp->getOperand(0).getType());
                    auto nceOpInType1 = mlir::cast<vpux::NDTypeInterface>(currentOp->getOperand(1).getType());
                    auto nceOpOutType = mlir::cast<vpux::NDTypeInterface>(currentOp->getResult(0).getType());
                    auto requiredCMX = VPU::getRequiredCMXSize(softMaxInType) +
                                       VPU::getRequiredCMXSize(softMaxOutType) + VPU::getRequiredCMXSize(nceOpInType0) +
                                       VPU::getRequiredCMXSize(nceOpInType1) + VPU::getRequiredCMXSize(nceOpOutType);
                    if (requiredCMX.count() < totalAvailableCMXSize) {
                        return;
                    }

                    patternOps[index].push_back(currentOp);
                }
            }
        }
    }

    auto isSymmetricConcat = [&patternOps] {
        if (patternOps.empty()) {
            return false;
        }

        const auto& firstBranch = patternOps.front();
        if (firstBranch.empty()) {
            return false;
        }
        for (const auto& branch : patternOps) {
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

void EfficientIROrderPass::safeRunOnFunc() {
    auto func = getOperation();

    if (hasVFBlock(func)) {
        // Reorder operations in every VF block for efficient execution
        func->walk([&](VPU::VerticalFusionOp vfOp) {
            reorderOperationsInVFBlock(vfOp);
        });
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
}

}  // namespace

//
// createEfficientIROrderPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createEfficientIROrderPass(bool enableReorderConcatBranches, Logger log) {
    return std::make_unique<EfficientIROrderPass>(enableReorderConcatBranches, log);
}
