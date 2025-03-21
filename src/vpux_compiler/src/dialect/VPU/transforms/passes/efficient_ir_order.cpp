//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_config.hpp"
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
    explicit EfficientIROrderPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

// for NCE ops move it close to their users or parents:
// for two-inputs operations place it right after the last parent in order not to
// execute it later than both inputs are ready. Usually misalignment happens
// when one of the input is executed too early
//               CONV0
//         |             |
//       CONV1         CONV2
//                       |
//                     CONV3
//         |             |
//               ...
//             ELTWISE
// so, it would have been more beneficial to execute eltwise earlier just after convolutions
// for other operations, place them close to users in order to execute them just before
// they are needed
void reorderOperations(ArrayRef<VPU::NCEOpInterface> operations) {
    for (auto origOp : operations | reversed) {
        if (origOp->hasTrait<VPU::EltwiseOp>()) {
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

        auto* firstUser = getFirstUser(origOp->getResult(0));

        if (firstUser != nullptr) {
            origOp->moveBefore(firstUser);
        }
    }
}

void reorderOperationsInVFBlock(VPU::VerticalFusionOp vfOp) {
    SmallVector<mlir::Operation*, 4> computeOpsInBlock;
    auto vfConfig = VFConfig(vfOp);
    for (auto op : vfConfig.getVFOperations()) {
        if (mlir::isa<VPU::NCEOpInterface, VPU::SWOpInterface>(op)) {
            computeOpsInBlock.push_back(op);
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

void EfficientIROrderPass::safeRunOnFunc() {
    auto func = getOperation();

    if (hasVFBlock(func)) {
        // Reorder operations in every VF block for efficient execution
        func->walk([&](VPU::VerticalFusionOp vfOp) {
            reorderOperationsInVFBlock(vfOp);
        });
        return;
    }

    auto operationsInBlock = to_small_vector(func.getOps<VPU::NCEOpInterface>());
    reorderOperations(operationsInBlock);
}

}  // namespace

//
// createEfficientIROrderPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createEfficientIROrderPass(Logger log) {
    return std::make_unique<EfficientIROrderPass>(log);
}
