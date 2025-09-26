//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/reorder_ir_utils.hpp"
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

void EfficientIROrderPass::safeRunOnFunc() {
    auto func = getOperation();

    if (hasVFBlock(func)) {
        // Reorder operations in every VF block for efficient execution
        func->walk([&](VPU::VerticalFusionOp vfOp) {
            reorderOperationsInVFBlock(vfOp);
        });
        return;
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

std::unique_ptr<mlir::Pass> vpux::VPU::createEfficientIROrderPass(Logger log) {
    return std::make_unique<EfficientIROrderPass>(log);
}
