//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_SCFFUSELASTVIEWLIKEOP
#define GEN_PASS_DEF_SCFFUSELASTVIEWLIKEOP
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// SCFFuseLastViewLikeOpPass
//

class SCFFuseLastViewLikeOpPass final : public VPU::impl::SCFFuseLastViewLikeOpBase<SCFFuseLastViewLikeOpPass> {
public:
    explicit SCFFuseLastViewLikeOpPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

using OffsetSizePair = std::pair<SmallVector<int64_t>, SmallVector<int64_t>>;

OffsetSizePair updateOffsetAndSize(VPU::PermuteCastOp permuteCastOp, ArrayRef<int64_t> origOffsets,
                                   ArrayRef<int64_t> origSizes) {
    auto permuteCastInType = mlir::cast<NDTypeInterface>(permuteCastOp.getInput().getType());
    auto permuteCastOutType = mlir::cast<NDTypeInterface>(permuteCastOp.getResult().getType());

    auto permuteArr = [&](ArrayRef<int64_t> arr) {
        const auto arrInMemOrder = permuteCastInType.getDimsOrder().toMemoryOrder(Shape(arr));
        const auto arrPermutedInMemOrder = applyPerm(arrInMemOrder, permuteCastOp.getMemPerm());
        return permuteCastOutType.getDimsOrder().toLogicalOrder(arrPermutedInMemOrder).raw();
    };

    const auto newOffsets = permuteArr(origOffsets);
    const auto newSizes = permuteArr(origSizes);

    return {newOffsets, newSizes};
}

void moveOpInsideForLoop(VPU::PermuteCastOp permuteCastOp, mlir::tensor::InsertSliceOp insertSliceOp) {
    mlir::OpBuilder bodyBuilder(insertSliceOp.getOperation());

    mlir::IRMapping mapper;
    mapper.map(permuteCastOp->getOperand(0), insertSliceOp.getSource());
    auto newPermuteCastOp = bodyBuilder.clone(*permuteCastOp.getOperation(), mapper);
    vpux::inferReturnTypes(newPermuteCastOp, vpux::InferShapedTypeMode::ALL);

    auto permuteCastOutType = mlir::cast<NDTypeInterface>(permuteCastOp.getResult().getType());
    insertSliceOp.setOperand(0, newPermuteCastOp->getResult(0));
    insertSliceOp.getDestMutable().get().setType(permuteCastOutType);
    insertSliceOp.getResult().setType(mlir::cast<mlir::RankedTensorType>(permuteCastOutType));

    auto [newOffsets, newSizes] =
            updateOffsetAndSize(permuteCastOp, insertSliceOp.getStaticOffsets(), insertSliceOp.getStaticSizes());

    insertSliceOp.setStaticOffsets(newOffsets);
    insertSliceOp.setStaticSizes(newSizes);
}

bool traverseNestedForLoops(mlir::scf::ForOp forOp, VPU::PermuteCastOp permuteCastOp, Logger log) {
    auto permuteCastOutType = mlir::cast<NDTypeInterface>(permuteCastOp.getResult().getType());
    auto body = forOp.getBody();

    auto yieldOps = body->getOps<mlir::scf::YieldOp>();
    assert(!yieldOps.empty() && "no scf.yield op in for loop");
    assert(std::distance(yieldOps.begin(), yieldOps.end()) == 1 && "only one scf.yield op allowed per loop");

    auto yieldOp = *yieldOps.begin();

    mlir::Operation* producer = yieldOp->getOperand(0).getDefiningOp();

    bool movedPermute = false;
    if (auto insertSliceOp = mlir::dyn_cast_or_null<mlir::tensor::InsertSliceOp>(producer)) {
        log.trace("Found InsertSliceOp at {0}", insertSliceOp->getLoc());
        moveOpInsideForLoop(permuteCastOp, insertSliceOp);
        movedPermute = true;
    }

    if (auto innerForOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(producer)) {
        log.trace("Found inner scf for loop at {0}", innerForOp->getLoc());
        movedPermute = traverseNestedForLoops(innerForOp, permuteCastOp, log);
    }

    if (movedPermute) {
        log.trace("Permute got moved, adapt return type for scf.for at ", forOp->getLoc());
        auto iterArg = forOp.getInitArgsMutable().begin()->get();
        iterArg.setType(permuteCastOutType);
        forOp.getResult(0).setType(permuteCastOutType);
    }
    return movedPermute;
}

void SCFFuseLastViewLikeOpPass::safeRunOnFunc() {
    auto func = getOperation();

    SmallVector<VPU::PermuteCastOp> permuteCastOpToErase;
    func->walk([&](VPU::PermuteCastOp permuteCastOp) {
        mlir::Operation* user = *permuteCastOp->getResult(0).getUsers().begin();
        const auto& nestedLog = _log.nest();
        _log.trace("Got '{0}' at '{1}'", permuteCastOp->getName(), permuteCastOp->getLoc());

        if (!mlir::isa<mlir::func::ReturnOp>(user)) {
            nestedLog.trace("Not used by a return operation");
            return;
        }

        mlir::Operation* producer = permuteCastOp->getOperand(0).getDefiningOp();
        if (!mlir::isa<mlir::scf::ForOp>(producer)) {
            nestedLog.trace("Not produced by a scf::ForOp");
            return;
        }

        auto forOp = mlir::cast<mlir::scf::ForOp>(producer);

        if (!forOp->getResult(0).hasOneUse()) {
            nestedLog.trace("scf::ForOp output has more than one use.");
            return;
        }

        const auto movedPermute = traverseNestedForLoops(forOp, permuteCastOp, nestedLog);

        if (!movedPermute) {
            nestedLog.trace("Could not move view op into parent scf.for.");
            return;
        }

        permuteCastOp->replaceAllUsesWith(forOp.getResults());

        if (permuteCastOp->getUses().empty()) {
            permuteCastOpToErase.push_back(permuteCastOp);
        }
        nestedLog.trace("Successfully fused last view op into scf::ForOp.");
    });

    for (auto permCast : llvm::make_early_inc_range(permuteCastOpToErase)) {
        permCast->erase();
    }
}

}  // namespace

//
// createSCFFuseLastViewLikeOpPass
//

std::unique_ptr<mlir::Pass> VPU::createSCFFuseLastViewLikeOpPass(Logger log) {
    return std::make_unique<SCFFuseLastViewLikeOpPass>(log);
}
