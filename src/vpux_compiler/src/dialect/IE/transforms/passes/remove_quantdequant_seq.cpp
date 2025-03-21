//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_REMOVEQUANTDEQUANTSEQ
#define GEN_PASS_DEF_REMOVEQUANTDEQUANTSEQ
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// RemoveQuantDequantSeqPass
//

class RemoveQuantDequantSeqPass final : public IE::impl::RemoveQuantDequantSeqBase<RemoveQuantDequantSeqPass> {
public:
    explicit RemoveQuantDequantSeqPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void RemoveQuantDequantSeqPass::safeRunOnFunc() {
    auto func = getOperation();
    // Remove remaining Quantize->Dequantize sequence to not perform explicit FakeQuantize.
    // This might have slight impact on accuracy but gives visible performance improvement
    // TODO: Evaluate possibility of replacing such sequence with ClampOp fused with DPU task
    // Quantize                          Quantize
    //   |                                  |
    //  ElemTypeInfoOpInterface         ElemTypeInfoOpInterface
    //    \                                  /
    //                Concat
    //                 |
    //            ElemTypeInfoOpInterface
    //                 |
    //               Dequant

    SmallVector<mlir::Operation*> opsToErase;

    func.walk([&](vpux::IE::ConcatOp concatOp) {
        if (!concatOp->hasOneUse()) {
            return;
        }

        SmallVector<std::pair<mlir::OpOperand*, mlir::Operation*>> quantizeOps;
        SmallVector<mlir::Operation*> consumerOps;

        for (auto& operand : concatOp->getOpOperands()) {
            auto parentOp = operand.get().getDefiningOp();

            if (!mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface, IE::QuantizeOp>(parentOp)) {
                return;
            }
            if (mlir::isa<IE::ConcatOp>(parentOp)) {
                return;
            }

            while (mlir::isa<IE::ElemTypeInfoOpInterface>(parentOp)) {
                parentOp = parentOp->getOperand(0).getDefiningOp();
                if (mlir::isa<IE::ConcatOp>(parentOp)) {
                    return;
                }
                if (!mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface, IE::QuantizeOp>(parentOp)) {
                    return;
                }
            }

            if (!mlir::isa_and_nonnull<IE::QuantizeOp>(parentOp)) {
                return;
            }

            quantizeOps.push_back(std::make_pair(&operand, parentOp));
        }

        mlir::Operation* operation = concatOp;
        mlir::Operation* dequantizeOp = nullptr;
        while (!operation->getUsers().empty() &&
               mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface, IE::DequantizeOp>(operation)) {
            auto consumer = *(operation->getResult(0).getUsers().begin());
            if (!consumer->getResult(0).hasOneUse()) {
                return;
            }
            if (mlir::isa<IE::ConcatOp>(consumer)) {
                return;
            }
            if (mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface>(consumer)) {
                consumerOps.push_back(consumer);
            }
            if (mlir::isa<IE::DequantizeOp>(consumer)) {
                dequantizeOp = mlir::dyn_cast<vpux::IE::DequantizeOp>(*consumer);
                break;
            }
            operation = consumer;
        }

        if (dequantizeOp == nullptr) {
            return;
        }

        for (auto [operand, quantOp] : quantizeOps) {
            auto childOp = *(quantOp->getResult(0).getUsers().begin());
            if (childOp == concatOp) {
                operand->set(quantOp->getOperand(0));
            } else {
                childOp->getOpOperand(0).set(quantOp->getOperand(0));
            }
            while (!mlir::isa<IE::ConcatOp>(childOp)) {
                inferReturnTypes(childOp, InferShapedTypeMode::ELEM_TYPE);
                childOp = *(childOp->getResult(0).getUsers().begin());
            }
        }

        inferReturnTypes(concatOp, InferShapedTypeMode::ELEM_TYPE);

        for (auto op : consumerOps) {
            inferReturnTypes(op, InferShapedTypeMode::ELEM_TYPE);
        }

        dequantizeOp->replaceAllUsesWith(dequantizeOp->getOperands());
        opsToErase.push_back(dequantizeOp);

        for (auto quantOp : quantizeOps) {
            opsToErase.push_back(quantOp.second);
        }
    });

    for (auto op : llvm::make_early_inc_range(opsToErase)) {
        op->erase();
    }

    func.walk([this](vpux::IE::QuantizeOp quantizeOp) {
        if (!quantizeOp->hasOneUse()) {
            return;
        }

        auto dequantizeOp = mlir::dyn_cast<vpux::IE::DequantizeOp>(*quantizeOp->getUsers().begin());
        if (dequantizeOp == nullptr) {
            SmallVector<mlir::Operation*> targetOps;
            mlir::Operation* operation = quantizeOp;
            _log.trace("Search target pattern for {0} at {1}", quantizeOp->getName(), quantizeOp->getLoc());
            while (operation && !operation->getUsers().empty()) {
                auto user = *(operation->getUsers().begin());

                if (mlir::isa<IE::ConcatOp>(user)) {
                    return;
                }

                if (!mlir::isa<IE::ElemTypeInfoOpInterface, IE::DequantizeOp>(user)) {
                    return;
                }

                if (mlir::isa<IE::ElemTypeInfoOpInterface>(user)) {
                    if (!user->hasOneUse()) {
                        return;
                    }
                    _log.trace("Push  ElemTypeInfoOpInterface {0} at {1}", user->getName(), user->getLoc());
                    targetOps.push_back(user);
                    operation = user;
                    continue;
                }

                if (mlir::isa<IE::DequantizeOp>(user)) {
                    _log.trace("Found dequantize user {0} at {1}, stop pattern searching", user->getName(),
                               user->getLoc());
                    dequantizeOp = mlir::dyn_cast<vpux::IE::DequantizeOp>(*user);
                    break;
                }
            }

            _log.trace("Capture the pattern for {0} at {1}", quantizeOp->getName(), quantizeOp->getLoc());

            //[Quantize]->[ElemTypeInfoOpInterface] ... ->[Dequantize] pattern is captured
            // Rewrite the sub-graph.
            targetOps.front()->getOpOperand(0).set(quantizeOp.getInput());
            for (auto op : targetOps) {
                inferReturnTypes(op, InferShapedTypeMode::ELEM_TYPE);
            }
            // Remove old Quantize & Dequantize ops.
            dequantizeOp.replaceAllUsesWith(targetOps.back());
            dequantizeOp.erase();
            quantizeOp.erase();
        } else {
            //[Quantize]->[Dequantize] pattern, remove it directly
            dequantizeOp.replaceAllUsesWith(quantizeOp.getInput());
        }
    });
}  // namespace

}  // namespace

//
// createRemoveQuantDequantSeqPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createRemoveQuantDequantSeqPass(Logger log) {
    return std::make_unique<RemoveQuantDequantSeqPass>(log);
}
