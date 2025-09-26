//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>

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
        SmallVector<SmallVector<mlir::Operation*>> nestedInterReturnTypesOps;

        for (auto& operand : concatOp->getOpOperands()) {
            auto parentOp = operand.get().getDefiningOp();
            mlir::OpOperand* parentOperand = &operand;
            SmallVector<mlir::Operation*> parentOps;

            if (!mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface, IE::QuantizeOp>(parentOp)) {
                return;
            }
            if (parentOp->getNumOperands() != 1) {
                return;
            }

            while (mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface>(parentOp)) {
                // Check if parentOp was already visited
                bool visitedParentOp = false;
                for (const auto& interReturnTypesOps1 : nestedInterReturnTypesOps) {
                    auto it = std::find(interReturnTypesOps1.begin(), interReturnTypesOps1.end(), parentOp);
                    if (it != interReturnTypesOps1.end()) {
                        visitedParentOp = true;
                    }
                }
                if (!visitedParentOp) {
                    parentOps.push_back(parentOp);
                }
                parentOperand = &parentOp->getOpOperand(0);
                parentOp = parentOperand->get().getDefiningOp();
                if (mlir::isa_and_nonnull<IE::ConcatOp>(parentOp)) {
                    return;
                }
                if (!mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface, IE::QuantizeOp>(parentOp)) {
                    return;
                }
                if (parentOp->getNumOperands() != 1) {
                    return;
                }
            }

            if (!mlir::isa_and_nonnull<IE::QuantizeOp>(parentOp)) {
                return;
            }
            bool visitedParentOp = false;
            for (auto [operand, quantOp] : quantizeOps) {
                if (quantOp == parentOp) {
                    visitedParentOp = true;
                }
            }
            if (visitedParentOp == false) {
                quantizeOps.push_back(std::make_pair(parentOperand, parentOp));
            }
            nestedInterReturnTypesOps.push_back(parentOps);
        }

        mlir::Operation* operation = concatOp;
        mlir::Operation* dequantizeOp = nullptr;
        while (!operation->getResult(0).getUsers().empty() &&
               mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface, IE::DequantizeOp>(operation)) {
            auto consumer = *(operation->getResult(0).getUsers().begin());
            if (!consumer->getResult(0).hasOneUse()) {
                return;
            }
            if (mlir::isa_and_nonnull<IE::ConcatOp>(consumer)) {
                return;
            }
            if (mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface>(consumer)) {
                consumerOps.push_back(consumer);
            }
            if (mlir::isa_and_nonnull<IE::DequantizeOp>(consumer)) {
                dequantizeOp = mlir::dyn_cast<vpux::IE::DequantizeOp>(*consumer);
                break;
            }
            operation = consumer;
        }

        if (dequantizeOp == nullptr) {
            return;
        }

        // Skip QuantizeOp by linking to above op
        for (auto [operand, quantOp] : quantizeOps) {
            operand->set(quantOp->getOperand(0));
            opsToErase.push_back(quantOp);
        }

        // Infer return types to the chain of ElemTypeInfoOpInterface that sets between QuantizeOp and ConcatOp
        for (auto& opsToInferReturnType1 : nestedInterReturnTypesOps) {
            for (auto op = opsToInferReturnType1.rbegin(); op != opsToInferReturnType1.rend(); ++op) {
                inferReturnTypes(*op, InferShapedTypeMode::ELEM_TYPE);
            }
        }

        inferReturnTypes(concatOp, InferShapedTypeMode::ELEM_TYPE);

        for (auto op : consumerOps) {
            inferReturnTypes(op, InferShapedTypeMode::ELEM_TYPE);
        }

        dequantizeOp->replaceAllUsesWith(dequantizeOp->getOperands());
        opsToErase.push_back(dequantizeOp);
    });

    for (auto op : llvm::make_early_inc_range(opsToErase)) {
        // Remove dangling ops
        if (op->getResult(0).getUsers().empty()) {
            op->erase();
        }
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

                if (mlir::isa_and_nonnull<IE::ConcatOp>(user)) {
                    return;
                }

                if (!mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface, IE::DequantizeOp>(user)) {
                    return;
                }

                if (mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface>(user)) {
                    if (!user->hasOneUse()) {
                        return;
                    }
                    _log.trace("Push  ElemTypeInfoOpInterface {0} at {1}", user->getName(), user->getLoc());
                    targetOps.push_back(user);
                    operation = user;
                    continue;
                }

                if (mlir::isa_and_nonnull<IE::DequantizeOp>(user)) {
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

    // Erase any fp16->fp16 Dequantize Ops created by previous alterations
    func.walk([&](vpux::IE::DequantizeOp dequantizeOp) {
        const auto inputElementType =
                mlir::cast<vpux::NDTypeInterface>(dequantizeOp->getOperand(0).getType()).getElementType();
        const auto outputElementType =
                mlir::cast<vpux::NDTypeInterface>(dequantizeOp->getResult(0).getType()).getElementType();

        if (!inputElementType.isF16() || !outputElementType.isF16()) {
            return;
        }

        dequantizeOp->replaceAllUsesWith(dequantizeOp->getOperands());
        dequantizeOp.erase();
    });

}  // namespace

}  // namespace

//
// createRemoveQuantDequantSeqPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createRemoveQuantDequantSeqPass(Logger log) {
    return std::make_unique<RemoveQuantDequantSeqPass>(log);
}
