//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Dialect/Quant/IR/Quant.h>

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

// Remove remaining Quantize->Dequantize sequence to not perform explicit FakeQuantize.
// This might have slight impact on accuracy but gives visible performance improvement
//
//         Quantize                     Quantize
//            |                            |
//  ElemTypeInfoOpInterface      ElemTypeInfoOpInterface
//            |                            |
//            \__________        __________/
//                       \      /
//                        Concat
//                          |
//                ElemTypeInfoOpInterface
//                          |
//                      Dequantize
//
// TODO: Evaluate possibility of replacing such sequence with ClampOp fused with DPU task
class RemoveQuantDequantSeqPass final : public IE::impl::RemoveQuantDequantSeqBase<RemoveQuantDequantSeqPass> {
public:
    explicit RemoveQuantDequantSeqPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

class ConcatRemoveQuantDequantRewriter final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    ConcatRemoveQuantDequantRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

class QuantizeRemoveQuantDequantRewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    QuantizeRemoveQuantDequantRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

class DequantizeRemoveQuantDequantRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeRemoveQuantDequantRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConcatRemoveQuantDequantRewriter::matchAndRewrite(IE::ConcatOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at loc '{1}'", origOp->getName(), origOp->getLoc());

    if (!origOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "Op has more than one use");
    }

    SmallVector<std::pair<mlir::OpOperand*, mlir::Operation*>> quantizeOps;
    SmallVector<mlir::Operation*> consumerOps;
    SmallVector<SmallVector<mlir::Operation*>> nestedInterReturnTypesOps;

    for (auto& operand : origOp->getOpOperands()) {
        auto parentOp = operand.get().getDefiningOp();
        mlir::OpOperand* parentOperand = &operand;

        SmallVector<mlir::Operation*> parentOps;
        if (!mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface, IE::QuantizeOp>(parentOp)) {
            return matchFailed(_log, rewriter, origOp,
                               "Concat input #{0} is not produced by Quantize or ElemTypeInfoOpInterface",
                               operand.getOperandNumber());
        }
        if (parentOp->getNumOperands() != 1) {
            return matchFailed(_log, rewriter, origOp,
                               "Producer '{0}' at loc '{1}' of Concat input #{2} has {3} operands, expected 1",
                               parentOp->getName(), parentOp->getLoc(), operand.getOperandNumber(),
                               parentOp->getNumOperands());
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
                _log.nest().trace("Concat input #{0}: push ElemTypeInfoOpInterface '{1}' at loc '{2}'",
                                  operand.getOperandNumber(), parentOp->getName(), parentOp->getLoc());
                parentOps.push_back(parentOp);
            }
            parentOperand = &parentOp->getOpOperand(0);
            parentOp = parentOperand->get().getDefiningOp();
            if (mlir::isa_and_nonnull<IE::ConcatOp>(parentOp)) {
                return matchFailed(_log, rewriter, origOp,
                                   "Chain of Concat input #{0} reaches another Concat at loc '{1}'",
                                   operand.getOperandNumber(), parentOp->getLoc());
            }
            if (!mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface, IE::QuantizeOp>(parentOp)) {
                return matchFailed(_log, rewriter, origOp,
                                   "Chain of Concat input #{0} reaches an op which is neither Quantize nor "
                                   "ElemTypeInfoOpInterface",
                                   operand.getOperandNumber());
            }
            if (parentOp->getNumOperands() != 1) {
                return matchFailed(_log, rewriter, origOp,
                                   "Producer '{0}' at loc '{1}' in the chain of Concat input #{2} has {3} operands, "
                                   "expected 1",
                                   parentOp->getName(), parentOp->getLoc(), operand.getOperandNumber(),
                                   parentOp->getNumOperands());
            }
        }

        if (!mlir::isa_and_nonnull<IE::QuantizeOp>(parentOp)) {
            return matchFailed(_log, rewriter, origOp, "Chain of Concat input #{0} does not start with Quantize",
                               operand.getOperandNumber());
        }
        _log.nest().trace("Concat input #{0}: found producer Quantize at loc '{1}' behind {2} ElemTypeInfoOpInterface "
                          "op(s)",
                          operand.getOperandNumber(), parentOp->getLoc(), parentOps.size());
        // Always add the operand-quantize pair, even if the quantize op was already seen
        // This handles cases like Concat(%0, %0) where both inputs come from the same quantize
        quantizeOps.push_back(std::make_pair(parentOperand, parentOp));
        nestedInterReturnTypesOps.push_back(parentOps);
    }

    mlir::Operation* operation = origOp;
    mlir::Operation* dequantizeOp = nullptr;
    while (!operation->getResult(0).getUsers().empty() &&
           mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface, IE::DequantizeOp>(operation)) {
        auto consumer = *(operation->getResult(0).getUsers().begin());
        if (!consumer->getResult(0).hasOneUse()) {
            return matchFailed(_log, rewriter, origOp, "Consumer '{0}' at loc '{1}' has more than one use",
                               consumer->getName(), consumer->getLoc());
        }
        if (mlir::isa_and_nonnull<IE::ConcatOp>(consumer)) {
            return matchFailed(_log, rewriter, origOp, "Consumer at loc '{0}' is another Concat", consumer->getLoc());
        }
        if (mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface>(consumer)) {
            _log.nest().trace("Push consumer ElemTypeInfoOpInterface '{0}' at loc '{1}'", consumer->getName(),
                              consumer->getLoc());
            consumerOps.push_back(consumer);
        }
        if (mlir::isa_and_nonnull<IE::DequantizeOp>(consumer)) {
            _log.nest().trace("Found Dequantize at loc '{0}', stop pattern searching", consumer->getLoc());
            dequantizeOp = mlir::dyn_cast<vpux::IE::DequantizeOp>(*consumer);
            break;
        }
        operation = consumer;
    }

    if (dequantizeOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "No Dequantize consumer found below Concat");
    }

    // Safety check: ensure multi-result ops in the chain don't have results with users
    // outside the current transformation chain
    llvm::SmallPtrSet<mlir::Operation*, 16> chainOpsSet;
    for (const auto& chainOps : nestedInterReturnTypesOps) {
        chainOpsSet.insert(chainOps.begin(), chainOps.end());
    }
    chainOpsSet.insert(origOp.getOperation());
    chainOpsSet.insert(consumerOps.begin(), consumerOps.end());
    chainOpsSet.insert(dequantizeOp);

    for (auto* op : chainOpsSet) {
        if (op->getNumResults() <= 1) {
            continue;
        }
        for (auto result : op->getResults()) {
            for (auto* user : result.getUsers()) {
                if (!chainOpsSet.contains(user)) {
                    return matchFailed(_log, rewriter, origOp,
                                       "Multi-result op '{0}' at loc '{1}' has user '{2}' at loc '{3}' outside of the "
                                       "captured pattern",
                                       op->getName(), op->getLoc(), user->getName(), user->getLoc());
                }
            }
        }
    }

    _log.trace("Captured pattern Quantize->[ElemTypeInfoOpInterface...]->Concat->[ElemTypeInfoOpInterface...]->"
               "Dequantize for Concat at loc '{0}': {1} Quantize input(s), {2} consumer(s)",
               origOp->getLoc(), quantizeOps.size(), consumerOps.size());

    for (size_t i = 0; i < quantizeOps.size(); i++) {
        auto operand = quantizeOps[i].first;
        auto quantOp = quantizeOps[i].second;

        _log.nest().trace("Bypassing Quantize at loc '{0}' for Concat input #{1}", quantOp->getLoc(),
                          operand->getOperandNumber());

        rewriter.modifyOpInPlace(operand->getOwner(), [&] {
            operand->set(quantOp->getOperand(0));
        });
    }

    for (auto& opsToInferReturnType1 : nestedInterReturnTypesOps) {
        for (auto op = opsToInferReturnType1.rbegin(); op != opsToInferReturnType1.rend(); ++op) {
            inferReturnTypes(*op, InferShapedTypeMode::ELEM_TYPE);
            _log.nest().trace("Inferred element type of '{0}' at loc '{1}': '{2}'", (*op)->getName(), (*op)->getLoc(),
                              (*op)->getResult(0).getType());
        }
    }

    inferReturnTypes(origOp, InferShapedTypeMode::ELEM_TYPE);
    _log.nest().trace("Inferred element type of '{0}' at loc '{1}': '{2}'", origOp->getName(), origOp->getLoc(),
                      origOp->getResult(0).getType());

    for (auto op : consumerOps) {
        inferReturnTypes(op, InferShapedTypeMode::ELEM_TYPE);
        _log.nest().trace("Inferred element type of '{0}' at loc '{1}': '{2}'", op->getName(), op->getLoc(),
                          op->getResult(0).getType());
    }

    _log.nest().trace("Removing Dequantize at loc '{0}' and replacing it with its operands", dequantizeOp->getLoc());
    rewriter.replaceOp(dequantizeOp, dequantizeOp->getOperands());

    return mlir::success();
}

mlir::LogicalResult QuantizeRemoveQuantDequantRewriter::matchAndRewrite(IE::QuantizeOp origOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at loc '{1}'", origOp->getName(), origOp->getLoc());

    if (!origOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "Op has more than one use");
    }

    auto dequantizeOp = mlir::dyn_cast<vpux::IE::DequantizeOp>(*origOp->getUsers().begin());
    if (dequantizeOp != nullptr) {
        _log.trace("Found pattern Quantize->Dequantize. Removing Quantize at loc '{0}' and Dequantize at loc '{1}', "
                   "replacing them with the Quantize input",
                   origOp->getLoc(), dequantizeOp->getLoc());
        rewriter.replaceOp(dequantizeOp, origOp.getInput());
        rewriter.eraseOp(origOp);
        return mlir::success();
    }

    _log.trace("Search target pattern for '{0}' at loc '{1}'", origOp->getName(), origOp->getLoc());

    SmallVector<mlir::Operation*> targetOps;
    mlir::Operation* operation = origOp;
    while (!operation->getUsers().empty()) {
        auto user = *(operation->getUsers().begin());

        if (mlir::isa_and_nonnull<IE::ConcatOp>(user)) {
            return matchFailed(_log, rewriter, origOp,
                               "User at loc '{0}' is a Concat, handled by ConcatRemoveQuantDequantRewriter",
                               user->getLoc());
        }

        if (!mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface, IE::DequantizeOp>(user)) {
            return matchFailed(_log, rewriter, origOp,
                               "User '{0}' at loc '{1}' is neither ElemTypeInfoOpInterface nor Dequantize",
                               user->getName(), user->getLoc());
        }

        if (mlir::isa_and_nonnull<IE::InterpolateOp>(user)) {
            auto seOp = mlir::dyn_cast<IE::SEOpInterface>(user);
            if (seOp && seOp.isSupported(emptyLogCb)) {
                return matchFailed(_log, rewriter, origOp,
                                   "Found Interpolate user at loc '{0}' that can be mapped to SE, stop pattern "
                                   "searching",
                                   user->getLoc());
            }
        }

        if (mlir::isa_and_nonnull<IE::ElemTypeInfoOpInterface>(user)) {
            if (!user->hasOneUse()) {
                return matchFailed(_log, rewriter, origOp, "User '{0}' at loc '{1}' has more than one use",
                                   user->getName(), user->getLoc());
            }
            _log.nest().trace("Push ElemTypeInfoOpInterface '{0}' at loc '{1}'", user->getName(), user->getLoc());
            targetOps.push_back(user);
            operation = user;
            continue;
        }

        if (mlir::isa_and_nonnull<IE::DequantizeOp>(user)) {
            _log.nest().trace("Found Dequantize at loc '{0}', stop pattern searching", user->getLoc());
            dequantizeOp = mlir::cast<vpux::IE::DequantizeOp>(*user);
            break;
        }
    }

    if (dequantizeOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "No Dequantize user found below Quantize");
    }

    _log.trace("Captured pattern Quantize->[ElemTypeInfoOpInterface...]->Dequantize for Quantize at loc '{0}' with "
               "{1} intermediate op(s)",
               origOp->getLoc(), targetOps.size());

    // Rewrite the sub-graph.
    _log.nest().trace("Bypassing Quantize at loc '{0}' for '{1}' at loc '{2}'", origOp->getLoc(),
                      targetOps.front()->getName(), targetOps.front()->getLoc());
    rewriter.modifyOpInPlace(targetOps.front(), [&] {
        targetOps.front()->getOpOperand(0).set(origOp.getInput());
    });
    for (auto op : targetOps) {
        inferReturnTypes(op, InferShapedTypeMode::ELEM_TYPE);
        _log.nest().trace("Inferred element type of '{0}' at loc '{1}': '{2}'", op->getName(), op->getLoc(),
                          op->getResult(0).getType());
    }

    auto replaceOp = targetOps.back();
    _log.nest().trace("Removing Dequantize at loc '{0}' and replacing it with '{1}' at loc '{2}'",
                      dequantizeOp->getLoc(), replaceOp->getName(), replaceOp->getLoc());
    rewriter.replaceOp(dequantizeOp, replaceOp);

    return mlir::success();
}

mlir::LogicalResult DequantizeRemoveQuantDequantRewriter::matchAndRewrite(IE::DequantizeOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at loc '{1}'", origOp->getName(), origOp->getLoc());

    const auto inputElementType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType()).getElementType();
    const auto outputElementType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType()).getElementType();
    if (!inputElementType.isF16() || !outputElementType.isF16()) {
        return matchFailed(_log, rewriter, origOp,
                           "Input element type is '{0}' and output element type is '{1}'. Both are required to be FP16",
                           inputElementType, outputElementType);
    }

    _log.trace("Removing FP16->FP16 Dequantize at loc '{0}' and replacing it with its operands", origOp->getLoc());
    rewriter.replaceOp(origOp, origOp->getOperands());

    return mlir::success();
}

}  // namespace

void RemoveQuantDequantSeqPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    {
        _log.trace("Stage 1: Fold Quantize->[...]->Concat->[...]->Dequantize patterns");
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<ConcatRemoveQuantDequantRewriter>(&ctx, _log.nest());
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }

    {
        _log.trace("Stage 2: Fold Quantize->[...]->Dequantize patterns");
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<QuantizeRemoveQuantDequantRewriter>(&ctx, _log.nest());
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }

    {
        _log.trace("Stage 3: Erase FP16->FP16 Dequantize ops left by the previous stages");
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<DequantizeRemoveQuantDequantRewriter>(&ctx, _log.nest());
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }
}

//
// createRemoveQuantDequantSeqPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createRemoveQuantDequantSeqPass(Logger log) {
    return std::make_unique<RemoveQuantDequantSeqPass>(log);
}
