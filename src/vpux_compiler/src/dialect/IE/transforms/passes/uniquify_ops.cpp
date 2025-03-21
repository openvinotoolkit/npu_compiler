//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux::IE {
#define GEN_PASS_DECL_UNIQUIFYOPS
#define GEN_PASS_DEF_UNIQUIFYOPS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// RemoveDuplicatingGeneric
//

template <typename ConcreteOp>
class RemoveDuplicatingGeneric : public mlir::OpRewritePattern<ConcreteOp> {
public:
    RemoveDuplicatingGeneric(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    virtual bool isDuplicatedOperation(ConcreteOp firstOp, ConcreteOp secondOp, Logger log) const;
    virtual void eliminateDuplicatedOperation(ConcreteOp firstOp, ConcreteOp secondOp,
                                              mlir::PatternRewriter& rewriter) const;
    Logger _log;
};

template <typename ConcreteOp>
bool RemoveDuplicatingGeneric<ConcreteOp>::isDuplicatedOperation(ConcreteOp firstOp, ConcreteOp secondOp,
                                                                 Logger) const {
    if (firstOp && secondOp) {
        if (firstOp.getType() == secondOp.getType()) {
            return true;
        }
    }
    return false;
}

template <typename ConcreteOp>
void RemoveDuplicatingGeneric<ConcreteOp>::eliminateDuplicatedOperation(ConcreteOp firstOp, ConcreteOp secondOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    rewriter.replaceOp(secondOp, firstOp->getResults());
}

template <typename ConcreteOp>
mlir::LogicalResult RemoveDuplicatingGeneric<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    ConcreteOp firstUser = origOp;
    for (auto user : origOp->getOperand(0).getUsers()) {
        if (auto currOp = mlir::dyn_cast<ConcreteOp>(user)) {
            if (currOp->isBeforeInBlock(firstUser.getOperation()) && isDuplicatedOperation(origOp, currOp, _log)) {
                firstUser = currOp;
            }
        }
    }

    mlir::DenseSet<mlir::Operation*> usersToRemove;

    for (auto user : llvm::make_early_inc_range(firstUser->getOperand(0).getUsers())) {
        if (user == firstUser) {
            continue;
        }

        if (auto currOp = mlir::dyn_cast<ConcreteOp>(user)) {
            if (isDuplicatedOperation(firstUser, currOp, _log)) {
                usersToRemove.insert(currOp);
            }
        }
    }

    if (usersToRemove.size() == 0) {
        return mlir::failure();
    }

    for (auto user : usersToRemove) {
        auto userOp = mlir::dyn_cast<ConcreteOp>(user);
        _log.trace("Current node has a duplicate. Eliminate usage of current node:\n{0} {1}\n{2} {3}",
                   firstUser.getLoc(), firstUser, userOp.getLoc(), userOp);
        eliminateDuplicatedOperation(firstUser, userOp, rewriter);
    }

    return mlir::success();
}

//
// RemoveDuplicatingConcat
//

class RemoveDuplicatingConcat final : public RemoveDuplicatingGeneric<IE::ConcatOp> {
public:
    RemoveDuplicatingConcat(mlir::MLIRContext* ctx, Logger log): RemoveDuplicatingGeneric<IE::ConcatOp>(ctx, log) {
    }

private:
    bool isDuplicatedOperation(IE::ConcatOp firstOp, IE::ConcatOp secondOp, Logger log) const override;
};

bool RemoveDuplicatingConcat::isDuplicatedOperation(IE::ConcatOp firstOp, IE::ConcatOp secondOp, Logger) const {
    auto inputNumber = firstOp.getInputs().size();
    if (inputNumber != secondOp.getInputs().size()) {
        return false;
    }

    for (size_t i = 0; i < inputNumber; i++) {
        if (firstOp.getInputs()[i] != secondOp.getInputs()[i]) {
            return false;
        }
    }

    if (firstOp.getType() != secondOp.getType()) {
        return false;
    }

    if (firstOp.getPerAxisAttr() != secondOp.getPerAxisAttr()) {
        return false;
    }

    if (firstOp.getStaticOffsetsAttr() != secondOp.getStaticOffsetsAttr()) {
        return false;
    }

    return true;
}

//
// RemoveDuplicatingCommutativeEltwise
//

// The class is for commutative eltwise operation like add, and, don't use it for subtract
template <typename ConcreteOp>
class RemoveDuplicatingCommutativeEltwise final : public RemoveDuplicatingGeneric<ConcreteOp> {
public:
    RemoveDuplicatingCommutativeEltwise(mlir::MLIRContext* ctx, Logger log)
            : RemoveDuplicatingGeneric<ConcreteOp>(ctx, log) {
    }

private:
    bool isDuplicatedOperation(ConcreteOp firstOp, ConcreteOp secondOp, Logger log) const override;
};

template <typename ConcreteOp>
bool RemoveDuplicatingCommutativeEltwise<ConcreteOp>::isDuplicatedOperation(ConcreteOp firstOp, ConcreteOp secondOp,
                                                                            Logger) const {
    if (firstOp.getType() != secondOp.getType()) {
        return false;
    }

    const auto firstOpInput1 = firstOp->getOperands()[0];
    const auto firstOpInput2 = firstOp->getOperands()[1];
    const auto secondOpInput1 = secondOp->getOperands()[0];
    const auto secondOpInput2 = secondOp->getOperands()[1];

    const auto inputsAreEqual = (firstOpInput1 == secondOpInput1) && (firstOpInput2 == secondOpInput2);
    const auto swappedInputsAreEqual = (firstOpInput1 == secondOpInput2) && (firstOpInput2 == secondOpInput1);

    return inputsAreEqual || swappedInputsAreEqual;
}

//
// RemoveDuplicatingPooling
//

// The class is for pooling operation like maxpool and avgpool
template <typename ConcreteOp>
class RemoveDuplicatingPooling final : public RemoveDuplicatingGeneric<ConcreteOp> {
public:
    RemoveDuplicatingPooling(mlir::MLIRContext* ctx, Logger log): RemoveDuplicatingGeneric<ConcreteOp>(ctx, log) {
    }

private:
    bool isDuplicatedOperation(ConcreteOp firstOp, ConcreteOp secondOp, Logger log) const override;
};

template <typename ConcreteOp>
bool RemoveDuplicatingPooling<ConcreteOp>::isDuplicatedOperation(ConcreteOp firstOp, ConcreteOp secondOp,
                                                                 Logger) const {
    if (firstOp.getType() != secondOp.getType()) {
        return false;
    }

    if (firstOp->getAttrDictionary() != secondOp->getAttrDictionary()) {
        return false;
    }

    return true;
}

//
// RemoveDuplicatingPermute
//

class RemoveDuplicatingPermute final : public RemoveDuplicatingGeneric<IE::MemPermuteOp> {
public:
    RemoveDuplicatingPermute(mlir::MLIRContext* ctx, Logger log): RemoveDuplicatingGeneric<IE::MemPermuteOp>(ctx, log) {
    }

private:
    bool isDuplicatedOperation(IE::MemPermuteOp firstOp, IE::MemPermuteOp secondOp, Logger log) const override;
    void eliminateDuplicatedOperation(IE::MemPermuteOp firstOp, IE::MemPermuteOp secondOp,
                                      mlir::PatternRewriter& rewriter) const override;
};

bool RemoveDuplicatingPermute::isDuplicatedOperation(IE::MemPermuteOp firstOp, IE::MemPermuteOp secondOp,
                                                     Logger log) const {
    if (firstOp == nullptr || secondOp == nullptr) {
        return false;
    }

    if (firstOp.getType() == secondOp.getType()) {
        return true;
    }

    // check if two MemPermute ops have the same merged permutation
    auto firstMergedPermAndShape = vpux::getMergedPermutationAndShape(
            firstOp.getInput().getType().cast<NDTypeInterface>(), firstOp.getMemPerm());
    auto secondMergedPermAndShape = vpux::getMergedPermutationAndShape(
            secondOp.getInput().getType().cast<NDTypeInterface>(), secondOp.getMemPerm());

    auto firstMergedPerm = firstMergedPermAndShape.first;
    auto secondMergedPerm = secondMergedPermAndShape.first;

    log.trace("[RemoveDuplicatingPermute]: firstMergedPerm {0}, secondMergedPerm {1}", firstMergedPerm,
              secondMergedPerm);
    if (firstMergedPerm == secondMergedPerm) {
        return true;
    }

    return false;
}

void RemoveDuplicatingPermute::eliminateDuplicatedOperation(IE::MemPermuteOp firstOp, IE::MemPermuteOp secondOp,
                                                            mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();

    rewriter.startOpModification(firstOp);
    rewriter.setInsertionPointAfter(firstOp);

    // Set destination order
    auto outputType = secondOp.getOutput().getType().cast<NDTypeInterface>();

    const auto targetLayout = outputType.getDimsOrder();
    const auto targetOrderAttr = mlir::AffineMapAttr::get(targetLayout.toAffineMap(ctx));
    const auto outLayoutCastLoc = appendLoc(secondOp.getLoc(), "_out_layout_cast");
    auto outLayoutCastOp = rewriter.create<IE::LayoutCastOp>(outLayoutCastLoc, firstOp.getOutput(), targetOrderAttr);

    // Set destination shape
    const auto outShapeCastLoc = appendLoc(secondOp.getLoc(), "_out_shape_cast");
    const auto targetShape = outputType.getShape();
    const auto targetShapeAttr = getIntArrayAttr(ctx, targetShape.raw());
    auto outShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(outShapeCastLoc, outputType, outLayoutCastOp.getOutput(), targetShapeAttr);
    rewriter.replaceOp(secondOp, outShapeCastOp->getResults());

    rewriter.finalizeOpModification(firstOp);
}

//
// RemoveDuplicatingSlice
//

class RemoveDuplicatingSlice final : public RemoveDuplicatingGeneric<IE::SliceOp> {
public:
    RemoveDuplicatingSlice(mlir::MLIRContext* ctx, Logger log): RemoveDuplicatingGeneric<IE::SliceOp>(ctx, log) {
    }

private:
    bool isDuplicatedOperation(IE::SliceOp firstOp, IE::SliceOp secondOp, Logger log) const override;
};

bool RemoveDuplicatingSlice::isDuplicatedOperation(IE::SliceOp firstOp, IE::SliceOp secondOp, Logger) const {
    if (firstOp.getType() != secondOp.getType()) {
        return false;
    }

    if (firstOp.getStaticOffsetsAttr() != secondOp.getStaticOffsetsAttr()) {
        return false;
    }

    if (firstOp.getStaticSizesAttr() != secondOp.getStaticSizesAttr()) {
        return false;
    }

    return true;
}

//
// RemoveDuplicatingExpand
//

class RemoveDuplicatingExpand final : public RemoveDuplicatingGeneric<IE::ExpandOp> {
public:
    RemoveDuplicatingExpand(mlir::MLIRContext* ctx, Logger log): RemoveDuplicatingGeneric<IE::ExpandOp>(ctx, log) {
    }

private:
    bool isDuplicatedOperation(IE::ExpandOp firstOp, IE::ExpandOp secondOp, Logger log) const override;
};

bool RemoveDuplicatingExpand::isDuplicatedOperation(IE::ExpandOp firstOp, IE::ExpandOp secondOp, Logger) const {
    if (firstOp.getType() != secondOp.getType()) {
        return false;
    }

    if (firstOp.getPadsBeginAttr() != secondOp.getPadsBeginAttr()) {
        return false;
    }

    if (firstOp.getPadsEndAttr() != secondOp.getPadsEndAttr()) {
        return false;
    }

    return true;
}

//
// UniquifyOpsPass
//

class UniquifyOpsPass final : public IE::impl::UniquifyOpsBase<UniquifyOpsPass> {
public:
    explicit UniquifyOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UniquifyOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RemoveDuplicatingGeneric<IE::ReorderOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::PermuteCastOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::ShapeCastOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::QuantizeCastOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::LayoutCastOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::ReshapeOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::AffineReshapeOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::PermuteQuantizeOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::TileOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::FloorOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::ConvertOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingCommutativeEltwise<IE::AddOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingCommutativeEltwise<IE::AndOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingPooling<IE::AvgPoolOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingPooling<IE::MaxPoolOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingConcat>(&ctx, _log);
    patterns.add<RemoveDuplicatingPermute>(&ctx, _log);
    patterns.add<RemoveDuplicatingSlice>(&ctx, _log);
    patterns.add<RemoveDuplicatingExpand>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUniquifyOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUniquifyOpsPass(Logger log) {
    return std::make_unique<UniquifyOpsPass>(log);
}
