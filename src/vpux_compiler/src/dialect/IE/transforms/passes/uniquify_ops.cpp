//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/SetVector.h>
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

    // Small/SetVector preserves insertion order while keeping uniqueness
    constexpr auto maxUsersEstimate{16};
    llvm::SmallSetVector<mlir::Operation*, maxUsersEstimate> usersToRemove;

    auto opUsers = firstUser->getOperand(0).getUsers();
    for (auto user : opUsers) {
        if (user == firstUser) {
            continue;
        }

        if (auto currOp = mlir::dyn_cast<ConcreteOp>(user)) {
            if (isDuplicatedOperation(firstUser, currOp, _log)) {
                usersToRemove.insert(currOp);
            }
        }
    }

    if (usersToRemove.empty()) {
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

    auto firstOpType = mlir::cast<NDTypeInterface>(firstOp.getType());
    auto secondOpType = mlir::cast<NDTypeInterface>(secondOp.getType());
    auto maybeMemPerm = tryToFindPermutationForPermuteCast(firstOpType, secondOpType.getDimsOrder(),
                                                           secondOpType.getShape(), firstOp.getContext());

    log.trace("[RemoveDuplicatingPermute]: valid permutation {0}.", maybeMemPerm.has_value() ? "found" : "not found");
    return maybeMemPerm.has_value();
}

void RemoveDuplicatingPermute::eliminateDuplicatedOperation(IE::MemPermuteOp firstOp, IE::MemPermuteOp secondOp,
                                                            mlir::PatternRewriter& rewriter) const {
    rewriter.setInsertionPointAfter(firstOp);

    // Set destination order
    auto outputType = mlir::cast<vpux::NDTypeInterface>(secondOp.getOutput().getType());
    const auto outPermuteCastLoc = appendLoc(secondOp.getLoc(), "_out_perm_cast");
    auto permuteCast = tryToFindPermuteCastOp(outPermuteCastLoc, firstOp.getOutput(), outputType.getDimsOrder(),
                                              outputType.getShape(), rewriter);
    assert(permuteCast.has_value() && "Valid PermuteCast must be generated. Condition checked beforehand.");
    rewriter.replaceOp(secondOp, permuteCast.value()->getResults());
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
    patterns.add<RemoveDuplicatingGeneric<IE::TransposeOp>>(&ctx, _log);
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
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
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
