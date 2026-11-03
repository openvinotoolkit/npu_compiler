//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/SetVector.h>
#include <mlir/IR/PatternMatch.h>

namespace vpux::IE {
#define GEN_PASS_DECL_UNIQUIFYSIMILAROPS
#define GEN_PASS_DEF_UNIQUIFYSIMILAROPS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

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
    const auto outPermuteCastLoc = takeOpLoc(secondOp, "permute_out");
    auto permuteCast = IE::tryToFindPermuteCastOp(outPermuteCastLoc, firstOp.getOutput(), outputType.getDimsOrder(),
                                                  outputType.getShape(), rewriter);
    assert(permuteCast.has_value() && "Valid PermuteCast must be generated. Condition checked beforehand.");
    rewriter.replaceOp(secondOp, permuteCast.value()->getResults());
}

class UniquifySimilarOpsPass final : public IE::impl::UniquifySimilarOpsBase<UniquifySimilarOpsPass> {
public:
    explicit UniquifySimilarOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UniquifySimilarOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RemoveDuplicatingPermute>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createUniquifySimilarOpsPass(Logger log) {
    return std::make_unique<UniquifySimilarOpsPass>(log);
}
