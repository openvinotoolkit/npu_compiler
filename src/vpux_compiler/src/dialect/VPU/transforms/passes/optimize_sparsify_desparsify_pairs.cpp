//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_OPTIMIZESPARSIFYDESPARSIFYPAIRS
#define GEN_PASS_DEF_OPTIMIZESPARSIFYDESPARSIFYPAIRS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// OptimizeSparsifyDesparsifyPairsPass
//

class OptimizeSparsifyDesparsifyPairsPass final :
        public VPU::impl::OptimizeSparsifyDesparsifyPairsBase<OptimizeSparsifyDesparsifyPairsPass> {
public:
    explicit OptimizeSparsifyDesparsifyPairsPass(const VPU::ActivationSparsityOptions& options, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(options);

        parseFromOptions();
    }

    explicit OptimizeSparsifyDesparsifyPairsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initializeOptions(StringRef options) final;

private:
    void safeRunOnFunc() final;
    void parseFromOptions();

private:
    VPU::ActivationSparsityProfile _sparsityProfile{VPU::ActivationSparsityProfile::S0};
};

mlir::LogicalResult OptimizeSparsifyDesparsifyPairsPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    parseFromOptions();

    return mlir::success();
}

void OptimizeSparsifyDesparsifyPairsPass::parseFromOptions() {
    const auto parsedSparsityProfile = VPU::symbolizeActivationSparsityProfile(this->actSparsityProfile.getValue());
    if (parsedSparsityProfile.has_value()) {
        _sparsityProfile = parsedSparsityProfile.value();
    }
}

//
// RemoveConcatWrappers
//

class RemoveConcatWrappers final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    RemoveConcatWrappers(mlir::MLIRContext* ctx, Logger log, VPU::ActivationSparsityProfile sparsityProfile)
            : mlir::OpRewritePattern<VPU::ConcatOp>(ctx), _log(log), _sparsityProfile(sparsityProfile) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPU::ActivationSparsityProfile _sparsityProfile;
};

// Legend:
// === sparse path
// --- dense path
// Convert VPU.Concat ops wrapped by Desparsify/Sparsify
// => Desparsify \                / Sparsify => SparseConsumer
// => Desparsify |-> VPU.Concat ->| Sparsify => SparseConsumer
// => Desparsify /                \ DenseConsumer(if S1 profile)
// To
// ==\                 /=> SparseConsumer
// ==| => VPU.Concat =>|=>  SparseConsumer
// ==/                 \=> Desparsify => DenseConsumer
mlir::LogicalResult RemoveConcatWrappers::matchAndRewrite(VPU::ConcatOp concatOp,
                                                          mlir::PatternRewriter& rewriter) const {
    // Allow mixed consumers depends on profile. TODO: E#53581
    const bool allowMixedConsumers = _sparsityProfile == VPU::ActivationSparsityProfile::S1;

    mlir::DenseSet<mlir::Operation*> consumerSparsifyOps;
    size_t totalConsumers = 0;
    for (auto user : concatOp.getOutput().getUsers()) {
        ++totalConsumers;
        if (mlir::isa<VPU::SparsifyOp>(user)) {
            consumerSparsifyOps.insert(user);
        }
    }

    if (consumerSparsifyOps.empty()) {
        return mlir::failure();
    }

    const auto canBeFullyConverted = consumerSparsifyOps.size() == totalConsumers;
    if (!allowMixedConsumers && !canBeFullyConverted) {
        _log.trace("Couldnt remove sparsity wrappers around '{0}', because mixed consumers are not allowed.", concatOp);
        return mlir::failure();
    }

    SmallVector<mlir::Value> newConcatInputs;
    SmallVector<VPU::DesparsifyOp> removableDesparsifyOps;
    for (mlir::Value input : concatOp.getInputs()) {
        if (auto producingDesparsifyOp = input.getDefiningOp<VPU::DesparsifyOp>()) {
            newConcatInputs.push_back(producingDesparsifyOp.getInput());
            removableDesparsifyOps.push_back(producingDesparsifyOp);
        }
    }

    if (newConcatInputs.size() != concatOp.getInputs().size()) {
        return mlir::failure();
    }

    _log.trace("Removed wrappers around '{0}'", concatOp);
    const auto newOutputType = (*consumerSparsifyOps.begin())->getResult(0).getType();
    auto newConcatOp = rewriter.create<VPU::ConcatOp>(concatOp->getLoc(), newOutputType, newConcatInputs,
                                                      concatOp.getPerAxisAttr(), concatOp.getStaticOffsetsAttr());

    if (!canBeFullyConverted) {
        auto desparsifyOp = rewriter.create<VPU::DesparsifyOp>(concatOp->getLoc(), newConcatOp.getOutput());
        concatOp.getOutput().replaceUsesWithIf(desparsifyOp.getOutput(), [&](mlir::OpOperand& opOperand) {
            return !consumerSparsifyOps.contains(opOperand.getOwner());
        });
    }

    for (auto consumingSparsifyOp : consumerSparsifyOps) {
        consumingSparsifyOp->getResult(0).replaceAllUsesWith(newConcatOp);
        rewriter.eraseOp(consumingSparsifyOp);
    }

    for (auto producingDesparsify : removableDesparsifyOps) {
        if (producingDesparsify.getOutput().use_empty()) {
            rewriter.eraseOp(producingDesparsify);
        }
    }

    return mlir::success();
}

//
// EliminateDesparsifySparsifyPairs
//

class EliminateDesparsifySparsifyPairs final : public mlir::OpRewritePattern<VPU::DesparsifyOp> {
public:
    EliminateDesparsifySparsifyPairs(mlir::MLIRContext* ctx, Logger log, VPU::ActivationSparsityProfile sparsityProfile)
            : mlir::OpRewritePattern<VPU::DesparsifyOp>(ctx), _log(log), _sparsityProfile(sparsityProfile) {
        setDebugName("EliminateDesparsifySparsifyPairs");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::DesparsifyOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPU::ActivationSparsityProfile _sparsityProfile;
};

// Eliminate Desparsify->Sparsify sequences. In most general case looks like
//                    /-> Sparsify => SparseConsumer
//     => Desparsify -|-> Sparsify => SparseConsumer
//                    \-> DenseConsumer
// To
//     => Desparsify -> DenseConsumer
//     ||=> SparseConsumer
//     ||=> SparseConsumer
// If Desparsify has no dense consumers (most common case) sequence will be completely eliminated
//
// For S0 and AUTO profiles, all Desparsify consumers have to be Sparsify operations in order for the optimization to
// take place. This is done in order to avoid standalone Desparsify ops as much as possible since pairs can be easily
// simplified. A concrete example:
//     Sparsify -> NCEConv -> Sparsify -> Desparsify -> Sparsify -> NCEEltwise -> Sparsify -> Desparsify -> [result 0]
//                                                   |           |>
//                                                   |> Sparsify |
//                                                   |
//                                                   |> MaxPool -> [output 1]
// Could be simplified to:
//     Sparsify -> NCEConv -> Sparsify -> Desparsify -> MaxPool -> [output 1]
//                                     |
//                                     |> NCEEltwise -> Sparsify -> Desparsify -> [result 0]
// NCEConv's output pair of Sparsify->Desparsify will be later simplified, similar to NCEEltwise's input Sparsify ops.
// Therefore, standalone Desparsify operations can be avoided with S0 profile.
mlir::LogicalResult EliminateDesparsifySparsifyPairs::matchAndRewrite(VPU::DesparsifyOp desparsifyOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    size_t totalUses = 0;
    llvm::SetVector<mlir::Operation*> removableOps{};
    for (const auto user : desparsifyOp->getUsers()) {
        ++totalUses;
        if (mlir::isa<VPU::SparsifyOp>(user)) {
            removableOps.insert(user);
        }
    }
    if (removableOps.empty()) {
        return matchFailed(_log, rewriter, desparsifyOp, "[{0}] '{1}' at '{2}' does not have Sparsify consumers",
                           getDebugName(), desparsifyOp->getName(), desparsifyOp->getLoc());
    }
    if (_sparsityProfile != VPU::ActivationSparsityProfile::S1) {
        if (removableOps.size() != totalUses) {
            return matchFailed(_log, rewriter, desparsifyOp,
                               "[{0}] '{1}' at '{2}' does not have only Sparsify consumers, which is needed for S0 and "
                               "AUTO profile",
                               getDebugName(), desparsifyOp->getName(), desparsifyOp->getLoc());
        }
    }
    auto sparseInput = desparsifyOp.getInput();
    for (auto sparsifyOp : removableOps) {
        sparsifyOp->getResult(0).replaceAllUsesWith(sparseInput);
        rewriter.eraseOp(sparsifyOp);
    }
    if (removableOps.size() == totalUses) {
        _log.trace("Pattern Desparsify->Sparsify completely erased with root at '{0}'", desparsifyOp);
        rewriter.eraseOp(desparsifyOp);
    }
    return mlir::success();
}

//
// safeRunOnFunc
//

void OptimizeSparsifyDesparsifyPairsPass::safeRunOnFunc() {
    using namespace VPU;
    using namespace VPU::NCESparsity;

    auto func = getOperation();
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<EliminateDesparsifySparsifyPairs>(&ctx, _log, _sparsityProfile);
    patterns.add<RemoveConcatWrappers>(&ctx, _log, _sparsityProfile);
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeSparsifyDesparsifyPairsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createOptimizeSparsifyDesparsifyPairsPass(Logger log) {
    return std::make_unique<OptimizeSparsifyDesparsifyPairsPass>(log);
}

std::unique_ptr<mlir::Pass> vpux::VPU::createOptimizeSparsifyDesparsifyPairsPass(
        const VPU::ActivationSparsityOptions& options, Logger log) {
    return std::make_unique<OptimizeSparsifyDesparsifyPairsPass>(options, log);
}
