//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <utility>

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEANDCLEANUPFQ
#define GEN_PASS_DEF_PROPAGATEANDCLEANUPFQ
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool isFQAgnosticOp(mlir::Operation* op) {
    if (op == nullptr) {
        // BlockArgument
        return false;
    }

    // Copypaste from nGraph
    return mlir::isa<IE::MaxPoolOp, IE::VariadicSplitOp, IE::ReorgYoloOp, IE::TransposeOp, IE::SqueezeOp,
                     IE::UnsqueezeOp, IE::DepthToSpaceOp>(op);
}

IE::FakeQuantizeOp cloneFQ(IE::FakeQuantizeOp currFqOp, mlir::Operation* topUser, mlir::Value input,
                           mlir::Location originLoc, StringRef suffix, mlir::PatternRewriter& rewriter,
                           const std::shared_ptr<std::set<IE::FakeQuantizeOp>>& propagatedFQ) {
    mlir::IRMapping mapper;
    mapper.map(currFqOp.getInput(), input);

    // TODO: #-155244
    // We set the insertion point to the user operation to mimic the exact behaviour of the original ngraph pass.
    // This is to circumvent a problem in FeasibleAllocationPass that would generate a different schedule otherwise.
    rewriter.setInsertionPoint(topUser);

    auto newFqOp = mlir::cast<IE::FakeQuantizeOp>(rewriter.clone(*currFqOp, mapper));
    vpux::inferReturnTypes(newFqOp, vpux::InferShapedTypeMode::ALL);
    newFqOp->setLoc(appendLoc(originLoc, suffix));

    propagatedFQ->insert(currFqOp);
    propagatedFQ->insert(newFqOp);

    return newFqOp;
}

//
// PropagateFQUp
//

class PropagateFQUp final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    PropagateFQUp(mlir::MLIRContext* ctx, Logger log, std::shared_ptr<std::set<IE::FakeQuantizeOp>> propagatedFQ)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log), _propagatedFQ(std::move(propagatedFQ)) {
        this->setDebugName("PropagateFQUp");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp fqOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    std::shared_ptr<std::set<IE::FakeQuantizeOp>> _propagatedFQ;
};

mlir::LogicalResult PropagateFQUp::matchAndRewrite(IE::FakeQuantizeOp fqOp, mlir::PatternRewriter& rewriter) const {
    if (!IE::hasStaticLowAndHighValues(fqOp)) {
        return mlir::failure();
    }

    if (!IE::isPerTensorFQ({fqOp})) {
        return mlir::failure();
    }

    bool wasPropagated = false;
    int counter = 0;
    auto originLoc = fqOp.getLoc();
    while (true) {
        auto producerOp = fqOp.getInput().getDefiningOp();
        if (!isFQAgnosticOp(producerOp)) {
            break;
        }

        // The pattern is simplified intentionally
        // In the case of multiple users, they must all be the same FQ
        // Probably it makes sense to use RemoveDuplicatingGeneric rewriter from UniquifyOpsPass
        if (!producerOp->hasOneUse()) {
            break;
        }

        auto prevOp = producerOp->getOperand(0).getDefiningOp();
        // Copypaste from nGraph
        if (mlir::isa_and_nonnull<IE::FakeQuantizeOp, Const::DeclareOp, IE::InterpolateOp, IE::ReshapeOp>(prevOp)) {
            break;
        }

        auto newFqOp = cloneFQ(fqOp, producerOp, producerOp->getOperand(0), originLoc,
                               formatv("_propagated_up_{0}", counter).str(), rewriter, _propagatedFQ);

        // replace operand with new FQ only for producer
        producerOp->setOperand(0, newFqOp.getOutput());

        counter++;

        wasPropagated = true;
        fqOp = newFqOp;
    }

    return mlir::success(wasPropagated);
}

//
// PropagateFQDown
//

class PropagateFQDown final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    PropagateFQDown(mlir::MLIRContext* ctx, Logger log, std::shared_ptr<std::set<IE::FakeQuantizeOp>> propagatedFQ)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log), _propagatedFQ(std::move(propagatedFQ)) {
        this->setDebugName("PropagateFQDown");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp fqOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    std::shared_ptr<std::set<IE::FakeQuantizeOp>> _propagatedFQ;
};

bool areValuesEqual(Const::DeclareOp inCstOp, Const::DeclareOp outCstOp) {
    auto inData = IE::getConst(inCstOp);
    auto outData = IE::getConst(outCstOp);

    if (inData.size() != outData.size()) {
        return false;
    }

    auto allOfValuesAreEqual = llvm::all_of(llvm::zip(inData, outData), [](auto pair) {
        return isFloatEqual(std::get<0>(pair), std::get<1>(pair));
    });

    return allOfValuesAreEqual;
}

bool isViewLikeOrFQ(mlir::Operation* op) {
    // Keep FQs at I/O to
    // - simplify tests
    // - preserve the original behavior of the nGraph pass

    if (op == nullptr) {
        // BlockArgument
        return false;
    }

    if (mlir::isa<mlir::func::ReturnOp>(op)) {
        return false;
    }

    // Using IE::isPureViewOp(op) results in some performance regressions
    return mlir::isa<IE::VariadicSplitOp, IE::StridedSliceOp, IE::SplitOp, IE::ReorgYoloOp, IE::TransposeOp,
                     IE::SqueezeOp, IE::ReshapeOp, IE::ConcatOp, IE::TileOp, IE::UnsqueezeOp, IE::ScatterNDUpdateOp>(
                   op) ||
           mlir::isa<IE::FakeQuantizeOp>(op);
}

bool isValidConcatOp(mlir::Operation* op) {
    auto concatOp = mlir::dyn_cast<IE::ConcatOp>(op);
    if (!concatOp) {
        return false;
    }

    SmallVector<IE::FakeQuantizeOp> siblingFqOps;
    for (auto input : concatOp.getInputs()) {
        auto fqOp = input.getDefiningOp<IE::FakeQuantizeOp>();
        if (!fqOp || !fqOp->hasOneUse()) {
            return false;
        }

        auto prevOp = fqOp.getInput().getDefiningOp();
        if (!isViewLikeOrFQ(prevOp)) {
            return false;
        }

        siblingFqOps.push_back(fqOp);
    }

    auto refFq = siblingFqOps.front();
    auto refFqConsts = std::make_tuple(refFq.getInputLow().getDefiningOp<Const::DeclareOp>(),
                                       refFq.getInputHigh().getDefiningOp<Const::DeclareOp>(),
                                       refFq.getOutputLow().getDefiningOp<Const::DeclareOp>(),
                                       refFq.getOutputHigh().getDefiningOp<Const::DeclareOp>());

    auto areTheSameFqOps = [&](IE::FakeQuantizeOp fqOp) {
        auto fqConsts = std::make_tuple(fqOp.getInputLow().getDefiningOp<Const::DeclareOp>(),
                                        fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>(),
                                        fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>(),
                                        fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>());

        return fqConsts == refFqConsts || (areValuesEqual(std::get<0>(fqConsts), std::get<0>(refFqConsts)) &&
                                           areValuesEqual(std::get<1>(fqConsts), std::get<1>(refFqConsts)) &&
                                           areValuesEqual(std::get<2>(fqConsts), std::get<2>(refFqConsts)) &&
                                           areValuesEqual(std::get<3>(fqConsts), std::get<3>(refFqConsts)));
    };

    return llvm::all_of(siblingFqOps, areTheSameFqOps);
}

mlir::LogicalResult PropagateFQDown::matchAndRewrite(IE::FakeQuantizeOp fqOp, mlir::PatternRewriter& rewriter) const {
    if (!IE::hasStaticLowAndHighValues(fqOp)) {
        return mlir::failure();
    }

    if (!IE::isPerTensorFQ({fqOp})) {
        return mlir::failure();
    }

    bool wasPropagated = false;
    int counter = 0;
    auto originLoc = fqOp.getLoc();
    while (true) {
        // The pattern is simplified intentionally
        // In the case of multiple users, only propagate FQ through FQAgnostic ops
        if (!fqOp.getOutput().hasOneUse()) {
            break;
        }

        auto consumerOp = *fqOp.getOutput().getUsers().begin();
        bool checkConcatOp = isValidConcatOp(consumerOp);
        if (!isFQAgnosticOp(consumerOp) && !checkConcatOp) {
            break;
        }

        auto allUsersAreFqOrReturn = llvm::all_of(consumerOp->getResult(0).getUsers(), [](mlir::Operation* op) {
            return mlir::isa_and_nonnull<IE::FakeQuantizeOp, mlir::func::ReturnOp>(op);
        });

        if (allUsersAreFqOrReturn) {
            break;
        }

        const auto topUser = getFirstUser(consumerOp->getResult(0));
        VPUX_THROW_WHEN(topUser == nullptr, "Unused operation: {0}", consumerOp);
        auto newFqOp = cloneFQ(fqOp, topUser, consumerOp->getResult(0), originLoc,
                               formatv("_propagated_down_{0}", counter).str(), rewriter, _propagatedFQ);

        if (checkConcatOp) {
            for (auto input : consumerOp->getOperands()) {
                auto siblingFqOp = input.getDefiningOp<IE::FakeQuantizeOp>();
                if (siblingFqOp != nullptr) {
                    _propagatedFQ->insert(siblingFqOp);
                }
            }
        }

        // replace operands for all users below
        rewriter.replaceUsesWithIf(consumerOp->getResult(0), newFqOp.getOutput(), [&](mlir::OpOperand& opOperand) {
            return opOperand.getOwner() != newFqOp;
        });

        counter++;

        wasPropagated = true;
        fqOp = newFqOp;
    }

    return mlir::success(wasPropagated);
}

bool isIdentityFQ(IE::FakeQuantizeOp fqOp) {
    auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();

    if (inLowConst == outLowConst && inHighConst == outHighConst) {
        return true;
    }

    auto isLowEqual = areValuesEqual(inLowConst, outLowConst);
    auto isHighEqual = areValuesEqual(inHighConst, outHighConst);

    return isLowEqual && isHighEqual;
}

//
// CleanupFQRewriter
//

class CleanupFQRewriter final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    CleanupFQRewriter(mlir::MLIRContext* ctx, Logger log, std::shared_ptr<std::set<IE::FakeQuantizeOp>> propagatedFQ)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log), _propagatedFQ(std::move(propagatedFQ)) {
        this->setDebugName("CleanupFQ");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    std::shared_ptr<std::set<IE::FakeQuantizeOp>> _propagatedFQ;
};

mlir::LogicalResult CleanupFQRewriter::matchAndRewrite(IE::FakeQuantizeOp fqOp, mlir::PatternRewriter& rewriter) const {
    if (!IE::hasStaticLowAndHighValues(fqOp)) {
        return mlir::failure();
    }

    // Skip for non-propagated FQ
    if (std::find(_propagatedFQ->begin(), _propagatedFQ->end(), fqOp) == _propagatedFQ->end()) {
        return mlir::failure();
    }

    auto levels = fqOp.getLevels();
    // Maximum number of levels that exceeds I8/U8 storage type
    if (!levels.has_value() || *levels > QuantizationLevels::QUANT_LEVELS_8BIT) {
        return mlir::failure();
    }

    if (!isViewLikeOrFQ(fqOp.getInput().getDefiningOp())) {
        return mlir::failure();
    }

    for (auto user : fqOp.getOutput().getUsers()) {
        if (!isViewLikeOrFQ(user)) {
            return mlir::failure();
        }
    }

    if (!isIdentityFQ(fqOp)) {
        return mlir::failure();
    }

    rewriter.replaceOp(fqOp, fqOp.getInput());

    return mlir::success();
}

//
// PropagateAndCleanUpFQ
//

class PropagateAndCleanUpFQ final : public IE::impl::PropagateAndCleanUpFQBase<PropagateAndCleanUpFQ> {
public:
    explicit PropagateAndCleanUpFQ(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PropagateAndCleanUpFQ::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto propagatedFQ = std::make_shared<std::set<IE::FakeQuantizeOp>>();

    mlir::RewritePatternSet propagateFQpatterns(&ctx);
    propagateFQpatterns.add<PropagateFQUp>(&ctx, _log, propagatedFQ);
    propagateFQpatterns.add<PropagateFQDown>(&ctx, _log, propagatedFQ);

    auto config = getDefaultGreedyRewriteConfig();
    if (mlir::failed(applyPatternsGreedily(func, std::move(propagateFQpatterns), config))) {
        signalPassFailure();
    }

    mlir::RewritePatternSet cleanUpFQpatterns(&ctx);
    cleanUpFQpatterns.add<CleanupFQRewriter>(&ctx, _log, propagatedFQ);

    config = getDefaultGreedyRewriteConfig();
    if (mlir::failed(applyPatternsGreedily(func, std::move(cleanUpFQpatterns), config))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateAndCleanUpFQPass(Logger log) {
    return std::make_unique<PropagateAndCleanUpFQ>(log);
}
