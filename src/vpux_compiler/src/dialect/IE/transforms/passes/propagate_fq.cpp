//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEFQ
#define GEN_PASS_DEF_PROPAGATEFQ
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
                           mlir::Location originLoc, StringRef suffix, mlir::PatternRewriter& rewriter) {
    mlir::IRMapping mapper;
    mapper.map(currFqOp.getInput(), input);

    // TODO: #-155244
    // We set the insertion point to the user operation to mimic the exact behaviour of the original ngraph pass.
    // This is to circumvent a problem in FeasibleAllocationPass that would generate a different schedule otherwise.
    rewriter.setInsertionPoint(topUser);

    auto newFqOp = mlir::cast<IE::FakeQuantizeOp>(rewriter.clone(*currFqOp, mapper));
    vpux::inferReturnTypes(newFqOp, vpux::InferShapedTypeMode::ALL);
    newFqOp->setLoc(appendLoc(originLoc, suffix));

    return newFqOp;
}

//
// PropagateFQUp
//

class PropagateFQUp final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    PropagateFQUp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        this->setDebugName("PropagateFQUp");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp fqOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PropagateFQUp::matchAndRewrite(IE::FakeQuantizeOp fqOp, mlir::PatternRewriter& rewriter) const {
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
                               formatv("_propagated_up_{0}", counter).str(), rewriter);

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
    PropagateFQDown(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        this->setDebugName("PropagateFQDown");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp fqOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PropagateFQDown::matchAndRewrite(IE::FakeQuantizeOp fqOp, mlir::PatternRewriter& rewriter) const {
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
        if (!isFQAgnosticOp(consumerOp)) {
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
                               formatv("_propagated_down_{0}", counter).str(), rewriter);

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

//
// PropagateFQ
//

class PropagateFQ final : public IE::impl::PropagateFQBase<PropagateFQ> {
public:
    explicit PropagateFQ(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PropagateFQ::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PropagateFQUp>(&ctx, _log);
    patterns.add<PropagateFQDown>(&ctx, _log);

    auto config = getDefaultGreedyRewriteConfig();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateFQPass(Logger log) {
    return std::make_unique<PropagateFQ>(log);
}
