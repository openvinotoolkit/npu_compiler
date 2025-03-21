//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include <mlir/IR/PatternMatch.h>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/utils/attributes_properties_conversion.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZEIDENTITYPOOL
#define GEN_PASS_DEF_OPTIMIZEIDENTITYPOOL
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// RemoveIdentityPool
//

template <typename ConcreteOp>
class RemoveIdentityPool final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    RemoveIdentityPool(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult RemoveIdentityPool<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{1}' at '{2}'", origOp->getName(), origOp->getLoc());
    if (!IE::isIdentityPooling(origOp)) {
        _log.nest().trace("Op not identity");
        return mlir::failure();
    }

    auto inputType = origOp.getInput().getType();
    auto outputType = origOp.getOutput().getType();
    if (inputType != outputType) {
        _log.nest().trace("Mismatched input/output type '{1}' with '{2}'", inputType, outputType);
        return mlir::failure();
    }

    _log.nest().trace("Replacing '{1}' with '{2}'", origOp->getName(), origOp.getInput());
    rewriter.replaceOp(origOp, origOp.getInput());
    return mlir::success();
}

bool isIdentityAvgPoolWithPostOp(IE::AvgPoolOp avgPoolOp) {
    if (avgPoolOp.getPostOpAttr() == nullptr) {
        return false;
    }

    auto postOpAttrName = avgPoolOp.getPostOpAttr().getName().getValue();
    if (postOpAttrName != IE::ClampOp::getOperationName() && postOpAttrName != IE::LeakyReluOp::getOperationName() &&
        postOpAttrName != IE::ReLUOp::getOperationName()) {
        return false;
    }

    auto inputType = avgPoolOp.getInput().getType();
    auto outputType = avgPoolOp.getOutput().getType();
    if (inputType != outputType) {
        return false;
    }

    const auto stride = parseIntArrayAttr<int64_t>(avgPoolOp.getStrides());
    const auto kernel = parseIntArrayAttr<int64_t>(avgPoolOp.getKernelSize());
    const auto padStart = parseIntArrayAttr<int64_t>(avgPoolOp.getPadsBegin());
    const auto padEnd = parseIntArrayAttr<int64_t>(avgPoolOp.getPadsEnd());
    const auto ones = SmallVector<int64_t>(kernel.size(), 1);
    const auto zeros = SmallVector<int64_t>(padStart.size(), 0);
    return (stride == ones && kernel == ones && padStart == zeros && padEnd == zeros);
}

//
// FuseIdentityAvgPoolWithPostOp
//

//
//               |
//            Conv w/o postOp
//               |                                 |
//     Identity AvgPool w/ postOp       ->      Conv w/ postOp
//               |                                 |
//

class FuseIdentityAvgPoolWithPostOp final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    FuseIdentityAvgPoolWithPostOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
        setDebugName("FuseIdentityAvgPoolWithPostOp");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp avgPoolOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseIdentityAvgPoolWithPostOp::matchAndRewrite(IE::AvgPoolOp avgPoolOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{1}' at '{2}'", avgPoolOp->getName(), avgPoolOp->getLoc());

    // Found the identity avgPool with postOp
    if (!isIdentityAvgPoolWithPostOp(avgPoolOp)) {
        _log.nest().trace("Op is not identity avgPool with postOp!");
        return mlir::failure();
    }

    if (!avgPoolOp->getOperand(0).hasOneUse()) {
        _log.nest().trace("avgPoolOp is not the only user of its input Value!");
        return mlir::failure();
    }

    // Check the parentOp supported postOp
    auto producerOp = avgPoolOp->getOperand(0).getDefiningOp<IE::LayerWithPostOpInterface>();
    if (producerOp == nullptr) {
        _log.nest().trace("avgPoolOp producer does not support post-processing!");
        return mlir::failure();
    }

    if (producerOp.getPostOp().has_value()) {
        _log.nest().trace("avgPoolOp producer already has post-processing!");
        return mlir::failure();
    }

    auto postOp = avgPoolOp->getResult(0).getDefiningOp();
    auto postOpAttrName = avgPoolOp.getPostOpAttr().getName().getValue();
    if (postOpAttrName == IE::ClampOp::getOperationName()) {
        IE::ClampOp::Adaptor clamp(std::nullopt, nullptr,
                                   toProperties<IE::ClampOp>(avgPoolOp.getPostOpAttr().getAttrs()));
        postOp = rewriter.create<IE::ClampOp>(avgPoolOp->getLoc(), avgPoolOp.getInput(), clamp.getMinAttr(),
                                              clamp.getMaxAttr());
    } else if (postOpAttrName == IE::LeakyReluOp::getOperationName()) {
        IE::LeakyReluOp::Adaptor leakyRelu(std::nullopt, nullptr,
                                           toProperties<IE::LeakyReluOp>(avgPoolOp.getPostOpAttr().getAttrs()));
        postOp = rewriter.create<IE::LeakyReluOp>(avgPoolOp->getLoc(), avgPoolOp.getInput(),
                                                  leakyRelu.getNegativeSlopeAttr());
    } else if (postOpAttrName == IE::ReLUOp::getOperationName()) {
        postOp = rewriter.create<IE::ReLUOp>(avgPoolOp->getLoc(), avgPoolOp.getInput());
    }

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };
    if (!producerOp.isSupportedPostOp(postOp, logCb)) {
        _log.nest().trace("avgPoolOp producer does not support the post-processing in avgPoolOp!");
        rewriter.eraseOp(postOp);
        return mlir::failure();
    }
    rewriter.eraseOp(postOp);

    // Set postOp for producer and then replace the avgPoolOp
    producerOp->setAttr("post_op", avgPoolOp.getPostOpAttr());
    rewriter.replaceOp(avgPoolOp, producerOp->getResult(0));

    return mlir::success();
}

//
// FuseIdentityQuantizedAvgPool
//

class FuseIdentityQuantizedAvgPool final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    FuseIdentityQuantizedAvgPool(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseIdentityQuantizedAvgPool::matchAndRewrite(IE::AvgPoolOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    if (!IE::isQuantizedAvgPoolPermutation(origOp)) {
        _log.trace("no quantized avg pool");
        return mlir::failure();
    }

    if (!origOp.getInput().hasOneUse()) {
        _log.trace("avgPoolOp is not the only user of its input Value!");
        return mlir::failure();
    }

    auto parentPoolOp = origOp.getInput().getDefiningOp<IE::AvgPoolOp>();

    if (parentPoolOp == nullptr || !isIdentityAvgPoolWithPostOp(parentPoolOp)) {
        _log.trace("There is no parent pool with postop");
        return mlir::failure();
    }

    origOp->setAttr("post_op", parentPoolOp.getPostOpAttr());
    origOp.setOperand(parentPoolOp.getInput());
    rewriter.eraseOp(parentPoolOp);

    return mlir::success();
}

//
// FuseIdentityWithQuantizedAdd
//

class FuseIdentityWithQuantizedAdd final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    FuseIdentityWithQuantizedAdd(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseIdentityWithQuantizedAdd::matchAndRewrite(IE::AddOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    if (!IE::isAddOutputQuantized(origOp)) {
        _log.trace("no quantized add");
        return mlir::failure();
    }

    auto hasMoreUses = llvm::any_of(origOp.getInput1().getUsers(), [&](auto userOp) {
        return userOp != origOp;
    });

    if (hasMoreUses || origOp.getInput1() != origOp.getInput2()) {
        _log.trace("ProducerOp has multiple users");
        return mlir::failure();
    }

    auto parentPoolOp = origOp.getInput1().getDefiningOp<IE::AvgPoolOp>();

    if (parentPoolOp == nullptr || !isIdentityAvgPoolWithPostOp(parentPoolOp)) {
        _log.trace("There is no parent pool with postop");
        return mlir::failure();
    }

    mlir::Operation* postOp;
    auto postOpAttrName = parentPoolOp.getPostOpAttr().getName().getValue();
    if (postOpAttrName == IE::LeakyReluOp::getOperationName()) {
        IE::LeakyReluOp::Adaptor leakyRelu(std::nullopt, nullptr,
                                           toProperties<IE::LeakyReluOp>(parentPoolOp.getPostOpAttr().getAttrs()));
        postOp = rewriter.create<IE::LeakyReluOp>(parentPoolOp->getLoc(), parentPoolOp.getInput(),
                                                  leakyRelu.getNegativeSlopeAttr());
    } else {
        _log.trace("No Prelu post op");
        return mlir::failure();
    }

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };
    if (!mlir::dyn_cast<IE::LayerWithPostOpInterface>(origOp.getOperation()).isSupportedPostOp(postOp, logCb)) {
        _log.nest().trace("avgPoolOp producer does not support the post-processing in AddOp!");
        rewriter.eraseOp(postOp);
        return mlir::failure();
    }
    rewriter.eraseOp(postOp);

    origOp->setAttr("post_op", parentPoolOp.getPostOpAttr());
    origOp.setOperand(0, parentPoolOp.getInput());
    origOp.setOperand(1, parentPoolOp.getInput());
    rewriter.eraseOp(parentPoolOp);

    return mlir::success();
}

//
// OptimizeIdentityPoolPass
//

class OptimizeIdentityPoolPass final : public IE::impl::OptimizeIdentityPoolBase<OptimizeIdentityPoolPass> {
public:
    explicit OptimizeIdentityPoolPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeIdentityPoolPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RemoveIdentityPool<IE::MaxPoolOp>>(&ctx, _log);
    patterns.add<RemoveIdentityPool<IE::AvgPoolOp>>(&ctx, _log);
    patterns.add<FuseIdentityAvgPoolWithPostOp>(&ctx, _log);
    patterns.add<FuseIdentityQuantizedAvgPool>(&ctx, _log);
    patterns.add<FuseIdentityWithQuantizedAdd>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// OptimizeIdentityPoolPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeIdentityPoolPass(Logger log) {
    return std::make_unique<OptimizeIdentityPoolPass>(log);
}
