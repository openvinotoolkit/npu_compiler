//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/IR/PatternMatch.h>

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
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    if (!IE::isIdentityPooling(origOp)) {
        _log.nest().trace("Op not identity");
        return mlir::failure();
    }

    auto inputType = origOp.getInput().getType();
    auto outputType = origOp.getOutput().getType();
    if (inputType != outputType) {
        _log.nest().trace("Mismatched input/output type '{0}' with '{1}'", inputType, outputType);
        return mlir::failure();
    }

    _log.nest().trace("Replacing '{0}' with '{1}'", origOp->getName(), origOp.getInput());
    rewriter.replaceOp(origOp, origOp.getInput());
    return mlir::success();
}

bool isIdentityAvgPoolWithPPE(IE::AvgPoolOp avgPoolOp) {
    const auto postOp = avgPoolOp.getPostOpAttr();
    const auto clamp = avgPoolOp.getClampAttr();

    if (clamp == nullptr && postOp == nullptr) {
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
//     Conv (w/clamp || w/o clamp) w/o postOp
//               |                                                       |
//     Identity AvgPool (w/clamp || w/o clamp) w/ postOp  ->  Conv (w/ clamp || w/o clamp) w/ postOp
//               |                                                       |
//
//  PPE tasks apply post-op first then clamp. If the following subgraph is given:
//  ```
//  Conv {clamp = {max = 1.0, min = -1.0} }
//   |
//  Identity AvgPool {clamp = {max = 0.5, min = -0.5}, post_op = #IE.ReLU<>}
//  ```
//
//  It can be decomposed as:
//  ```
//  Conv
//   |
//  Clamp1
//   |
//  Identity AvgPool
//   |
//  ReLU
//   |
//  Clamp2
//  ```
//
//  AvgPool can be removed:
//  ```
//  Conv
//   |
//  Clamp1
//   |
//  ReLU
//   |
//  Clamp2
//  ```
//
//  The correct order to guarantee correctness while remaining performant:
//  1. Fuse ReLU into Conv
//  2. Execute ReLU on min and max of Clamp1 and fuse into Conv
//  3. Intersect Clamp2 with Clamp1 into Conv

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
    _log.trace("Got '{0}' at '{1}'", avgPoolOp->getName(), avgPoolOp->getLoc());

    // Found the identity avgPool with PPE
    if (!isIdentityAvgPoolWithPPE(avgPoolOp)) {
        _log.nest().trace("Op is not identity avgPool with PPE!");
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

    if (producerOp.getPostOp() != nullptr && avgPoolOp.getPostOpAttr() != nullptr) {
        _log.nest().trace("avgPoolOp producer already has post-processing!");
        return mlir::failure();
    }

    auto avgPoolPostOp = avgPoolOp.getPostOpAttr();
    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };
    if (avgPoolPostOp != nullptr && !producerOp.isSupportedPostOpAttr(avgPoolPostOp, logCb)) {
        _log.nest().trace("avgPoolOp producer does not support the post_op in avgPoolOp");
        return mlir::failure();
    }

    auto fusedClamp = IE::getFusedClamp(_log, producerOp.getClampAttr(), avgPoolPostOp, avgPoolOp.getClampAttr(),
                                        producerOp, avgPoolOp.getInput().getType());
    if (!fusedClamp.has_value()) {
        return mlir::failure();
    }

    rewriter.modifyOpInPlace(producerOp, [&] {
        if (avgPoolPostOp != nullptr) {
            producerOp.setPostOpAttr(avgPoolPostOp);
        }

        if (*fusedClamp != nullptr) {
            producerOp.setClampAttr(fusedClamp.value());
        }
    });

    // Replace the AvgPool with its producer, clamp/post_op fused
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

    if (parentPoolOp == nullptr || !isIdentityAvgPoolWithPPE(parentPoolOp)) {
        _log.trace("There is no parent pool with postop");
        return mlir::failure();
    }

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };
    auto layerWithPostOp = mlir::cast<IE::LayerWithPostOpInterface>(origOp.getOperation());
    const auto parentPostOp = parentPoolOp.getPostOpAttr();
    const auto origPostOp = origOp.getPostOpAttr();

    if (parentPostOp != nullptr && origPostOp != nullptr) {
        _log.trace("Both avgPoolOp and its parent pool have a post_op!");
        return mlir::failure();
    }

    if (parentPostOp != nullptr && !layerWithPostOp.isSupportedPostOpAttr(parentPostOp, logCb)) {
        _log.trace("avgPoolOp does not support the post_op of its parent pool!");
        return mlir::failure();
    }

    const auto fusedClamp = IE::getFusedClamp(_log, parentPoolOp.getClampAttr(), origPostOp, origOp.getClampAttr(),
                                              layerWithPostOp, origOp.getOutput().getType());
    if (!fusedClamp.has_value()) {
        return mlir::failure();
    }

    if (fusedClamp.value() != nullptr) {
        origOp.setClampAttr(fusedClamp.value());
    }

    if (parentPostOp != nullptr) {
        origOp.setPostOpAttr(parentPostOp);
    }

    origOp.setOperand(0, parentPoolOp.getInput());
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

    if (parentPoolOp == nullptr || !isIdentityAvgPoolWithPPE(parentPoolOp) || parentPoolOp.getPostOpAttr() == nullptr) {
        _log.trace("There is no parent pool with postop");
        return mlir::failure();
    }

    const auto leakyRelu = mlir::dyn_cast<IE::LeakyReluAttr>(parentPoolOp.getPostOpAttr());
    if (leakyRelu == nullptr) {
        _log.trace("No Prelu post op");
        return mlir::failure();
    }

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    if (origOp.getPostOpAttr() != nullptr) {
        _log.trace("AddOp already has a post_op");
        return mlir::failure();
    }

    auto layerWithPostOp = mlir::cast<IE::LayerWithPostOpInterface>(origOp.getOperation());
    if (!layerWithPostOp.isSupportedPostOpAttr(leakyRelu, logCb)) {
        _log.nest().trace("avgPoolOp producer does not support the post-op in AddOp");
        return mlir::failure();
    }

    const auto fusedClamp = IE::getFusedClamp(_log, parentPoolOp.getClampAttr(), origOp.getPostOpAttr(),
                                              origOp.getClampAttr(), layerWithPostOp, origOp.getOutput().getType());
    if (!fusedClamp.has_value()) {
        return mlir::failure();
    }

    if (fusedClamp.value() != nullptr) {
        origOp.setClampAttr(fusedClamp.value());
    }

    origOp.setPostOpAttr(parentPoolOp.getPostOpAttr());
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

// NOTE: LayerWithPostOpInterface operations can have PPE which means,
//       they could have clamp and post_op separately. When fusing these
//       ops it's important to ensure to account PPE's on both operations
void OptimizeIdentityPoolPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<FuseIdentityQuantizedAvgPool>(&ctx, _log);
        patterns.add<FuseIdentityWithQuantizedAdd>(&ctx, _log);
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }

    {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<FuseIdentityAvgPoolWithPostOp>(&ctx, _log);
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }

    {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<RemoveIdentityPool<IE::MaxPoolOp>>(&ctx, _log);
        patterns.add<RemoveIdentityPool<IE::AvgPoolOp>>(&ctx, _log);
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }
}

}  // namespace

//
// OptimizeIdentityPoolPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeIdentityPoolPass(Logger log) {
    return std::make_unique<OptimizeIdentityPoolPass>(log);
}
