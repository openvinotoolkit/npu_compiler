//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <deque>

namespace vpux::IE {
#define GEN_PASS_DECL_SPLITCONVWITHMULTIPLEFQ
#define GEN_PASS_DEF_SPLITCONVWITHMULTIPLEFQ
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool checkConvolution(mlir::Operation* convOp, Logger log) {
    log.trace("Check Convolution operation:");
    if (!mlir::isa_and_nonnull<IE::ConvolutionOp>(convOp)) {
        return false;
    }

    auto convolution = mlir::cast<IE::ConvolutionOp>(convOp);
    log.nest().trace("Got IE::Convolution Operation at '{0}'", convolution->getLoc());
    auto inputOp = convolution.getInput().getDefiningOp();
    auto filterOp = convolution.getFilter().getDefiningOp();
    auto dequantizeInput = mlir::isa_and_nonnull<IE::FakeQuantizeOp, IE::DequantizeOp>(inputOp);
    auto dequantizeFilter = mlir::isa_and_nonnull<IE::FakeQuantizeOp, IE::DequantizeOp>(filterOp);

    return dequantizeInput && dequantizeFilter;
}

bool checkBias(mlir::Operation* biasOp, Logger log) {
    log.trace("Check Bias operation:");
    if (!mlir::isa_and_nonnull<IE::ScaleShiftOp, IE::AddOp>(biasOp)) {
        return false;
    }

    if (auto scaleOp = mlir::dyn_cast<IE::ScaleShiftOp>(biasOp)) {
        log.nest().trace("Got IE::ScaleShift Operation at '{0}'", scaleOp->getLoc());
        if (scaleOp.getWeights() != nullptr) {
            return false;
        }

        auto convOp = scaleOp.getInput().getDefiningOp();
        if (!mlir::isa_and_nonnull<IE::ConvolutionOp>(convOp) || convOp->getNumOperands() != 2) {
            return false;
        }

        const auto convOutShape = getShape(convOp->getOpResult(0));
        const auto biasShape = getShape(scaleOp.getBiases());
        if (biasShape.size() != 4 || biasShape[Dims4D::Act::N] != 1 ||
            biasShape[Dims4D::Act::C] != convOutShape[Dims4D::Act::C] || biasShape[Dims4D::Act::H] != 1 ||
            biasShape[Dims4D::Act::W] != 1) {
            return false;
        }
        const auto logCb = [&](const formatv_object_base& msg) {
            log.trace("{0}", msg.str());
        };
        auto mainOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(convOp);
        if (mainOp == nullptr || !mainOp.isSupportedPostOp(scaleOp, logCb)) {
            return false;
        }
    }

    if (auto addOp = mlir::dyn_cast<IE::AddOp>(biasOp)) {
        log.nest().trace("Got IE::Add Operation at '{0}'", addOp->getLoc());
        auto addOutShape = getShape(addOp.getOutput());
        auto biasesShape = getShape(addOp.getInput2());
        if (addOutShape.size() != 4 || biasesShape.size() != 4) {
            return false;
        }
        if (biasesShape[Dims4D::Act::N] != 1 || biasesShape[Dims4D::Act::H] != 1 || biasesShape[Dims4D::Act::W] != 1) {
            return false;
        }
    }

    return true;
}

bool checkPost(mlir::Operation* postOp, Logger log) {
    log.trace("Check post operation:");
    if (postOp == nullptr || postOp->getNumOperands() != 1) {
        return false;
    }

    auto producerOp = postOp->getOperand(0).getDefiningOp();
    if (producerOp == nullptr) {
        return false;
    }
    if (!mlir::isa<IE::ConvolutionOp>(producerOp)) {
        producerOp = producerOp->getOperand(0).getDefiningOp();
        if (!mlir::isa_and_nonnull<IE::ConvolutionOp>(producerOp)) {
            return false;
        }
    }
    log.nest().trace("Got producer IE::Convolution Operation at '{0}'", producerOp->getLoc());

    auto mainOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(producerOp);
    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };
    if (mainOp == nullptr || !mainOp.isSupportedPostOp(postOp, logCb) || mainOp.getPostOp() != nullptr) {
        return false;
    }
    log.nest().trace("Got supported Post Operation at '{0}' ", postOp->getLoc());

    return true;
}

bool checkQuantize(mlir::Operation* fqOp, Logger log) {
    log.trace("Check Quantize operation:");
    return mlir::isa_and_nonnull<IE::FakeQuantizeOp, IE::QuantizeOp>(fqOp);
}

//
// SplitConvWithOnlyFakeQuantConsumers
//

using Branch = std::deque<mlir::Operation*>;

void splitBranch(Branch& branch, int64_t idx, mlir::PatternRewriter& rewriter, Logger log) {
    log.trace("Split branch:");
    mlir::Operation* producerOp = rewriter.clone(*branch.front());
    extendOpLoc(producerOp, llvm::formatv("_branch_{0}", idx));
    log.nest().trace("Got root operation: {0}", producerOp->getLoc());
    branch.pop_front();
    mlir::Operation* lastOp = branch.back();
    branch.pop_back();

    mlir::IRMapping mapper;
    for (auto item : branch | indexed) {
        auto op = item.value();
        mapper.map(op->getOperand(0), producerOp->getResult(0));
        auto newOp = rewriter.clone(*op, mapper);
        extendOpLoc(newOp, llvm::formatv("_branch_{0}_{1}", idx, item.index()));
        log.nest().trace("Next operation in branch: {0}", newOp->getLoc());
        producerOp = newOp;
    }
    lastOp->setOperand(0, producerOp->getResult(0));
    extendOpLoc(lastOp, llvm::formatv("_branch_{0}", idx));
    log.nest().trace("Got last operation: {0}", lastOp->getLoc());
}

class SplitConvWithOnlyFakeQuantConsumers : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    SplitConvWithOnlyFakeQuantConsumers(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("SplitConvWithOnlyFakeQuantConsumers");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

    bool matchPattern(FuncRef<bool(mlir::Operation*, Logger)> check, mlir::SmallVector<Branch>& branches, bool optional,
                      bool closing, Logger log) const;
};

mlir::LogicalResult SplitConvWithOnlyFakeQuantConsumers::matchAndRewrite(IE::ConvolutionOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE::Convolution Operation '{1}'", getDebugName(), origOp->getLoc());
    auto innerLog = _log.nest();
    auto branches = mlir::SmallVector<Branch>{{origOp}};
    if (!matchPattern(checkConvolution, branches, false, false, innerLog)) {
        return mlir::failure();
    }
    if (!matchPattern(checkBias, branches, true, false, innerLog)) {
        return mlir::failure();
    }
    if (!matchPattern(checkPost, branches, true, false, innerLog)) {
        return mlir::failure();
    }
    if (!matchPattern(checkQuantize, branches, false, true, innerLog)) {
        return mlir::failure();
    }

    branches.pop_back();
    if (branches.empty()) {
        return mlir::failure();
    }
    for (auto item : branches | indexed) {
        auto branch = item.value();
        auto idx = item.index();
        splitBranch(branch, idx, rewriter, innerLog);
    }

    return mlir::success();
}

bool SplitConvWithOnlyFakeQuantConsumers::matchPattern(FuncRef<bool(mlir::Operation*, Logger)> check,
                                                       mlir::SmallVector<Branch>& branches, bool optional, bool closing,
                                                       Logger log) const {
    log.trace("Match pattern:");
    auto innerLog = log.nest();
    auto branchesUpdated = mlir::SmallVector<Branch>();
    for (auto branch : branches) {
        auto back = branch.back();
        innerLog.trace("Got branch with last operation: {0}", back->getLoc());
        if (!check(back, innerLog)) {
            if (!optional) {
                return false;
            } else {
                innerLog.trace("Branch unchanged");
                branchesUpdated.push_back(branch);
                continue;
            }
        }

        if (closing) {
            innerLog.trace("Branch unchanged");
            branchesUpdated.push_back(branch);
            continue;
        }

        for (auto consumer : back->getUsers()) {
            innerLog.trace("Got next operation in branch: {0}", consumer->getLoc());
            auto branchNextOp = branch;
            branchNextOp.push_back(consumer);
            branchesUpdated.push_back(branchNextOp);
        }
    }

    branches = std::move(branchesUpdated);
    return true;
}

//
// SplitConvWithPostOpAndFakeQuant
//

class SplitConvWithPostOpAndFakeQuant : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    SplitConvWithPostOpAndFakeQuant(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("SplitConvWithPostOpAndFakeQuant");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitConvWithPostOpAndFakeQuant::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got FakeQuantize Operation: '{1}'", getDebugName(), origOp->getLoc());
    auto innerLog = _log.nest();
    auto postOp = origOp->getOperand(0).getDefiningOp();
    if (!checkPost(postOp, innerLog)) {
        return mlir::failure();
    }
    innerLog.trace("Got Post Operation: '{0}'", postOp->getLoc());
    auto branch = Branch{postOp};

    auto biasOp = postOp->getOperand(0).getDefiningOp();
    auto convOp = biasOp;
    if (checkBias(biasOp, innerLog)) {
        branch.push_front(biasOp);
        convOp = biasOp->getOperand(0).getDefiningOp();
        innerLog.trace("Got Bias Operation: '{0}'", biasOp->getLoc());
    }

    if (!checkConvolution(convOp, innerLog)) {
        return mlir::failure();
    }
    innerLog.trace("Got Convolution Operation: '{0}'", convOp->getLoc());
    branch.push_front(convOp);

    bool hasOneConsumer = true;
    for (auto op : branch) {
        if (!op->getResult(0).hasOneUse()) {
            hasOneConsumer = false;
            break;
        }
    }

    if (hasOneConsumer) {
        return mlir::failure();
    }

    branch.push_back(origOp);
    splitBranch(branch, 0, rewriter, innerLog);

    return mlir::success();
}

//
// SplitConvWithMultipleFQPass
//

class SplitConvWithMultipleFQPass final : public IE::impl::SplitConvWithMultipleFQBase<SplitConvWithMultipleFQPass> {
public:
    explicit SplitConvWithMultipleFQPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SplitConvWithMultipleFQPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SplitConvWithPostOpAndFakeQuant>(&ctx, _log);
    patterns.add<SplitConvWithOnlyFakeQuantConsumers>(&ctx, _log);

    auto func = getOperation();
    auto config = getDefaultGreedyRewriteConfig();
    config.useTopDownTraversal = false;
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitConvWithMultipleFQPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSplitConvWithMultipleFQPass(Logger log) {
    return std::make_unique<SplitConvWithMultipleFQPass>(log);
}
