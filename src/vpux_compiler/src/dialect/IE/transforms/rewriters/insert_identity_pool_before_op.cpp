//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_INSERTIDENTITYPOOLBEFOREOP
#define GEN_PASS_DEF_INSERTIDENTITYPOOLBEFOREOP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool isEligiblePostOp(mlir::Operation* op, Logger log) {
    auto postOpInterface = op->getOperand(0).getDefiningOp<IE::LayerWithPostOpInterface>();
    if (postOpInterface == nullptr || postOpInterface.hasPPE() || !postOpInterface->getResult(0).hasOneUse()) {
        return true;
    }

    const auto inElemType =
            mlir::cast<vpux::NDTypeInterface>(postOpInterface->getOperand(0).getType()).getElementType();
    const auto outElemType =
            mlir::cast<vpux::NDTypeInterface>(postOpInterface->getResult(0).getType()).getElementType();
    // Because of the convert to float, the prelu shift will be bypassed. Check PPE diagram
    if (mlir::isa<mlir::quant::QuantizedType>(inElemType) && !mlir::isa<mlir::quant::QuantizedType>(outElemType) &&
        mlir::isa<IE::PReluOp, IE::LeakyReluOp>(op)) {
        log.trace("A PRelu or LeakyRely at {0} has mixed precision producer, and because of this the prelu shift will "
                  "be skiped",
                  op->getLoc());
        return true;
    }

    // Insert AvgPool between MaxPool and Clamp since MaxPool fused with Clamp is not fully supported
    // by firmware. Tracking Number: E#-145636
    auto parentMaxPoolOp = op->getOperand(0).getDefiningOp<IE::MaxPoolOp>();
    if (parentMaxPoolOp != nullptr && mlir::isa<IE::ClampOp>(op)) {
        log.trace("A MaxPool Op followed by a Clamp Op at {0} will not trigger a fusion, insert an AvgPool Op then",
                  op->getLoc());
        return true;
    }

    log.trace("A PostOp at {0} has already got a suitable producer", op->getLoc());
    return false;
}

//
// InsertIdPoolRewriter
//

template <typename ConcreteOp>
class InsertIdPoolRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    InsertIdPoolRewriter(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<ConcreteOp>(ctx, benefit), _log(log) {
        this->setDebugName("InsertIdPoolRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(ConcreteOp concreteOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult InsertIdPoolRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp concreteOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    if (!isEligiblePostOp(concreteOp.getOperation(), _log)) {
        return mlir::failure();
    }

    auto input = concreteOp->getOperand(0);
    const auto inElemType = mlir::cast<vpux::NDTypeInterface>(input.getType()).getElementType();
    if (!IE::isAvgPoolSupportedElementType(inElemType)) {
        _log.trace("Skip inserting identity AvgPool at {0}: unsupported input element type '{1}'", concreteOp->getLoc(),
                   inElemType);
        return mlir::failure();
    }

    auto* identityOp =
            IE::createIdentityAvgPool(input, input.getType(), rewriter, takeOpLoc(concreteOp, "identity_avgpool"));
    // The identity AvgPool is inserted so it can later be fused into a DPU (NCE) AvgPool.
    // If the created AvgPool is not supported by NCEAveragePoolOp, it would just be a useless op.
    auto avgPoolOp = mlir::dyn_cast_if_present<IE::AvgPoolOp>(identityOp);
    if (avgPoolOp == nullptr) {
        return mlir::failure();
    }
    if (!VPU::NCEAveragePoolOp::isSupported(avgPoolOp, vpux::emptyLogCb, /*checkLayout=*/false,
                                            /*checkChannelAlignment=*/false)) {
        _log.trace("Skip inserting identity AvgPool at {0}: not supported by NCE AvgPool", concreteOp->getLoc());
        rewriter.eraseOp(identityOp);
        return mlir::failure();
    }

    mlir::IRMapping mapper;
    const SmallVector<mlir::Value> inputsToMap = {identityOp->getResult(0)};
    mapper.map(concreteOp->getOperands(), ArrayRef(inputsToMap));
    auto* newLayerOp = rewriter.clone(*concreteOp, mapper);

    newLayerOp->setLoc(takeOpLoc(concreteOp, "with_identity_avgpool"));
    rewriter.replaceOp(concreteOp, newLayerOp->getResult(0));

    return mlir::success();
}

//
// InsertIdentityPoolBeforeOpPass
//

class InsertIdentityPoolBeforeOpPass final :
        public IE::impl::InsertIdentityPoolBeforeOpBase<InsertIdentityPoolBeforeOpPass> {
public:
    explicit InsertIdentityPoolBeforeOpPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void InsertIdentityPoolBeforeOpPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    // LeakyReLU and Clamp can bypass pooling checks.
    // The channels of resulting poolings will be aligned in the future passes.
    // This not the case for MemPermute.
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<InsertIdPoolRewriter<IE::LeakyReluOp>>(&ctx, _log);
    patterns.add<InsertIdPoolRewriter<IE::ClampOp>>(&ctx, _log);
    patterns.add<InsertIdPoolRewriter<IE::ReLUOp>>(&ctx, _log);

    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createInsertIdentityPoolBeforeOpPass(Logger log) {
    return std::make_unique<InsertIdentityPoolBeforeOpPass>(log);
}

void vpux::IE::registerInsertIdentityPoolBeforeOpRewriters(RewriterRegistry& registry,
                                                           ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index,
                                                           Logger log) {
    registry.registerRewriterSet("insert-identity-pool-before-op-set", [&registry, benefitLevels, index, log]() {
        registry.registerRewriter<InsertIdPoolRewriter<IE::LeakyReluOp>>("insert-id-pool-rewriter-leakyrelu", log,
                                                                         benefitLevels[index]);
        registry.registerRewriter<InsertIdPoolRewriter<IE::ClampOp>>("insert-id-pool-rewriter-clamp", log,
                                                                     benefitLevels[index]);
        registry.registerRewriter<InsertIdPoolRewriter<IE::ReLUOp>>("insert-id-pool-rewriter-relu", log,
                                                                    benefitLevels[index]);
    });
}
