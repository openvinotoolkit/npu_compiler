//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/insert_identity_pool_before_op.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
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

bool vpux::IE::isEligiblePostOp(mlir::Operation* op, Logger log) {
    auto postOpInterface = op->getOperand(0).getDefiningOp<IE::LayerWithPostOpInterface>();
    if (postOpInterface == nullptr || postOpInterface.getPostOp() != nullptr ||
        !postOpInterface->getResult(0).hasOneUse()) {
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

mlir::LogicalResult vpux::IE::genericIdInserter(mlir::Operation* concreteOp, const InsertIdFunctor& insertId,
                                                mlir::PatternRewriter& rewriter, Logger log) {
    mlir::Operation* identityOp = insertId(concreteOp, rewriter, log);
    if (identityOp == nullptr) {
        return mlir::failure();
    }

    mlir::IRMapping mapper;
    const SmallVector<mlir::Value> inputsToMap = {identityOp->getResult(0)};
    mapper.map(concreteOp->getOperands(), ArrayRef(inputsToMap));
    auto* newLayerOp = rewriter.clone(*concreteOp, mapper);
    rewriter.replaceOp(concreteOp, newLayerOp->getResult(0));

    return mlir::success();
}

using namespace vpux;

namespace {

mlir::Operation* insertAvgPool(mlir::Operation* concreteOp, mlir::PatternRewriter& rewriter, Logger log) {
    if (!IE::isEligiblePostOp(concreteOp, log)) {
        return nullptr;
    }

    auto input = concreteOp->getOperand(0);
    return IE::createIdentityAvgPool(input, input.getType(), rewriter, concreteOp->getLoc());
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
    patterns.add<IE::InsertIdPoolRewriter<IE::LeakyReluOp>>(&ctx, insertAvgPool, _log);
    patterns.add<IE::InsertIdPoolRewriter<IE::ClampOp>>(&ctx, insertAvgPool, _log);
    patterns.add<IE::InsertIdPoolRewriter<IE::ReLUOp>>(&ctx, insertAvgPool, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createInsertIdentityPoolBeforeOpPass(Logger log) {
    return std::make_unique<InsertIdentityPoolBeforeOpPass>(log);
}
