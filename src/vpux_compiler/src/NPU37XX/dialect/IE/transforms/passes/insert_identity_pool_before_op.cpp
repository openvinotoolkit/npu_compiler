//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/insert_identity_pool_before_op.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"

#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE::arch37xx {
#define GEN_PASS_DECL_INSERTIDENTITYPOOLBEFOREOP
#define GEN_PASS_DEF_INSERTIDENTITYPOOLBEFOREOP
#include "vpux/compiler/NPU37XX/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE::arch37xx

using namespace vpux;

namespace {

bool isEligiblePermute(mlir::Operation* op, Logger log) {
    const auto permuteInterface = getFusableLayerWithPermuteInterface(op);
    if (permuteInterface != nullptr && permuteInterface->getResult(0).hasOneUse()) {
        log.trace("A permutation at {0} has already got a suitable producer", op->getLoc());
        return false;
    }
    const auto expectedLayout = DimsOrder::NHWC;
    const auto inputLayout = DimsOrder::fromValue(op->getOperand(0));
    if (inputLayout != expectedLayout) {
        log.trace("A permutation at {0} has got an unsupported input layout {1}, expected {2}", op->getLoc(),
                  inputLayout, expectedLayout);
        return false;
    }
    const auto shapeRef = getShape(op->getOperand(0));
    const auto batchSize = shapeRef[Dims4D::Act::N];
    if (batchSize != 1) {
        log.trace("A permutation at {0} has an unsupported batch size {1}", op->getLoc(), batchSize);
        return false;
    }

    // Activation shave version of MemPermute seems to be more efficient when MaxPool cannot be split over height.
    constexpr int64_t MIN_DIM_SIZE_FOR_TILING = 2;
    const auto height = shapeRef[Dims4D::Act::H];
    if (height < MIN_DIM_SIZE_FOR_TILING) {
        log.trace("A permutation at {0} has an unsupported height {1}", op->getLoc(), height);
        return false;
    }

    return true;
}

mlir::Operation* insertAvgPool(mlir::Operation* concreteOp, mlir::PatternRewriter& rewriter, Logger log) {
    if (!IE::isEligiblePostOp(concreteOp, log)) {
        return nullptr;
    }

    auto input = concreteOp->getOperand(0);
    return IE::createIdentityAvgPool(input, input.getType(), rewriter, concreteOp->getLoc());
}

bool checkPooling(mlir::Operation* op, mlir::Operation* maybePermute) {
    auto permuteInterface = mlir::dyn_cast_or_null<IE::LayerWithPermuteInterface>(op);
    if (permuteInterface == nullptr) {
        return true;
    }
    if (!permuteInterface.isSupportedPermutation(maybePermute)) {
        return false;
    }
    const auto outputType = op->getResult(0).getType().template cast<vpux::NDTypeInterface>();
    if (mlir::isa<mlir::IntegerType>(outputType.getElementType())) {
        return false;
    }
    auto alignInterface = mlir::dyn_cast_or_null<IE::AlignedChannelsOpInterface>(op);
    if (alignInterface == nullptr) {
        return true;
    }
    return alignInterface.verifyChannels().succeeded();
}

mlir::Operation* insertMaxPool(mlir::Operation* concreteOp, mlir::PatternRewriter& rewriter, Logger log) {
    if (!isEligiblePermute(concreteOp, log)) {
        return nullptr;
    }
    const auto outputType = concreteOp->getOperand(0).getType();
    auto identityOp = IE::createIdentityMaxPool(concreteOp->getOperand(0), outputType, rewriter);

    // This check is required to avoid processing IE.MemPermute with misaligned channels.
    if (!checkPooling(identityOp, concreteOp)) {
        rewriter.eraseOp(identityOp);
        return nullptr;
    }

    return identityOp;
}

//
// InsertIdentityPoolBeforeOpPass
//

class InsertIdentityPoolBeforeOpPass final :
        public IE::arch37xx::impl::InsertIdentityPoolBeforeOpBase<InsertIdentityPoolBeforeOpPass> {
public:
    explicit InsertIdentityPoolBeforeOpPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
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
    // IE.MaxPool operations are more efficient for IE.MemPermute fusion.
    patterns.add<IE::InsertIdPoolRewriter<IE::MemPermuteOp>>(&ctx, insertMaxPool, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::arch37xx::createInsertIdentityPoolBeforeOpPass(Logger log) {
    return std::make_unique<InsertIdentityPoolBeforeOpPass>(log);
}
