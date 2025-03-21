//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <utility>

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSELSTMCELL
#define GEN_PASS_DEF_DECOMPOSELSTMCELL
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// LSTMCellRewriter
//

class LSTMCellRewriter final : public mlir::OpRewritePattern<IE::LSTMCellOp> {
public:
    LSTMCellRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::LSTMCellOp>(ctx), _log(std::move(log)) {
        this->setDebugName("LSTMCellRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::LSTMCellOp lstmCell, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMCellRewriter::matchAndRewrite(IE::LSTMCellOp lstmCell, mlir::PatternRewriter& rewriter) const {
    if (VPU::LSTMCellOp::isSupported(lstmCell)) {
        return mlir::failure();
    }
    _log.trace("Got op {0} at {1}", lstmCell->getName(), lstmCell->getLoc());

    mlir::Value newInput = lstmCell.getInputData();
    if (lstmCell.getWeights()) {
        newInput = rewriter.create<IE::MatMulOp>(takeOpLoc(lstmCell, "in_mul"), lstmCell.getInputData(),
                                                 lstmCell.getWeights(), false, true, nullptr);
    }

    if (lstmCell.getBiases()) {
        newInput =
                rewriter.create<IE::AddOp>(takeOpLoc(lstmCell, "bias"), newInput, lstmCell.getBiases(),
                                           IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NUMPY),
                                           nullptr, nullptr, nullptr, nullptr);
    }

    const mlir::Value matMulHiddenState =
            rewriter.create<IE::MatMulOp>(takeOpLoc(lstmCell, "mul_hid"), lstmCell.getInitialHiddenState(),
                                          lstmCell.getRecurrenceWeights(), false, true, nullptr);

    const mlir::Value lstmGatesInput = rewriter.create<IE::AddOp>(
            takeOpLoc(lstmCell, "gates"), newInput, matMulHiddenState,
            IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT), nullptr, nullptr,
            nullptr, nullptr);

    rewriter.replaceOpWithNewOp<IE::LSTMGatesOp>(lstmCell, lstmGatesInput, lstmCell.getInitialCellState());

    return mlir::success();
}

//
// DecomposeLSTMCellPass
//

class DecomposeLSTMCellPass final : public IE::impl::DecomposeLSTMCellBase<DecomposeLSTMCellPass> {
public:
    explicit DecomposeLSTMCellPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void DecomposeLSTMCellPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<LSTMCellRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeLSTMCellPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeLSTMCellPass(Logger log) {
    return std::make_unique<DecomposeLSTMCellPass>(std::move(log));
}
