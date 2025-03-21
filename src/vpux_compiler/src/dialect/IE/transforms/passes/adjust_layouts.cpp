//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/utils/adjust_layout_utils.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_ADJUSTLAYOUTS
#define GEN_PASS_DEF_ADJUSTLAYOUTS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

using InputToReordersMap = DenseMap<mlir::Value, std::unordered_map<DimsOrder, IE::ReorderOp>>;

//
// LayerRewriter
//

class LayerRewriter final : public mlir::OpInterfaceRewritePattern<IE::LayoutInfoOpInterface> {
public:
    LayerRewriter(mlir::MLIRContext* ctx, Logger log, const bool seOpsEnabled, const bool seExperimentalOpsEnabled)
            : mlir::OpInterfaceRewritePattern<IE::LayoutInfoOpInterface>(ctx),
              _log(log),
              _seOpsEnabled(seOpsEnabled),
              _seExperimentalOpsEnabled(seExperimentalOpsEnabled) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayoutInfoOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _seOpsEnabled;
    bool _seExperimentalOpsEnabled;
};

mlir::LogicalResult LayerRewriter::matchAndRewrite(IE::LayoutInfoOpInterface origOp,
                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("Rewrite layer operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto orderInfo = origOp.getLayoutInfo();
    origOp.inferLayoutInfo(orderInfo, _seOpsEnabled, _seExperimentalOpsEnabled);

    rewriter.startOpModification(origOp);

    const auto inputs = origOp->getOpOperands();
    for (auto i : irange(inputs.size())) {
        auto& input = inputs[i];

        const auto curOrder = DimsOrder::fromValue(input.get());
        const auto supportedOrder = orderInfo.getInput(i);

        _log.nest(1).trace("Process input #{0}", i);
        if (curOrder != supportedOrder) {
            insertReorderForInput(origOp, input, supportedOrder, rewriter, _log.nest());
        }
    }

    const auto outputs = origOp->getOpResults();
    for (auto i : irange(outputs.size())) {
        auto output = outputs[i];

        const auto curOrder = DimsOrder::fromValue(output);
        const auto supportedOrder = orderInfo.getOutput(i);

        _log.nest(1).trace("Process output #{0}", i);
        if (curOrder != supportedOrder) {
            changeDimsOrder(output, supportedOrder, _log.nest());
            insertReorderForOutput(origOp, output, curOrder, rewriter, _log.nest());
        }
    }

    rewriter.finalizeOpModification(origOp);

    return mlir::success();
}

//
// AdjustLayoutsPass
//

class AdjustLayoutsPass final : public IE::impl::AdjustLayoutsBase<AdjustLayoutsPass> {
public:
    explicit AdjustLayoutsPass(const bool seOpsEnabled, const bool seExperimentalOpsEnabled, Logger log)
            : _seOpsEnabled(seOpsEnabled), _seExperimentalOpsEnabled(seExperimentalOpsEnabled) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _seOpsEnabled;
    bool _seExperimentalOpsEnabled;
};

mlir::LogicalResult AdjustLayoutsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (seOpsEnabled.hasValue()) {
        _seOpsEnabled = seOpsEnabled.getValue();
    }
    if (seExperimentalOpsEnabled.hasValue()) {
        _seExperimentalOpsEnabled = seExperimentalOpsEnabled.getValue();
    }

    return mlir::success();
}

void AdjustLayoutsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (auto iface = mlir::dyn_cast<IE::LayoutInfoOpInterface>(op)) {
            _log.trace("Check layer operation '{0}' at '{1}'", op->getName(), op->getLoc());

            auto orderInfo = iface.getLayoutInfo();
            _log.nest().trace("Current layouts: {0}", orderInfo);

            iface.inferLayoutInfo(orderInfo, _seOpsEnabled, _seExperimentalOpsEnabled);
            _log.nest().trace("Required layouts: {0}", orderInfo);

            return !orderInfo.hasChanges();
        }

        return true;
    });
    target.addLegalOp<IE::ReorderOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<LayerRewriter>(&ctx, _log.nest(), _seOpsEnabled, _seExperimentalOpsEnabled);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAdjustLayoutsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustLayoutsPass(const bool seOpsEnabled,
                                                              const bool seExperimentalOpsEnabled, Logger log) {
    return std::make_unique<AdjustLayoutsPass>(seOpsEnabled, seExperimentalOpsEnabled, log);
}
