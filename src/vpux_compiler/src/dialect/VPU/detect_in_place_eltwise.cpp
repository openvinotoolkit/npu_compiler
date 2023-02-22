//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// DetectInPlaceEltwise
//

class DetectInPlaceEltwise final : public mlir::OpRewritePattern<VPU::NCEEltwiseOp> {
public:
    DetectInPlaceEltwise(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::NCEEltwiseOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEEltwiseOp eltwiseOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DetectInPlaceEltwise::matchAndRewrite(VPU::NCEEltwiseOp eltwiseOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Check Eltwise op {0} for inplace execution", eltwiseOp->getLoc());

    if (eltwiseOp.is_inplace().getValueOr(false)) {
        return mlir::failure();
    }

    auto output = eltwiseOp.output();
    auto eltwiseAllInputs = eltwiseOp.getInputs();

    // #65421
    if (eltwiseOp.fitIntoCMX(eltwiseAllInputs[0].getType().cast<NDTypeInterface>(),
                             eltwiseAllInputs[1].getType().cast<NDTypeInterface>(),
                             output.getType().cast<NDTypeInterface>())) {
        return mlir::failure();
    }

    for (auto input : eltwiseAllInputs) {
        _log.nest().trace("Checking input {0}", input.getType());
        if (!input.hasOneUse()) {
            // This input is used by another operation, try next input
            continue;
        }

        auto nestLog = _log.nest(2);
        // Check that input is not block argument
        if (input.isa<mlir::BlockArgument>()) {
            nestLog.trace("Input is a block argument - not supported");
            continue;
        }

        // Check that input tensor is compatible with output
        auto inInterface = input.getType().cast<NDTypeInterface>();
        auto outInterface = output.getType().cast<NDTypeInterface>();
        if (!isCompatibleForInplaceOp(inInterface, outInterface, nestLog)) {
            continue;
        }

        auto nceOpInterface = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(eltwiseOp.getOperation());
        if (nceOpInterface == nullptr) {
            return mlir::failure();
        }

        eltwiseOp.is_inplaceAttr(mlir::BoolAttr::get(rewriter.getContext(), true));
        _log.trace("EltwiseOp attribute set to inplace {0}", eltwiseOp->getLoc());
        return mlir::success();
    }

    return mlir::failure();
}

//
// DetectInPlaceEltwisePass
//

class DetectInPlaceEltwisePass final : public VPU::DetectInPlaceEltwiseBase<DetectInPlaceEltwisePass> {
public:
    explicit DetectInPlaceEltwisePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DetectInPlaceEltwisePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto function = getFunction();

    // #65420
    auto arch = VPU::getArch(function);
    if ((arch != VPU::ArchKind::VPUX37XX)) {
        return;
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DetectInPlaceEltwise>(&ctx, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(function, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDetectInPlaceEltwisePass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createDetectInPlaceEltwisePass(Logger log) {
    return std::make_unique<DetectInPlaceEltwisePass>(log);
}
