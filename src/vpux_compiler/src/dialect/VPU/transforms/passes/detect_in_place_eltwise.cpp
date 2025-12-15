//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_DETECTINPLACEELTWISE
#define GEN_PASS_DEF_DETECTINPLACEELTWISE
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

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

    if (eltwiseOp.getIsInplace().value_or(false)) {
        return mlir::failure();
    }

    auto output = eltwiseOp.getOutput();
    auto eltwiseAllInputs = eltwiseOp.getInputs();

    // #65421
    if (eltwiseOp.fitIntoCMX(mlir::cast<vpux::NDTypeInterface>(eltwiseAllInputs[0].getType()),
                             mlir::cast<vpux::NDTypeInterface>(eltwiseAllInputs[1].getType()),
                             mlir::cast<vpux::NDTypeInterface>(output.getType()))) {
        return mlir::failure();
    }

    // sprLUT adds additional dummy DPU task, that writes garbage to the output
    // (see InsertDelayDPUVariant pass). In case of in-place operation it will
    // write into the input, corrupting its data.
    if (auto ppeAttr = mlir::dyn_cast_or_null<VPU::PPEFpAttr>(eltwiseOp.getPpeAttr())) {
        if (ppeAttr.getSprlut()) {
            return mlir::failure();
        }
    }

    for (auto input : eltwiseAllInputs) {
        _log.nest().trace("Checking input {0}", input.getType());
        if (!input.hasOneUse()) {
            // This input is used by another operation, try next input
            continue;
        }

        auto nestLog = _log.nest(2);
        // Check that input is not block argument
        if (mlir::isa<mlir::BlockArgument>(input)) {
            nestLog.trace("Input is a block argument - not supported");
            continue;
        }

        // Check that input tensor is compatible with output
        auto inInterface = mlir::cast<vpux::NDTypeInterface>(input.getType());
        auto outInterface = mlir::cast<vpux::NDTypeInterface>(output.getType());
        if (!isCompatibleForInplaceOp(inInterface, outInterface, nestLog)) {
            continue;
        }

        auto nceOpInterface = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(eltwiseOp.getOperation());
        if (nceOpInterface == nullptr) {
            return mlir::failure();
        }

        eltwiseOp.setIsInplaceAttr(mlir::BoolAttr::get(rewriter.getContext(), true));
        _log.trace("EltwiseOp attribute set to inplace {0}", eltwiseOp->getLoc());
        return mlir::success();
    }

    return mlir::failure();
}

//
// DetectInPlaceEltwisePass
//

class DetectInPlaceEltwisePass final : public VPU::impl::DetectInPlaceEltwiseBase<DetectInPlaceEltwisePass> {
public:
    explicit DetectInPlaceEltwisePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DetectInPlaceEltwisePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto function = getOperation();

    // TODO: #65420
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DetectInPlaceEltwise>(&ctx, _log);

    collectOpsAndApplyPatterns(function, std::move(patterns));
}

}  // namespace

//
// createDetectInPlaceEltwisePass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createDetectInPlaceEltwisePass(Logger log) {
    return std::make_unique<DetectInPlaceEltwisePass>(log);
}
