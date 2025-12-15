//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_FUSECLAMPPASS
#define GEN_PASS_DEF_FUSECLAMPPASS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// ClampConverter
//

class ClampConverter final : public mlir::OpRewritePattern<VPU::ClampOp> {
public:
    ClampConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::ClampOp>(ctx), _log(log) {
        this->setDebugName("ClampConverter");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::ClampOp clampOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ClampConverter::matchAndRewrite(VPU::ClampOp clampOp, mlir::PatternRewriter& rewriter) const {
    auto nceOp = mlir::cast<VPU::NCEOpInterface>(clampOp.getInput().getDefiningOp());
    const auto clampMin = clampOp.getMin().convertToDouble();
    const auto clampMax = clampOp.getMax().convertToDouble();
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType()).getElementType();

    // Update PPE attribute clamps
    auto ppeAttr = nceOp.getPPE();
    const auto& adapter = VPU::PpeVersionConfig::getFactoryAs<vpux::VPU::IPpeAdapterClamp>();
    ppeAttr = adapter.intersectClamps(ppeAttr, clampMin, clampMax, outElemType);

    auto newOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(rewriter.clone(*nceOp));
    newOp.setPPE(ppeAttr);
    rewriter.replaceOp(clampOp, newOp->getResult(0));
    if (nceOp.use_empty()) {
        rewriter.eraseOp(nceOp);
    }

    return mlir::success();
}

bool isLegalClamp(VPU::ClampOp clampOp) {
    // Clamp operations without NCE producer are legal.
    auto nceOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(clampOp.getInput().getDefiningOp());
    if (nceOp == nullptr) {
        return true;
    }

    // If the operation is quantized, only per-tensor quantization is supported.
    // If the operation is has float16 or bfloat16 precision, the lower bound of the clamp must be zero.
    // Other data types are not supported.
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType()).getElementType();
    if (mlir::isa<mlir::quant::QuantizedType>(outElemType)) {
        return mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outElemType);
    } else if (outElemType.isF16() || outElemType.isBF16()) {
        const auto clampMin = clampOp.getMin().convertToDouble();
        return !isDoubleEqual(clampMin, 0.0);
    } else {
        return true;
    }
}

//
// FuseClampPass
//

class FuseClampPass final : public VPU::impl::FuseClampPassBase<FuseClampPass> {
public:
    explicit FuseClampPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseClampPass::safeRunOnFunc() {
    // TODO: #70644

    auto func = getOperation();
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    // Adding the entire dialect here since the VPU.Clamp will be replaced with one of VPU.NCE operations.
    target.addLegalDialect<VPU::VPUDialect>();
    target.addDynamicallyLegalOp<VPU::ClampOp>(isLegalClamp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ClampConverter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createFuseClampPass(Logger log) {
    return std::make_unique<FuseClampPass>(log);
}
