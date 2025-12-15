//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTGRNTONORMALIZEL2
#define GEN_PASS_DEF_CONVERTGRNTONORMALIZEL2
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertGRNToNormalizeL2
//

class ConvertGRNToNormalizeL2 final : public mlir::OpRewritePattern<IE::GRNOp> {
public:
    ConvertGRNToNormalizeL2(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GRNOp>(ctx), _log(log) {
        setDebugName("ConvertGRNToNormalizeL2");
    }

    mlir::LogicalResult matchAndRewrite(IE::GRNOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};
mlir::LogicalResult ConvertGRNToNormalizeL2::matchAndRewrite(IE::GRNOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto inputShape = to_small_vector(getShape(origOp.getInput()));
    const auto outputShape = to_small_vector(getShape(origOp.getOutput()));
    mlir::MLIRContext* ctx = origOp->getContext();

    VPUX_THROW_UNLESS((inputShape.size() >= 2) && (inputShape.size() <= 4),
                      "GRN input rank {0} need to be >= than 2 and <= than 4", inputShape.size());
    VPUX_THROW_UNLESS(inputShape.size() == outputShape.size(),
                      "GRN input rank {0} need to be equal with output rank {1}", inputShape.size(),
                      outputShape.size());

    auto epsAttr = origOp.getBiasAttr();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    if (inputType.getElementType().isF16() && outputType.getElementType().isF16()) {
        auto minEpsilon = static_cast<double>(std::numeric_limits<type::float16>::smallest_mixed_precision_eps);
        auto epsilon = origOp.getBias().convertToDouble();
        if (epsilon < minEpsilon) {
            epsAttr = getFPAttr(origOp->getContext(), minEpsilon);
        }
    }

    SmallVector<mlir::Attribute> axesAttrVec;
    auto intType = getSInt64Type(ctx);
    SmallVector<mlir::Attribute> axes = {mlir::IntegerAttr::get(intType, 1)};
    const auto axesInput = mlir::ArrayAttr::get(ctx, axes);
    const auto epsModeAttr = IE::EpsModeAttr::get(ctx, IE::EpsMode::ADD);
    rewriter.replaceOpWithNewOp<IE::NormalizeL2Op>(origOp, origOp.getInput(), nullptr, axesInput, epsAttr, epsModeAttr);
    return mlir::success();
}

//
// ConvertGRNToNormalizeL2Pass
//

class ConvertGRNToNormalizeL2Pass final : public IE::impl::ConvertGRNToNormalizeL2Base<ConvertGRNToNormalizeL2Pass> {
public:
    explicit ConvertGRNToNormalizeL2Pass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertGRNToNormalizeL2Pass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::GRNOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::NormalizeL2Op>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertGRNToNormalizeL2>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertGRNToNormalizeL2Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGRNToNormalizeL2Pass(Logger log) {
    return std::make_unique<ConvertGRNToNormalizeL2Pass>(log);
}
