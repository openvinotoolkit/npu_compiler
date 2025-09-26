//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_ADJUSTNCEOPSWITHI32INPUTS
#define GEN_PASS_DEF_ADJUSTNCEOPSWITHI32INPUTS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertPrecisionToFP16
//

template <class ConcreteOp>
class ConvertPrecisionToFP16 final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ConvertPrecisionToFP16(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("ConvertPrecisionToFP16");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult ConvertPrecisionToFP16<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Found '{0}' Operation at '{1}'", origOp->getName(), origOp->getLoc());
    auto outputElemType = mlir::dyn_cast<vpux::NDTypeInterface>(origOp->getResult(0).getType()).getElementType();

    mlir::IRMapping mapper;
    for (auto inputIter : origOp->getOperands() | indexed) {
        auto input = inputIter.value();
        auto index = inputIter.index();
        const auto inputElemType = mlir::dyn_cast<vpux::NDTypeInterface>(input.getType()).getElementType();
        const auto elemTypeFP16 = mlir::Float16Type::get(inputElemType.getContext());
        auto inputLoc = appendLoc(origOp->getLoc(), "_Input_Convert_{0}", index);
        const auto inputCvtToFP16 =
                rewriter.createOrFold<IE::ConvertOp>(inputLoc, input, mlir::TypeAttr::get(elemTypeFP16));
        mapper.map(origOp->getOperand(index), inputCvtToFP16);
    }

    auto newOp = rewriter.clone(*origOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ELEM_TYPE);
    auto outputLoc = appendLoc(origOp->getLoc(), "_Output_Convert_0");
    const auto outputCvtToOrig =
            rewriter.createOrFold<IE::ConvertOp>(outputLoc, newOp->getResult(0), mlir::TypeAttr::get(outputElemType));
    rewriter.replaceOp(origOp, outputCvtToOrig);

    return mlir::success();
}

//
// AdjustNCEOpsWithI32InputsPass
//

class AdjustNCEOpsWithI32InputsPass final :
        public IE::impl::AdjustNCEOpsWithI32InputsBase<AdjustNCEOpsWithI32InputsPass> {
public:
    explicit AdjustNCEOpsWithI32InputsPass(Logger log, bool enableConvertFCToConv)
            : _enableConvertFCToConv(enableConvertFCToConv) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;

    bool _enableConvertFCToConv;
};

mlir::LogicalResult AdjustNCEOpsWithI32InputsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (enableConvertFCToConv.hasValue()) {
        _enableConvertFCToConv = enableConvertFCToConv.getValue();
    }

    return mlir::success();
}

void AdjustNCEOpsWithI32InputsPass::safeRunOnModule() {
    auto& ctx = getContext();

    const auto isLegalOp = [](mlir::Operation* op) {
        auto outputElemType = mlir::dyn_cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getElementType();

        // Currently only encounter si32 type, if any other types found later, extend for other types.
        if (outputElemType.isSignedInteger(32)) {
            return false;
        }
        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::ConvertOp>();
    target.addDynamicallyLegalOp<IE::MatMulOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>(isLegalOp);

    if (_enableConvertFCToConv) {
        target.addDynamicallyLegalOp<IE::FullyConnectedOp>(isLegalOp);
    }

    target.markUnknownOpDynamicallyLegal([](mlir::Operation*) {
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertPrecisionToFP16<IE::MatMulOp>>(&ctx, _log);
    patterns.add<ConvertPrecisionToFP16<IE::ConvolutionOp>>(&ctx, _log);
    patterns.add<ConvertPrecisionToFP16<IE::GroupConvolutionOp>>(&ctx, _log);
    patterns.add<ConvertPrecisionToFP16<IE::FullyConnectedOp>>(&ctx, _log);

    auto module = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAdjustNCEOpsWithI32InputsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustNCEOpsWithI32InputsPass(Logger log, bool enableConvertFCToConv) {
    return std::make_unique<AdjustNCEOpsWithI32InputsPass>(log, enableConvertFCToConv);
}
