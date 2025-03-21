//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTMINMAXTOCLAMP
#define GEN_PASS_DEF_CONVERTMINMAXTOCLAMP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertMinMaxToClampPass
//

class ConvertMinMaxToClampPass final : public IE::impl::ConvertMinMaxToClampBase<ConvertMinMaxToClampPass> {
public:
    explicit ConvertMinMaxToClampPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// MinMaxConverter
//

template <class ConcreteOp>
class MinMaxConverter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MinMaxConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("MinMaxConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult MinMaxConverter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    auto nonScalarOperand = origOp.getInput2();
    auto scalarOperand = origOp.getInput1();
    if (origOp.getInput1().getType().template cast<vpux::NDTypeInterface>().getNumElements() != 1) {
        nonScalarOperand = scalarOperand;
        scalarOperand = origOp.getInput2();
    }
    if (scalarOperand.getType().template cast<vpux::NDTypeInterface>().getNumElements() != 1) {
        return mlir::failure();
    }
    float scalarValue = 0.0f;
    auto scalarInputConst = scalarOperand.template getDefiningOp<Const::DeclareOp>();

    if (scalarInputConst == nullptr) {
        IE::FakeQuantizeOp fqOp = scalarOperand.template getDefiningOp<IE::FakeQuantizeOp>();
        if (fqOp == nullptr) {
            _log.trace("Only constant input is supported for scalar_input'{0}'", origOp->getLoc());
            return mlir::failure();
        }

        auto fqScalarInputConst = fqOp.getInput().template getDefiningOp<Const::DeclareOp>();
        if (fqScalarInputConst == nullptr) {
            _log.trace("Only constant input is supported for scalar_input'{0}'", fqOp->getLoc());
            return mlir::failure();
        }
        if (!fqOp.getLevels().has_value()) {
            _log.nest().trace("FakeQuantize without levels at '{0}'", fqOp->getLoc());
            return mlir::failure();
        }
        auto concreteScalarInputContentAttr = fqScalarInputConst.getContentAttr();
        if (!concreteScalarInputContentAttr.isSplat()) {
            _log.nest().trace("Constant Concrete input must be scalar at '{0}'", fqOp.getLoc());
            return mlir::failure();
        }
        auto inputVal = concreteScalarInputContentAttr.fold().template getSplatValue<float>();
        auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
        auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
        auto outLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
        auto outHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();

        if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
            _log.nest().trace("Got non constant parameters of FakeQuantize at '{0}'", fqOp->getLoc());
            return mlir::failure();
        }

        const auto inLowConstContent = inLowConst.getContent();
        const auto inHighConstContent = inHighConst.getContent();
        const auto outLowConstContent = outLowConst.getContent();
        const auto outHighConstContent = outHighConst.getContent();

        if (!inLowConstContent.isSplat() || !inHighConstContent.isSplat() || !outLowConstContent.isSplat() ||
            !outHighConstContent.isSplat()) {
            _log.nest().trace("Got non scalar fake quantize range '{0}'", fqOp->getLoc());
            return mlir::failure();
        }

        const auto inLowConstContentVal = inLowConstContent.getSplatValue<float>();
        const auto inHighConstContentVal = inHighConstContent.getSplatValue<float>();
        const auto outLowConstContentVal = outLowConstContent.getSplatValue<float>();
        const auto outHighConstContentVal = outHighConstContent.getSplatValue<float>();

        auto levels = fqOp.getLevels();
        float fLevels = checked_cast<float>(levels.value());

        scalarValue = vpux::fakeQuantize(inputVal, inLowConstContentVal, inHighConstContentVal, outLowConstContentVal,
                                         outHighConstContentVal, fLevels);
    } else {
        if (const auto& attr = scalarInputConst.getContentAttr(); !attr.isSplat()) {
            _log.trace("Only splat input is supported for 'scalar_input' '{0}'", origOp->getLoc());
            return mlir::failure();
        }
        auto scalarInputContent = scalarInputConst.getContent();
        scalarValue = scalarInputContent.template getSplatValue<float>();
    }

    mlir::FloatAttr clampMax;
    mlir::FloatAttr clampMin;
    if (mlir::isa<IE::MaximumOp>(origOp)) {
        clampMax =
                getFPAttr(origOp->getContext(), checked_cast<double>(std::numeric_limits<vpux::type::float16>::max()));
        clampMin = getFPAttr(origOp->getContext(), scalarValue);
    } else if (mlir::isa<IE::MinimumOp>(origOp)) {
        clampMax = getFPAttr(origOp->getContext(), scalarValue);
        clampMin = getFPAttr(origOp->getContext(),
                             checked_cast<double>(std::numeric_limits<vpux::type::float16>::lowest()));
    }

    auto newClampOp = rewriter.create<IE::ClampOp>(origOp->getLoc(), nonScalarOperand, clampMin, clampMax);
    rewriter.replaceOp(origOp, newClampOp.getOutput());

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertMinMaxToClampPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MinMaxConverter<IE::MinimumOp>>(&ctx, _log);
    patterns.add<MinMaxConverter<IE::MaximumOp>>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        _log.debug("Failed to replace Min or Max operation with Clamp");
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertMinMaxToClampPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertMinMaxToClampPass(Logger log) {
    return std::make_unique<ConvertMinMaxToClampPass>(log);
}
