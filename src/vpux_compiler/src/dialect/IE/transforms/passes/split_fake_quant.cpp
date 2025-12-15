//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_SPLITFAKEQUANT
#define GEN_PASS_DEF_SPLITFAKEQUANT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool checkRange(const Const::ContentAttr& lowConst, const Const::ContentAttr& highConst,
                IE::AutoBroadcastType broadcast, bool (*predicate)(const double low, const double high)) {
    const auto lowAttr = lowConst.fold();
    const auto highAttr = highConst.fold();
    if (lowAttr.isSplat() && highAttr.isSplat()) {
        const auto low = lowAttr.getSplatValue<double>();
        const auto high = highAttr.getSplatValue<double>();
        return predicate(low, high);
    }

    const auto lowVals = lowAttr.getValues<double>();
    const auto highVals = highAttr.getValues<double>();

    SmallVector<double> lows(lowVals);
    SmallVector<double> highs(highVals);
    broadcastRange(lows, highs, broadcast);

    for (auto p : zip(lows, highs)) {
        const auto lowVal = std::get<0>(p);
        const auto highVal = std::get<1>(p);
        if (!predicate(lowVal, highVal)) {
            return false;
        }
    }

    return true;
}

bool containsValueZero(const Const::ContentAttr& lowConst, const Const::ContentAttr& highConst,
                       IE::AutoBroadcastType broadcast) {
    // Also rMin > rMax ranges are valid; they are used to signal the presence of negative scales.
    // Currently there is no other information in FakeQuantize operation, to deduce back the negative scale
    // based on just the range information.
    auto containsZero = [](const double low, const double high) {
        return (low <= 0 && high >= 0) || (low >= 0 && high <= 0);
    };

    return checkRange(lowConst, highConst, broadcast, containsZero);
}

// Ranges without value zero lead to a negative zero-point which is not supported in the DPU PPE
bool hasRangeWithoutZero(IE::FakeQuantizeOp fqOp) {
    auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();

    if (!containsValueZero(inLowConst.getContentAttr(), inHighConst.getContentAttr(), fqOp.getAutoBroadcast()) ||
        !containsValueZero(outLowConst.getContentAttr(), outHighConst.getContentAttr(), fqOp.getAutoBroadcast())) {
        return true;
    }

    return false;
}

// Scalar like [7, 7] is handled separately and zero value is not required.
// In this case ZP=0, scale=scalar.
bool isScalar(IE::FakeQuantizeOp fqOp) {
    auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();

    auto isScalarLambda = [](const double low, const double high) {
        return std::fabs(high - low) < std::numeric_limits<double>::epsilon();
    };

    return checkRange(inLowConst.getContentAttr(), inHighConst.getContentAttr(), fqOp.getAutoBroadcast(),
                      isScalarLambda);
}

//
// UseQuantDequant
//

class UseQuantDequant final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    UseQuantDequant(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("UseQuantDequant");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UseQuantDequant::matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got FakeQuantize Operation '{1}'", getDebugName(), origOp->getLoc());
    auto innerLog = _log.nest();

    if (origOp.getInput().getDefiningOp<Const::DeclareOp>() != nullptr) {
        return matchFailed(innerLog, rewriter, origOp, "Got constant input");
    }

    auto inLowConst = origOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = origOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = origOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = origOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return matchFailed(innerLog, rewriter, origOp, "Got non constant parameters");
    }

    innerLog.trace("Try to use Quantize/[QuantizeCast]/Dequantize operations");

    const auto realType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto realElemType = mlir::cast<mlir::FloatType>(realType.getElementType());

    auto lowFpType = origOp.getLowFpType();
    VPUX_THROW_WHEN(
            (lowFpType.has_value() && !mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(lowFpType.value())),
            "[{0}] Unsupported low precision type"),
            getDebugName();
    auto isSigned = lowFpType.has_value() && mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(lowFpType.value());

    if (!lowFpType.has_value()) {
        mlir::Value value = origOp.getInput();
        auto op = value.getDefiningOp();
        // Go upwards through the traversing chain comprising
        // BlockArgument->ViewLikeOp*N->ConvertOp->ViewLikeOp*N->FakeQuantOp
        while (op != nullptr && mlir::isa<IE::ViewLikeOpInterface, IE::TransposeOp>(op)) {
            value = *op->getOperands().begin();
            op = value.getDefiningOp();
        }
        if (op != nullptr && mlir::isa<IE::ConvertOp>(op)) {
            value = *op->getOperands().begin();
            if (!mlir::isa<mlir::BlockArgument>(value)) {
                op = value.getDefiningOp();
                while (op != nullptr && mlir::isa<IE::ViewLikeOpInterface>(op)) {
                    value = *op->getOperands().begin();
                    op = value.getDefiningOp();
                }
            }
            if (mlir::isa<mlir::BlockArgument>(value)) {
                auto valueElemType = mlir::cast<vpux::NDTypeInterface>(value.getType()).getElementType();
                if (auto intElemType = mlir::dyn_cast_or_null<mlir::IntegerType>(valueElemType)) {
                    isSigned = mlir::cast<mlir::IntegerType>(valueElemType).isSigned();
                }
            }
        }
    }

    if (IE::keepIntTypeForSIWeightsAsInput(origOp)) {
        isSigned = true;
    }

    const auto inQuantizeElemType = getQuantizedType(
            inLowConst.getContentAttr(), inHighConst.getContentAttr(), origOp.getLevels(), origOp.getLowFpType(),
            realElemType, isSigned, origOp.getLoc(), origOp.getAutoBroadcast(), /*ignoreZPCheck=*/false, innerLog);

    const auto outQuantizeElemType = getQuantizedType(
            outLowConst.getContentAttr(), outHighConst.getContentAttr(), origOp.getLevels(), origOp.getLowFpType(),
            realElemType, isSigned, origOp.getLoc(), origOp.getAutoBroadcast(), /*ignoreZPCheck=*/false, innerLog);

    innerLog.trace("Insert Quantize op '{0}' -> '{1}'", realElemType, inQuantizeElemType);
    auto quantizeOp =
            rewriter.create<IE::QuantizeOp>(takeOpLoc(origOp, "quant_in"), origOp.getInput(), inQuantizeElemType);

    auto result = quantizeOp.getResult();
    if (inQuantizeElemType != outQuantizeElemType) {
        innerLog.trace("Insert QuantizeCast op '{0}' -> '{1}'", inQuantizeElemType, outQuantizeElemType);
        auto quantizeCastOp = rewriter.create<IE::QuantizeCastOp>(origOp.getLoc(), result, outQuantizeElemType);
        result = quantizeCastOp.getResult();
    }

    innerLog.trace("Insert Dequantize op '{0}' -> '{1}'", outQuantizeElemType, realElemType);
    rewriter.replaceOpWithNewOp<IE::DequantizeOp>(origOp, result, realElemType);

    return mlir::success();
}

//
// UseConstDequant
//

class UseConstDequant final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    UseConstDequant(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("UseConstDequant");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::FailureOr<float> getCommonRatio(const Const::Content& content1, const Const::Content& content2) {
    const auto vals1 = content1.getValues<float>();
    const auto vals2 = content2.getValues<float>();

    if (vals1.size() != vals2.size()) {
        return mlir::failure();
    }

    if (std::equal(vals1.begin(), vals1.end(), vals2.begin(), isFloatEqual)) {
        return 1.0f;
    }

    SmallVector<float> ratios;
    ratios.reserve(vals1.size());

    std::transform(vals1.begin(), vals1.end(), vals2.begin(), std::back_inserter(ratios), std::divides<>{});

    // check that all ratios are equal
    if (std::adjacent_find(ratios.begin(), ratios.end(), [](float a, float b) {
            return !isFloatEqual(a, b);
        }) == ratios.end()) {
        return ratios[0];
    } else {
        // Input and output limits has per channel ratio
        return mlir::failure();
    }
}

mlir::LogicalResult UseConstDequant::matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got FakeQuantize Operation '{1}'", getDebugName(), origOp->getLoc());
    auto innerLog = _log.nest();

    auto inConst = origOp.getInput().getDefiningOp<Const::DeclareOp>();
    if (inConst == nullptr) {
        return matchFailed(innerLog, rewriter, origOp, "Got non constant input");
    }

    auto inLowConst = origOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = origOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = origOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = origOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return matchFailed(innerLog, rewriter, origOp, "Got non constant parameters");
    }

    auto inConstAttr = inConst.getContentAttr();
    const auto& inBaseVals = inConstAttr.getBaseContent();
    const auto inBaseElemType = inBaseVals.getShapedType().getElementType();

    const auto inLowContent = inLowConst.getContent();
    const auto inHighContent = inHighConst.getContent();

    // TODO: make this check more reliable
    if (!mlir::isa<mlir::IntegerType>(inBaseElemType)) {
        if (!inLowContent.isSplat() || !inHighContent.isSplat()) {
            innerLog.warning("Legacy model, original input values are not integer");

            // Workaround for Float weights, it lacks generality but is ok for old networks
            // Check if FQ can be removed for float weights
            const auto outLowContent = outLowConst.getContent();
            const auto outHighContent = outHighConst.getContent();

            const auto ratioLow = getCommonRatio(inLowContent, outLowContent);
            const auto ratioHigh = getCommonRatio(inHighContent, outHighContent);

            if (mlir::failed(ratioLow) || mlir::failed(ratioHigh)) {
                return matchFailed(innerLog, rewriter, origOp,
                                   "In and out limits differ and has per channel ratio, do not support");
            } else if (!isFloatEqual(ratioLow.value(), ratioHigh.value())) {
                return matchFailed(innerLog, rewriter, origOp, "Unsupported case, ratioHigh={0} != ratioLow={1}",
                                   ratioHigh, ratioLow);
            }

            if (ratioHigh.value() == 1.0f) {
                // FQ input and output ranges are equal, only remove FQ
                rewriter.replaceOpWithNewOp<Const::DeclareOp>(origOp, origOp.getType(), inConst.getContentAttr())
                        ->setLoc(inConst->getLoc());
            } else {
                // FQ input and output ranges are NOT equal, rescale weights
                innerLog.trace("Rescale weights");
                auto newConstAttr = inConst.transformContentAttr().rescale(ratioHigh.value()).get();
                rewriter.replaceOpWithNewOp<Const::DeclareOp>(origOp, origOp.getType(), std::move(newConstAttr))
                        ->setLoc(inConst->getLoc());
            }

            return mlir::success();
        }
    }

    innerLog.trace("Try to use constant dequantize");

    const auto realType = mlir::cast<vpux::NDTypeInterface>(inConstAttr.getType());
    const auto realElemType = mlir::cast<mlir::FloatType>(realType.getElementType());

    const auto multiZeroPoint = !IE::hasFQSameZeroPoint(origOp);

    {
        // This function patches malfunctioning FQ operation (where in_low and
        // in_high do not align with the FQ levels) that may appear in IR due to
        // some other compiler pass producing garbage (likely candidate is
        // --convert-subtract-to-add). Yet, the compiler has to
        // deal with this somehow and the best-effort currently is to manually
        // patch the weights of such FQ to align them to the input range of the
        // FQ block. After this, one can split FQ into QDQ.

        // Check if FakeQuantize input range is equal to a low precision storage type and if not apply FakeQuantize
        // mathematical formula to the FakeQuantize input Const to quantize the data to the targeted low precision type
        // TODO: E#122705 Add Quantize transformation in Const dialect
        auto levels = origOp.getLevels();
        if (levels.has_value() && inLowContent.isSplat() && inHighContent.isSplat()) {
            const auto isSigned = Const::hasNegativeValues(inLowContent);
            if (!isLowPrecisionTypeRange(getContext(), ArrayRef(inLowContent.getSplatValue<float>()),
                                         ArrayRef(inHighContent.getSplatValue<float>()), *levels, isSigned)) {
                // #E128147: add subbyte type support
                if (levels > 16) {
                    const auto quantizedElemType =
                            getQuantizedType(inLowConst.getContentAttr(), inHighConst.getContentAttr(), levels,
                                             origOp.getLowFpType(), realElemType, isSigned, origOp.getLoc(),
                                             origOp.getAutoBroadcast(), multiZeroPoint, innerLog);
                    inConstAttr = inConst.transformContentAttr().quantize(quantizedElemType).get();
                } else {
                    double intQLow = 0.;
                    double intQHigh = 0.;
                    std::tie(intQLow, intQHigh, std::ignore) = getStorageParams(getContext(), levels.value(), isSigned);
                    auto qLow = checked_cast<float>(intQLow);
                    auto qHigh = checked_cast<float>(intQHigh);

                    auto inLow = inLowContent.getSplatValue<float>();
                    auto inHigh = inHighContent.getSplatValue<float>();
                    const auto inConstContent = inConst.getContent();
                    const auto inVals = inConstContent.getValues<float>();
                    SmallVector<float> quantizedVals(inVals.size());
                    for (size_t i = 0; i < inVals.size(); ++i) {
                        quantizedVals[i] = fakeQuantize(inVals[i], inLow, inHigh, qLow, qHigh, levels.value());
                    }

                    // Generate the Const::ContentAttr with the adjusted constant content
                    const auto inConstStorageType = mlir::dyn_cast<mlir::RankedTensorType>(inConstContent.getType());
                    const auto quantizedConstElementVal =
                            Const::createConstContent(inConstStorageType, ArrayRef(quantizedVals));
                    inConstAttr = Const::ContentAttr::get(quantizedConstElementVal);
                }
            }
        }
    }

    const auto qElemType =
            getQuantizedType(outLowConst.getContentAttr(), outHighConst.getContentAttr(), origOp.getLevels(),
                             origOp.getLowFpType(), realElemType, Const::hasNegativeValues(inLowContent),
                             origOp.getLoc(), origOp.getAutoBroadcast(), multiZeroPoint, innerLog);
    if (qElemType == nullptr) {
        return mlir::failure();
    }

    innerLog.trace("Use quantized element type '{0}'", qElemType);

    const auto qType = realType.changeElemType(qElemType);

    auto newInConstAttrSetup = inConstAttr.transform();
    newInConstAttrSetup = newInConstAttrSetup.castElemType(qElemType);
    const auto moduleOp = getModuleOp(origOp.getOperation());
    const auto isAsymmetricPerChannelZeroPointSupported = config::asymmetricPerChannelZeroPointSupported(moduleOp);
    // Fuse dequantize to const directly since it could not convert to HW for multi Zero Point case
    if (!isAsymmetricPerChannelZeroPointSupported && multiZeroPoint) {
        newInConstAttrSetup = newInConstAttrSetup.dequantize();
        rewriter.replaceOpWithNewOp<Const::DeclareOp>(origOp, origOp.getType(), newInConstAttrSetup.get())
                ->setLoc(inConst->getLoc());
        return mlir::success();
    }

    auto newInOp = rewriter.create<Const::DeclareOp>(inConst->getLoc(), qType, newInConstAttrSetup.get());
    rewriter.replaceOpWithNewOp<IE::DequantizeOp>(origOp, newInOp.getOutput(), realElemType);
    return mlir::success();
}

//
// SplitFakeQuantPass
//

class SplitFakeQuantPass final : public IE::impl::SplitFakeQuantBase<SplitFakeQuantPass> {
public:
    explicit SplitFakeQuantPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SplitFakeQuantPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);

    // per-channel quantization with different zero points is not supported on HW (E#65130)
    // if this is the case, the FQ op will be marked legal and will later be executed as SW op
    target.addDynamicallyLegalOp<IE::FakeQuantizeOp>([](IE::FakeQuantizeOp fqOp) {
        if (!IE::hasStaticLowAndHighValues(fqOp)) {
            return true;
        }

        auto maybeConstOp = fqOp.getInput().getDefiningOp<Const::DeclareOp>();
        bool isConstInput = (maybeConstOp != nullptr);

        // #E-122320 support fuse range withoutZero to const.
        if (hasRangeWithoutZero(fqOp) && !isScalar(fqOp)) {
            return true;
        }

        auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
        auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
        auto outLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
        auto outHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
        const auto realType = mlir::cast<vpux::NDTypeInterface>(fqOp.getInput().getType());
        const auto realElemType = mlir::cast<mlir::FloatType>(realType.getElementType());

        // Although HW could not support multi Zero Point, but if the input is const, we can fuse the fq to const
        // So here ignore multi Zero Point check for const input
        const auto inQuantizeElemType = getQuantizedType(
                inLowConst.getContentAttr(), inHighConst.getContentAttr(), fqOp.getLevels(), fqOp.getLowFpType(),
                realElemType, false, fqOp.getLoc(), fqOp.getAutoBroadcast(), /*ignoreZPCheck=*/isConstInput);

        if (inQuantizeElemType == nullptr) {
            return true;
        }

        const auto outQuantizeElemType = getQuantizedType(
                outLowConst.getContentAttr(), outHighConst.getContentAttr(), fqOp.getLevels(), fqOp.getLowFpType(),
                realElemType, false, fqOp.getLoc(), fqOp.getAutoBroadcast(), /*ignoreZPCheck=*/isConstInput);

        return outQuantizeElemType == nullptr;
    });

    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::QuantizeOp>();
    target.addLegalOp<IE::QuantizeCastOp>();
    target.addLegalOp<IE::DequantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UseQuantDequant>(&ctx, _log);
    patterns.add<UseConstDequant>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitFakeQuantPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSplitFakeQuantPass(Logger log) {
    return std::make_unique<SplitFakeQuantPass>(log);
}
