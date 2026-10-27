//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_dequantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

#include <cmath>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTDYNAMICDEQUANTIZETOFAKEQUANTIZE
#define GEN_PASS_DEF_CONVERTDYNAMICDEQUANTIZETOFAKEQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertDynamicDequantizeToFakeQuantize
//
// Bridge pass. Reconstructs the exact `IE.FakeQuantize` that `WeightsDequantizeToFakeQuantize` would have
// produced from an `IE.DynamicDequantize` emitted by `WeightsDequantizeToDynamicDequantize` for a static-weights
// chain. This shields the downstream (FakeQuantize-centric) pipeline while the weights-import rewriter switches to
// DynamicDequantize.
//
class ConvertDynamicDequantizeToFakeQuantize final : public mlir::OpRewritePattern<IE::DynamicDequantizeOp> {
public:
    ConvertDynamicDequantizeToFakeQuantize(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DynamicDequantizeOp>(ctx), _log(log) {
        setDebugName("ConvertDynamicDequantizeToFakeQuantize");
    }

    mlir::LogicalResult matchAndRewrite(IE::DynamicDequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertDynamicDequantizeToFakeQuantize::matchAndRewrite(IE::DynamicDequantizeOp origOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    // Only convert DynamicDequantize ops that the weights import created (and marked).
    if (!origOp->hasAttr(IE::WEIGHTS_IMPORT_DYN_DEQUANT_ATTR)) {
        return matchFailed(_log, rewriter, origOp, "not a weights-import DynamicDequantize (missing marker)");
    }

    // The weights are either a compile-time constant (static weights-as-constant) or a block argument
    // (weights-as-input, e.g. groupwise-quantized MatMul). Both must be reconstructed to FakeQuantize;
    // the only difference is how the float FakeQuantize input is built (see below).
    auto weightsConst = origOp.getInput().getDefiningOp<Const::DeclareOp>();

    const auto inElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getElementType();
    // The input is either quantized (e.g. quant.uniform / quant.uniform_per_axis / ...)
    // or a raw integer / low-fp / quantile type.
    const auto quantType = mlir::dyn_cast<mlir::quant::QuantizedType>(inElemType);
    const auto storageType = quantType != nullptr ? quantType.getStorageType() : inElemType;

    // Scale is a compile-time constant (static weights case).
    auto scaleConst = origOp.getScale().getDefiningOp<Const::DeclareOp>();
    if (scaleConst == nullptr) {
        return matchFailed(_log, rewriter, origOp,
                           "weights-import DynamicDequantize scale is not a compile-time constant");
    }

    // Optional integer ZP (asymmetric weights).
    Const::ContentAttr zpShiftAttr = nullptr;
    SmallVector<int32_t> preciseZeroPoints;
    if (origOp.getZp() != nullptr) {
        auto zpConst = origOp.getZp().getDefiningOp<Const::DeclareOp>();
        if (zpConst == nullptr) {
            return matchFailed(_log, rewriter, origOp, "weights-import DynamicDequantize has non-const ZP");
        }
        zpShiftAttr =
                zpConst.getContentAttr().transform().castElemType(mlir::Float32Type::get(rewriter.getContext())).get();

        const auto zpElemType = mlir::cast<mlir::ShapedType>(zpConst.getType()).getElementType();
        if (!mlir::isa<mlir::IntegerType>(zpElemType)) {
            return matchFailed(_log, rewriter, origOp, "DynamicDequantize ZP element type '{0}' is not an integer",
                               zpElemType);
        }

        // Capture the ZP values through the f32 view used to build the output limits below, then round to the
        // nearest integer. If this wasn't done then reading it as an integer would mean the value would be truncated.
        // There was a pattern observed where the ZP was 0.99999994 and reading it as an integer would yield 0, but the
        // correct value is 1. SplitFakeQuant would then attach a quantized type whose zero-point is off by one for
        // every weight.
        const auto zpFolded = zpShiftAttr.fold();
        const auto zpValues = zpFolded.getValues<float>();
        preciseZeroPoints.resize(zpValues.size());
        std::transform(zpValues.begin(), zpValues.end(), preciseZeroPoints.begin(), [](float zp) {
            return checked_cast<int32_t>(std::llround(zp));
        });
    }

    const auto ctx = rewriter.getContext();
    const auto loc = origOp.getLoc();
    const auto dstElemType = origOp.getDstElemType();
    const auto scaleElemType = mlir::cast<vpux::NDTypeInterface>(scaleConst.getType()).getElementType();
    // Keep bridge output precision aligned with the original weights-dequant path.
    // If DynDeq carries fp32 dst but the dequant scale is already fp16, rebuild
    // the equivalent FakeQuantize in fp16 to avoid propagating fp32 through
    // downstream conv/eltwise chains.
    const auto fqElemType = (mlir::isa<mlir::Float32Type>(dstElemType) && mlir::isa<mlir::Float16Type>(scaleElemType))
                                    ? scaleElemType
                                    : dstElemType;

    // Recover the FakeQuantize input interval + levels/low_fp_type from the storage type, mirroring
    // WeightsDequantizeToFakeQuantize::commonMatchAndRewrite.

    float inLow = 0.0f;
    float inHigh = 0.0f;
    mlir::IntegerAttr levelsAttr = nullptr;
    mlir::TypeAttr lowFpTypeAttr = nullptr;

    const auto intervalResult =
            mlir::TypeSwitch<mlir::Type, mlir::LogicalResult>(storageType)
                    .Case<mlir::quant::QuantileType>([&](mlir::quant::QuantileType quantileType) {
                        const auto quantileTable = quantileType.getQuantiles();
                        inLow = static_cast<float>(quantileTable.front());
                        inHigh = static_cast<float>(quantileTable.back());
                        lowFpTypeAttr = mlir::TypeAttr::get(quantileType);
                        return mlir::success();
                    })
                    .Case<mlir::IntegerType>([&](mlir::IntegerType intType) {
                        constexpr float halfDivisor = 2.0f;
                        const auto levels = IE::getQuantizationLevels(intType);
                        levelsAttr = getIntAttr(ctx, levels);
                        const bool isSigned = quantType != nullptr ? quantType.isSigned() : intType.isSignedInteger();
                        inLow = isSigned ? -(static_cast<float>(levels) / halfDivisor) : 0.0f;
                        inHigh = static_cast<float>(levels) + inLow - 1.0f;
                        return mlir::success();
                    })
                    .Default([&](mlir::Type t) -> mlir::LogicalResult {
                        if (!isLowFpType(t)) {
                            return matchFailed(_log, rewriter, origOp,
                                               "Unsupported weights storage type for FakeQuantize reconstruction");
                        }
                        const auto storageParams = getStorageParams(t);
                        if (mlir::failed(storageParams)) {
                            return matchFailed(_log, rewriter, origOp,
                                               "Failed to get storage params for the weights storage type");
                        }
                        std::tie(inLow, inHigh, std::ignore) = *storageParams;
                        lowFpTypeAttr = mlir::TypeAttr::get(t);
                        return mlir::success();
                    });
    if (mlir::failed(intervalResult)) {
        return mlir::failure();
    }

    // Rebuild the float FakeQuantize input.
    // - Constant weights: append a float CastElemType to the existing transform chain, preserving the
    //   provenance chain (and materializing a QuantileType/NF4 codebook to its float values).
    // - Block-argument weights (weights-as-input): the raw integer/quantile argument cannot carry a
    //   transform chain, so insert an IE.Convert to the FakeQuantize expressed type instead.
    const auto floatWeightsType =
            mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).changeElemType(fqElemType);
    mlir::Value floatWeights;
    if (weightsConst != nullptr) {
        const auto contentAttr = weightsConst.getContentAttr();
        const auto floatWeightsContent = contentAttr.transform().castElemType(fqElemType).get();
        floatWeights =
                Const::DeclareOp::create(rewriter, appendLoc(loc, "fq_weights"),
                                         mlir::cast<mlir::RankedTensorType>(floatWeightsType), floatWeightsContent)
                        .getOutput();
    } else {
        floatWeights = IE::ConvertOp::create(rewriter, appendLoc(loc, "fq_weights"), origOp.getInput(), fqElemType)
                               .getOutput();
    }

    // input_low / input_high: splat consts of the weights float type, shape [1, ..., 1].
    // Use f32 base with CastElemType to match the representation produced by
    // WeightsDequantizeToFakeQuantize, preserving f32 precision for downstream passes.
    const auto inRank = mlir::cast<vpux::NDTypeInterface>(floatWeightsType).getRank();
    const auto scalarType = mlir::RankedTensorType::get(SmallVector<int64_t>(inRank, 1), fqElemType);
    const auto f32ScalarType =
            mlir::RankedTensorType::get(SmallVector<int64_t>(inRank, 1), mlir::Float32Type::get(ctx));
    auto makeF32BasedConst = [&](mlir::Location constLoc, float val) -> mlir::Value {
        if (mlir::isa<mlir::Float32Type>(fqElemType)) {
            return Const::createFloatConst(rewriter, constLoc, scalarType, ArrayRef<float>({val}));
        }
        const auto base = Const::createConstContent(f32ScalarType, ArrayRef<float>({val}));
        const auto attr = Const::ContentAttr::get(base, {Const::CastElemTypeAttr::get(fqElemType)});
        return Const::DeclareOp::create(rewriter, constLoc, scalarType, attr).getOutput();
    };
    auto inLowConst = makeF32BasedConst(appendLoc(loc, "fq_in_low"), inLow);
    auto inHighConst = makeF32BasedConst(appendLoc(loc, "fq_in_high"), inHigh);

    // output_low / output_high = (in - zp) * scale.
    // This pass runs late in the pipeline, so the scale constant has usually already been sliced by an earlier
    // pass (ResolveStridedSlice, UnrollSDPAPattern, ...) and carries a transform chain such as
    // `dense_resource<...> : tensor<2304x1xf32>, [#const.SubView<[2240, 0], [64, 1]>]`. Folding that content to
    // build the bounds would collapse the base to the sliced size and drop the chain, which downstream passes
    // cannot recover. Instead, scale the base data in f32 and re-attach the original transform chain, keeping
    // the same base resource and SubView the FakeQuantize-based import path ends up with.
    const auto scaleAttr = scaleConst.getContentAttr();
    const auto outStorageType = mlir::cast<mlir::RankedTensorType>(scaleConst.getType());

    mlir::Value outLowConst, outHighConst;
    bool builtViaRescale = false;

    auto buildScaledBounds = [&](float lowFactor, float highFactor) {
        const auto scaleTransforms = scaleAttr.getTransformations();
        const auto scaleBase = scaleAttr.getBaseContent();
        const auto scaleBaseType = mlir::cast<mlir::RankedTensorType>(scaleBase.getType());

        // Preserve only view-like transform chains. Arithmetic/element-type transforms
        // must be materialized via revertScaleShift to keep values exact.
        for (auto tr : scaleTransforms) {
            if (!mlir::isa<Const::SubViewAttr, Const::ReshapeAttr, Const::AffineReshapeAttr, Const::TransposeAttr,
                           Const::ReorderAttr, Const::MemPermuteAttr>(tr)) {
                return;
            }
        }

        if (!mlir::isa<mlir::Float32Type>(scaleBaseType.getElementType())) {
            return;
        }

        SmallVector<float> baseVals;
        if (scaleBase.isSplat()) {
            baseVals.push_back(scaleBase.getSplatValue<float>());
        } else {
            baseVals = to_small_vector(scaleBase.getValues<float>());
        }
        SmallVector<float> lowVals(baseVals.size()), highVals(baseVals.size());
        for (size_t i = 0; i < baseVals.size(); ++i) {
            lowVals[i] = baseVals[i] * lowFactor;
            highVals[i] = baseVals[i] * highFactor;
        }

        const auto lowBase = Const::createConstContent(scaleBaseType, ArrayRef<float>(lowVals));
        const auto highBase = Const::createConstContent(scaleBaseType, ArrayRef<float>(highVals));
        const auto outLowAttr = Const::ContentAttr::get(lowBase, scaleTransforms);
        const auto outHighAttr = Const::ContentAttr::get(highBase, scaleTransforms);
        outLowConst = Const::DeclareOp::create(rewriter, appendLoc(loc, "fq_out_low"), outStorageType, outLowAttr);
        outHighConst = Const::DeclareOp::create(rewriter, appendLoc(loc, "fq_out_high"), outStorageType, outHighAttr);
        builtViaRescale = true;
    };

    const auto scaleFolded = scaleAttr.fold();

    if (zpShiftAttr == nullptr && scaleFolded.isSplat() && scaleFolded.getSplatValue<float>() == 1.0f) {
        // Exact unit scale: output bounds are identical to input bounds, reuse constants directly.
        outLowConst = inLowConst;
        outHighConst = inHighConst;
        builtViaRescale = true;
    } else if (zpShiftAttr == nullptr) {
        // Symmetric case: outBound = scale * inBound.
        // Use the chain-preserving path when the scale transform list is safe for re-attachment,
        // otherwise fall through to revertScaleShift which always materializes correctly.
        buildScaledBounds(inLow, inHigh);
    } else {
        const auto zpContent = zpShiftAttr.fold();
        if (zpContent.isSplat()) {
            // Splat ZP + commuting scale chain: outBound = scale * (inBound - zp)
            const float zpSplat = zpContent.getSplatValue<float>();
            buildScaledBounds(inLow - zpSplat, inHigh - zpSplat);
        } else {
            // Per-channel/asymmetric ZP requires exact elementwise (in - zp) * scale;
            // force fallback to revertScaleShift below.
        }
    }

    if (!builtViaRescale) {
        // Fallback for per-channel ZP or a chain that must not be re-attached: materialize all values
        // (provenance chain is lost, but values are correct).
        const auto reverted = IE::revertScaleShift(ctx, scaleAttr, zpShiftAttr, inLow, inHigh, scalarType, _log);
        if (mlir::failed(reverted)) {
            return matchFailed(_log, rewriter, origOp,
                               "Failed to revert scale-shift for the FakeQuantize output interval");
        }
        const auto& [outLow, outHigh] = reverted.value();
        // Note: shape could've changed due to scale / zp broadcasting.
        const auto materializedStorageType = mlir::cast<mlir::RankedTensorType>(outLow.getType());
        const auto outLowValues = to_small_vector(outLow.getValues<float>());
        const auto outHighValues = to_small_vector(outHigh.getValues<float>());
        // Store as f32 base with CastElemType to preserve precision for downstream passes.
        const auto fallbackElemType = materializedStorageType.getElementType();
        const bool fallbackNeedsF32 = !mlir::isa<mlir::Float32Type>(fallbackElemType);
        if (fallbackNeedsF32) {
            const auto f32StorageType =
                    mlir::RankedTensorType::get(materializedStorageType.getShape(), mlir::Float32Type::get(ctx));
            const auto castTransform = Const::CastElemTypeAttr::get(fallbackElemType);
            const auto lowBase = Const::createConstContent(f32StorageType, ArrayRef<float>(outLowValues));
            const auto highBase = Const::createConstContent(f32StorageType, ArrayRef<float>(outHighValues));
            outLowConst = Const::DeclareOp::create(rewriter, appendLoc(loc, "fq_out_low"), materializedStorageType,
                                                   Const::ContentAttr::get(lowBase, {castTransform}));
            outHighConst = Const::DeclareOp::create(rewriter, appendLoc(loc, "fq_out_high"), materializedStorageType,
                                                    Const::ContentAttr::get(highBase, {castTransform}));
        } else {
            outLowConst = Const::createFloatConst(rewriter, appendLoc(loc, "fq_out_low"), materializedStorageType,
                                                  ArrayRef(outLowValues));
            outHighConst = Const::createFloatConst(rewriter, appendLoc(loc, "fq_out_high"), materializedStorageType,
                                                   ArrayRef(outHighValues));
        }
    }

    const auto broadCastAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);
    auto fqOp =
            rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(origOp, floatWeights, inLowConst, inHighConst, outLowConst,
                                                            outHighConst, levelsAttr, lowFpTypeAttr, broadCastAttr);

    // Capture the original fp32 scales/ZPs from the DynamicDequantize operands as discardable attributes on
    // the FQ op. ConvertPrecisionToFP later rounds the output-limit constants to f16; SplitFakeQuant reads
    // these attributes to avoid recomputing scales from those rounded limits.
    // Content::getValues<float>() dispatches on the storage element type (f16/f32/f64) and converts on the fly.
    const auto preciseScaleFolded = scaleConst.getContentAttr().fold();
    SmallVector<float> preciseScales(to_small_vector(preciseScaleFolded.getValues<float>()));
    if (preciseZeroPoints.empty()) {
        preciseZeroPoints.assign(preciseScales.size(), int32_t(0));
    }

    // DynamicDequantize broadcasts scale against ZP, so one operand may be per-tensor while the other is
    // per-channel. SplitFakeQuant needs one ZP per scale, so materialize that broadcast here instead of
    // storing arrays of different lengths.
    if (preciseScales.size() == 1 && preciseZeroPoints.size() > 1) {
        const auto scale = preciseScales.front();
        preciseScales.assign(preciseZeroPoints.size(), scale);
    } else if (preciseZeroPoints.size() == 1 && preciseScales.size() > 1) {
        const auto zeroPoint = preciseZeroPoints.front();
        preciseZeroPoints.assign(preciseScales.size(), zeroPoint);
    }

    // A mismatch that is not a scalar broadcast spans more than one axis and cannot be expressed as the flat
    // per-channel array the attributes hold. Leave them unset - SplitFakeQuant then recomputes the parameters
    // from the output limits, which is less precise but always valid.
    if (preciseScales.size() != preciseZeroPoints.size()) {
        _log.trace("Not caching precise quant params at '{0}': {1} scales cannot be paired with {2} zero-points", loc,
                   preciseScales.size(), preciseZeroPoints.size());
    } else {
        IE::setPreciseScalesZeroPoints(fqOp, preciseScales, preciseZeroPoints);
    }

    _log.trace("Converted DynamicDequantize to FakeQuantize at '{0}'", loc);
    return mlir::success();
}

//
// ConvertDynamicDequantizeToFakeQuantizePass
//

class ConvertDynamicDequantizeToFakeQuantizePass final :
        public IE::impl::ConvertDynamicDequantizeToFakeQuantizeBase<ConvertDynamicDequantizeToFakeQuantizePass> {
public:
    explicit ConvertDynamicDequantizeToFakeQuantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertDynamicDequantizeToFakeQuantizePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertDynamicDequantizeToFakeQuantize>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertDynamicDequantizeToFakeQuantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertDynamicDequantizeToFakeQuantizePass(Logger log) {
    return std::make_unique<ConvertDynamicDequantizeToFakeQuantizePass>(log);
}
