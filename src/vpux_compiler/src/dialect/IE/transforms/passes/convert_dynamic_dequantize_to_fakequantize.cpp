//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_dequantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

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
        return matchFailed(rewriter, origOp, "not a weights-import DynamicDequantize (missing marker)");
    }

    // The weights are either a compile-time constant (static weights-as-constant) or a block argument
    // (weights-as-input, e.g. groupwise-quantized MatMul). Both must be reconstructed to FakeQuantize;
    // the only difference is how the float FakeQuantize input is built (see below).
    auto weightsConst = origOp.getInput().getDefiningOp<Const::DeclareOp>();

    const auto inElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getElementType();
    // The input is either a quant.uniform-wrapped integer (produced by an earlier import rewriter) or a
    // raw integer / low-fp / quantile type.
    const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(inElemType);
    const auto storageType = uniformType != nullptr ? uniformType.getStorageType() : inElemType;

    // Scale is a compile-time constant (static weights case).
    auto scaleConst = origOp.getScale().getDefiningOp<Const::DeclareOp>();
    if (scaleConst == nullptr) {
        return matchFailed(rewriter, origOp, "weights-import DynamicDequantize scale is not a compile-time constant");
    }

    // Optional integer ZP (asymmetric weights).
    Const::ContentAttr zpShiftAttr = nullptr;
    if (origOp.getZp() != nullptr) {
        auto zpConst = origOp.getZp().getDefiningOp<Const::DeclareOp>();
        if (zpConst == nullptr) {
            return matchFailed(rewriter, origOp, "weights-import DynamicDequantize has non-const ZP");
        }
        zpShiftAttr =
                zpConst.getContentAttr().transform().castElemType(mlir::Float32Type::get(rewriter.getContext())).get();
    }

    const auto ctx = rewriter.getContext();
    const auto loc = origOp.getLoc();
    const auto dstElemType = origOp.getDstElemType();

    // Recover the FakeQuantize input interval + levels/low_fp_type from the storage type, mirroring
    // WeightsDequantizeToFakeQuantize::commonMatchAndRewrite.

    float inLow = 0.0f;
    float inHigh = 0.0f;
    mlir::IntegerAttr levelsAttr = nullptr;
    mlir::TypeAttr lowFpTypeAttr = nullptr;

    const auto intervalResult =
            mlir::TypeSwitch<mlir::Type, mlir::LogicalResult>(storageType)
                    .Case<vpux::type::QuantileType>([&](vpux::type::QuantileType quantileType) {
                        const auto quantileTable = quantileType.getQuantiles();
                        inLow = quantileTable.front();
                        inHigh = quantileTable.back();
                        lowFpTypeAttr = mlir::TypeAttr::get(quantileType);
                        return mlir::success();
                    })
                    .Case<mlir::IntegerType>([&](mlir::IntegerType intType) {
                        const auto levels = IE::getQuantizationLevels(intType);
                        levelsAttr = getIntAttr(ctx, levels);
                        const bool isSigned =
                                uniformType != nullptr ? uniformType.isSigned() : intType.isSignedInteger();
                        inLow = isSigned ? -(static_cast<float>(levels) / 2.0f) : 0.0f;
                        inHigh = static_cast<float>(levels) + inLow - 1.0f;
                        return mlir::success();
                    })
                    .Default([&](mlir::Type t) -> mlir::LogicalResult {
                        if (!isLowFpType(t)) {
                            return matchFailed(rewriter, origOp,
                                               "Unsupported weights storage type for FakeQuantize reconstruction");
                        }
                        const auto storageParams = getStorageParams(t);
                        if (mlir::failed(storageParams)) {
                            return matchFailed(rewriter, origOp,
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
            mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).changeElemType(dstElemType);
    mlir::Value floatWeights;
    if (weightsConst != nullptr) {
        const auto contentAttr = weightsConst.getContentAttr();
        auto floatWeightsContent = contentAttr.transform().castElemType(dstElemType).get();
        floatWeights = rewriter.create<Const::DeclareOp>(appendLoc(loc, "fq_weights"),
                                                         mlir::cast<mlir::RankedTensorType>(floatWeightsType),
                                                         std::move(floatWeightsContent))
                               .getOutput();
    } else {
        floatWeights = rewriter.create<IE::ConvertOp>(appendLoc(loc, "fq_weights"), origOp.getInput(), dstElemType)
                               .getOutput();
    }

    // input_low / input_high: splat consts of the weights float type, shape [1, ..., 1].
    const auto inRank = mlir::cast<vpux::NDTypeInterface>(floatWeightsType).getRank();
    const auto scalarType = mlir::RankedTensorType::get(SmallVector<int64_t>(inRank, 1), dstElemType);
    auto inLowConst =
            Const::createFloatConst(rewriter, appendLoc(loc, "fq_in_low"), scalarType, ArrayRef<float>({inLow}));
    auto inHighConst =
            Const::createFloatConst(rewriter, appendLoc(loc, "fq_in_high"), scalarType, ArrayRef<float>({inHigh}));

    // output_low / output_high = (in - zp) * scale.
    const auto scaleAttr = scaleConst.getContentAttr();
    const auto reverted = IE::revertScaleShift(ctx, scaleAttr, zpShiftAttr, inLow, inHigh, scalarType, _log);
    if (mlir::failed(reverted)) {
        return matchFailed(rewriter, origOp, "Failed to revert scale-shift for the FakeQuantize output interval");
    }
    const auto& [outLow, outHigh] = reverted.value();
    const auto outStorageType = mlir::cast<mlir::RankedTensorType>(outLow.getType());
    const auto outLowValues = to_small_vector(outLow.getValues<float>());
    const auto outHighValues = to_small_vector(outHigh.getValues<float>());
    auto outLowConst =
            Const::createFloatConst(rewriter, appendLoc(loc, "fq_out_low"), outStorageType, ArrayRef(outLowValues));
    auto outHighConst =
            Const::createFloatConst(rewriter, appendLoc(loc, "fq_out_high"), outStorageType, ArrayRef(outHighValues));

    const auto broadCastAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);
    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(origOp, floatWeights, inLowConst, inHighConst, outLowConst,
                                                    outHighConst, levelsAttr, lowFpTypeAttr, broadCastAttr);

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
