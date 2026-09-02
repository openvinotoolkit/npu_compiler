//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_dequantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

using namespace vpux;

namespace {

bool isSupportedWeightsElemType(mlir::Type elemType) {
    if (isLowFpType(elemType) || mlir::isa<vpux::type::QuantileType>(elemType)) {
        return true;
    }

    const auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType);
    if (intType == nullptr) {
        return false;
    }

    const auto supportedBitWidth =
            intType.getWidth() == 16 || intType.getWidth() == 8 || intType.getWidth() == 4 || intType.getWidth() == 2;
    if (!supportedBitWidth) {
        return false;
    }

    // DynamicDequantize verifier rejects signless integer input/zp.
    return !intType.isSignless();
}

// Build an integer zero-point constant from a (possibly scalar / lower-rank) shift constant, reshaped to
// `inputRank` so it satisfies the IE.DynamicDequantize verifier (ZP rank must equal input rank; dims may be 1).
mlir::Value createRankAlignedZeroPoint(mlir::PatternRewriter& rewriter, mlir::Location loc, Const::DeclareOp shiftConst,
                                       mlir::Type zpElemType, int64_t inputRank) {
    const auto shiftShape = mlir::cast<vpux::NDTypeInterface>(shiftConst.getType()).getShape().raw();
    VPUX_THROW_WHEN(checked_cast<int64_t>(shiftShape.size()) > inputRank,
                    "Shift rank {0} exceeds input rank {1}; cannot build zero-point", shiftShape.size(), inputRank);

    SmallVector<int64_t> alignedShape(inputRank, 1);
    const auto offset = inputRank - checked_cast<int64_t>(shiftShape.size());
    for (auto i : irange(shiftShape.size())) {
        alignedShape[offset + i] = shiftShape[i];
    }

    auto contentSetup = shiftConst.getContentAttr().transform().castElemType(zpElemType);
    if (checked_cast<int64_t>(shiftShape.size()) != inputRank) {
        contentSetup = std::move(contentSetup).reshape(ShapeRef(alignedShape));
    }
    auto zpContentAttr = std::move(contentSetup).get();
    const auto zpType = mlir::cast<vpux::NDTypeInterface>(shiftConst.getType())
                                .changeShapeElemType(ShapeRef(alignedShape), zpElemType);
    return rewriter.create<Const::DeclareOp>(loc, mlir::cast<mlir::RankedTensorType>(zpType), zpContentAttr)
            .getOutput();
}

// Builds an implicit splat-1.0 scale constant, shaped per-output-channel (dim 0 matches `rawWeights`'
// leading dim, all other dims are 1) rather than fully-collapsed to rank-many 1s. Downstream consumers
// (e.g. the New Weight Table ScaleTable builder) require the scale's Dim(0) to equal the number of
// output channels; a fully-collapsed [1, 1, ..., 1] shape breaks that invariant even though the value
// is numerically identical (a splat is stored as a single value regardless of its declared shape).
mlir::Value createImplicitUnitScale(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Type dstType,
                                    mlir::Value rawWeights) {
    const auto weightsShape = mlir::cast<vpux::NDTypeInterface>(rawWeights.getType()).getShape();
    SmallVector<int64_t> scaleShape(weightsShape.size(), 1);
    scaleShape[0] = weightsShape[Dim(0)];
    const auto scaleType = mlir::RankedTensorType::get(scaleShape, dstType);
    return Const::createFloatConst(rewriter, loc, scaleType, ArrayRef<float>({1.0f}));
}

// Finalizes a matched WD chain into an `IE.DynamicDequantize`, shared by the const (compile-time
// weights) and block-arg (weights-as-input) rewriters. `rawWeights` is the already-correctly-typed
// raw value each rewriter locates/builds on its own (a freshly re-cast `Const::DeclareOp` for the
// former, `origOp.getInput()` directly for the latter); `zpElemType` is the element type the
// zero-point constant must be cast to.
mlir::LogicalResult finishDynamicDequantizeRewrite(mlir::PatternRewriter& rewriter,
                                                   const IE::WeightsDequantizeStructureInfo& wdInfo,
                                                   mlir::Value rawWeights, mlir::Type zpElemType, Logger log) {
    auto lastOp = wdInfo.getLastOp();
    const auto loc = lastOp->getLoc();
    const auto dstType = mlir::cast<vpux::NDTypeInterface>(lastOp->getResult(0).getType()).getElementType();
    const auto rawRank = mlir::cast<vpux::NDTypeInterface>(rawWeights.getType()).getRank();

    mlir::Value zpValue = nullptr;
    if (auto staticShift = wdInfo.getStaticShift()) {
        auto shiftConst = staticShift.getDefiningOp<Const::DeclareOp>();
        if (shiftConst == nullptr) {
            log.trace("Non-const shift; cannot build integer ZP.");
            return mlir::failure();
        }
        // The IE.DynamicDequantize verifier requires the ZP rank to equal the input rank (dims may
        // be 1).
        zpValue = createRankAlignedZeroPoint(rewriter, appendLoc(loc, "ddq_zp"), shiftConst, zpElemType, rawRank);
    }

    // The IE.DynamicDequantize scale operand is mandatory. A shift-only chain (no Multiply) has no
    // static scale to reuse, so synthesize an implicit splat-1.0 one.
    const auto scaleValue = wdInfo.getStaticScale() ? wdInfo.getStaticScale()
                                                    : createImplicitUnitScale(rewriter, appendLoc(loc, "ddq_scale"),
                                                                              dstType, rawWeights);

    auto dynamicDequant =
            rewriter.create<IE::DynamicDequantizeOp>(appendLoc(loc, "ddq"), rawWeights, scaleValue, zpValue, dstType);

    // Mark the op for the DynDeq -> FakeQuantize bridge pass.
    if (wdInfo.isKVcachedPattern() || !zpElemType.isInteger(2)) {
        dynamicDequant->setAttr(IE::WEIGHTS_IMPORT_DYN_DEQUANT_ATTR, mlir::UnitAttr::get(rewriter.getContext()));
    }

    rewriter.replaceAllOpUsesWith(lastOp, dynamicDequant.getResult());
    wdInfo.cleanUpCurrentWdChain(rewriter);

    return mlir::success();
}

//
// WeightsDequantizeToDynamicDequantizeConstRewriter
//

class WeightsDequantizeToDynamicDequantizeConstRewriter final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    WeightsDequantizeToDynamicDequantizeConstRewriter(mlir::MLIRContext* ctx, Logger log,
                                                      mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<Const::DeclareOp>(ctx, benefit), _log(log) {
        setDebugName("WeightsDequantizeToDynamicDequantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("Got {0} at `{1}`.", origOp->getName(), origOp->getLoc());

        auto maybeWdInfo = IE::WeightsDequantizeStructureInfo::create(origOp, _log.nest());
        if (mlir::failed(maybeWdInfo)) {
            _log.trace("Failed to match WeightsDequantize structure");
            return mlir::failure();
        }
        auto wdInfo = maybeWdInfo.value();

        const auto zpElemType = IE::getTrueElemType(origOp);
        const auto suitableType = isSupportedWeightsElemType(zpElemType);
        if (!suitableType) {
            _log.trace("Input data type {0} is not supported.", zpElemType);
            return mlir::failure();
        }
        if (wdInfo.getDynamicScale()) {
            _log.trace("Can't create DynamicDequantize with dynamic scale");
            return mlir::failure();
        }
        if (wdInfo.isQuantizedConsumedByGather()) {
            _log.trace("WD chain feeds a Gather op; deferring to native DynamicDequantize path.");
            return mlir::failure();
        }

        const auto users = origOp->getUsers();
        if (users.empty()) {
            _log.trace("No users of the matched DeclareOp");
            return mlir::failure();
        }

        // The chain may have no Multiply at all: a per-channel scale of exactly 1.0 is folded away by
        // IE::MultiplyOp::fold() before this rewriter ever runs (Const -> Convert -> Subtract -> Conv).

        auto lastOp = wdInfo.getLastOp();

        const auto loc = lastOp->getLoc();

        const auto baseInputElemType = origOp.getContentAttr().getBaseContent().getElementType();

        const auto quantileType = mlir::isa<vpux::type::QuantileType>(zpElemType)
                                          ? mlir::cast<vpux::type::QuantileType>(zpElemType)
                                          : IE::tryParsingNF4(origOp);

        const auto rawWeightsElemType =
                quantileType != nullptr ? mlir::cast<mlir::Type>(quantileType) : baseInputElemType;

        const auto rawType = mlir::dyn_cast<vpux::NDTypeInterface>(origOp.getType()).changeElemType(rawWeightsElemType);

        auto rawContentAttr = origOp.getContentAttr().transform().castElemType(rawWeightsElemType).get();

        auto rawDeclareOp = rewriter.create<Const::DeclareOp>(loc, rawType, rawContentAttr);

        if (!rawDeclareOp) {
            _log.trace("Invalid rawDeclareOp for DynDeq");
            return mlir::failure();
        }

        return finishDynamicDequantizeRewrite(rewriter, wdInfo, rawDeclareOp.getOutput(), zpElemType, _log);
    }

private:
    Logger _log;
};

//
// WeightsDequantizeToDynamicDequantizeBlockArgumentRewriter
//

class WeightsDequantizeToDynamicDequantizeBlockArgumentRewriter final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    WeightsDequantizeToDynamicDequantizeBlockArgumentRewriter(mlir::MLIRContext* ctx, Logger log,
                                                              mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::ConvertOp>(ctx, benefit), _log(log) {
        setDebugName("WeightsDequantizeToDynamicDequantizeBlockArgumentRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("Got {0} at `{1}`.", origOp->getName(), origOp->getLoc());

        auto maybeWdInfo = IE::WeightsDequantizeStructureInfo::create(origOp, _log.nest());
        if (mlir::failed(maybeWdInfo)) {
            return mlir::failure();
        }
        auto wdInfo = std::move(*maybeWdInfo);

        const auto inputElemType = IE::getTrueElemType(origOp);

        // Keep parity with the historical WD->FQ supported-type whitelist.
        const auto suitableType = isSupportedWeightsElemType(inputElemType);
        if (!suitableType) {
            _log.trace("Input data type {0} is not supported.", inputElemType);
            return mlir::failure();
        }
        if (wdInfo.getDynamicScale() != nullptr || wdInfo.getDynamicShift() != nullptr) {
            _log.trace("Dynamic scale or shift; cannot produce compile-time DynDeq.");
            return mlir::failure();
        }
        if (wdInfo.getStaticScale() == nullptr && wdInfo.getStaticShift() == nullptr) {
            _log.trace("No static scale or shift; cannot produce DynDeq.");
            return mlir::failure();
        }
        if (wdInfo.getQuantizedAxisCount() < 2) {
            _log.trace("Per-tensor or per-channel; handled by const rewriter.");
            return mlir::failure();
        }

        return finishDynamicDequantizeRewrite(rewriter, wdInfo, origOp.getInput(), inputElemType, _log);
    }

private:
    Logger _log;
};

}  // namespace

//
// registerWeightsDequantizeToDynamicDequantizeRewriters
//

void vpux::IE::registerWeightsDequantizeToDynamicDequantizeRewriters(RewriterRegistry& registry,
                                                                     ArrayRef<mlir::PatternBenefit> benefitLevels,
                                                                     size_t index, Logger log) {
    registry.registerRewriterSet("weights-dequantize-to-dynamic-dequantize", [&]() {
        registry.registerRewriter<WeightsDequantizeToDynamicDequantizeConstRewriter>(
                "weights-dequantize-to-dynamic-dequantize-const", log, benefitLevels[index]);
        registry.registerRewriter<WeightsDequantizeToDynamicDequantizeBlockArgumentRewriter>(
                "weights-dequantize-to-dynamic-dequantize-block-arg", log, benefitLevels[index]);
        IE::registerConvertOpRewriters(registry);
    });
}
