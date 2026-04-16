//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/impl/weights_dequantize_to_fakequantize_strategy.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

template <typename OriginalOp>
mlir::LogicalResult commonMatchAndRewrite(OriginalOp origOp, IE::WeightsDequantizeStructureInfo& wdInfo,
                                          mlir::PatternRewriter& rewriter) {
    const auto loc = wdInfo.getLastOp()->getLoc();
    const auto ctx = rewriter.getContext();

    // The only supported weights data type are I8, U8, I4, U4, I2, U2, F8E4M3FN, F8E5M2 and NF4
    const auto inputElemType = IE::getTrueElemType(origOp);
    const auto suitableType = inputElemType.isInteger(8) || inputElemType.isInteger(4) || inputElemType.isInteger(2) ||
                              isFloat8(inputElemType) || mlir::isa<vpux::type::QuantileFloatType>(inputElemType);
    if (!suitableType) {
        wdInfo.log.trace("Input data type {0} is not supported.", inputElemType);
        return mlir::failure();
    }

    // Compute input low, input high constants of FakeQuantize using the value interval of the weights type
    auto inLow = 0.0f;
    auto inHigh = 0.0f;
    mlir::IntegerAttr levelsAttr = nullptr;
    mlir::TypeAttr lowFpTypeAttr = nullptr;
    auto quantileFloatType = [&]() -> vpux::type::QuantileFloatType {
        if (mlir::isa<vpux::type::QuantileFloatType>(inputElemType)) {
            return mlir::cast<vpux::type::QuantileFloatType>(inputElemType);
        }
        return IE::tryParsingNF4(origOp);
    }();

    if (quantileFloatType) {
        // Quantile float case
        auto quantileTable = quantileFloatType.getQuantiles();
        inLow = quantileTable.front();
        inHigh = quantileTable.back();
        lowFpTypeAttr = mlir::TypeAttr::get(quantileFloatType);
    } else if (mlir::isa<mlir::IntegerType>(inputElemType)) {
        // Integer case
        const auto levels = IE::getQuantizationLevels(inputElemType);
        levelsAttr = getIntAttr(ctx, levels);
        inLow = (inputElemType.isSignedInteger() ? -(levels / 2) : 0);
        inHigh = (levels + inLow - 1);
    } else {
        // Floating point case
        const auto lowFpRangeOrFail = vpux::getLowFpRange(inputElemType);
        VPUX_THROW_WHEN(mlir::failed(lowFpRangeOrFail), "Unexpected data type {0}", inputElemType);

        std::tie(inLow, inHigh) = lowFpRangeOrFail.value();
        lowFpTypeAttr = mlir::TypeAttr::get(inputElemType);
    }

    const auto [inLowConst, inHighConst] =
            wdInfo.getInputQuantizationInterval(rewriter, appendLoc(loc, "artificial_fq_in_param"), inLow, inHigh);

    // Compute output low and output high constants of FakeQuantize by applying a reverse scale-shift to the inputs
    const auto [outLowConst, outHighConst] =
            wdInfo.getOutputQuantizationInterval(rewriter, appendLoc(loc, "artificial_fq_out_param"), inLow, inHigh);

    const auto broadCastAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);

    // sanity checks:
    VPUX_THROW_WHEN(origOp->getNumResults() != 1, "Unexpected number of results {0} in operation {1}",
                    origOp->getNumResults(), origOp->getName());
    VPUX_THROW_WHEN(wdInfo.getLastOp()->getNumResults() != 1, "Unexpected number of results {0} in operation {1}",
                    wdInfo.getLastOp()->getNumResults(), wdInfo.getLastOp()->getName());
    if (bool isConstFq = mlir::isa<Const::DeclareOp>(origOp); !isConstFq) {
        // E#132447: later passes rely on FQ block with constant weights input
        // being placed near the constant declarations. Once the issue is
        // resolved, the setting of insertion point should be universal.
        rewriter.setInsertionPoint(wdInfo.getLastOp());
    } else {
        // in case we only have a DeclareOp that's treated as WD block (this is
        // a supported case somehow), we need to set the insertion point
        // correctly to prevent domination errors.
        rewriter.setInsertionPointAfter(origOp);
    }

    // Create the FakeQuantize to replace the WD pattern with either levels or fp8 type
    auto fqOp = rewriter.create<IE::FakeQuantizeOp>(appendLoc(loc, "artificial_fq"), wdInfo.getInput(), inLowConst,
                                                    inHighConst, outLowConst, outHighConst, levelsAttr, lowFpTypeAttr,
                                                    broadCastAttr);
    const auto oldOutput = wdInfo.getLastOp()->getResult(0);
    rewriter.replaceAllUsesExcept(oldOutput, fqOp.getResult(), fqOp);

    wdInfo.cleanUpCurrentWdChain(rewriter);

    return mlir::success();
}

class WeightsDequantizeToFakeQuantizeConstRewriter final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    WeightsDequantizeToFakeQuantizeConstRewriter(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<Const::DeclareOp>(ctx, benefit), _log(log) {
        setDebugName("WeightsDequantizeToFakeQuantizeRewriter");
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
        if (wdInfo.getDynamicScale()) {
            _log.trace("Can't create FakeQuantize with dynamic scale");
            return mlir::failure();
        }
        // Constant WD chains feeding a Gather are routed to DynamicDequantize.
        // The Gather accesses a small subset of rows, so offline dequantization wastes blob size.
        if (wdInfo.isI4ConsumedByGather()) {
            _log.trace("WD chain feeds a Gather op with i4/ui4 weights; deferring to DynamicDequantize path.");
            return mlir::failure();
        }
        if (!wdInfo.isKVcachedPattern() && IE::getTrueElemType(origOp).isInteger(2)) {
            // Force to use DynamicDequantize for u2 WaC groupwise prefill model
            _log.trace("Got u2 weights-as-constant groupwise prefill pattern.");
            return mlir::failure();
        }

        return commonMatchAndRewrite(origOp, wdInfo, rewriter);
    }

private:
    Logger _log;
};

class MultiAxisQuantizedBlockArgumentRewriter final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    MultiAxisQuantizedBlockArgumentRewriter(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::ConvertOp>(ctx, benefit), _log(log) {
        setDebugName("MultiAxisQuantizedBlockArgumentRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("Got {0} at `{1}`.", origOp->getName(), origOp->getLoc());

        auto maybeWdInfo = IE::WeightsDequantizeStructureInfo::create(origOp, _log.nest());
        if (mlir::failed(maybeWdInfo)) {
            _log.trace("Failed to match WeightsDequantize structure");
            return mlir::failure();
        }
        auto wdInfo = std::move(*maybeWdInfo);
        if (wdInfo.getDynamicScale() != nullptr || wdInfo.getDynamicShift() != nullptr) {
            _log.trace("Can't create FakeQuantize with dynamic scale or shift");
            return mlir::failure();
        }
        if (wdInfo.getStaticScale() == nullptr && wdInfo.getStaticShift() == nullptr) {
            _log.trace("Can't create FakeQuantize without static scale and shift");
            return mlir::failure();
        }
        if (wdInfo.getQuantizedAxisCount() < 2) {
            // ConsolidateWeightsDequantization cannot handle higher-dimensionality cases.
            // TODO: E#171775 Change <2 to <3.
            _log.trace("Got per-tensor or per-channel quantization.");
            return mlir::failure();
        }
        if (!wdInfo.isKVcachedPattern() && IE::getTrueElemType(origOp).isInteger(2)) {
            // Force to use DynamicDequantize both when quant params are as constants as well as when they're not as
            // constants for the u2 WaI groupwise prefill model
            _log.trace("Got u2 groupwise prefill pattern.");
            return mlir::failure();
        }

        return commonMatchAndRewrite(origOp, wdInfo, rewriter);
    }

private:
    Logger _log;
};

}  // namespace

//
// WeightsDequantizeToFakeQuantizeStrategy
//

void IE::arch50xx::WeightsDequantizeToFakeQuantizeStrategy::registerRewriters(RewriterRegistry& registry,
                                                                              Logger& log) const {
    registry.registerRewriterSet("weights-dequantize-to-fq", [&]() {
        registry.registerRewriter<WeightsDequantizeToFakeQuantizeConstRewriter>("weights-dequantize-to-fq-const", log,
                                                                                _benefitLevels[_index]);
        registry.registerRewriter<MultiAxisQuantizedBlockArgumentRewriter>("multi-axis-quantize-block-arg", log,
                                                                           _benefitLevels[_index]);
        IE::registerConvertOpRewriters(registry);
    });
}
