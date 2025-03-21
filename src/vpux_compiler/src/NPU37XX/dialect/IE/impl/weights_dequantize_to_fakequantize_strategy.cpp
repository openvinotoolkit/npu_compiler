//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/weights_dequantize_to_fakequantize_strategy.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

template <typename OriginalOp>
mlir::LogicalResult commonMatchAndRewrite(OriginalOp origOp, IE::WeightsDequantizeStructureInfo& wdInfo,
                                          mlir::PatternRewriter& rewriter) {
    const auto loc = wdInfo.getLastOp()->getLoc();

    const auto inputElemType = IE::getTrueElemTypeOfWeights(origOp);
    // The commonMatchAndRewrite supported weights data type are I8, U8, I4, U4 and NF4
    if (!inputElemType.isInteger(8) && !inputElemType.isInteger(4) &&
        !mlir::isa<vpux::type::QuantileFloatType>(inputElemType)) {
        return mlir::failure();
    }

    // Compute input low, input high constants of FakeQuantize using the value interval of the weights type
    auto inLow = 0.0f;
    auto inHigh = 0.0f;
    mlir::IntegerAttr levelsAttr = nullptr;
    mlir::TypeAttr lowFpTypeAttr = nullptr;
    if (mlir::isa<mlir::IntegerType>(inputElemType)) {
        // Integer case
        const auto levels = IE::getQuantizationLevels(inputElemType);
        levelsAttr = getIntAttr(rewriter.getContext(), levels);
        inLow = (inputElemType.isSignedInteger() ? -(levels / 2) : 0);
        inHigh = (levels + inLow - 1);
    } else if (auto quantileFloatType = mlir::dyn_cast<vpux::type::QuantileFloatType>(inputElemType)) {
        // Quantile float case
        auto width = quantileFloatType.getWidth();
        quantileFloatType = vpux::type::QuantileFloatType::getQuantileFloat(rewriter.getContext(), width);
        auto quantileTable = quantileFloatType.getQuantiles();
        inLow = quantileTable.front();
        inHigh = quantileTable.back();
        lowFpTypeAttr = mlir::TypeAttr::get(quantileFloatType);
    }

    const auto [inLowConst, inHighConst] =
            wdInfo.getInputQuantizationInterval(rewriter, appendLoc(loc, "artificial_fq_in_param"), inLow, inHigh);

    // Compute output low and output high constants of FakeQuantize by applying a reverse scale-shift to the inputs
    const auto [outLowConst, outHighConst] =
            wdInfo.getOutputQuantizationInterval(rewriter, appendLoc(loc, "artificial_fq_out_param"), inLow, inHigh);

    const auto broadCastAttr = IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), IE::AutoBroadcastType::NUMPY);

    // sanity checks:
    VPUX_THROW_WHEN(origOp->getNumResults() != 1, "Unexpected number of results {0} in operation {1}",
                    origOp->getNumResults(), origOp->getName());
    VPUX_THROW_WHEN(wdInfo.getLastOp()->getNumResults() != 1, "Unexpected number of results {0} in operation {1}",
                    wdInfo.getLastOp()->getNumResults(), wdInfo.getLastOp()->getName());

    const auto oldOutput = wdInfo.getLastOp()->getResult(0);
    mlir::Value fqInput = origOp->getResult(0);
    if (wdInfo.getInput() != nullptr) {
        fqInput = wdInfo.getInput();
    }

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
    auto fqOp =
            rewriter.create<IE::FakeQuantizeOp>(appendLoc(loc, "artificial_fq"), fqInput, inLowConst, inHighConst,
                                                outLowConst, outHighConst, levelsAttr, lowFpTypeAttr, broadCastAttr);
    rewriter.replaceAllUsesExcept(oldOutput, fqOp.getResult(), fqOp);

    wdInfo.cleanUpCurrentWdChain(rewriter);

    return mlir::success();
}

class WeightsDequantizeToFakeQuantizeConstRewriter final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    WeightsDequantizeToFakeQuantizeConstRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<Const::DeclareOp>(ctx), _log(log) {
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
        return commonMatchAndRewrite(origOp, wdInfo, rewriter);
    }

private:
    Logger _log;
};

class WeightsDequantizeToFakeQuantizeBlockRewriter final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    WeightsDequantizeToFakeQuantizeBlockRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        setDebugName("WeightsDequantizeToFakeQuantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final {
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
        if (!wdInfo.getInput()) {
            _log.trace("Can't create FakeQuantize without a valid input");
            return mlir::failure();
        }
        if (!wdInfo.getScale() && !wdInfo.getShift()) {
            _log.trace("Can't create FakeQuantize without static scale and shift");
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

IE::arch37xx::WeightsDequantizeToFakeQuantizeStrategy::WeightsDequantizeToFakeQuantizeStrategy(
        bool enableWDBlockArgumentInput) noexcept
        : _enableWDBlockArgumentInput(enableWDBlockArgumentInput) {
}

void IE::arch37xx::WeightsDequantizeToFakeQuantizeStrategy::addPatterns(mlir::RewritePatternSet& patterns,
                                                                        Logger& log) const {
    auto ctx = patterns.getContext();

    IE::ConvertOp::getCanonicalizationPatterns(patterns, ctx);

    patterns.add<WeightsDequantizeToFakeQuantizeConstRewriter>(ctx, log);
    if (_enableWDBlockArgumentInput) {
        patterns.add<WeightsDequantizeToFakeQuantizeBlockRewriter>(ctx, log);
    }
}
