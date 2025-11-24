//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// ConcatOpConverter
//

// This rewriter creates `FakeQuantize` operation, which combines per-channel quantization from `Concat` inputs,
// and places it after the `Concat` operation. For example:
// The following `Concat`:
// ```
//     FQ 1x256x128x128 -> Concat <- FQ 1x48x128x128
//                             |
//                         GroupConv 1x304x128x128
// ```
// will be transformed into:
// ```
//     FQ 1x256x128x128 -> Concat <- FQ 1x48x128x128
//                             |
//                             FQ 1x304x128x128
//                             |
//                         GroupConv 1x304x128x128
// ```

class ConcatOpConverter final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    ConcatOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isPerAxisFqValue(mlir::Value input) {
    auto maybeFqOp = input.getDefiningOp<IE::FakeQuantizeOp>();
    if (maybeFqOp == nullptr) {
        return false;
    }

    return !IE::isPerTensorFQ({maybeFqOp});
}

bool isPerAxisFqOp(mlir::Operation* op) {
    auto maybeFqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op);
    if (maybeFqOp == nullptr) {
        return false;
    }

    return !IE::isPerTensorFQ({maybeFqOp});
}

bool isLegalConcat(IE::ConcatOp origConcatOp) {
    const auto concatUsers = origConcatOp.getOutput().getUsers();
    if (std::all_of(concatUsers.begin(), concatUsers.end(), isPerAxisFqOp)) {
        return true;
    }
    const auto concatInputList = origConcatOp.getInputs();
    return !std::all_of(concatInputList.begin(), concatInputList.end(), isPerAxisFqValue);
}

void appendFqValues(mlir::Value fqInput, std::vector<float>& totalValues) {
    // Fetch values from given FQ input and concatenate them with destination vector
    auto inConst = fqInput.getDefiningOp<Const::DeclareOp>();
    auto inConstContent = inConst.getContentAttr().fold();
    auto inValues = inConstContent.getValues<float>();
    std::copy(inValues.begin(), inValues.end(), std::back_inserter(totalValues));
}

mlir::Value createFqTensor(mlir::Location loc, const std::vector<float>& totalFqValues,
                           mlir::PatternRewriter& rewriter) {
    // Build FQ input using concatenated values
    const auto tensorType = mlir::RankedTensorType::get({1, checked_cast<int64_t>(totalFqValues.size()), 1, 1},
                                                        mlir::Float32Type::get(rewriter.getContext()));
    return Const::createConst(rewriter, loc, tensorType, ArrayRef(totalFqValues));
}

mlir::LogicalResult ConcatOpConverter::matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at `{1}`.", origOp->getName(), origOp->getLoc());

    if (isLegalConcat(origOp)) {
        return mlir::failure();
    }

    auto concatInputList = origOp.getInputs();
    if (concatInputList.empty()) {
        return mlir::failure();
    }
    _log.nest().trace("Got {0} FQs as input.", concatInputList.size());

    auto firstFq = concatInputList.front().getDefiningOp<IE::FakeQuantizeOp>();
    auto firstFqAxis = getFQAxisIndex(firstFq, _log);
    auto concatAxis = getConcatAxis(origOp);

    if (!concatAxis.has_value() || !firstFqAxis.has_value() || concatAxis.value().ind() != firstFqAxis.value()) {
        _log.nest().trace("Concat axis does not match FakeQuantize axis.");
        return mlir::failure();
    }

    const auto levels = firstFq.getLevels();
    const auto lowFpType = firstFq.getLowFpType();
    const auto autoBroadcast = firstFq.getAutoBroadcast();
    std::vector<float> totalInLo;
    std::vector<float> totalInHi;
    std::vector<float> totalOutLo;
    std::vector<float> totalOutHi;

    for (const auto& concatInput : concatInputList) {
        auto fqOp = concatInput.getDefiningOp<IE::FakeQuantizeOp>();
        auto curFqAxis = getFQAxisIndex(fqOp, _log);

        if (levels != fqOp.getLevels() || lowFpType != fqOp.getLowFpType() ||
            autoBroadcast != fqOp.getAutoBroadcast() || !curFqAxis.has_value() ||
            curFqAxis.value() != firstFqAxis.value()) {
            _log.nest().trace("Got FQs with different levels, lowFpTypes, fqAxis or autobroadcasts.");
            return mlir::failure();
        }

        appendFqValues(fqOp.getInputLow(), totalInLo);
        appendFqValues(fqOp.getInputHigh(), totalInHi);
        appendFqValues(fqOp.getOutputLow(), totalOutLo);
        appendFqValues(fqOp.getOutputHigh(), totalOutHi);
    }

    auto concatOp = rewriter.create<IE::ConcatOp>(origOp->getLoc(), concatInputList, origOp.getPerAxisAttr(),
                                                  origOp.getStaticOffsetsAttr());

    auto inLowOp = createFqTensor(takeOpLoc(origOp, "in_low"), totalInLo, rewriter);
    auto inHighOp = createFqTensor(takeOpLoc(origOp, "in_high"), totalInHi, rewriter);
    auto outLowOp = createFqTensor(takeOpLoc(origOp, "out_low"), totalOutLo, rewriter);
    auto outHighOp = createFqTensor(takeOpLoc(origOp, "out_high"), totalOutHi, rewriter);

    auto levelsAttr = levels.has_value() ? getIntAttr(origOp.getContext(), *levels) : nullptr;
    auto lowFpTypeAttr = lowFpType.has_value() ? mlir::TypeAttr::get(*lowFpType) : nullptr;

    auto fqOp =
            rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(origOp, concatOp.getOutput(), inLowOp, inHighOp, outLowOp,
                                                            outHighOp, levelsAttr, lowFpTypeAttr, autoBroadcast);
    extendOpLoc(fqOp, "common_fq");

    return mlir::success();
}

}  // namespace

void vpux::IE::registerPerAxisFQConcatRewriters(RewriterRegistry& registry, Logger log) {
    registry.registerRewriter<ConcatOpConverter>("per-axis-fq-concat", log);
}
