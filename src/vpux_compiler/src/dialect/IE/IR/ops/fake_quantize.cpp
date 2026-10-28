//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_types/types.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/transpose_op_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::FakeQuantizeOp::verify() {
    const auto levels = getLevels();
    const auto lowFpType = getLowFpType();

    if (!levels.has_value()) {
        if (!lowFpType.has_value()) {
            return errorAt(*this, "Missing both levels and low precision floating type");
        }
        if (!isLowFpType(lowFpType.value()) && !mlir::isa<mlir::quant::QuantileType>(lowFpType.value())) {
            return errorAt(*this, "Unsupported low floating point type {0}", *lowFpType);
        }
    } else {
        if (lowFpType.has_value()) {
            return errorAt(*this,
                           "Contradicting attributes, both levels and low precision floating type were provided");
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::FakeQuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::FakeQuantizeOpAdaptor quantize(operands, attrs, prop);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<mlir::ShapedType>(quantize.getInput().getType());
    const auto inputLowType = mlir::cast<mlir::ShapedType>(quantize.getInputLow().getType());
    const auto inputHighType = mlir::cast<mlir::ShapedType>(quantize.getInputHigh().getType());
    const auto outputLowType = mlir::cast<mlir::ShapedType>(quantize.getOutputLow().getType());
    const auto outputHighType = mlir::cast<mlir::ShapedType>(quantize.getOutputHigh().getType());
    const auto autob = quantize.getAutoBroadcast();

    const auto inputTensorType = mlir::cast<mlir::RankedTensorType>(inputType);
    const auto outDesc = vpux::getTensorAttr(inputTensorType);

    const auto outShapeOrResult =
            IE::broadcastEltwiseShape({inputType.getShape(), inputLowType.getShape(), inputHighType.getShape(),
                                       outputLowType.getShape(), outputHighType.getShape()},
                                      autob, loc);

    if (mlir::succeeded(outShapeOrResult)) {
        inferredReturnShapes.emplace_back(outShapeOrResult.value(), inputType.getElementType(), outDesc);
    }

    return outShapeOrResult;
}

mlir::LogicalResult vpux::IE::FakeQuantizeOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    reifiedReturnShapes.emplace_back(reifyTrivialTensor(builder, getInput(), getLoc()));
    return mlir::success();
}

mlir::OpFoldResult vpux::IE::FakeQuantizeOp::fold(FoldAdaptor) {
    if (auto fakeQuantize = getInput().getDefiningOp<IE::FakeQuantizeOp>()) {
        const auto cstMinInSecondFQ = getInputLow();
        const auto cstMaxInSecondFQ = getInputHigh();
        const auto cstMinOutSecondFQ = getOutputLow();
        const auto cstMaxOutSecondFQ = getOutputHigh();
        const auto cstMinInFirstFQ = fakeQuantize.getInputLow();
        const auto cstMaxInFirstFQ = fakeQuantize.getInputHigh();
        const auto cstMinOutFirstFQ = fakeQuantize.getOutputLow();
        const auto cstMaxOutFirstFQ = fakeQuantize.getOutputHigh();
        if (cstMinInSecondFQ == cstMinInFirstFQ && cstMaxInSecondFQ == cstMaxInFirstFQ &&
            cstMinOutSecondFQ == cstMinOutFirstFQ && cstMaxOutSecondFQ == cstMaxOutFirstFQ) {
            return getInput();
        }
    }

    return nullptr;
}

namespace {
class TransposeGroups final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    TransposeGroups(mlir::MLIRContext* ctx): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(Logger::global()) {
        setDebugName("TransposeGroups");
    }

public:
    SmallVector<unsigned> getTransposition(mlir::Value output) const;
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp fqOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

SmallVector<unsigned> TransposeGroups::getTransposition(mlir::Value output) const {
    SmallVector<unsigned> fallbackValue = {2, 0, 1};
    if (!output.hasOneUse()) {
        return fallbackValue;
    }
    const auto consumers = output.getUsers();
    auto maybeReshape = mlir::dyn_cast_or_null<IE::AffineReshapeOp>(*consumers.begin());
    if (maybeReshape == nullptr) {
        return fallbackValue;
    }
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(maybeReshape.getDimMapping());
    const auto isGatherDim = [](const ArrayRef<int64_t> map) -> bool {
        return map.size() != 1;
    };
    if (std::any_of(dimMapping.begin(), dimMapping.end(), isGatherDim)) {
        return fallbackValue;
    }
    const auto flattenMap = [](const ArrayRef<int64_t> map) -> int64_t {
        return map.front();
    };
    SmallVector<int64_t> flatMapping;
    std::transform(dimMapping.begin(), dimMapping.end(), std::back_inserter(flatMapping), flattenMap);
    SmallVector<unsigned> permutationMap = std::move(fallbackValue);
    // 1. FQ [1280, 20, 128] * [1280, 1, 128] -> Reshape [[0], [1], [1]] -> [1280, 2560]
    // Transpose [1280, 20, 128] to [20, 128, 1280]
    // 2. FQ [1280, 20, 128] * [1280, 1, 128] -> Reshape [[0], [0], [1]] -> [2560, 128]
    // Transpose [1280, 20, 128] to [128, 1280, 20]
    // 3. FQ [1280, 20, 128] * [1280, 20, 1]  -> Reshape [[0], [1], [1]] -> [1280, 2560]
    // Transpose [1280, 20, 128] to [20, 128, 1280]
    if (flatMapping == SmallVector<int64_t>{0, 0, 1}) {
        permutationMap = {2, 0, 1};
    } else if (flatMapping == SmallVector<int64_t>{0, 1, 1}) {
        permutationMap = {1, 2, 0};
    }
    return permutationMap;
}

mlir::LogicalResult TransposeGroups::matchAndRewrite(IE::FakeQuantizeOp fqOp, mlir::PatternRewriter& rewriter) const {
    const auto quantAxes = IE::findAxes(fqOp);
    // Process only group quantization cases.
    if (quantAxes.size() != 2) {
        return matchFailed(_log, rewriter, fqOp, "FakeQuantize does not provide group quantization");
    }
    if (quantAxes.count(0) != 1) {
        return matchFailed(_log, rewriter, fqOp, "The first dimension must be quantized");
    }
    for (auto operand : fqOp->getOperands() | indexed) {
        auto constOp = operand.value().getDefiningOp<Const::DeclareOp>();
        if (constOp == nullptr) {
            return matchFailed(_log, rewriter, fqOp, "FakeQuantize input is not const");
        }
        const auto outType = mlir::cast<vpux::NDTypeInterface>(operand.value().getType());
        if (outType.getRank() != 3) {
            return matchFailed(_log, rewriter, fqOp, "FakeQuantize input rank does not meet the requirement");
        }

        // Check if it is benefical to propagate.
        // The first dimension must be the largest for all the operands except for the data operand.
        // The data operand is skipped because it may trigger the transposition on an unrelated axis.
        // For example:
        // %data = const.Declare tensor<32x128x64xf32>
        // %in_lo = const.Declare tensor<1x1x1xf32>
        // %in_hi = const.Declare tensor<1x1x1xf32>
        // %out_lo = const.Declare tensor<32x1x64xf32>
        // %out_hi = const.Declare tensor<32x1x64xf32>
        // IE.FakeQuantize(%data, %in_lo, %in_hi, %out_lo, %out_hi)
        //
        // Note that for 32x1x64 the transposition is not required because 32 is less than 64.
        // This is not the case for 32x128x64, where 128 is greater than 32.
        if (operand.index() != 0) {
            const auto outShape = outType.getShape();
            // In the example above, 32 must be the least non-trivial dimension.
            // There's no point to compare the first dimension with dimensions equal to one.
            // Exclude such dimensions from the shape.
            Shape filteredShape;
            const auto isNonTrivialDim = [](const int64_t dim) -> bool {
                return dim != 1;
            };
            std::copy_if(outShape.begin(), outShape.end(), std::back_inserter(filteredShape), isNonTrivialDim);
            if (filteredShape.empty()) {
                // Skip 1x1x1 shapes, they don't have axes.
                continue;
            }

            // The shape must begin with the minimal dimension.
            // 64x1x32 is beneficial to transpose.
            // 64x32x1 is beneficial to transpose.
            // 32x64x1 is not beneficial to transpose.
            // 32x1x64 is not beneficial to transpose.
            const auto minElementIt = std::min_element(filteredShape.begin(), filteredShape.end());
            if (minElementIt == filteredShape.begin()) {
                return matchFailed(_log, rewriter, fqOp, "Not benefical to propagate to const");
            }
        }
    }

    auto order = getTransposition(fqOp.getOutput());
    auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(order, getContext()));
    auto input = IE::TransposeOp::create(rewriter, appendLoc(fqOp.getInput().getLoc(), "transposed"), fqOp.getInput(),
                                         nullptr, orderAttr);
    auto inLow = IE::TransposeOp::create(rewriter, appendLoc(fqOp.getInputLow().getLoc(), "transposed"),
                                         fqOp.getInputLow(), nullptr, orderAttr);
    auto inHigh = IE::TransposeOp::create(rewriter, appendLoc(fqOp.getInputHigh().getLoc(), "transposed"),
                                          fqOp.getInputHigh(), nullptr, orderAttr);
    auto outLow = IE::TransposeOp::create(rewriter, appendLoc(fqOp.getOutputLow().getLoc(), "transposed"),
                                          fqOp.getOutputLow(), nullptr, orderAttr);
    auto outHigh = IE::TransposeOp::create(rewriter, appendLoc(fqOp.getOutputHigh().getLoc(), "transposed"),
                                           fqOp.getOutputHigh(), nullptr, orderAttr);
    auto newFqOp = IE::FakeQuantizeOp::create(
            rewriter, fqOp->getLoc(), input.getOutput(), inLow.getOutput(), inHigh.getOutput(), outLow.getOutput(),
            outHigh.getOutput(), fqOp.getLevelsAttr(), fqOp.getLowFpTypeAttr(), fqOp.getAutoBroadcastAttr());
    const auto inverseOrder = IE::deduceInverseOrder(input);
    auto restoreOrderAttr = mlir::AffineMapAttr::get(inverseOrder.toAffineMap(getContext()));
    auto out = IE::TransposeOp::create(rewriter, appendLoc(newFqOp.getOutput().getLoc(), "transposed"),
                                       newFqOp.getOutput(), nullptr, restoreOrderAttr);
    rewriter.replaceOp(fqOp, out.getOutput());

    return mlir::success();
}
}  // namespace

void vpux::IE::FakeQuantizeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                           mlir::MLIRContext* context) {
    patterns.add<TransposeGroups>(context);
}
