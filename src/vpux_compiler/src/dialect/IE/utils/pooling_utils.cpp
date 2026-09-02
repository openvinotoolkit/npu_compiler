//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

#include <numeric>

using namespace vpux;

//
// isAvgPoolSupportedElementType
//

bool IE::isAvgPoolSupportedElementType(mlir::Type elemType) {
    // Must stay in sync with IE_AvgPoolOp input constraint in pooling.td:
    //   RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32, SI64, SI8, UI8]>
    return elemType.isF16() || elemType.isF32() || elemType.isF64() ||
           mlir::isa<mlir::quant::QuantizedType>(elemType) || elemType.isSignedInteger(32) ||
           elemType.isSignedInteger(64) || elemType.isSignedInteger(8) || elemType.isUnsignedInteger(8);
}

//
// createIdentityAvgPool
//

mlir::Operation* IE::createIdentityAvgPool(mlir::Value input, mlir::Type outType, mlir::OpBuilder& builder,
                                           mlir::Location loc) {
    const SmallVector<int64_t> poolStrides = {1, 1};
    const SmallVector<int64_t> poolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    auto ctx = builder.getContext();

    return builder.create<IE::AvgPoolOp>(loc, outType, input, nullptr, getIntArrayAttr(ctx, poolKernels),
                                         getIntArrayAttr(ctx, poolStrides), getIntArrayAttr(ctx, pads),
                                         getIntArrayAttr(ctx, pads),
                                         vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR),
                                         mlir::UnitAttr::get(ctx), nullptr, nullptr, nullptr, nullptr, nullptr);
}

//
// createIdentityMaxPool
//

mlir::Operation* IE::createIdentityMaxPool(mlir::Value input, mlir::Type outType, mlir::PatternRewriter& rewriter) {
    const SmallVector<int64_t> poolStrides = {1, 1};
    const SmallVector<int64_t> poolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    auto ctx = rewriter.getContext();

    return rewriter.create<IE::MaxPoolOp>(
            appendLoc(input.getLoc(), "to_maxpool"), outType, input, nullptr, getIntArrayAttr(ctx, poolKernels),
            getIntArrayAttr(ctx, poolStrides), getIntArrayAttr(ctx, pads), getIntArrayAttr(ctx, pads),
            IE::RoundingTypeAttr::get(ctx, IE::RoundingType::FLOOR), nullptr, nullptr, nullptr, nullptr, nullptr);
}

//
// isQuantizedPurposeAvgPool
//
bool IE::isQuantizedPurposeAvgPool(IE::AvgPoolOp avgPool) {
    if (!isIdentityPooling(avgPool)) {
        return false;
    }

    auto inputType = mlir::cast<vpux::NDTypeInterface>(avgPool.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(avgPool.getOutput().getType());
    if (!inputType.getElementType().isF16() ||
        !mlir::isa<mlir::quant::UniformQuantizedType>(outputType.getElementType())) {
        return false;
    }

    return inputType.getDimsOrder() == outputType.getDimsOrder();
}

//
// isQuantizedAvgPoolPermutation
//
bool IE::isQuantizedAvgPoolPermutation(IE::AvgPoolOp avgPool) {
    if (!isIdentityPooling(avgPool)) {
        return false;
    }

    auto inputType = mlir::cast<vpux::NDTypeInterface>(avgPool.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(avgPool.getOutput().getType());

    // do not check order, cause the pool might be used for permutation as well
    return inputType.getElementType().isF16() &&
           mlir::isa<mlir::quant::UniformQuantizedType>(outputType.getElementType());
}

//
// isAddOutputQuantized
//
bool IE::isAddOutputQuantized(IE::AddOp add) {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(add.getInput1().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(add.getOutput().getType());

    // do not check order, cause the pool might be used for permutation as well
    return inputType.getElementType().isF16() &&
           mlir::isa<mlir::quant::UniformQuantizedType>(outputType.getElementType());
}

//
// createConvertPoolingForScaleTable
//

mlir::Value IE::createConvertPoolingForScaleTable(mlir::Value scale, mlir::PatternRewriter& rewriter) {
    if (scale == nullptr) {
        return nullptr;
    }

    const auto scaleType = mlir::cast<vpux::NDTypeInterface>(scale.getType());
    if (scaleType.getElementType().isF32()) {
        return scale;
    }

    auto* ctx = rewriter.getContext();

    // Use an identity MaxPool (1x1 kernel, stride 1, no padding) to convert the scale
    // from f16 to f32, preserving accuracy compared to a plain ConvertOp.
    // Reshape the scale tensor so channels <= 256; remaining elements are
    // distributed as equally as possible across H and W.
    const auto origShape = scaleType.getShape();
    const int64_t totalElems =
            std::accumulate(origShape.begin(), origShape.end(), int64_t(1), std::multiplies<int64_t>());

    // Find the largest supported channel count that divides totalElems.
    const auto& strategyFactory = VPU::getVPUStrategyFactory(ctx);
    SmallVector<int64_t> supportedChannels(strategyFactory->getSupportedChannelsDW());
    int64_t C = 1;
    for (const int64_t candidate : supportedChannels) {
        if (candidate <= totalElems && totalElems % candidate == 0) {
            C = candidate;
            break;
        }
    }

    // Split the remaining elements into H and W (roughly equal): getFactorsList returns all
    // (larger, smaller) factor pairs of rem for smaller in [1, sqrt(rem)], in increasing order of
    // smaller, so the last entry is the pair closest to a square split.
    const auto rem = totalElems / C;
    const auto factors = getFactorsList(rem);
    VPUX_THROW_WHEN(factors.empty(), "Cannot factorize a zero-element scale/bias tensor '{0}'", scaleType);
    const auto factor = factors.back();
    const auto W = factor.first;
    const auto H = factor.second;

    std::array<int64_t, 4> poolShape = {1, C, H, W};

    // Reshape scale into [1, C, H, W] for the MaxPool.
    auto reshapedScale = rewriter.createOrFold<IE::ReshapeOp>(appendLoc(scale.getLoc(), "scale_reshape_in"), scale,
                                                              getIntArrayAttr(ctx, poolShape));

    // Identity MaxPool outputs fp32.
    const auto reshapedType = mlir::cast<vpux::NDTypeInterface>(reshapedScale.getType());
    const auto fp32PoolType = reshapedType.changeElemType(mlir::Float32Type::get(ctx));
    auto maxPoolOut = IE::createIdentityMaxPool(reshapedScale, fp32PoolType, rewriter)->getResult(0);

    std::array<int64_t, 4> wtScaleShape = {totalElems, 1, 1, 1};
    return rewriter.createOrFold<IE::ReshapeOp>(appendLoc(scale.getLoc(), "scale_reshape_out"), maxPoolOut,
                                                getIntArrayAttr(ctx, wtScaleShape));
}
