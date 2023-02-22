//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/sparsity_strategy.hpp"
#include "vpux/compiler/dialect/VPU/types.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/init.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <gtest/gtest.h>
#include <mlir/IR/MLIRContext.h>
#include <memory>

using namespace vpux;
using namespace VPU::NCESparsity;

namespace {

mlir::Operation* createConstOp(mlir::MLIRContext* context, const std::initializer_list<int64_t>& rawShape,
                               double ratio) {
    const Shape shape(rawShape);
    const auto dataType = mlir::RankedTensorType::get(shape.raw(), mlir::Float32Type::get(context));

    std::vector<float> content(shape.totalSize(), 1);
    const auto numSparse = static_cast<int64_t>(shape.totalSize() * ratio);
    std::fill_n(content.begin(), numSparse, 0);

    const auto dataAttr = mlir::DenseElementsAttr::get(dataType, mlir::ArrayRef<float>(content));

    mlir::OpBuilder builder(context);
    return builder.create<Const::DeclareOp>(mlir::UnknownLoc::get(context), dataType,
                                            Const::ContentAttr::get(dataAttr));
}

std::unique_ptr<BaseWeightsSparsityStrategy> getRatioBasedStrategy(
        double floatRatio = WEIGHTS_SPARSITY_FLOAT_RATIO_THRESHOLD,
        double intRatio = WEIGHTS_SPARSITY_INT_RATIO_THRESHOLD) {
    return std::unique_ptr<BaseWeightsSparsityStrategy>(new RatioBasedWeightsSparsityStrategy(floatRatio, intRatio));
}

std::unique_ptr<BaseWeightsSparsityStrategy> getCMXBasedStrategy(int64_t cmxSizeBytes) {
    vpux::Byte cmxSize(cmxSizeBytes);
    return std::unique_ptr<BaseWeightsSparsityStrategy>(
            new CMXConsumptionBasedWeightsSparsityStrategy(cmxSize, CMX_BASED_STRATEGY_DEFAULT_INTERVALS));
}

struct ThresholdPair {
    ThresholdPair(double reference)
            : _lowerThreshold(std::max(0., reference / 2.)), _upperThreshold(std::min(1., (1. + reference) / 2.)) {
    }
    double _lowerThreshold;
    double _upperThreshold;
};

};  // namespace

void ratioBasedStrategyTestTemplate(double threshold, bool isFloat) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    ctx.loadDialect<Const::ConstDialect>();

    Logger log("weights-sparsity-test", LogLevel::Debug);

    const auto defaultFloatRatioThreshold = ThresholdPair(threshold);

    const auto lowSparsity =
            createConstOp(&ctx, {1, 64, 32, 32}, defaultFloatRatioThreshold._lowerThreshold)->getResult(0);
    const auto highSparsity =
            createConstOp(&ctx, {1, 64, 32, 32}, defaultFloatRatioThreshold._upperThreshold)->getResult(0);

    // LOWER_FLOAT_THRESHOLD < threshold -> no sparsity
    EXPECT_EQ(getRatioBasedStrategy()->shouldSparsifyWeights(log, lowSparsity, isFloat), false);
    // UPPER_FLOAT_THRESHOLD > threshold -> enable sparsity
    EXPECT_EQ(getRatioBasedStrategy()->shouldSparsifyWeights(log, highSparsity, isFloat), true);
}

TEST(MLIR_WeightsSparsity, RatioBasedStrategyFloatInput) {
    ratioBasedStrategyTestTemplate(WEIGHTS_SPARSITY_FLOAT_RATIO_THRESHOLD, true);
}

TEST(MLIR_WeightsSparsity, RatioBasedStrategyIntInput) {
    ratioBasedStrategyTestTemplate(WEIGHTS_SPARSITY_INT_RATIO_THRESHOLD, false);
}

TEST(MLIR_WeightsSparsity, CMXBasedStrategy) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    ctx.loadDialect<Const::ConstDialect>();

    Logger log("weights-sparsity-test", LogLevel::Debug);

    const auto firstInterval = *CMX_BASED_STRATEGY_DEFAULT_INTERVALS.begin();
    VPUX_THROW_WHEN(CMX_BASED_STRATEGY_DEFAULT_INTERVALS.size() < 2 ||
                            firstInterval._floatRatioThreshold != CMXBasedSparsityThreshold::DISABLED_SPARSITY_RATIO,
                    "CMX_BASED_STRATEGY_DEFAULT_INTERVALS should contain at least 2 intervals and first of them should "
                    "disable sparsity");

    const auto lightWeightAlmostDense = createConstOp(&ctx, {1, 1, 32, 32}, 0.01)->getResult(0);
    const auto lightWeightAlmostSparse = createConstOp(&ctx, {1, 1, 32, 32}, 0.99)->getResult(0);
    const auto bigCMX = static_cast<int64_t>(4. * 32. * 32. / (firstInterval._cmxSizeRatio + 0.0001) * 2.);

    // No sparsity regardless to content in first interval
    EXPECT_EQ(getCMXBasedStrategy(bigCMX)->shouldSparsifyWeights(log, lightWeightAlmostDense, true), false);
    EXPECT_EQ(getCMXBasedStrategy(bigCMX)->shouldSparsifyWeights(log, lightWeightAlmostSparse, true), false);

    auto it = CMX_BASED_STRATEGY_DEFAULT_INTERVALS.begin();
    const auto end = CMX_BASED_STRATEGY_DEFAULT_INTERVALS.end();
    for (++it; it != end;) {
        const auto sparsityThreshold = ThresholdPair(it->_intRatioThreshold);
        const double lowerCMX = it->_cmxSizeRatio;
        const double upperCMX = ++it == end ? 1. : it->_cmxSizeRatio;
        const double targetCMXRatio = (lowerCMX + upperCMX) / 2.;
        // Generating CMX size in such way, to get in just in the middle of intervals, for example
        // 5-50% => weights should have 22% of CMX size
        const auto cmxSize = static_cast<int64_t>(4 * 32. * 32. / targetCMXRatio);

        const auto denseOp = createConstOp(&ctx, {1, 1, 32, 32}, sparsityThreshold._lowerThreshold)->getResult(0);
        EXPECT_EQ(getCMXBasedStrategy(cmxSize)->shouldSparsifyWeights(log, denseOp, false), false);
        const auto sparseOp = createConstOp(&ctx, {1, 1, 32, 32}, sparsityThreshold._upperThreshold)->getResult(0);
        EXPECT_EQ(getCMXBasedStrategy(cmxSize)->shouldSparsifyWeights(log, sparseOp, false), true);
    }
}
