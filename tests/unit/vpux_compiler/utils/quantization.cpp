//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/utils/quantization.hpp"
#include "common/utils.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/type/float16.hpp"

#include <gtest/gtest.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <cstdint>

using namespace vpux;

using MLIR_QuantizationUtilsTest = MLIR_UnitBase;

void checkScalesAndZps(mlir::Type tiledType, ArrayRef<double> expectedScales, ArrayRef<int64_t> expectedZps) {
    auto perAxisQuant = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(tiledType);
    EXPECT_NE(perAxisQuant, nullptr);

    const auto scales = perAxisQuant.getScales();
    EXPECT_EQ(scales, expectedScales);

    const auto zps = perAxisQuant.getZeroPoints();
    EXPECT_EQ(zps, expectedZps);
}

void createSubchannelTypeWithDataType(mlir::MLIRContext& ctx, const mlir::Type& floatType,
                                      const mlir::IntegerType& intType) {
    ctx.loadDialect<mlir::quant::QuantDialect>();
    ctx.loadDialect<Const::ConstDialect>();

    vpux::SmallVector<int32_t> dims = {2, 3};
    vpux::SmallVector<int64_t> blockSizes = {8, 1};
    vpux::SmallVector<int64_t> scalesShape = {2, 4};
    SmallVector<mlir::Attribute, 8> scales;
    SmallVector<mlir::Attribute, 8> zeroPoints;

    for (auto idx : irange(scalesShape[0] * scalesShape[1])) {
        scales.push_back(mlir::FloatAttr::get(floatType, idx));
        zeroPoints.push_back(mlir::IntegerAttr::get(intType, idx));
    }

    auto denseAllScales = mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(scalesShape, floatType), scales);
    auto denseAllZeroPoints =
            mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(scalesShape, intType), zeroPoints);

    unsigned bitWidth = intType.getWidth();
    int64_t minValue = 0;
    int64_t maxValue = (1LL << bitWidth) - 1;

    const auto quantType = mlir::quant::UniformQuantizedSubChannelType::get(
            0, intType, floatType, denseAllScales, denseAllZeroPoints, dims, blockSizes, minValue, maxValue);

    auto test = getPerAxisTypeForBlock(quantType, 0);
    auto testBlock2 = getPerAxisTypeForBlock(quantType, 1);

    auto zeroPointsExpected1 = bitWidth == 2 ? SmallVector<int64_t>{0, 1, -2, -1} : SmallVector<int64_t>{0, 1, 2, 3};
    auto zeroPointsExpected2 = bitWidth == 2 ? SmallVector<int64_t>{0, 1, -2, -1} : SmallVector<int64_t>{4, 5, 6, 7};

    auto expectedPerAxisType = mlir::quant::UniformQuantizedPerAxisType::get(
            quantType.getFlags(), quantType.getStorageType(), quantType.getExpressedType(),
            ArrayRef<double>{0.0f, 1.0f, 2.0f, 3.0f}, zeroPointsExpected1, 3, quantType.getStorageTypeMin(),
            quantType.getStorageTypeMax());
    auto expectedPerAxisTypeBlock2 = mlir::quant::UniformQuantizedPerAxisType::get(
            quantType.getFlags(), quantType.getStorageType(), quantType.getExpressedType(),
            ArrayRef<double>{4.0f, 5.0f, 6.0f, 7.0f}, zeroPointsExpected2, 3, quantType.getStorageTypeMin(),
            quantType.getStorageTypeMax());

    EXPECT_EQ(test, expectedPerAxisType);
    EXPECT_EQ(testBlock2, expectedPerAxisTypeBlock2);
}
// TESTS FOR SUBCHANNEL --> PerAxisType
//  FP16 - UINT16
//  FP16 - INT8
//  FP16 - UINT4
//  FP16 - INT4
//  FP32 - INT2

TEST_F(MLIR_QuantizationUtilsTest, SubchannelTypeScalesFP16ZpUINT16) {
    mlir::MLIRContext ctx(registry);
    createSubchannelTypeWithDataType(ctx, mlir::Float16Type::get(&ctx),
                                     mlir::IntegerType::get(&ctx, 16, mlir::IntegerType::Unsigned));
}

TEST_F(MLIR_QuantizationUtilsTest, SubchannelTypeScalesFP16ZpINT8) {
    mlir::MLIRContext ctx(registry);
    createSubchannelTypeWithDataType(ctx, mlir::Float16Type::get(&ctx),
                                     mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed));
}

TEST_F(MLIR_QuantizationUtilsTest, SubchannelTypeScalesFP16ZpUINT4) {
    mlir::MLIRContext ctx(registry);
    createSubchannelTypeWithDataType(ctx, mlir::Float16Type::get(&ctx),
                                     mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::Unsigned));
}

TEST_F(MLIR_QuantizationUtilsTest, SubchannelTypeScalesFP16ZpINT4) {
    mlir::MLIRContext ctx(registry);
    createSubchannelTypeWithDataType(ctx, mlir::Float16Type::get(&ctx),
                                     mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::Signed));
}

TEST_F(MLIR_QuantizationUtilsTest, SubchannelTypeScalesFP32ZpINT2) {
    mlir::MLIRContext ctx(registry);
    createSubchannelTypeWithDataType(ctx, mlir::Float32Type::get(&ctx),
                                     mlir::IntegerType::get(&ctx, 2, mlir::IntegerType::Signed));
}

TEST_F(MLIR_QuantizationUtilsTest, TileScalesAndZp) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<mlir::quant::QuantDialect>();

    constexpr int64_t axisSize = 32;
    SmallVector<double> scales(axisSize, 0.01);
    SmallVector<int64_t> zeroPoints(axisSize, 1);

    for (auto idx : irange(axisSize)) {
        scales[idx] *= idx;      // scales = 0.01 0.02 0.03 etc.
        zeroPoints[idx] *= idx;  // zp = 1 2 3 4 5 etc.
    }

    const auto quantType = mlir::quant::UniformQuantizedPerAxisType::get(
            0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx), scales, zeroPoints, 0, 0, 255);

    {
        // Test case 0: tile contiguous section on quant axis
        const SmallVector<double> expectedScales = {0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24};
        const SmallVector<int64_t> expectedZPs = {15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

        const auto shape = Shape({10, 2, 3, 1});
        const auto offsets = Shape({15, 0, 0, 0});
        auto tiledTypeContiguous = tileScalesAndZP(quantType, shape, offsets);
        checkScalesAndZps(tiledTypeContiguous, expectedScales, expectedZPs);

        const auto strides = Shape({1, 1, 1, 1});
        tiledTypeContiguous = tileScalesAndZP(quantType, shape, offsets, strides);
        checkScalesAndZps(tiledTypeContiguous, expectedScales, expectedZPs);
    }

    {
        // Test case 1: tile strided section on quant axis
        const SmallVector<double> expectedScalesOdd = {0.15, 0.18, 0.21, 0.24, 0.27, 0.3};
        const SmallVector<int64_t> expectedZPsOdd = {15, 18, 21, 24, 27, 30};

        const auto shape = Shape({6, 2, 3, 1});
        const auto offsets = Shape({15, 0, 0, 0});
        const auto stridesOdd = Shape({3, 1, 1, 1});
        auto tiledTypeStridedOdd = tileScalesAndZP(quantType, shape, offsets, stridesOdd);
        checkScalesAndZps(tiledTypeStridedOdd, expectedScalesOdd, expectedZPsOdd);

        const SmallVector<double> expectedScalesEven = {0.15, 0.17, 0.19, 0.21, 0.23, 0.25};
        const SmallVector<int64_t> expectedZPsEven = {15, 17, 19, 21, 23, 25};

        const auto stridesEven = Shape({2, 1, 1, 1});
        auto tiledTypeStridedEven = tileScalesAndZP(quantType, shape, offsets, stridesEven);
        checkScalesAndZps(tiledTypeStridedEven, expectedScalesEven, expectedZPsEven);
    }

    {
        // Test case 2: stride axis is not quantization axis
        const SmallVector<double> expectedScales = {0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24};
        const SmallVector<int64_t> expectedZPs = {15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

        const auto shape = Shape({10, 2, 3, 1});
        const auto offsets = Shape({15, 0, 2, 0});
        const auto strides = Shape({1, 1, 2, 1});
        auto tiledTypeContiguous = tileScalesAndZP(quantType, shape, offsets, strides);
        checkScalesAndZps(tiledTypeContiguous, expectedScales, expectedZPs);
    }

    {
        // Test case 3: slice axis is not quantization axis
        const auto shape = Shape({32, 2, 3, 1});
        const auto offsets = Shape({0, 0, 2, 0});
        const auto strides = Shape({1, 1, 2, 1});
        auto noTilingTypeOnQuantAxisType = tileScalesAndZP(quantType, shape, offsets, strides);
        checkScalesAndZps(noTilingTypeOnQuantAxisType, scales, zeroPoints);
    }

    {
        // Test case 4: multiple tiles on quantization axis
        //   [Tile0]   [Tile1  ]
        // 0,[1,2,3],4,[5,6,7,8], ...
        SmallVector<int64_t> offsets = {1, 5};
        SmallVector<int64_t> sizes = {3, 4};
        const SmallVector<double> expectedScales = {0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08};
        const SmallVector<int64_t> expectedZPs = {1, 2, 3, 5, 6, 7, 8};
        auto multiTilingTypeOnQuantAxisType = tileScalesAndZP(quantType, offsets, sizes);
        checkScalesAndZps(multiTilingTypeOnQuantAxisType, expectedScales, expectedZPs);
    }
}

// A per-axis quantized constant weight with a zeroed-out channel (out_low == out_high == 0) gets scale = 1.0 from
// getQuantizedType (the guard used for the quantize direction to avoid division by zero). When the type is used to
// dequantize the constant, IE::keepZeroScaleForConstDequant must restore the true zero scale for that channel so it
// dequantizes to 0 (R = (Q - zp) * 0) instead of leaking the raw integer code that scale = 1.0 would produce.
TEST_F(MLIR_QuantizationUtilsTest, KeepZeroScaleForConstDequant) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<mlir::quant::QuantDialect>();
    ctx.loadDialect<Const::ConstDialect>();

    const auto f16Type = mlir::Float16Type::get(&ctx);
    const auto f32Type = mlir::Float32Type::get(&ctx);
    const auto i8Type = mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed);

    // Middle channel got the scale = 1.0 substitution because its output range is degenerate [0, 0].
    const SmallVector<double> scales = {5.000000e-01, 1.0, 2.500000e-01};
    const SmallVector<int64_t> zeroPoints = {0, 0, 0};
    const auto perAxisType = mlir::quant::UniformQuantizedPerAxisType::get(
            mlir::quant::QuantizationFlags::Signed, i8Type, f16Type, scales, zeroPoints, /*quantizedDimension=*/0,
            /*storageTypeMin=*/-128, /*storageTypeMax=*/127);

    // Per-channel output ranges; only the middle channel is degenerate [0, 0].
    const auto rangeType = mlir::RankedTensorType::get({3, 1}, f32Type);
    const SmallVector<float> lowData = {-1.000000e-01f, 0.000000e+00f, -8.000000e-02f};
    const SmallVector<float> highData = {1.000000e-01f, 0.000000e+00f, 8.000000e-02f};
    const auto outLow = Const::ContentAttr::get(Const::createConstContent(rangeType, ArrayRef<float>(lowData)));
    const auto outHigh = Const::ContentAttr::get(Const::createConstContent(rangeType, ArrayRef<float>(highData)));

    const auto corrected = IE::keepZeroScaleForConstDequant(perAxisType, outLow, outHigh, IE::AutoBroadcastType::NUMPY);

    auto correctedPerAxis = mlir::dyn_cast_if_present<mlir::quant::UniformQuantizedPerAxisType>(corrected);
    ASSERT_TRUE(correctedPerAxis);

    // Only the zeroed-out channel is rewritten to 0.0; the other channels are untouched.
    const SmallVector<double> expectedScales = {5.000000e-01, 0.0, 2.500000e-01};
    EXPECT_EQ(correctedPerAxis.getScales(), ArrayRef<double>(expectedScales));
    EXPECT_EQ(correctedPerAxis.getZeroPoints(), ArrayRef<int64_t>(zeroPoints));

    // Show the accuracy impact directly by dequantizing an arbitrary stored code in the zeroed-out channel.
    const double storedCode = 127.0;

    // Before correction (bug reproduced): the substituted scale 1.0 leaks the raw integer code as a weight.
    EXPECT_EQ(perAxisType.getScales()[1], 1.0);
    EXPECT_EQ((storedCode - perAxisType.getZeroPoints()[1]) * perAxisType.getScales()[1], storedCode);

    // After correction: the true zero scale makes the channel dequantize to 0.
    EXPECT_EQ(correctedPerAxis.getScales()[1], 0.0);
    EXPECT_EQ((storedCode - correctedPerAxis.getZeroPoints()[1]) * correctedPerAxis.getScales()[1], 0.0);
}
