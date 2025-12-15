//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/utils/quantization.hpp"
#include "common/utils.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

#include <gtest/gtest.h>
#include <mlir/Dialect/Quant/IR/Quant.h>

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
