//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <cstdint>

using namespace vpux;
using MLIR_RetrieveScaleFromWeightTableUnitTest = MLIR_UnitBase;

void compareScaleRoundtrip(mlir::MLIRContext* ctx, double scale, config::ArchKind arch, bool isNewWeightsTable) {
    constexpr double approximationErr{0.0000001};
    constexpr int64_t bitWidth = 8;
    auto scaleConverter = VPU::NCESparsity::getPPEConverterCb(arch, isNewWeightsTable);
    auto scaleRetriever = VPU::NCESparsity::getScaleRetrieveCb(arch, isNewWeightsTable);

    if (arch <= config::ArchKind::NPU40XX) {
        auto floatType = getFp16Type(ctx);
        // Dummy quantized type, to pass to the scaleConverter
        const auto storageMin = mlir::quant::QuantizedType::getDefaultMinimumForInteger(true, bitWidth);
        const auto storageMax = mlir::quant::QuantizedType::getDefaultMaximumForInteger(true, bitWidth);
        const auto storageType = mlir::IntegerType::get(ctx, bitWidth, mlir::IntegerType::Signed);
        auto quantType = mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, storageType,
                                                                floatType, 1.2, 0, storageMin, storageMax);

        auto quantApprox = QuantizationApproximation(scale);

        auto convertRes = scaleConverter(checked_cast<uint8_t>(quantApprox.shift()),
                                         checked_cast<int16_t>(quantApprox.mult()), scale, floatType);
        const auto floatRes = scaleRetriever(convertRes, floatType);
        EXPECT_NEAR(scale, floatRes, approximationErr);

        convertRes = scaleConverter(checked_cast<uint8_t>(quantApprox.shift()),
                                    checked_cast<int16_t>(quantApprox.mult()), scale, quantType);
        const auto quantRes = scaleRetriever(convertRes, quantType);
        EXPECT_NEAR(scale, quantRes, approximationErr);
        return;
    }

    if (arch <= config::ArchKind::NPU60XX && !isNewWeightsTable) {
        auto convertRes = scaleConverter(0, 1, scale, nullptr);
        const auto floatRes = scaleRetriever(convertRes, nullptr);
        EXPECT_FLOAT_EQ(scale, floatRes);
        return;
    }

    if (arch >= config::ArchKind::NPU60XX && isNewWeightsTable) {
        mlir::Type scalesType = mlir::Float32Type::get(ctx);
        auto convertRes = scaleConverter(0, 1, scale, scalesType);
        const auto floatRes = scaleRetriever(convertRes, scalesType);
        EXPECT_FLOAT_EQ(scale, floatRes);

        scalesType = mlir::Float8E4M3FNType::get(ctx);
        convertRes = scaleConverter(0, 1, scale, scalesType);
        const auto fp8e4m3Res = scaleRetriever(convertRes, scalesType);
        EXPECT_FLOAT_EQ(scale, fp8e4m3Res);

        scalesType = mlir::Float8E5M2Type::get(ctx);
        convertRes = scaleConverter(0, 1, scale, scalesType);
        const auto fp8e5m2Res = scaleRetriever(convertRes, scalesType);
        EXPECT_FLOAT_EQ(scale, fp8e5m2Res);
    }

    ASSERT_FALSE(arch > config::ArchKind::NPU60XX && !isNewWeightsTable);
}

TEST_F(MLIR_RetrieveScaleFromWeightTableUnitTest, retrieveScaleFromTable) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<mlir::quant::QuantDialect>();

    compareScaleRoundtrip(&ctx, 2.5, config::ArchKind::NPU37XX, false);
    compareScaleRoundtrip(&ctx, -0.000625, config::ArchKind::NPU40XX, false);
    compareScaleRoundtrip(&ctx, 1.025, config::ArchKind::NPU50XX, false);

    compareScaleRoundtrip(&ctx, -3.12, config::ArchKind::NPU60XX, false);
    compareScaleRoundtrip(&ctx, 0.5, config::ArchKind::NPU60XX, true);
}
