//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_types/types.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/kernel_data_type.hpp"

#include <gtest/gtest.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

using namespace vpux;

namespace {

class MLIR_KernelDataTypeTest : public testing::Test {
protected:
    MLIR_KernelDataTypeTest() {
        ctx.loadDialect<mlir::quant::QuantDialect>();
    }

    mlir::MLIRContext ctx;
};

}  // namespace

TEST_F(MLIR_KernelDataTypeTest, ConvertFP32ToFP16IsShaveOnlyBeforeNPU40XX) {
    const auto fp32 = mlir::Float32Type::get(&ctx);
    const auto fp16 = mlir::Float16Type::get(&ctx);
    const auto i4 = mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::Signed);
    const auto i8 = mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed);
    const auto u8 = mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Unsigned);
    const auto i32 = mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
    const auto u32 = mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Unsigned);
    const auto i16 = mlir::IntegerType::get(&ctx, 16, mlir::IntegerType::Signed);
    const auto i64 = mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Signed);
    const auto fp64 = mlir::Float64Type::get(&ctx);

    EXPECT_TRUE(isConvertConfigSupported(fp32, fp16, config::ArchKind::NPU37XX));
    EXPECT_FALSE(isConvertConfigSupported(fp32, fp16, config::ArchKind::NPU40XX));
    EXPECT_FALSE(isConvertConfigSupported(fp32, fp16, config::ArchKind::NPU50XX));
    EXPECT_TRUE(isConvertConfigSupported(fp16, fp32, config::ArchKind::NPU40XX));
    EXPECT_TRUE(isConvertConfigSupported(i64, fp64, config::ArchKind::NPU40XX));
    EXPECT_FALSE(isConvertConfigSupported(fp32, i16, config::ArchKind::NPU40XX));
    EXPECT_FALSE(isConvertConfigSupported(fp16, i4, config::ArchKind::NPU40XX));

    EXPECT_TRUE(isConvertConfigSupported(u8, u32, config::ArchKind::NPU40XX));
    EXPECT_TRUE(isConvertConfigSupported(i8, u32, config::ArchKind::NPU40XX));
    EXPECT_TRUE(isConvertConfigSupported(fp32, u32, config::ArchKind::NPU40XX));
    EXPECT_TRUE(isConvertConfigSupported(fp16, u32, config::ArchKind::NPU40XX));
    EXPECT_TRUE(isConvertConfigSupported(i64, u32, config::ArchKind::NPU40XX));
    EXPECT_TRUE(isConvertConfigSupported(u32, u8, config::ArchKind::NPU40XX));
    EXPECT_TRUE(isConvertConfigSupported(u32, i8, config::ArchKind::NPU40XX));
    EXPECT_TRUE(isConvertConfigSupported(u32, i32, config::ArchKind::NPU40XX));
    EXPECT_TRUE(isConvertConfigSupported(u32, fp16, config::ArchKind::NPU40XX));
    EXPECT_TRUE(isConvertConfigSupported(u32, fp32, config::ArchKind::NPU40XX));
    EXPECT_TRUE(isConvertConfigSupported(u32, i64, config::ArchKind::NPU40XX));
}

TEST_F(MLIR_KernelDataTypeTest, QuantizeSupportedConfigs) {
    const auto fp32 = mlir::Float32Type::get(&ctx);
    const auto fp16 = mlir::Float16Type::get(&ctx);
    const auto u8 = mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Unsigned);
    const auto i8 = mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed);
    const auto u16 = mlir::IntegerType::get(&ctx, 16, mlir::IntegerType::Unsigned);
    const auto i16 = mlir::IntegerType::get(&ctx, 16, mlir::IntegerType::Signed);
    const auto u4 = mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::Unsigned);
    const auto i4 = mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::Signed);

    // FP32 -> U8 is supported on all arches; FP32 -> I8 is not supported.
    EXPECT_TRUE(isQuantizeConfigSupported(fp32, u8, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isQuantizeConfigSupported(fp32, u8, config::ArchKind::NPU40XX));
    EXPECT_FALSE(isQuantizeConfigSupported(fp32, i8, config::ArchKind::NPU40XX));

    // FP16 -> U8/I8/U16/I16 supported on all arches.
    EXPECT_TRUE(isQuantizeConfigSupported(fp16, u8, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isQuantizeConfigSupported(fp16, i8, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isQuantizeConfigSupported(fp16, u16, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isQuantizeConfigSupported(fp16, i16, config::ArchKind::NPU37XX));

    // FP16 -> U4/I4 requires NPU40XX+ and cannot be innermost axis (quantDim == 0).
    EXPECT_FALSE(isQuantizeConfigSupported(fp16, u4, config::ArchKind::NPU37XX));
    EXPECT_FALSE(isQuantizeConfigSupported(fp16, i4, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isQuantizeConfigSupported(fp16, u4, config::ArchKind::NPU40XX));
    EXPECT_TRUE(isQuantizeConfigSupported(fp16, i4, config::ArchKind::NPU40XX));
    EXPECT_FALSE(isQuantizeConfigSupported(fp16, u4, config::ArchKind::NPU40XX, /*quantDim=*/0));
    EXPECT_FALSE(isQuantizeConfigSupported(fp16, i4, config::ArchKind::NPU40XX, /*quantDim=*/0));
    EXPECT_TRUE(isQuantizeConfigSupported(fp16, u4, config::ArchKind::NPU40XX, /*quantDim=*/1));
    EXPECT_TRUE(isQuantizeConfigSupported(fp16, i4, config::ArchKind::NPU40XX, /*quantDim=*/1));
}

TEST_F(MLIR_KernelDataTypeTest, DequantizeSupportedConfigs) {
    const auto fp32 = mlir::Float32Type::get(&ctx);
    const auto fp16 = mlir::Float16Type::get(&ctx);
    const auto u8 = mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Unsigned);
    const auto i8 = mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed);
    const auto u16 = mlir::IntegerType::get(&ctx, 16, mlir::IntegerType::Unsigned);
    const auto i16 = mlir::IntegerType::get(&ctx, 16, mlir::IntegerType::Signed);
    const auto u4 = mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::Unsigned);
    const auto i4 = mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::Signed);
    const auto bf8 = mlir::Float8E5M2Type::get(&ctx);
    const auto hf8 = mlir::Float8E4M3FNType::get(&ctx);

    // U8 -> FP32 supported on all arches.
    EXPECT_TRUE(isDequantizeConfigSupported(u8, fp32, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isDequantizeConfigSupported(u8, fp32, config::ArchKind::NPU40XX));

    // U8/I8/U16/I16 -> FP16 supported on all arches.
    EXPECT_TRUE(isDequantizeConfigSupported(u8, fp16, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isDequantizeConfigSupported(i8, fp16, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isDequantizeConfigSupported(u16, fp16, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isDequantizeConfigSupported(i16, fp16, config::ArchKind::NPU37XX));

    // I4/U4 -> FP16 requires NPU40XX+.
    EXPECT_FALSE(isDequantizeConfigSupported(i4, fp16, config::ArchKind::NPU37XX));
    EXPECT_FALSE(isDequantizeConfigSupported(u4, fp16, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isDequantizeConfigSupported(i4, fp16, config::ArchKind::NPU40XX));
    EXPECT_TRUE(isDequantizeConfigSupported(u4, fp16, config::ArchKind::NPU40XX));

    // BF8/HF8 -> FP16 requires NPU50XX+.
    EXPECT_FALSE(isDequantizeConfigSupported(bf8, fp16, config::ArchKind::NPU40XX));
    EXPECT_FALSE(isDequantizeConfigSupported(hf8, fp16, config::ArchKind::NPU40XX));
    EXPECT_TRUE(isDequantizeConfigSupported(bf8, fp16, config::ArchKind::NPU50XX));
    EXPECT_TRUE(isDequantizeConfigSupported(hf8, fp16, config::ArchKind::NPU50XX));
}

TEST_F(MLIR_KernelDataTypeTest, DynamicDequantizeSupportedConfigs) {
    const auto fp32 = mlir::Float32Type::get(&ctx);
    const auto fp16 = mlir::Float16Type::get(&ctx);
    const auto u4 = mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::Unsigned);
    const auto i4 = mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::Signed);
    const auto i8 = mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed);
    const auto u8 = mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Unsigned);
    const auto u2 = mlir::IntegerType::get(&ctx, 2, mlir::IntegerType::Unsigned);
    const auto u16 = mlir::IntegerType::get(&ctx, 16, mlir::IntegerType::Unsigned);
    const auto nf4 = vpux::type::NF4Type::get(&ctx, u4, fp16);

    // I4/I8/U8/NF4 -> FP16 supported on all arches.
    EXPECT_TRUE(isDynamicDequantizeConfigSupported(i4, fp16, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isDynamicDequantizeConfigSupported(i8, fp16, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isDynamicDequantizeConfigSupported(u8, fp16, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isDynamicDequantizeConfigSupported(nf4, fp16, config::ArchKind::NPU37XX));

    // U16 -> FP16 has a vectorized path on NPU40XX+ and a generic per-element fallback below it,
    // but neither handles an explicit zero-point: the kernel would read it as U16.
    EXPECT_TRUE(isDynamicDequantizeConfigSupported(u16, fp16, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isDynamicDequantizeConfigSupported(u16, fp16, config::ArchKind::NPU40XX));
    EXPECT_FALSE(isDynamicDequantizeConfigSupported(u16, fp16, config::ArchKind::NPU40XX, u16));
    EXPECT_FALSE(isDynamicDequantizeConfigSupported(u16, fp16, config::ArchKind::NPU40XX, i8));

    // I4/I8/U8 -> FP32 supported on all arches.
    EXPECT_TRUE(isDynamicDequantizeConfigSupported(i4, fp32, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isDynamicDequantizeConfigSupported(i8, fp32, config::ArchKind::NPU37XX));
    EXPECT_TRUE(isDynamicDequantizeConfigSupported(u8, fp32, config::ArchKind::NPU37XX));

    // NF4/U16 -> FP32 not supported.
    EXPECT_FALSE(isDynamicDequantizeConfigSupported(nf4, fp32, config::ArchKind::NPU40XX));
    EXPECT_FALSE(isDynamicDequantizeConfigSupported(u16, fp32, config::ArchKind::NPU40XX));

    // U2 -> FP16 requires NPU40XX+ and mandates a U2 zero-point operand.
    EXPECT_TRUE(isDynamicDequantizeConfigSupported(u2, fp16, config::ArchKind::NPU40XX, u2));
    EXPECT_FALSE(isDynamicDequantizeConfigSupported(u2, fp16, config::ArchKind::NPU40XX));
    EXPECT_FALSE(isDynamicDequantizeConfigSupported(u2, fp16, config::ArchKind::NPU40XX, i8));
    EXPECT_FALSE(isDynamicDequantizeConfigSupported(u2, fp16, config::ArchKind::NPU37XX, u2));
}
