//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/dialect.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <string>
#include <utility>
#include <vector>

using namespace vpux;

TEST(MLIR_ElementTypeToString, CoversAllSupportedElementTypes) {
    mlir::DialectRegistry registry;
    registry.insert<vpux::type::QuantileDialect>();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::type::QuantileDialect>();

    // NF4 stores values as 4-bit codes, so the LUT must provide 2^4 = 16 entries for codes [0..15].
    const SmallVector<double> quantileLUT = {0.0, 1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                                             8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    const auto nf4Type = vpux::type::QuantileType::get(&ctx, getSInt8Type(&ctx), getUInt4Type(&ctx), quantileLUT);

    const std::vector<std::pair<mlir::Type, std::string>> testCases = {
            {mlir::Float64Type::get(&ctx), "f64"},
            {mlir::Float32Type::get(&ctx), "f32"},
            {mlir::Float16Type::get(&ctx), "f16"},
            {mlir::BFloat16Type::get(&ctx), "bf16"},
            {mlir::Float8E5M2Type::get(&ctx), "f8E5M2"},
            {mlir::Float8E4M3FNType::get(&ctx), "f8E4M3FN"},
            {mlir::IntegerType::get(&ctx, 8), "i8"},
            {mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Signed), "si64"},
            {mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed), "si32"},
            {mlir::IntegerType::get(&ctx, 16, mlir::IntegerType::Signed), "si16"},
            {mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed), "si8"},
            {mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::Signed), "si4"},
            {mlir::IntegerType::get(&ctx, 2, mlir::IntegerType::Signed), "si2"},
            {mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Unsigned), "ui64"},
            {mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Unsigned), "ui32"},
            {mlir::IntegerType::get(&ctx, 16, mlir::IntegerType::Unsigned), "ui16"},
            {mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Unsigned), "ui8"},
            {mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::Unsigned), "ui4"},
            {mlir::IntegerType::get(&ctx, 2, mlir::IntegerType::Unsigned), "ui2"},
            {mlir::IntegerType::get(&ctx, 1, mlir::IntegerType::Unsigned), "ui1"},
    };

    for (const auto& [type, expectedString] : testCases) {
        EXPECT_EQ(elementTypeToString(type), expectedString);
    }

    // For NF4 type, verify it prints with QuantileType signature
    const auto nf4TypeStr = elementTypeToString(nf4Type);
    EXPECT_TRUE(nf4TypeStr.find("QuantileType") != std::string::npos);

    // NoneType should print as "none"
    EXPECT_EQ(elementTypeToString(mlir::NoneType::get(&ctx)), "none");
}
