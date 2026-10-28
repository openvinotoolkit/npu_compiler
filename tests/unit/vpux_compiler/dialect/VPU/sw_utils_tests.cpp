//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"

#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_VPU_SWUtils = vpux::VPU::arch40xx::UnitTest;

TEST_F(MLIR_VPU_SWUtils, GetSWOpAlignmentUsesBroadcastZpShape) {
    constexpr llvm::StringLiteral IR = R"(
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>
module @test attributes {config.compilationMode = #config.compilation_mode<DefaultHW>,
                          config.platform = #config.platform<NPU4000>} {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.ExecutorResource 2 of @SHAVE_ACT
        config.ExecutorResource 1 of @DPU
    }
        func.func @main(%arg0: tensor<1x1x4x2x!qElemType>,
                %arg1: tensor<1x1x4x2xf16>,
                %arg2: tensor<1x1x4x1xsi4>) -> tensor<1x1x4x2xf16> {
        %0 = VPU.DynamicDequantize(%arg0, %arg1, %arg2) {dstElemType = f16}
            : tensor<1x1x4x2x!qElemType>, tensor<1x1x4x2xf16>, tensor<1x1x4x1xsi4>
            -> tensor<1x1x4x2xf16>
        return %0 : tensor<1x1x4x2xf16>
    }
}
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    VPU::DynamicDequantizeOp dequantizeOp = nullptr;
    module->walk([&](VPU::DynamicDequantizeOp op) {
        dequantizeOp = op;
    });
    ASSERT_TRUE(dequantizeOp != nullptr);

    const auto divisors = Shape({1, 1, 4, 1});
    const auto alignment = VPU::getSWOpAlignment(dequantizeOp.getOperation(), divisors);

    ASSERT_TRUE(alignment.has_value());
    EXPECT_EQ(alignment.value(), SmallVector<int64_t>({1, 1, 2, 1}));
}
