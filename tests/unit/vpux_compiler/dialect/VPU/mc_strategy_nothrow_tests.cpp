//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_manager.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_VPU_ClusteringStrategyNoThrow = vpux::VPU::arch40xx::UnitTest;

TEST_F(MLIR_VPU_ClusteringStrategyNoThrow, SWLayer_ClusteringStrategy) {
    constexpr llvm::StringLiteral inputIR = R"(
#loc0 = loc(unknown)
    module @main attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
        IE.TileResource 6 of @NCE at 1.700000e+03 MHz {
            IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
            IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
            IE.ExecutorResource 2 of @SHAVE_ACT
            IE.ExecutorResource 1 of @DPU
        }
        IE.ExecutorResource 1 of @M2I
        IE.ExecutorResource 2 of @DMA_NN
        IE.MemoryResource 2306867200 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
        func.func @main(%softmax_in: tensor<1x8x4x76xf16>,
                        %power_f16_in: tensor<1x16x256x256xf16>,
                        %power_f16_pow: tensor<1x16x1x1xf16>,
                        %power_f32_in: tensor<1x16x256x256xf32>,
                        %power_f32_pow: tensor<1x16x1x1xf32>,
                        %power_u16_in: tensor<1x16x256x256xui16>,
                        %power_u16_pow: tensor<1x16x1x1xui16>,
                        %power_u32_in: tensor<1x16x256x256xui32>,
                        %power_u32_pow: tensor<1x16x1x1xui32>,
                        %power_i32_in: tensor<1x16x256x256xsi32>,
                        %power_i32_pow: tensor<1x16x1x1xsi32>,
                        %mvn_f16_in: tensor<1x32x15x64xf16>,
                        %mvn_f32_in: tensor<1x32x15x64xf32>,
                        %div_f16_0: tensor<1x16x256x256xf16>,
                        %div_f16_1: tensor<1x16x1x1xf16>,
                        %div_f32_0: tensor<1x16x256x256xf32>,
                        %div_f32_1: tensor<1x16x1x1xf32>,
                        %div_u8_0: tensor<1x16x256x256xui8>,
                        %div_u8_1: tensor<1x16x1x1xui8>) ->
                        (tensor<1x8x4x76xf16>,
                        tensor<1x16x256x256xf16>,
                        tensor<1x16x256x256xf32>,
                        tensor<1x16x256x256xui16>,
                        tensor<1x16x256x256xui32>,
                        tensor<1x16x256x256xsi32>,
                        tensor<1x32x15x64xf16>,
                        tensor<1x32x15x64xf32>,
                        tensor<1x16x256x256xf16>,
                        tensor<1x16x256x256xf32>,
                        tensor<1x16x256x256xui8>){
        %softmax = VPU.SoftMax(%softmax_in) {axisInd = 3} : tensor<1x8x4x76xf16> -> tensor<1x8x4x76xf16>
        %pow_f16 = VPU.Power(%power_f16_in, %power_f16_pow) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x256xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x256x256xf16>
        %pow_f32 = VPU.Power(%power_f32_in, %power_f32_pow) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x256xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x256x256xf32>
        %pow_u16 = VPU.Power(%power_u16_in, %power_u16_pow) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x256xui16>, tensor<1x16x1x1xui16> -> tensor<1x16x256x256xui16>
        %pow_u32 = VPU.Power(%power_u32_in, %power_u32_pow) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x256xui32>, tensor<1x16x1x1xui32> -> tensor<1x16x256x256xui32>
        %pow_i32 = VPU.Power(%power_i32_in, %power_i32_pow) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x256xsi32>, tensor<1x16x1x1xsi32> -> tensor<1x16x256x256xsi32>
        %mvn_f16 = VPU.MVN6(%mvn_f16_in) {axes = [1, 3], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
        %mvn_f32 = VPU.MVN6(%mvn_f32_in) {axes = [1, 3], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf32> -> tensor<1x32x15x64xf32>
        %div_f16 = VPU.Divide(%div_f16_0, %div_f16_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x256xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x256x256xf16>
        %div_f32 = VPU.Divide(%div_f32_0, %div_f32_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x256xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x256x256xf32>
        %div_u8  = VPU.Divide(%div_u8_0, %div_u8_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x256xui8>, tensor<1x16x1x1xui8> -> tensor<1x16x256x256xui8>
        return %softmax, %pow_f16, %pow_f32, %pow_u16, %pow_u32, %pow_i32, %mvn_f16, %mvn_f32, %div_f16, %div_f32, %div_u8 :
                                                                        tensor<1x8x4x76xf16>,
                                                                        tensor<1x16x256x256xf16>,
                                                                        tensor<1x16x256x256xf32>,
                                                                        tensor<1x16x256x256xui16>,
                                                                        tensor<1x16x256x256xui32>,
                                                                        tensor<1x16x256x256xsi32>,
                                                                        tensor<1x32x15x64xf16>,
                                                                        tensor<1x32x15x64xf32>,
                                                                        tensor<1x16x256x256xf16>,
                                                                        tensor<1x16x256x256xf32>,
                                                                        tensor<1x16x256x256xui8>
    }
    }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto tileOp = IE::getTileExecutor(module.get());
    VPUX_THROW_UNLESS(tileOp != nullptr, "Failed to get NCE_Cluster information");
    VPUX_THROW_UNLESS(tileOp.getCount() > 1, "Cannot assign multi-cluster strategy to single-cluster module ops");
    bool enablePrefetchTiling = true;

    auto siblingsOpsAnalysis = vpux::VPU::SiblingOpsAnalysis(func);
    vpux::VPU::StrategyManager strategyManager(func, tileOp.getCount(), enablePrefetchTiling,
                                               VPU::MCOptimizationScope::SUBGRAPH, vpux::Logger::global(),
                                               siblingsOpsAnalysis);
    EXPECT_NO_THROW(strategyManager.assignMultiClusterStrategy(true));
}
