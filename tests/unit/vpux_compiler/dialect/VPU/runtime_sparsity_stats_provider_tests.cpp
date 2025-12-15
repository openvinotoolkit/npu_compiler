//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using vpux::config::ArchKind;
using namespace vpux;

using MLIR_VPU_RT_SPARSITY_STATS_PROVIDER = vpux::VPU::arch37xx::UnitTest;

TEST_F(MLIR_VPU_RT_SPARSITY_STATS_PROVIDER, MissedStats) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
    module @main {
        func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
            %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [16, 16, 1, 1],
                    strides = [1, 1]
                } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_1", "t_Convolution"])

            return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>
        }
    }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(ArchKind::NPU37XX, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto logger = vpux::Logger::global();
    auto rtStatsProvider = VPU::NCESparsity::RuntimeSparsityStatsProvider(func, logger);
    func->walk([&](VPU::NCEConvolutionOp convOp) {
        ASSERT_FALSE(rtStatsProvider.likelySparsityConsumer(convOp, 0));
        ASSERT_FALSE(rtStatsProvider.likelySparsityConsumer(convOp, 1));
    });
}

TEST_F(MLIR_VPU_RT_SPARSITY_STATS_PROVIDER, WithStats) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
    module @main {
        func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_100", "t_Convolution"])
        %2 = VPU.NCE.Eltwise(%1, %1) {
                    op_type = #VPU.eltwise_type<ADD>,
                    ppe = #VPU.PPEStub<>
                } -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Add_1", "t_Convolution"])
        %3 = VPU.MaxPool(%1) {
            kernel_size = [3, 3],
            pads_begin = [1, 1],
            ppe = #VPU.PPEStub<>,
            pads_end = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Maxpool_1", "t_Convolution", "fused"])

        return %2, %3 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>
    }
    net.SparsityStatistics sparsityInfo : {
        net.SparsityInfo 0.8 at input 0 of "Conv_1" loc(#loc0)
        net.SparsityInfo 0.05 at input 0 of "Add_1" loc(#loc0)
        net.SparsityInfo 0.5 at input 1 of "Add_1" loc(#loc0)
        net.SparsityInfo 0.33 at input 0 of "Maxpool_1" loc(#loc0)
    }

    }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(ArchKind::NPU37XX, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto logger = vpux::Logger::global();
    auto rtStatsProvider = VPU::NCESparsity::RuntimeSparsityStatsProvider(func, logger);

    // Location of op Conv_100, while stats provided for Conv_1
    func->walk([&](VPU::NCEConvolutionOp convOp) {
        ASSERT_FALSE(rtStatsProvider.likelySparsityConsumer(convOp, 0));
        ASSERT_FALSE(rtStatsProvider.likelySparsityConsumer(convOp, 1));
    });
    // Different sparsity ratio for inputs
    func->walk([&](VPU::NCEEltwiseOp eltOp) {
        ASSERT_FALSE(rtStatsProvider.likelySparsityConsumer(eltOp, 0));
        ASSERT_TRUE(rtStatsProvider.likelySparsityConsumer(eltOp, 1));
    });
    // Extra suffix shouldn't prevent layer from matching
    func->walk([&](VPU::MaxPoolOp poolOp) {
        ASSERT_TRUE(rtStatsProvider.likelySparsityConsumer(poolOp, 0));
    });
}
