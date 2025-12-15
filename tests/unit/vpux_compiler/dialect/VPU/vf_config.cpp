//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using vpux::config::ArchKind;
using namespace vpux;

using MLIR_VPU_VFConfig = vpux::VPU::arch37xx::UnitTest;

TEST_F(MLIR_VPU_VFConfig, VF_ConfigSimple) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
    module @main {
       func.func @main(%arg0: tensor<1x48x256x16xf16, {order = #NHWC}>) -> tensor<1x1024x256x16xf16, {order = #NHWC}> {
            %cst = const.Declare tensor<1024x48x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1024x48x1x1xf16>, [#const.Reorder<#NHWC>]
            %cst_0 = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>

            %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x48x256x16xf16, {order = #NHWC}>, %cst as %arg3: tensor<1024x48x1x1xf16, {order = #NHWC}>,
            %cst_0 as %arg4: tensor<1024x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 5, 1]}
                -> tensor<1x1024x256x16xf16, {order = #NHWC}> {
            %1 = VPU.NCE.Convolution(%arg2, %arg3, %arg4)
                {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [1024, 48, 1, 1], strides = [1, 1]}
                : tensor<1x48x256x16xf16, {order = #NHWC}>, tensor<1024x48x1x1xf16, {order = #NHWC}>, tensor<1024x1x1x4xsi32> -> tensor<1x1024x256x16xf16, {order = #NHWC}>
            %2 = VPU.SoftMax(%1)
                {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x256x16xf16, {order = #NHWC}>
                -> tensor<1x1024x256x16xf16, {order = #NHWC}>
                VPU.Yield %2
            }
            return %0 : tensor<1x1024x256x16xf16, {order = #NHWC}>
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

    func->walk([&](VPU::VerticalFusionOp vfOp) {
        auto config = VPU::VF::v1::VFConfig(vfOp);
        EXPECT_EQ(config.getVFOperations().size(), 2);
        EXPECT_EQ(config.getInputs().size(), 1);
        EXPECT_EQ(config.getOutputs().size(), 1);
        EXPECT_TRUE(mlir::isa<VPU::SoftMaxOp>(config.getLargestOp()));
        EXPECT_EQ(config.getSubgraph(), vfOp);
        EXPECT_FALSE(config.isPipelined());
    });
}

TEST_F(MLIR_VPU_VFConfig, VF_ConfigPipelined) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
    module @main {
        func.func @main(%arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>,
        %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>, %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
            %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>
            %cst_2 = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

            %0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x1024x4xf16, {order = #NHWC}>, %arg1 as %arg4: tensor<4096x48x1x1xf16, {order = #NHWC}>, %cst_0 as %arg5: tensor<4096x1x1x4xsi32>, %arg1 as %arg6: tensor<48x4096x1x1xf16, {order = #NHWC}>, %cst_2 as %arg7: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 24, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}>
            { %1 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
                {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>, tensor<4096x1x1x4xsi32> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
                %2 = VPU.SoftMax(%1) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
                %3 = VPU.NCE.Convolution(%2, %arg6, %arg7) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
                VPU.Yield %3
            }

            return %0 : tensor<1x48x1024x4xf16, {order = #NHWC}>
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

    func->walk([&](VPU::VerticalFusionOp vfOp) {
        auto config = VPU::VF::v1::VFConfig(vfOp);
        EXPECT_EQ(config.getVFOperations().size(), 3);
        EXPECT_EQ(config.getInputs().size(), 1);
        EXPECT_EQ(config.getOutputs().size(), 1);
        EXPECT_EQ(config.getSubgraph(), vfOp);
        EXPECT_TRUE(config.isPipelined());
    });
}

TEST_F(MLIR_VPU_VFConfig, VF_GenericConfigPipelined) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
    module @main {
        func.func @main(%arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>,
        %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
            %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>

            %0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x1024x4xf16, {order = #NHWC}>, %arg1 as %arg4: tensor<4096x48x1x1xf16, {order = #NHWC}>, %cst_0 as %arg5: tensor<4096x1x1x4xsi32>, %arg1 as %arg6: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 24, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
            {   %1 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
                {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>, tensor<4096x1x1x4xsi32> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
                %2 = VPU.SoftMax(%1) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
                VPU.Yield %2
            }

            return %0 : tensor<1x4096x1024x4xf16, {order = #NHWC}>
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

    func->walk([&](VPU::VerticalFusionOp vfOp) {
        auto config = VPU::VF::v2::VFConfig(vfOp);
        EXPECT_EQ(config.getVFOperations().size(), 2);
        EXPECT_EQ(config.getInputs().size(), 1);
        EXPECT_EQ(config.getOutputs().size(), 1);
        EXPECT_EQ(config.getSubgraph(), vfOp);
        EXPECT_TRUE(config.isPipelined());
    });
}
