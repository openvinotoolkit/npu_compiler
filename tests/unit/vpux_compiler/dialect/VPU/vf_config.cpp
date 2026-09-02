//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/json_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using vpux::config::Platform;
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
            %1 = VPU.NCE.Convolution(%arg2, %arg3, %arg4) rawFilterShape [1024, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                 strides = [1, 1]}
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
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU3720, config::CompilationMode::DefaultHW);

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
            { %1 = VPU.NCE.Convolution(%arg3, %arg4, %arg5) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                     strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>, tensor<4096x1x1x4xsi32> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
                %2 = VPU.SoftMax(%1) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
                %3 = VPU.NCE.Convolution(%2, %arg6, %arg7) rawFilterShape [48, 4096, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,  strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
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
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU3720, config::CompilationMode::DefaultHW);

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
            {   %1 = VPU.NCE.Convolution(%arg3, %arg4, %arg5) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                     strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>, tensor<4096x1x1x4xsi32> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
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
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU3720, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    func->walk([&](VPU::VerticalFusionOp vfOp) {
        VPU::VF::v2::VFCacheAnalysis cache(vfOp.getOperation());
        auto config = VPU::VF::v2::VFConfig(vfOp, cache);
        EXPECT_EQ(config.getVFOperations().size(), 2);
        EXPECT_EQ(config.getInputs().size(), 1);
        EXPECT_EQ(config.getOutputs().size(), 1);
        EXPECT_EQ(config.getSubgraph(), vfOp);
        EXPECT_TRUE(config.isPipelined());
    });
}

TEST_F(MLIR_VPU_VFConfig, VF_ManualConfiguration) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
    module @main {
    func.func @main(%arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>, %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
          %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>
          %0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x1024x4xf16, {order = #NHWC}>, %arg1 as %arg4: tensor<4096x48x1x1xf16, {order = #NHWC}>, %cst_0 as %arg5: tensor<4096x1x1x4xsi32>, %arg1 as %arg6: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 24, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
          {   %1 = VPU.NCE.Convolution(%arg3, %arg4, %arg5) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>,
                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                   strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>, tensor<4096x1x1x4xsi32> -> tensor<1x4096x1024x4xf16, {order = #NHWC}> loc(fused<{name = "Conv", type = "Convolution"}>["Conv", "_1"])
              VPU.Yield %1
          }
          %2 = VPU.VerticalFusion (%0 as %arg3: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 24, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}>{
              %1 = VPU.SoftMax(%arg3) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}> loc(fused<{name = "Softmax", type = "SoftMax"}>["Softmax", "_1"])
              VPU.Yield %1
          }
          return %2 : tensor<1x4096x1024x4xf16, {order = #NHWC}>
       }
    }
    )";

    constexpr llvm::StringLiteral manualStrategyJSON = R"(
        {
          "Conv?t_Convolution/_1": {
            "layerType": "VPU.NCE.Convolution",
            "multiClusterStrategy": "SplitOverHeight",
            "tilingStrategy": {
              "C": 1,
              "H": 32,
              "N": 1,
              "W": 1
            },
            "VFScenario": "FULL_PREFETCHING",
            "verticalFusion": "True",
            "verticalFusionHash": "0x1111"
          },
          "Softmax?t_SoftMax/_1": {
            "layerType": "VPU.SoftMax",
            "multiClusterStrategy": "SplitOverHeight",
             "tilingStrategy": {
              "C": 1,
              "H": 32,
              "N": 1,
              "W": 1
            },
            "VFScenario": "FULL_PREFETCHING",
            "verticalFusion": "True",
            "verticalFusionHash": "0x1111"
          }
        }
    )";

    constexpr llvm::StringLiteral diableVFManualStrategyJSON = R"(
        {
          "Conv?t_Convolution/_1": {
            "layerType": "VPU.NCE.Convolution",
            "multiClusterStrategy": "SplitOverHeight",
            "tilingStrategy": {
              "C": 1,
              "H": 32,
              "N": 1,
              "W": 1
            },
            "verticalFusion": "False"
          },
          "Softmax?t_SoftMax/_1": {
            "layerType": "VPU.SoftMax",
            "multiClusterStrategy": "SplitOverHeight",
             "tilingStrategy": {
              "C": 1,
              "H": 32,
              "N": 1,
              "W": 1
            },
            "verticalFusion": "False"
          }
        }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU4000, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    llvm::MapVector<mlir::Location, mlir::Operation*> operations;
    llvm::MapVector<mlir::Location, mlir::Operation*> outputPipeliningOps;
    collectAllComputeOps(func, operations, outputPipeliningOps, true);

    auto manualStrategy = llvm::json::parse(manualStrategyJSON);
    ASSERT_TRUE(manualStrategy.operator bool());
    VPU::overwriteManualStrategy(manualStrategy.get(), operations);

    auto vfOp = to_small_vector(func.getOps<VPU::VerticalFusionOp>());
    ASSERT_EQ(vfOp.size(), 1);
    auto tilingStrategy = parseIntArrayAttr<int64_t>(vfOp.front().getTilingStrategyAttr());
    ASSERT_EQ(tilingStrategy[Dims4D::Act::H.ind()], 32);

    // reload IR
    module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    operations.clear();
    outputPipeliningOps.clear();
    collectAllComputeOps(func, operations, outputPipeliningOps, true);

    manualStrategy = llvm::json::parse(diableVFManualStrategyJSON);
    ASSERT_TRUE(manualStrategy.operator bool());
    VPU::overwriteManualStrategy(manualStrategy.get(), operations);
    vfOp = to_small_vector(func.getOps<VPU::VerticalFusionOp>());
    ASSERT_EQ(vfOp.size(), 2);
}

TEST_F(MLIR_VPU_VFConfig, VF_ManualConfigurationForDuplicatedLoc) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
    module @main {
    func.func @main(%arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>, %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
          %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>
          %0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x1024x4xf16, {order = #NHWC}>, %arg1 as %arg4: tensor<4096x48x1x1xf16, {order = #NHWC}>, %cst_0 as %arg5: tensor<4096x1x1x4xsi32>, %arg1 as %arg6: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 24, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
          {   %1 = VPU.NCE.Convolution(%arg3, %arg4, %arg5) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>,
                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                   strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>, tensor<4096x1x1x4xsi32> -> tensor<1x4096x1024x4xf16, {order = #NHWC}> loc(fused<{name = "Conv", type = "Convolution"}>["Conv", "_1"])
              VPU.Yield %1
          }
          %2 = VPU.VerticalFusion (%0 as %arg3: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 24, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}>{
              %1 = VPU.SoftMax(%arg3) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}> loc(fused<{name = "Conv", type = "Convolution"}>["Conv", "_1"])
              VPU.Yield %1
          }
          %3 = VPU.VerticalFusion (%2 as %arg3: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 24, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}>{
              %1 = VPU.SoftMax(%arg3) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}> loc(fused<{name = "Conv", type = "Convolution"}>["Conv", "_1"])
              VPU.Yield %1
          }
          return %3 : tensor<1x4096x1024x4xf16, {order = #NHWC}>
       }
    }
    )";

    // when enable stratey dump, the duplicated loc will add suffix "_unique_[N]" in saved json
    constexpr llvm::StringLiteral manualStrategyJSON = R"(
        {
          "Conv?t_Convolution/_1": {
            "layerType": "VPU.NCE.Convolution",
            "multiClusterStrategy": "SplitOverHeight",
            "tilingStrategy": {
              "C": 1,
              "H": 32,
              "N": 1,
              "W": 1
            },
            "VFScenario": "FULL_PREFETCHING",
            "verticalFusion": "True",
            "verticalFusionHash": "0x1111"
          },
          "Conv?t_Convolution/_1/unique_0": {
            "layerType": "VPU.SoftMax",
            "multiClusterStrategy": "SplitOverHeight",
             "tilingStrategy": {
              "C": 1,
              "H": 32,
              "N": 1,
              "W": 1
            },
            "VFScenario": "FULL_PREFETCHING",
            "verticalFusion": "True",
            "verticalFusionHash": "0x1111"
          },
          "Conv?t_Convolution/_1/unique_1": {
            "layerType": "VPU.SoftMax",
            "multiClusterStrategy": "SplitOverHeight",
             "tilingStrategy": {
              "C": 1,
              "H": 32,
              "N": 1,
              "W": 1
            },
            "VFScenario": "FULL_PREFETCHING",
            "verticalFusion": "True",
            "verticalFusionHash": "0x1111"
          }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU4000, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    llvm::MapVector<mlir::Location, mlir::Operation*> operations;
    llvm::MapVector<mlir::Location, mlir::Operation*> outputPipeliningOps;
    collectAllComputeOps(func, operations, outputPipeliningOps, true);

    auto manualStrategy = llvm::json::parse(manualStrategyJSON);
    ASSERT_TRUE(manualStrategy.operator bool());
    VPU::overwriteManualStrategy(manualStrategy.get(), operations);

    auto vfOp = to_small_vector(func.getOps<VPU::VerticalFusionOp>());
    ASSERT_EQ(vfOp.size(), 1);
    auto tilingStrategy = parseIntArrayAttr<int64_t>(vfOp.front().getTilingStrategyAttr());
    ASSERT_EQ(tilingStrategy[Dims4D::Act::H.ind()], 32);
}

TEST_F(MLIR_VPU_VFConfig, VF_ManualConfigurationWithViewOp) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qtype = !quant.uniform<u8:f32, 1.000000e+00>

#loc0 = loc(unknown)
    module @main {
    func.func @main(%arg0: tensor<1x16x800x1280x!qtype, {order = #NHWC}>, %arg1: tensor<1x4x1600x2560x!qtype, {order = #NHWC}>) -> tensor<1x16x1600x640x!qtype, {order = #NHWC}> {
          %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x16x800x1280x!qtype, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 8, 1]} -> tensor<1x4x1600x2560x!qtype, {order = #NHWC}> {
            %inner = VPU.DepthToSpace(%arg2) {
                       block_size = 2 : i64,
                       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>,
                       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x800x1280x!qtype, {order = #NHWC}> -> tensor<1x4x1600x2560x!qtype, {order = #NHWC}> loc(fused<{name = "d2s", type = "DepthToSpace"}>["d2s", "_1"])
            VPU.Yield %inner
          }
          %1 = VPU.ShapeCast {shape = [1, 16, 1600, 640]} inputs(%0 : tensor<1x4x1600x2560x!qtype, {order = #NHWC}>) -> tensor<1x16x1600x640x!qtype, {order = #NHWC}>
          %2 = VPU.ShapeCast {shape = [1, 16, 1600, 640]} inputs(%arg1 : tensor<1x4x1600x2560x!qtype, {order = #NHWC}>) -> tensor<1x16x1600x640x!qtype, {order = #NHWC}>
          %3 = VPU.VerticalFusion (%1 as %arg2: tensor<1x16x1600x640x!qtype, {order = #NHWC}>, %2 as %arg3: tensor<1x16x1600x640x!qtype, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 8, 1]} -> tensor<1x16x1600x640x!qtype, {order = #NHWC}> {
            %inner = VPU.NCE.Eltwise(%arg2, %arg3) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                      is_inplace = true,
                      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                      op_type = #VPU.eltwise_type<ADD>,
                      ppe = #VPU.PPEFp<mode = <NOOP>,
                      clamp_low = -4.200000e+01 : f64,
                      clamp_high = 2.130000e+02 : f64,
                      scale = 2.1645799279212952E-5 : f64,
                      prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 4.200000e+01 : f64, in1_mult = [1.641600e+04], in2_mult = [3.288200e+04]>} -> tensor<1x16x1600x640x!qtype, {order = #NHWC}>  loc(fused<{name = "add", type = "Add"}>["add", "_1"])
            VPU.Yield %inner
          }
          return %3 : tensor<1x16x1600x640x!qtype, {order = #NHWC}>
       }
    }
    )";

    constexpr llvm::StringLiteral manualStrategyJSON = R"(
        {
          "d2s?t_DepthToSpace/_1": {
            "layerType": "VPU.DepthToSpace",
            "multiClusterStrategy": "SplitOverHeight",
            "tilingStrategy": {
              "C": 1,
              "H": 32,
              "N": 1,
              "W": 1
            },
            "VFScenario": "FULL_PREFETCHING",
            "verticalFusion": "True",
            "verticalFusionHash": "0x1111"
          },
          "add?t_Add/_1": {
            "layerType": "VPU.NCE.Eltwise",
            "multiClusterStrategy": "SplitOverHeight",
             "tilingStrategy": {
              "C": 1,
              "H": 32,
              "N": 1,
              "W": 1
            },
            "VFScenario": "FULL_PREFETCHING",
            "verticalFusion": "True",
            "verticalFusionHash": "0x1111"
          }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU4000, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    llvm::MapVector<mlir::Location, mlir::Operation*> operations;
    llvm::MapVector<mlir::Location, mlir::Operation*> outputPipeliningOps;
    collectAllComputeOps(func, operations, outputPipeliningOps, true);

    auto manualStrategy = llvm::json::parse(manualStrategyJSON);
    ASSERT_TRUE(manualStrategy.operator bool());
    VPU::overwriteManualStrategy(manualStrategy.get(), operations);

    auto vfOp = to_small_vector(func.getOps<VPU::VerticalFusionOp>());
    ASSERT_EQ(vfOp.size(), 1);
    auto tilingStrategy = parseIntArrayAttr<int64_t>(vfOp.front().getTilingStrategyAttr());
    ASSERT_EQ(tilingStrategy[Dims4D::Act::H.ind()], 32);
}
