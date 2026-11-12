//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/json_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/minimal_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/init/hw_strategy_registry.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using vpux::config::ArchKind;
using namespace vpux;

using MLIR_VPU_VFUtils = MLIR_UnitBase;

TEST_F(MLIR_VPU_VFUtils, VF_UtilsTilingLimitWithPermutation) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#loc0 = loc(unknown)
    module @test {
       func.func @main(%arg0: tensor<1x32x256x128xf16, {order = #NHWC}>,
                       %arg1 : tensor<256x32x3x3xf16, {order = #NHWC}>)
       -> tensor<1x128x256x256xf16> {

            %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x32x256x128xf16, {order = #NHWC}>,
%arg1 as %arg3 : tensor<256x32x3x3xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 3, 1]}
                -> tensor<1x128x256x256xf16> {

                %1 = VPU.NCE.Convolution(%arg2, %arg3) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                     strides = [1, 1]} : tensor<1x32x256x128xf16, {order = #NHWC}>,
                    tensor<256x32x3x3xf16, {order = #NHWC}>
                    -> tensor<1x256x256x128xf16>

                %2 = VPU.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NCHW}
                    : tensor<1x256x256x128xf16> -> tensor<1x128x256x256xf16, {order = #NHWC}>

                %3 = VPU.NCE.MaxPool(%2) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    kernel_size = [3, 3], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                    ppe = #VPU.PPEStub<>, strides = [1, 1]
                    } -> tensor<1x128x256x256xf16>


                VPU.Yield %3
            }

            return %0 : tensor<1x128x256x256xf16>
       }
    }
    )";
    const auto platform = config::Platform::NPU5010;
    vpux::VPU::registerStrategies(registry, platform);
    auto interfacesRegistry = vpux::createInterfacesRegistry(platform);
    interfacesRegistry->registerInterfaces(registry);
    vpux::VPU::initializeSingletons(registry, platform);

    mlir::MLIRContext ctx(registry);
    ctx.appendDialectRegistry(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(platform, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    func->walk([&](VPU::VerticalFusionOp vfOp) {
        VPU::VF::v2::VFCacheAnalysis cache(vfOp.getOperation());
        auto config = VPU::VF::v2::VFConfig(vfOp, cache);
        auto tilingLimit = getTilingLimit(Dims4D::Act::H, config, /*multiDimTiling=*/false);
        EXPECT_EQ(tilingLimit, 8);
    });
}

TEST_F(MLIR_VPU_VFUtils, VF_UtilsDynamicDequantConvCanUseCHTilingStrategy) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

module @main {
    func.func @main(%activation: tensor<1x4096x256x4xf16, {order = #NHWC}>,
                    %weights: tensor<1x32x4096x128x!qElemType>,
                    %scale: tensor<1x4096x32x1xf16>,
                    %weightsTable: tensor<4096x1x1x4xsi32>) -> tensor<1x4096x256x4xf16, {order = #NHWC}> {
        %0 = VPU.VerticalFusion (%weights as %arg0: tensor<1x32x4096x128x!qElemType>,
                                 %scale as %arg1: tensor<1x4096x32x1xf16>,
                                 %activation as %arg2: tensor<1x4096x256x4xf16, {order = #NHWC}>,
                                 %weightsTable as %arg3: tensor<4096x1x1x4xsi32>)
            attributes {tilingStrategy = [1, 1, 1, 1]}  -> tensor<1x4096x256x4xf16, {order = #NHWC}> {
            %permute = VPU.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHCW, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
                : tensor<1x32x4096x128x!qElemType> -> tensor<1x4096x32x128x!qElemType>
            %dq = VPU.DynamicDequantize(%permute, %arg1) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
                : tensor<1x4096x32x128x!qElemType>, tensor<1x4096x32x1xf16> -> tensor<1x4096x32x128xf16>
            %filter = VPU.AffineReshape(%dq) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [4096, 4096, 1, 1]}
                : tensor<1x4096x32x128xf16> -> tensor<4096x4096x1x1xf16>
            %filter_nhwc = VPU.PermuteCast(%filter) {dst_order = #NHWC, mem_perm = #NHWC}
                : tensor<4096x4096x1x1xf16> -> tensor<4096x4096x1x1xf16, {order = #NHWC}>
                    %conv = VPU.NCE.Convolution(%arg2, %filter_nhwc, %arg3) rawFilterShape [4096, 4096, 1, 1]
                {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                     ppe = #VPU.PPEStub<>, strides = [1, 1]}
                : tensor<1x4096x256x4xf16, {order = #NHWC}>, tensor<4096x4096x1x1xf16, {order = #NHWC}>,
                tensor<4096x1x1x4xsi32> -> tensor<1x4096x256x4xf16, {order = #NHWC}>
            VPU.Yield %conv
        }
        return %0 : tensor<1x4096x256x4xf16, {order = #NHWC}>
    }
}
    )";

    const auto platform = config::Platform::NPU6010;
    vpux::VPU::registerStrategies(registry, platform);
    auto interfacesRegistry = vpux::createInterfacesRegistry(platform);
    interfacesRegistry->registerInterfaces(registry);
    vpux::VPU::initializeSingletons(registry, platform);

    mlir::MLIRContext ctx(registry);
    ctx.appendDialectRegistry(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(platform, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto vfOps = to_small_vector(func.getOps<VPU::VerticalFusionOp>());
    ASSERT_EQ(vfOps.size(), 1);

    VPU::VF::v2::VFCacheAnalysis cache(vfOps.front().getOperation());
    auto config = VPU::VF::v2::VFConfig(vfOps.front(), cache);
    auto tilingStorage = std::make_unique<VPU::TilingOperationStorage>();

    const SmallVector<int64_t> validTiling = {1, 8, 32, 1};
    ASSERT_TRUE(mlir::succeeded(
            VPU::VF::v2::calculateTilingRegions(config, validTiling, vpux::Logger::global(), tilingStorage)));

    VPU::VF::v2::VFSplit split = {{Dims4D::Act::H, 32}, {Dims4D::Act::C, std::nullopt}};
    auto schedulingChecks = VPU::VF::v2::getSchedulingScenarios(config, Logger::global(), VFMergeCostMode::PERFORMANCE);
    auto minNumCalc = [](Dim, const VPU::VF::v2::VFSplit&) {
        return VPU::MINIMUM_LENGTH_TILING;
    };
    const auto maxNumCalc = [&config](Dim dim, const VPU::VF::v2::VFSplit&) {
        return VPU::VF::v2::getTilingLimit(dim, config);
    };
    auto vfCase = VPU::VF::v2::getVFCaseWithTiling(config, Dims4D::Act::C, split, minNumCalc, maxNumCalc, false,
                                                   vpux::Logger::global(), schedulingChecks);
    ASSERT_TRUE(vfCase.isInitialized());
    ASSERT_EQ(parseIntArrayAttr<int64_t>(vfCase.getTiling()), validTiling);
}

TEST_F(MLIR_VPU_VFUtils, PatternBasedVFMergeDynamicDequantConv) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

module @main {
    func.func @main(%activation: tensor<1x4096x256x4xf16, {order = #NHWC}>,
                    %weights: tensor<1x32x4096x128x!qElemType>,
                    %scale: tensor<1x4096x32x1xf16>,
                    %weightsTable: tensor<4096x1x1x4xsi32>) -> tensor<1x4096x256x4xf16, {order = #NHWC}> {
        %memPermute = VPU.VerticalFusion (%weights as %arg0: tensor<1x32x4096x128x!qElemType>)
            attributes {tilingStrategy = [1, 3, 1, 1]} -> tensor<1x4096x32x128x!qElemType> {
            %permute = VPU.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHCW, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
                : tensor<1x32x4096x128x!qElemType> -> tensor<1x4096x32x128x!qElemType>
            VPU.Yield %permute
        }
        %dequant = VPU.VerticalFusion (%memPermute as %arg0: tensor<1x4096x32x128x!qElemType>,
                                       %scale as %arg1: tensor<1x4096x32x1xf16>)
            attributes {tilingStrategy = [1, 11, 1, 1]} -> tensor<1x4096x32x128xf16> {
            %dq = VPU.DynamicDequantize(%arg0, %arg1) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
                : tensor<1x4096x32x128x!qElemType>, tensor<1x4096x32x1xf16> -> tensor<1x4096x32x128xf16>
            VPU.Yield %dq
        }
        %conv = VPU.VerticalFusion (%dequant as %arg2: tensor<1x4096x32x128xf16>,
                                   %activation as %arg3: tensor<1x4096x256x4xf16, {order = #NHWC}>,
                                   %weightsTable as %arg4: tensor<4096x1x1x4xsi32>)
            attributes {tilingStrategy = [1, 8, 32, 1]} -> tensor<1x4096x256x4xf16, {order = #NHWC}> {
            %filter = VPU.AffineReshape(%arg2) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [4096, 4096, 1, 1]}
                : tensor<1x4096x32x128xf16> -> tensor<4096x4096x1x1xf16>
            %filter_nhwc = VPU.PermuteCast(%filter) {dst_order = #NHWC, mem_perm = #NHWC}
                : tensor<4096x4096x1x1xf16> -> tensor<4096x4096x1x1xf16, {order = #NHWC}>
                    %result = VPU.NCE.Convolution(%arg3, %filter_nhwc, %arg4) rawFilterShape [4096, 4096, 1, 1]
                {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                     ppe = #VPU.PPEStub<>, strides = [1, 1]}
                : tensor<1x4096x256x4xf16, {order = #NHWC}>, tensor<4096x4096x1x1xf16, {order = #NHWC}>,
                tensor<4096x1x1x4xsi32> -> tensor<1x4096x256x4xf16, {order = #NHWC}>
            VPU.Yield %result
        }
        return %conv : tensor<1x4096x256x4xf16, {order = #NHWC}>
    }
}
    )";

    const auto platform = config::Platform::NPU6010;
    vpux::VPU::registerStrategies(registry, platform);
    auto interfacesRegistry = vpux::createInterfacesRegistry(platform);
    interfacesRegistry->registerInterfaces(registry);
    vpux::VPU::initializeSingletons(registry, platform);

    mlir::MLIRContext ctx(registry);
    ctx.appendDialectRegistry(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(platform, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    ASSERT_EQ(to_small_vector(func.getOps<VPU::VerticalFusionOp>()).size(), 3);

    mlir::PassManager patternPM(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    patternPM.addPass(VPU::createPatternBasedVFPass(WorkloadManagementMode::FWLM_V1_PAGES, vpux::Logger::global()));
    ASSERT_TRUE(mlir::succeeded(patternPM.run(module.get())));

    auto vfOps = to_small_vector(func.getOps<VPU::VerticalFusionOp>());
    ASSERT_EQ(vfOps.size(), 1);

    SmallVector<mlir::Operation*> innerOps;
    for (auto& op : vfOps.front().getBody()->without_terminator()) {
        innerOps.push_back(&op);
    }

    ASSERT_EQ(innerOps.size(), 5);
    EXPECT_TRUE(mlir::isa<VPU::MemPermuteOp>(innerOps[0]));
    EXPECT_TRUE(mlir::isa<VPU::DynamicDequantizeOp>(innerOps[1]));
    EXPECT_TRUE(mlir::isa<VPU::AffineReshapeOp>(innerOps[2]));
    EXPECT_TRUE(mlir::isa<VPU::PermuteCastOp>(innerOps[3]));
    EXPECT_TRUE(mlir::isa<VPU::NCEConvolutionOp>(innerOps[4]));
    const SmallVector<int64_t> expectedTiling = {1, 8, 32, 1};
    ASSERT_EQ(parseIntArrayAttr<int64_t>(vfOps.front().getTilingStrategy()), expectedTiling);
}

// Test that view op that preserves segmented distribution doesn't prevent
// vertical fusion region merging between SOK NCE operation and SOK SwKernel.
TEST_F(MLIR_VPU_VFUtils, SOKSegmentedOutputCompatible_ThroughAffineReshapeToSWConsumer) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @main {
    func.func @main(%arg0: tensor<1x64x16x16xf16, {order = #NHWC}>,
                    %weights: tensor<64x64x1x1xf16, {order = #NHWC}>)
            -> tensor<1x64x256x1xf16, {order = #NHWC}> {
        %conv = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [64, 64, 1, 1] {
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>, strides = [1, 1]}
            : tensor<1x64x16x16xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x16xf16, {order = #NHWC}>
        %reshape = VPU.AffineReshape(%conv) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 64, 256, 1]}
            : tensor<1x64x16x16xf16, {order = #NHWC}> -> tensor<1x64x256x1xf16, {order = #NHWC}>
        %swish = VPU.Swish(%reshape) {
                beta_value = 1.000000e+00 : f64,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
            : tensor<1x64x256x1xf16, {order = #NHWC}> -> tensor<1x64x256x1xf16, {order = #NHWC}>
        return %swish : tensor<1x64x256x1xf16, {order = #NHWC}>
    }
}
    )";

    const auto platform = config::Platform::NPU5010;
    vpux::VPU::registerStrategies(registry, platform);
    auto interfacesRegistry = vpux::createInterfacesRegistry(platform);
    interfacesRegistry->registerInterfaces(registry);
    vpux::VPU::initializeSingletons(registry, platform);

    mlir::MLIRContext ctx(registry);
    ctx.appendDialectRegistry(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(platform, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    VPU::NCEConvolutionOp convOp;
    func->walk([&](VPU::NCEConvolutionOp op) {
        convOp = op;
    });
    ASSERT_TRUE(convOp != nullptr);

    // The SOK-compatible SW consumer (Swish) is only reachable through the AffineReshape, so the
    // producer should still be considered SOK-segmented-output-compatible.
    EXPECT_TRUE(VPU::isSOKSegmentedOutputCompatible(convOp.getOperation()));
}
