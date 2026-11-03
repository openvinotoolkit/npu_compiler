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
