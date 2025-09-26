//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/factories/cost_model_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/op_tiling_cache.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/interfaces_registry.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <vpu_layer_strategy.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_PreSplitCostTest = vpux::VPU::arch40xx::UnitTest;

const static llvm::StringLiteral inputIRSOK = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {} {
        func.func @main(%arg0: tensor<1x128x32x32xf16, {order = #NHWC}>) -> tensor<1x64x32x32xf16, {order = #NHWC}> {
            %cst = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
            %cst_0 = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
            %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} : tensor<1x128x32x32xf16, {order = #NHWC}>, tensor<64x128x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32>
                -> tensor<1x64x32x32xf16, {order = #NHWC}>
            return %0 : tensor<1x64x32x32xf16, {order = #NHWC}>
        }
})";

const static llvm::StringLiteral inputIRClustering = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {} {
        func.func @main(%arg0: tensor<1x128x32x32xf16, {order = #NHWC}>) -> tensor<1x64x32x32xf16, {order = #NHWC}> {
            %cst = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
            %cst_0 = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
            %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} : tensor<1x128x32x32xf16, {order = #NHWC}>, tensor<64x128x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32>
                -> tensor<1x64x32x32xf16, {order = #NHWC}>
            return %0 : tensor<1x64x32x32xf16, {order = #NHWC}>
        }
})";

TEST_F(MLIR_PreSplitCostTest, SamePreSplitCostForSOK) {
    auto registry = vpux::createDialectRegistry();
    const auto arch = config::ArchKind::NPU40XX;
    auto interfacesRegistry = vpux::createInterfacesRegistry(arch);
    interfacesRegistry->registerInterfaces(registry);
    // set cost model factory
    VPU::CostModelConfig::setFactory(config::ArchKind::NPU40XX);

    mlir::MLIRContext ctx(registry);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIRSOK, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(arch, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto numTiles = 4;
    auto numDPUs = 1;

    auto nceOps = to_small_vector(func.getOps<vpux::VPU::NCEOpInterface>());
    ASSERT_TRUE(nceOps.size() == 1);
    auto nceOp = nceOps[0];

    auto outShape = getShape(nceOp->getResult(0));
    Shape offsets(outShape.size(), 0);
    Shape axis(outShape.size(), 1);
    TileInfo tileInfo(outShape, offsets, axis);
    OutputTiling outputTiling = OutputTiling{tileInfo};

    auto strategy = VPU::MultiClusterStrategy::SplitOverKernel;

    const auto costParams = VPU::getWorkloadCostParam(mlir::dyn_cast<VPU::NCEOpInterface>(nceOp.getOperation()), arch,
                                                      numDPUs, numTiles);
    const auto vpunnStrategy = VPU::getVPULayerStrategy(strategy, numDPUs, numTiles, arch, 1, true);

    auto layerCostModel = VPU::CostModelConfig::createLayerCostModel(arch);

    auto& cache = vpux::VPU::OpTilingCache::instance();
    cache.enableIfNecessary(false);

    auto dpuCostsOld = getDPUCostForNCEOp(nceOp, strategy, outputTiling, costParams, vpunnStrategy, layerCostModel,
                                          Logger::global());

    auto dpuCostsPreSplit = getDPUCostForNCEOpPreSplit(nceOp, outputTiling, costParams, vpunnStrategy.tiling_strategy,
                                                       layerCostModel, numDPUs, Logger::global());

    // For evenly split SOK strategy with no tiling strategy, the costs of old API and pre-split should be equal
    ASSERT_TRUE(dpuCostsOld.size() == 1 && dpuCostsPreSplit.size() == 1);
    ASSERT_TRUE(dpuCostsOld[0] == dpuCostsPreSplit[0]);
}

TEST_F(MLIR_PreSplitCostTest, SamePreSplitCostForClustering) {
    auto registry = vpux::createDialectRegistry();
    const auto arch = config::ArchKind::NPU40XX;
    auto interfacesRegistry = vpux::createInterfacesRegistry(arch);
    interfacesRegistry->registerInterfaces(registry);
    // set cost model factory
    VPU::CostModelConfig::setFactory(config::ArchKind::NPU40XX);

    mlir::MLIRContext ctx(registry);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIRClustering, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(arch, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto numTiles = 4;
    auto numDPUs = 1;

    auto nceOps = to_small_vector(func.getOps<vpux::VPU::NCEOpInterface>());
    ASSERT_TRUE(nceOps.size() == 1);
    auto nceOp = nceOps[0];

    auto outShape = getShape(nceOp->getResult(0));
    Shape offsets(outShape.size(), 0);
    Shape axis(outShape.size(), 1);
    TileInfo tileInfo(outShape, offsets, axis);
    OutputTiling outputTiling = OutputTiling{tileInfo};

    auto strategy = VPU::MultiClusterStrategy::Clustering;

    const auto costParams = VPU::getWorkloadCostParam(mlir::dyn_cast<VPU::NCEOpInterface>(nceOp.getOperation()), arch,
                                                      numDPUs, numTiles);
    const auto vpunnStrategy = VPU::getVPULayerStrategy(strategy, numDPUs, numTiles, arch, 1, true);

    auto layerCostModel = VPU::CostModelConfig::createLayerCostModel(arch);

    auto& cache = vpux::VPU::OpTilingCache::instance();
    cache.enableIfNecessary(false);

    auto dpuCostsOld = getDPUCostForNCEOp(nceOp, strategy, outputTiling, costParams, vpunnStrategy, layerCostModel,
                                          Logger::global());

    auto dpuCostsPreSplit = getDPUCostForNCEOpPreSplit(nceOp, outputTiling, costParams, vpunnStrategy.tiling_strategy,
                                                       layerCostModel, numDPUs, Logger::global());

    // For Clustering strategy with no tiling strategy, the costs of old API and pre-split are equal
    // because no per-cluster split is required
    ASSERT_TRUE(dpuCostsOld.size() == 1 && dpuCostsPreSplit.size() == 1);
    ASSERT_TRUE(dpuCostsOld[0] == dpuCostsPreSplit[0]);
}
