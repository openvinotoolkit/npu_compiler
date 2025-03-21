//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/op_tiling_cache.hpp"

#include "vpux/compiler/interfaces_registry.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_OpTilingCacheTest = vpux::VPU::arch37xx::UnitTest;

llvm::StringLiteral inputIR = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
        func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
          -> tensor<1x16x8x8xf16, {order = #NHWC}> {

            %0 = VPU.NCE.AveragePool(%arg0) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %1 = VPU.NCE.AveragePool(%0) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            return %1 : tensor<1x16x8x8xf16, {order = #NHWC}>
        }
})";

TEST_F(MLIR_OpTilingCacheTest, OutputTilingTest) {
    auto registry = vpux::createDialectRegistry();
    auto interfacesRegistry = vpux::createInterfacesRegistry(VPU::ArchKind::NPU40XX);
    interfacesRegistry->registerInterfaces(registry);

    mlir::MLIRContext ctx(registry);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);
    module.get()->removeAttr("VPU.arch");

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(VPU::ArchKind::NPU40XX, VPU::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto nceOps = to_small_vector(func.getOps<vpux::VPU::NCEAveragePoolOp>());
    ASSERT_TRUE(nceOps.size() == 2);

    auto& cache = vpux::VPU::OpTilingCache::instance();
    cache.enableIfNecessary(true);
    cache.cleanUp();

    const auto tileDimOrder = getTileDimOrder(nceOps[0], vpux::TilingMode::ISOLATED, Logger::global());
    auto outputTiling1 = vpux::getHWLayerTilingStrategy(nceOps[0], vpux::TilingMode::ISOLATED, Logger::global());
    ASSERT_TRUE(mlir::succeeded(outputTiling1));

    auto opHash = cache.calculateOpHash(nceOps[1], tileDimOrder);
    auto opHashWithTilingMode = cache.updateOpHashWithTilingMode(nceOps[1], opHash, vpux::TilingMode::ISOLATED);
    auto outputTiling2 = cache.getOutputTiling(opHashWithTilingMode, nceOps[1], getShape(nceOps[1]->getResult(0)));
    ASSERT_TRUE(outputTiling2.has_value());
    auto outputTiling2Value = outputTiling2.value();
    ASSERT_TRUE(mlir::succeeded(outputTiling2Value));
    ASSERT_EQ(outputTiling2Value.value(), outputTiling1.value());
}

TEST_F(MLIR_OpTilingCacheTest, OpDPUCostTest) {
    auto registry = vpux::createDialectRegistry();
    auto interfacesRegistry = vpux::createInterfacesRegistry(VPU::ArchKind::NPU40XX);
    interfacesRegistry->registerInterfaces(registry);

    mlir::MLIRContext ctx(registry);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);
    module.get()->removeAttr("VPU.arch");

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(VPU::ArchKind::NPU40XX, VPU::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto tileOp = vpux::IE::getTileExecutor(module.get());
    ASSERT_TRUE(tileOp != nullptr);
    auto dpuExec = tileOp.getSubExecutor(VPU::ExecutorKind::DPU);
    auto numTiles = tileOp.getCount();
    auto numDPUs = dpuExec.getCount();

    auto nceOps = to_small_vector(func.getOps<vpux::VPU::NCEOpInterface>());
    ASSERT_TRUE(nceOps.size() == 2);

    auto outShape = getShape(nceOps[0]->getResult(0));
    Shape offsets(outShape.size(), 0);
    Shape axis(outShape.size(), 1);
    TileInfo tileInfo(outShape, offsets, axis);
    OutputTiling outputTiling = OutputTiling{tileInfo};

    auto& cache = vpux::VPU::OpTilingCache::instance();
    cache.enableIfNecessary(true);
    cache.cleanUp();

    auto strategy =
            mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOps[0].getOperation()).getMultiClusterStrategy().value();

    const auto costParams = VPU::getWorkloadCostParam(mlir::dyn_cast<VPU::NCEOpInterface>(nceOps[0].getOperation()),
                                                      VPU::ArchKind::NPU40XX, numDPUs);
    const auto vpunnStrategy = VPU::getVPULayerStrategy(strategy, numDPUs, numTiles, VPU::ArchKind::NPU40XX, 1, true);

    auto layerCostModel = VPU::createLayerCostModel(VPU::ArchKind::NPU40XX);

    auto dpuCost1 = getDPUCostForNCEOp(nceOps[0], strategy, outputTiling, costParams, vpunnStrategy, layerCostModel,
                                       Logger::global());

    auto opHash = cache.calculateOpHash(nceOps[1].getOperation(), std::nullopt, outputTiling);
    auto dpuCost2 = cache.getOpDpuCost(opHash);
    ASSERT_TRUE(dpuCost2.has_value());
    ASSERT_EQ(dpuCost2.value(), dpuCost1);
}
