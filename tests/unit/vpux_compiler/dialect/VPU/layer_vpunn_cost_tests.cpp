//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/factories/cost_model_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <vpu/shave/SHVSoftmax.h>
#include <vpu_cost_model.h>
#include <vpu_layer_cost_model.h>

#include <gtest/gtest.h>

using vpux::config::ArchKind;
using vpux::VPU::MultiClusterStrategy;
using namespace vpux;

using MLIR_VPU_LayerVPUNNCost = vpux::VPU::arch37xx::UnitTest;

VPU::StrategyCost getSWVPUNNCost(std::shared_ptr<VPUNN::SWOperation> vpunnLayer, mlir::ModuleOp module,
                                 VPU::MultiClusterStrategy mcStrategy) {
    const auto archKind = config::getArch(module);
    const auto vpunnCostFunction = VPU::CostModelConfig::createLayerCostModel(archKind);

    auto tileOp = config::getTileExecutor(module);
    auto dpuExec = tileOp.getSubExecutor(VPU::ExecutorKind::DPU);

    auto shaveActExec = tileOp.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT);
    VPUX_THROW_WHEN(shaveActExec == nullptr, "Act shave kernels are not supported for the platform {0}", archKind);

    auto vpunnStrategy = VPU::getVPULayerStrategy(mcStrategy, dpuExec.getCount(), tileOp.getCount(), archKind,
                                                  shaveActExec.getCount(), false);
    return vpunnCostFunction->Layer(*vpunnLayer, vpunnStrategy);
}

VPUNN::CyclesInterfaceType getHWVPUNNCost(VPUNN::DPULayer& vpunnLayer, mlir::ModuleOp module,
                                          VPU::MultiClusterStrategy mcStrategy) {
    const auto archKind = config::getArch(module);
    const auto vpunnCostFunction = VPU::CostModelConfig::createLayerCostModel(archKind);

    auto tileOp = config::getTileExecutor(module);
    auto dpuExec = tileOp.getSubExecutor(VPU::ExecutorKind::DPU);

    auto shaveActExec = tileOp.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT);
    VPUX_THROW_WHEN(shaveActExec == nullptr, "Act shave kernels are not supported for the platform {0}", archKind);

    auto vpunnStrategy = VPU::getVPULayerStrategy(mcStrategy, dpuExec.getCount(), tileOp.getCount(), archKind,
                                                  shaveActExec.getCount(), true);
    return vpunnCostFunction->Layer(vpunnLayer, vpunnStrategy);
}

VPUNN::CyclesInterfaceType getSimpleCost(mlir::Operation* op, mlir::ModuleOp module,
                                         VPU::MultiClusterStrategy strategy) {
    auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    auto tileOp = config::getTileExecutor(module);

    return outputType.getTotalAllocSize().count() /
           (strategy == VPU::MultiClusterStrategy::Clustering ? 1 : tileOp.getCount());
}

VPUNN::CyclesInterfaceType getWeightsDMACost(VPU::NCEOpInterface nceOp, mlir::ModuleOp module) {
    auto weightsVal = nceOp.getWeightsOperand();
    if (weightsVal == nullptr) {
        return 0;
    }
    const auto weightsType = mlir::cast<vpux::NDTypeInterface>(weightsVal.getType());
    const auto archKind = config::getArch(module);
    const auto vpunnCostModel = VPU::CostModelConfig::createCostModel(archKind);
    const auto vpunnDevice = VPU::getVPUDeviceType(archKind);
    const auto numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();
    return checked_cast<VPUNN::CyclesInterfaceType>(getDMACost(weightsType, vpunnDevice, vpunnCostModel, numDMAPorts));
}

TEST_F(MLIR_VPU_LayerVPUNNCost, DPU_LayerCost) {
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
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_100", "t_Convolution"])

        return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>
    }
    }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto archKind = ArchKind::NPU37XX;

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(archKind, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    VPU::LayerVPUNNCost layerCost(func);

    auto tileOp = config::getTileExecutor(module.get());
    auto dpuExec = tileOp.getSubExecutor(VPU::ExecutorKind::DPU);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto nceOp = mlir::cast<VPU::NCEOpInterface>(convOp.getOperation());
        const auto costParam = VPU::getWorkloadCostParam(nceOp, archKind, dpuExec.getCount());
        auto dpuLayer = VPU::getDPULayer(costParam);

        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(convOp.getOperation());
        const auto getStrategyCost = [&](VPU::MultiClusterStrategy strategy) {
            if (clusteredOp != nullptr) {
                clusteredOp.setMultiClusterStrategy(strategy);
            }
            const auto cost = layerCost.getStrategyCost(convOp, strategy);
            if (clusteredOp != nullptr) {
                clusteredOp->removeAttr(VPU::multiClusterStrategy);
            }
            return cost;
        };

        auto weightsDMACost = getWeightsDMACost(nceOp, module.get());

        EXPECT_EQ(getStrategyCost(VPU::MultiClusterStrategy::Clustering),
                  getHWVPUNNCost(dpuLayer, module.get(), VPU::MultiClusterStrategy::Clustering) + weightsDMACost);
        EXPECT_EQ(getStrategyCost(VPU::MultiClusterStrategy::SplitOverHeight),
                  getHWVPUNNCost(dpuLayer, module.get(), VPU::MultiClusterStrategy::SplitOverHeight) + weightsDMACost);
        EXPECT_EQ(getStrategyCost(VPU::MultiClusterStrategy::HKSwitch),
                  getHWVPUNNCost(dpuLayer, module.get(), VPU::MultiClusterStrategy::HKSwitch) + weightsDMACost);
    });
}

TEST_F(MLIR_VPU_LayerVPUNNCost, SWKernel_LayerCost) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
    module @main {
        func.func @main(%arg0: tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}> {
        %0 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x8x4x76xf16, {order = #NHWC}>

        return %0 : tensor<1x8x4x76xf16, {order = #NHWC}>
    }
    }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto archKind = ArchKind::NPU37XX;

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(archKind, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    VPU::LayerVPUNNCost layerCost(func);

    auto vpuDevice = VPU::getVPUDeviceType(archKind);

    func->walk([&](VPU::SoftMaxOp kernelOp) {
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(kernelOp.getInput().getType());
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(kernelOp.getOutput().getType());

        const auto inputTensor = VPU::getVPUTensor(inputType.getShape(), inputType.getElementType());
        ;
        const auto outputTensor = VPU::getVPUTensor(outputType.getShape(), outputType.getElementType());
        const auto vpunnLayer = std::make_shared<VPUNN::SHVSoftmax>(vpuDevice, inputTensor, outputTensor);

        EXPECT_EQ(layerCost.getStrategyCost(kernelOp, VPU::MultiClusterStrategy::Clustering),
                  getSWVPUNNCost(vpunnLayer, module.get(), VPU::MultiClusterStrategy::Clustering));
        EXPECT_EQ(layerCost.getStrategyCost(kernelOp, VPU::MultiClusterStrategy::SplitOverHeight),
                  getSWVPUNNCost(vpunnLayer, module.get(), VPU::MultiClusterStrategy::SplitOverHeight));
        EXPECT_EQ(layerCost.getStrategyCost(kernelOp, VPU::MultiClusterStrategy::SplitOverKernel),
                  getSWVPUNNCost(vpunnLayer, module.get(), VPU::MultiClusterStrategy::SplitOverKernel));
    });
}

TEST_F(MLIR_VPU_LayerVPUNNCost, SWKernel_SimpleCost) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
    module @main {
        func.func @main(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x48x160x80xf16> {
        %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
        return %0 : tensor<1x48x160x80xf16>
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

    VPU::LayerVPUNNCost layerCost(func);

    func->walk([&](VPU::ConvertOp kernelOp) {
        EXPECT_EQ(layerCost.getStrategyCost(kernelOp, VPU::MultiClusterStrategy::Clustering),
                  getSimpleCost(kernelOp, module.get(), VPU::MultiClusterStrategy::Clustering));
        EXPECT_EQ(layerCost.getStrategyCost(kernelOp, VPU::MultiClusterStrategy::SplitOverHeight),
                  getSimpleCost(kernelOp, module.get(), VPU::MultiClusterStrategy::SplitOverHeight));
    });
}

TEST_F(MLIR_VPU_LayerVPUNNCost, DMA_Cost) {
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
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_100", "t_Convolution"])
        %2 = VPU.MaxPool(%1) {
            kernel_size = [3, 3],
            ppe = #VPU.PPEStub<>,
            pads_begin = [1, 1],
            pads_end = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Maxpool_1", "t_Convolution", "fused"])

        return %2 : tensor<1x16x16x16xf16, {order = #NHWC}>
    }
    }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto archKind = ArchKind::NPU37XX;

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(archKind, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    VPU::LayerVPUNNCost layerCost(func);

    auto vpuDevice = VPU::getVPUDeviceType(archKind);
    const auto vpunnCostFunction = VPU::CostModelConfig::createLayerCostModel(archKind);
    const auto dmaPorts = config::getAvailableExecutor(module.get(), VPU::ExecutorKind::DMA_NN).getCount();

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto spillWriteCostPerTile = layerCost.getSpillingWriteCostsForAllTiles(convOp.getOperation(),
                                                                                VPU::MultiClusterStrategy::Clustering);
        auto spillWriteCosts = std::accumulate(spillWriteCostPerTile.begin(), spillWriteCostPerTile.end(), 0);
        auto spillRefCost = getDMACost(mlir::cast<vpux::NDTypeInterface>(convOp.getResult().getType()), vpuDevice,
                                       vpunnCostFunction->get_TheoreticalDMA_cost_model_shared(), dmaPorts);

        EXPECT_EQ(spillWriteCosts, spillRefCost);
        EXPECT_EQ(layerCost.getSpillingWriteCost(convOp.getOperation(), VPU::MultiClusterStrategy::Clustering),
                  spillRefCost);
    });

    func->walk([&](VPU::NCEMaxPoolOp poolOp) {
        const auto spillRefCost =
                getDMACost(mlir::cast<vpux::NDTypeInterface>(poolOp.getOperand(0).getType()), vpuDevice,
                           vpunnCostFunction->get_TheoreticalDMA_cost_model_shared(), dmaPorts);
        const auto findOperand = [](mlir::Value operand) {
            return operand.getDefiningOp() != nullptr;
        };
        auto spillReadCostPerTiles = layerCost.getSpillingReadCostsForAllTiles(
                poolOp.getOperation(), VPU::MultiClusterStrategy::Clustering, nullptr, findOperand);
        auto spillReadCosts = std::accumulate(spillReadCostPerTiles.begin(), spillReadCostPerTiles.end(), 0);

        EXPECT_EQ(spillReadCosts, spillRefCost);

        EXPECT_EQ(layerCost.getSpillingReadCost(poolOp.getOperation(), VPU::MultiClusterStrategy::Clustering,
                                                poolOp.getOperand(0).getDefiningOp()),
                  spillRefCost);
        EXPECT_EQ(layerCost.getSpillingReadCost(poolOp.getOperation(), VPU::MultiClusterStrategy::Clustering, nullptr,
                                                findOperand),
                  spillRefCost);
    });
}
