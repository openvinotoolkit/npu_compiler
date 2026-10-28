//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
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

using namespace vpux;
using namespace vpux::VPU;
using config::Platform;

using MLIR_VPU_LayerVPUNNCost = vpux::VPU::arch37xx::UnitTest;

namespace {

VPU::StrategyCost getSWVPUNNCost(std::shared_ptr<VPUNN::SHAVEWorkload> vpunnLayer, mlir::ModuleOp module,
                                 VPU::MultiClusterStrategy mcStrategy) {
    const auto archKind = config::getArch(module);
    const auto vpunnCostFunction = VPU::CostModelConfig::createLayerCostModel(module);

    auto tileOp = config::getTileExecutor(module);
    auto dpuExec = tileOp.getSubExecutor(config::ExecutorKind::DPU);

    auto shaveActExec = tileOp.getSubExecutor(config::ExecutorKind::SHAVE_ACT);
    VPUX_THROW_WHEN(shaveActExec == nullptr, "Act shave kernels are not supported for the platform {0}", archKind);

    auto vpunnStrategy = VPU::getVPULayerStrategy(mcStrategy, dpuExec.getCount(), tileOp.getCount(), archKind,
                                                  shaveActExec.getCount(), false);
    return vpunnCostFunction->Layer(*vpunnLayer, vpunnStrategy);
}

VPUNN::CyclesInterfaceType getHWVPUNNCost(VPUNN::DPULayer& vpunnLayer, mlir::ModuleOp module,
                                          VPU::MultiClusterStrategy mcStrategy) {
    const auto archKind = config::getArch(module);
    const auto vpunnCostFunction = VPU::CostModelConfig::createLayerCostModel(module);

    auto tileOp = config::getTileExecutor(module);
    auto dpuExec = tileOp.getSubExecutor(config::ExecutorKind::DPU);

    auto shaveActExec = tileOp.getSubExecutor(config::ExecutorKind::SHAVE_ACT);
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
    const auto vpunnCostModel = VPU::CostModelConfig::createCostModel(module);
    const auto vpuDevice = VPU::getVPUDeviceType(module);
    const auto numDMAPorts = config::getAvailableExecutor(module, config::ExecutorKind::DMA_NN).getCount();
    return checked_cast<VPUNN::CyclesInterfaceType>(getDMACost(weightsType, config::getArch(module), vpuDevice,
                                                               vpunnCostModel, numDMAPorts,
                                                               vpux::DmaDirectionType::DDR_TO_CMX));
}

}  // namespace

TEST_F(MLIR_VPU_LayerVPUNNCost, DPU_LayerCost) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
    module @main {
        func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

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

    const auto platform = Platform::NPU3720;
    const auto archKind = config::getArch(platform);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(platform, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    VPU::LayerVPUNNCost layerCost(func);

    auto tileOp = config::getTileExecutor(module.get());
    auto dpuExec = tileOp.getSubExecutor(config::ExecutorKind::DPU);

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
        func.func @main(%arg0: tensor<1x80x4000x76xf16, {order = #NHWC}>) -> tensor<1x80x4000x76xf16, {order = #NHWC}> {
        %0 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x80x4000x76xf16, {order = #NHWC}> -> tensor<1x80x4000x76xf16, {order = #NHWC}>

        return %0 : tensor<1x80x4000x76xf16, {order = #NHWC}>
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

    VPU::LayerVPUNNCost layerCost(func);

    const auto vpuDevice = VPU::getVPUDeviceType(*module);

    func->walk([&](VPU::SoftMaxOp kernelOp) {
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(kernelOp.getInput().getType());
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(kernelOp.getOutput().getType());

        const auto inputTensor = VPU::getVPUTensor(inputType.getShape(), inputType.getElementType());
        const auto outputTensor = VPU::getVPUTensor(outputType.getShape(), outputType.getElementType());

        const auto inputTensors = std::vector<VPUNN::VPUTensor>{inputTensor};
        const auto outputTensors = std::vector<VPUNN::VPUTensor>{outputTensor};
        VPUNN::SHAVEWorkload::Parameters params = {1};
        const auto vpunnLayer =
                std::make_shared<VPUNN::SHAVEWorkload>("SoftMax", vpuDevice, inputTensors, outputTensors, params);

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
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU3720, config::CompilationMode::DefaultHW);

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
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

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

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU3720, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    VPU::LayerVPUNNCost layerCost(func);

    const auto vpuDevice = VPU::getVPUDeviceType(*module);
    const auto archKind = config::getArch(*module);

    const auto vpunnCostFunction = VPU::CostModelConfig::createLayerCostModel(module.get());
    const auto dmaPorts = config::getAvailableExecutor(module.get(), config::ExecutorKind::DMA_NN).getCount();

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto spillWriteCostPerTile = layerCost.getSpillingWriteCostsForAllTiles(convOp.getOperation(),
                                                                                VPU::MultiClusterStrategy::Clustering);
        auto spillWriteCosts = std::accumulate(spillWriteCostPerTile.begin(), spillWriteCostPerTile.end(), 0);
        auto spillRefCost = getDMACost(mlir::cast<vpux::NDTypeInterface>(convOp.getResult(0).getType()), archKind,
                                       vpuDevice, vpunnCostFunction->get_TheoreticalDMA_cost_model_shared(), dmaPorts,
                                       vpux::DmaDirectionType::CMX_TO_DDR);

        EXPECT_EQ(spillWriteCosts, spillRefCost);
        EXPECT_EQ(layerCost.getSpillingWriteCost(convOp.getOperation(), VPU::MultiClusterStrategy::Clustering),
                  spillRefCost);
    });

    func->walk([&](VPU::NCEMaxPoolOp poolOp) {
        const auto spillRefCost =
                getDMACost(mlir::cast<vpux::NDTypeInterface>(poolOp.getOperand(0).getType()), archKind, vpuDevice,
                           vpunnCostFunction->get_TheoreticalDMA_cost_model_shared(), dmaPorts,
                           vpux::DmaDirectionType::DDR_TO_CMX);
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

// ---------------------------------------------------------------------------
// DMA cost direction tests
// ---------------------------------------------------------------------------

class MLIR_VPU_DMACost_MultiPlatform : public testing::TestWithParam<Platform> {
public:
    void SetUp() override {
        registry = vpux::createDialectRegistry();
        auto interfacesRegistry = vpux::createInterfacesRegistry(GetParam());
        interfacesRegistry->registerInterfaces(registry);
        VPU::initializeSingletons(registry, GetParam());

        ctx = std::make_unique<mlir::MLIRContext>(registry);
    }

    mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringLiteral ir) {
        auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, ctx.get());
        if (!module) {
            return {};
        }
        mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
        auto opts = VPU::InitCompilerOptions(GetParam(), config::CompilationMode::DefaultHW);
        VPU::buildInitCompilerPipeline(pm, opts, Logger::global());
        if (mlir::failed(pm.run(module.get()))) {
            return {};
        }
        return module;
    }

    mlir::DialectRegistry registry;
    std::unique_ptr<mlir::MLIRContext> ctx;
};

// Exercises every DmaDirectionType on multiple tensor shapes and pins how the VPUNN DMA cost model
// reacts to the source/destination memory locations. Guards the cost API against silent changes in
// direction dependent cost
TEST_P(MLIR_VPU_DMACost_MultiPlatform, DMA_Cost_PerDirection) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

    module @main {
        func.func @main(
            %arg0: tensor<1x16x8x8xf16, {order = #NHWC}>,
            %arg1: tensor<1x64x64x64xf16, {order = #NHWC}>,
            %arg2: tensor<1x128x32x32xf16, {order = #NHWC}>,
            %arg3: tensor<1x256x16x16xf16, {order = #NHWC}>,
            %arg4: tensor<1x512x8x8xf16, {order = #NHWC}>
        ) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
        return %arg0 : tensor<1x16x8x8xf16, {order = #NHWC}>
    }
    }
    )";
    auto module = parse(inputIR);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto vpuDevice = VPU::getVPUDeviceType(*module);
    const auto archKind = config::getArch(*module);
    const auto costModel = VPU::CostModelConfig::createCostModel(module.get());
    const auto dmaPorts = config::getAvailableExecutor(module.get(), config::ExecutorKind::DMA_NN).getCount();

    for (unsigned argIdx = 0; argIdx < func.getNumArguments(); ++argIdx) {
        const auto tensorType = mlir::cast<vpux::NDTypeInterface>(func.getArgument(argIdx).getType());

        const auto costFor = [&](vpux::DmaDirectionType direction) {
            return getDMACost(tensorType, archKind, vpuDevice, costModel, dmaPorts, direction);
        };

        const auto cmxToDdrCost = costFor(vpux::DmaDirectionType::CMX_TO_DDR);
        const auto ddrToCmxCost = costFor(vpux::DmaDirectionType::DDR_TO_CMX);
        const auto cmxToCmxCost = costFor(vpux::DmaDirectionType::CMX_TO_CMX);
        const auto ddrToDdrCost = costFor(vpux::DmaDirectionType::DDR_TO_DDR);

        // Every direction must produce a usable cost.
        EXPECT_GT(cmxToDdrCost, 0u) << "arg" << argIdx;
        EXPECT_GT(ddrToCmxCost, 0u) << "arg" << argIdx;
        EXPECT_GT(cmxToCmxCost, 0u) << "arg" << argIdx;
        EXPECT_GT(ddrToDdrCost, 0u) << "arg" << argIdx;

        // CMX->DDR and DDR->CMX are symmetrical in current VPUNN
        EXPECT_EQ(cmxToDdrCost, ddrToCmxCost) << "arg" << argIdx;

        // CMX->CMX should be cheaper than transfers involving DDR
        EXPECT_LT(cmxToCmxCost, cmxToDdrCost) << "arg" << argIdx;
    }
}

// Exercises getDMACost with distributed tensor types across different distribution modes and all DmaDirectionType
// values. Verifies that every combination returns a non-zero cost and that the CMX<->DDR direction symmetry holds
TEST_P(MLIR_VPU_DMACost_MultiPlatform, DMA_Cost_PerDirection_Distributed) {
    constexpr llvm::StringLiteral inputIR = R"(
    module @main {
        func.func @main(%arg0: tensor<1x64x16x16xf16>) -> tensor<1x64x16x16xf16> {
        return %arg0 : tensor<1x64x16x16xf16>
    }
    }
    )";
    auto module = parse(inputIR);
    ASSERT_TRUE(module.get() != nullptr);

    const auto vpuDevice = VPU::getVPUDeviceType(*module);
    const auto archKind = config::getArch(*module);
    const auto costModel = VPU::CostModelConfig::createCostModel(module.get());
    const auto dmaPorts = config::getAvailableExecutor(module.get(), config::ExecutorKind::DMA_NN).getCount();

    // Common attributes
    const auto shape = SmallVector<int64_t>({1, 64, 16, 16});
    const auto elemType = mlir::Float16Type::get(ctx.get());
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx.get()));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(ctx.get(), "CMX_NN");
    const auto numClusters = getIntAttr(ctx.get(), 4);
    const auto numTiles = getIntArrayAttr(ctx.get(), SmallVector<int64_t>({1, 1, 4, 1}));

    struct DistCase {
        StringRef name;
        VPU::DistributionInfoAttr distAttr;
    };

    // SEGMENTED
    const auto segmentedAttr = VPU::DistributionInfoAttr::get(
            ctx.get(), VPU::DistributionModeAttr::get(ctx.get(), VPU::DistributionMode::SEGMENTED), numTiles, nullptr,
            nullptr, nullptr, numClusters, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    // DUPLICATED
    const auto duplicatedAttr = VPU::DistributionInfoAttr::get(
            ctx.get(), VPU::DistributionModeAttr::get(ctx.get(), VPU::DistributionMode::DUPLICATED), numTiles, nullptr,
            nullptr, nullptr, numClusters, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    // OVERLAPPED (needs kernel/pads/strides)
    const auto kernel = getIntArrayAttr(ctx.get(), SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(ctx.get(), getIntAttr(ctx.get(), 1), getIntAttr(ctx.get(), 1),
                                            getIntAttr(ctx.get(), 1), getIntAttr(ctx.get(), 1));
    const auto strides = getIntArrayAttr(ctx.get(), SmallVector<int64_t>({1, 1}));
    const auto overlappedAttr = VPU::DistributionInfoAttr::get(
            ctx.get(), VPU::DistributionModeAttr::get(ctx.get(), VPU::DistributionMode::OVERLAPPED), numTiles, kernel,
            pads, strides, numClusters, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    // DUPLICATED|SEGMENTED
    const auto dupSegAttr = VPU::DistributionInfoAttr::get(
            ctx.get(),
            VPU::DistributionModeAttr::get(ctx.get(),
                                           VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED),
            numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr);

    const SmallVector<DistCase> cases = {
            {"SEGMENTED", segmentedAttr},
            {"DUPLICATED", duplicatedAttr},
            {"OVERLAPPED", overlappedAttr},
            {"DUPLICATED|SEGMENTED", dupSegAttr},
    };

    for (const auto& tc : cases) {
        const auto distType = mlir::cast<vpux::NDTypeInterface>(
                VPU::DistributedTensorType::get(ctx.get(), shape, elemType, dimsOrder, dimsSpace, tc.distAttr));
        ASSERT_TRUE(distType != nullptr) << tc.name.str();

        const auto costFor = [&](vpux::DmaDirectionType direction) {
            return getDMACost(distType, archKind, vpuDevice, costModel, dmaPorts, direction);
        };

        const auto cmxToDdrCost = costFor(vpux::DmaDirectionType::CMX_TO_DDR);
        const auto ddrToCmxCost = costFor(vpux::DmaDirectionType::DDR_TO_CMX);
        const auto cmxToCmxCost = costFor(vpux::DmaDirectionType::CMX_TO_CMX);
        const auto ddrToDdrCost = costFor(vpux::DmaDirectionType::DDR_TO_DDR);

        // Every direction must produce a usable cost.
        EXPECT_GT(cmxToDdrCost, 0u) << tc.name.str();
        EXPECT_GT(ddrToCmxCost, 0u) << tc.name.str();
        EXPECT_GT(cmxToCmxCost, 0u) << tc.name.str();
        EXPECT_GT(ddrToDdrCost, 0u) << tc.name.str();

        // CMX->DDR and DDR->CMX are symmetrical in current VPUNN
        EXPECT_EQ(cmxToDdrCost, ddrToCmxCost) << tc.name.str();

        // CMX->CMX should be cheaper than transfers involving DDR
        EXPECT_LT(cmxToCmxCost, cmxToDdrCost) << tc.name.str();
    }
}

namespace {
auto dmaCostPlatformName = [](const testing::TestParamInfo<Platform>& info) {
    return config::stringifyEnum(info.param).str();
};
}  // namespace

INSTANTIATE_TEST_SUITE_P(NPU40XX, MLIR_VPU_DMACost_MultiPlatform, ::testing::Values(Platform::NPU4000),
                         dmaCostPlatformName);
INSTANTIATE_TEST_SUITE_P(NPU50XX, MLIR_VPU_DMACost_MultiPlatform, ::testing::Values(Platform::NPU5010),
                         dmaCostPlatformName);

// ---------------------------------------------------------------------------
// LayerCostModel::hasSpilling tests
// ---------------------------------------------------------------------------
