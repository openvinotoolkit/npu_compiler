//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/dma_cost_model_utils.hpp"

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
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <vpu/shave/SHVSoftmax.h>
#include <vpu_cost_model.h>
#include <vpu_layer_cost_model.h>

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <string>
#include <utility>

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

        const auto inputTensor = VPU::getVPUTensor(inputType.getShape(), inputType.getElementType(), vpuDevice);
        const auto outputTensor = VPU::getVPUTensor(outputType.getShape(), outputType.getElementType(), vpuDevice);

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

// The contiguous VPUNN interface prices CMX->DDR and DDR->CMX identically. The stride-aware descriptor
// interface charges a DDR read slightly above a DDR write, so the two directions only match up to a small
// fixed latency there.
static void expectCmxDdrDirectionSymmetry(size_t cmxToDdrCost, size_t ddrToCmxCost, config::ArchKind archKind,
                                          StringRef label) {
    constexpr double maxDirectionLatencyDeltaInCycles = 32.0;
    if (vpux::usesStrideAwareDMACostModel(archKind)) {
        EXPECT_NEAR(static_cast<double>(cmxToDdrCost), static_cast<double>(ddrToCmxCost),
                    maxDirectionLatencyDeltaInCycles)
                << label.str();
        return;
    }
    EXPECT_EQ(cmxToDdrCost, ddrToCmxCost) << label.str();
}

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

        expectCmxDdrDirectionSymmetry(cmxToDdrCost, ddrToCmxCost, archKind, "arg" + std::to_string(argIdx));

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

        expectCmxDdrDirectionSymmetry(cmxToDdrCost, ddrToCmxCost, archKind, tc.name);

        // CMX->CMX should be cheaper than transfers involving DDR
        EXPECT_LT(cmxToCmxCost, cmxToDdrCost) << tc.name.str();
    }
}

// Pins the round-robin port assignment of the stride-aware multi-cluster cost: cluster i is charged whole to
// port i % numDMAPorts and the busiest port sets the cost, with no tail splitting.
//
// The cluster count is deliberately not a multiple of the port count, which is what separates this estimate
// from the legacy optimistic one. Three equal per-cluster transfers of cost c give:
//   - 3 ports -> one cluster per port                        -> c
//   - 2 ports -> port 0 takes clusters 0 and 2, port 1 one   -> 2c   (the legacy sum/ports gives 1.5c)
//   - 1 port  -> everything serialized                       -> 3c
// An even cluster/port ratio cannot tell the two formulas apart, hence the uneven one here.
TEST_P(MLIR_VPU_DMACost_MultiPlatform, DMA_Cost_Distributed_BusiestPortSetsTheCost) {
    constexpr llvm::StringLiteral inputIR = R"(
    module @main {
        func.func @main(%arg0: tensor<1x64x48x16xf16>) -> tensor<1x64x48x16xf16> {
        return %arg0 : tensor<1x64x48x16xf16>
    }
    }
    )";
    auto module = parse(inputIR);
    ASSERT_TRUE(module.get() != nullptr);

    const auto vpuDevice = VPU::getVPUDeviceType(*module);
    const auto archKind = config::getArch(*module);
    const auto costModel = VPU::CostModelConfig::createCostModel(module.get());

    if (!vpux::usesStrideAwareDMACostModel(archKind)) {
        // Only the stride-aware path aggregates per-cluster costs round-robin. The contiguous path prices the
        // whole distributed buffer in one query and divides by the port count, so it has nothing to pin here.
        return;
    }

    // 48 rows split over 3 clusters gives 3 equal per-cluster transfers, so the per-port loads differ only
    // because of the round-robin assignment and not because of the cluster sizes.
    constexpr int64_t numClustersValue = 3;
    const auto shape = SmallVector<int64_t>({1, 64, 48, 16});
    const auto elemType = mlir::Float16Type::get(ctx.get());
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx.get()));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(ctx.get(), "CMX_NN");
    const auto numClusters = getIntAttr(ctx.get(), numClustersValue);
    const auto numTiles = getIntArrayAttr(ctx.get(), SmallVector<int64_t>({1, 1, numClustersValue, 1}));

    const auto segmentedAttr = VPU::DistributionInfoAttr::get(
            ctx.get(), VPU::DistributionModeAttr::get(ctx.get(), VPU::DistributionMode::SEGMENTED), numTiles, nullptr,
            nullptr, nullptr, numClusters, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto distType = mlir::cast<vpux::NDTypeInterface>(
            VPU::DistributedTensorType::get(ctx.get(), shape, elemType, dimsOrder, dimsSpace, segmentedAttr));
    ASSERT_TRUE(distType != nullptr);

    const auto costOnPorts = [&](int64_t numDMAPorts) {
        return getDMACost(distType, archKind, vpuDevice, costModel, numDMAPorts, vpux::DmaDirectionType::DDR_TO_CMX);
    };

    // One cluster per port: the busiest port carries a single per-cluster transfer.
    const auto perClusterCost = costOnPorts(numClustersValue);
    ASSERT_GT(perClusterCost, 0u);

    // Round-robin over two ports: port 0 gets clusters 0 and 2, port 1 gets cluster 1. Charging the tail to
    // its own port is what distinguishes this from the legacy estimate, which would return 1.5x instead.
    EXPECT_EQ(costOnPorts(2), 2 * perClusterCost);

    // A single port serializes every cluster.
    EXPECT_EQ(costOnPorts(1), static_cast<size_t>(numClustersValue) * perClusterCost);
}

// ---------------------------------------------------------------------------
// Stride-aware DMA descriptor tests
// ---------------------------------------------------------------------------

namespace {

// Two buffers with the exact same logical shape (1x16x16x16 f16, NHWC) and element type:
//   - %arg0 is densely packed
//   - %arg1 is the same 16 channel slice seen inside a 32 channel parent buffer
// In NHWC the innermost memory dimension is C, so the packed buffer is one contiguous 8192 byte
// block while the strided one only holds 16 * 2 = 32 contiguous bytes before every gap.
constexpr llvm::StringLiteral compactAndStridedIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

    module @main {
        func.func @main(
            %arg0: memref<1x16x16x16xf16, {order = #NHWC}, @DDR>,
            %arg1: memref<1x16x16x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @DDR>
        ) -> memref<1x16x16x16xf16, {order = #NHWC}, @DDR> {
        return %arg0 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
    }
    }
    )";

constexpr int32_t expectedNumDims = 4;
// VPUDMATensor stores dimensions outermost-first, i.e. in NHWC memory order: N, H, W, C.
constexpr std::array<int32_t, expectedNumDims> expectedDescriptorShape{1, 16, 16, 16};
constexpr std::array<int32_t, expectedNumDims> expectedCompactByteStrides{8192, 512, 32, 2};
constexpr std::array<int32_t, expectedNumDims> expectedStridedByteStrides{16384, 1024, 64, 2};
constexpr int32_t expectedCompactContiguousBytes = 8192;
constexpr int32_t expectedStridedContiguousBytes = 32;

}  // namespace

// The new VPUDMADescriptor interface carries per-dimension byte strides, so a strided buffer is
// described differently from a packed one. The contiguous VPUTensor (the DMAWorkload interface)
// only carries a 4D element count and collapses both buffers into the very same object.
TEST_P(MLIR_VPU_DMACost_MultiPlatform, DMA_Descriptor_CarriesStrides) {
    auto module = parse(compactAndStridedIR);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);
    ASSERT_EQ(func.getNumArguments(), 2u);

    const auto vpuDevice = VPU::getVPUDeviceType(*module);
    const auto compactType = mlir::cast<vpux::NDTypeInterface>(func.getArgument(0).getType());
    const auto stridedType = mlir::cast<vpux::NDTypeInterface>(func.getArgument(1).getType());

    // Same logical shape, different strides: only the descriptor interface can tell them apart.
    ASSERT_EQ(compactType.getShape(), stridedType.getShape());
    ASSERT_NE(compactType.getStrides(), stridedType.getStrides());

    const auto compactTensor = vpux::getVPUDMATensor(compactType, vpuDevice);
    const auto stridedTensor = vpux::getVPUDMATensor(stridedType, vpuDevice);

    EXPECT_EQ(compactTensor.num_dims, expectedNumDims);
    EXPECT_EQ(stridedTensor.num_dims, expectedNumDims);

    for (int32_t dim = 0; dim < expectedNumDims; ++dim) {
        EXPECT_EQ(compactTensor.shape[dim], expectedDescriptorShape[dim]) << "compact shape, dim " << dim;
        EXPECT_EQ(stridedTensor.shape[dim], expectedDescriptorShape[dim]) << "strided shape, dim " << dim;

        // The strides are populated, and they mirror the parent buffer of the strided view.
        EXPECT_EQ(compactTensor.byte_strides[dim], expectedCompactByteStrides[dim]) << "compact stride, dim " << dim;
        EXPECT_EQ(stridedTensor.byte_strides[dim], expectedStridedByteStrides[dim]) << "strided stride, dim " << dim;
    }

    // The layout is populated as well, which is what enables permutation detection in VPUNN.
    EXPECT_TRUE(compactTensor.layout.has_value());
    EXPECT_TRUE(stridedTensor.layout.has_value());
    EXPECT_EQ(compactTensor.layout, stridedTensor.layout);

    // Consequence of the populated strides: VPUNN sees one contiguous block for the packed buffer
    // and 32 byte chunks for the strided one.
    EXPECT_EQ(compactTensor.getContiguousBytes(), expectedCompactContiguousBytes);
    EXPECT_EQ(stridedTensor.getContiguousBytes(), expectedStridedContiguousBytes);

    const auto stridedDescriptor = vpux::buildDMADescriptor(vpuDevice, stridedTensor, stridedTensor,
                                                            VPUNN::MemoryLocation::DRAM, VPUNN::MemoryLocation::CMX);
    EXPECT_EQ(stridedDescriptor.getContiguousBytesSrc(), expectedStridedContiguousBytes);
    EXPECT_EQ(stridedDescriptor.getContiguousBytesDst(), expectedStridedContiguousBytes);

    // Contiguous interface for comparison: it is built from the logical shape only, so the packed and
    // the strided buffer produce identical workload inputs - there is nowhere to put the strides.
    const auto contiguousCompactTensor =
            VPUNN::VPUTensor({static_cast<unsigned int>(compactType.getShape()[Dims4D::Act::W]),
                              static_cast<unsigned int>(compactType.getShape()[Dims4D::Act::H]),
                              static_cast<unsigned int>(compactType.getShape()[Dims4D::Act::C]),
                              static_cast<unsigned int>(compactType.getShape()[Dims4D::Act::N])},
                             VPUNN::DataType::FLOAT16);
    const auto contiguousStridedTensor =
            VPUNN::VPUTensor({static_cast<unsigned int>(stridedType.getShape()[Dims4D::Act::W]),
                              static_cast<unsigned int>(stridedType.getShape()[Dims4D::Act::H]),
                              static_cast<unsigned int>(stridedType.getShape()[Dims4D::Act::C]),
                              static_cast<unsigned int>(stridedType.getShape()[Dims4D::Act::N])},
                             VPUNN::DataType::FLOAT16);
    EXPECT_TRUE(contiguousCompactTensor == contiguousStridedTensor);
}

// The contiguous interface prices a strided buffer exactly like a packed one, while the stride-aware
// descriptor interface charges the fragmentation penalty on top.
TEST_P(MLIR_VPU_DMACost_MultiPlatform, DMA_Cost_HigherWhenStridedInterfaceIsOn) {
    auto module = parse(compactAndStridedIR);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);
    ASSERT_EQ(func.getNumArguments(), 2u);

    const auto vpuDevice = VPU::getVPUDeviceType(*module);
    const auto archKind = config::getArch(*module);
    const auto costModel = VPU::CostModelConfig::createCostModel(module.get());
    const auto dmaPorts = config::getAvailableExecutor(module.get(), config::ExecutorKind::DMA_NN).getCount();

    const auto compactType = mlir::cast<vpux::NDTypeInterface>(func.getArgument(0).getType());
    const auto stridedType = mlir::cast<vpux::NDTypeInterface>(func.getArgument(1).getType());

    // `useStrided` picks the interface explicitly, whatever the platform default is; std::nullopt follows
    // the platform. Every assertion below relies on that, so the same test runs on all platforms.
    const auto costFor = [&](vpux::NDTypeInterface type, std::optional<bool> useStrided) {
        return getDMACost(type, archKind, vpuDevice, costModel, dmaPorts, vpux::DmaDirectionType::DDR_TO_CMX,
                          useStrided);
    };

    // Contiguous interface, forced on every platform through the explicit override.
    const auto compactContiguousCost = costFor(compactType, /*useStrided=*/false);
    const auto stridedContiguousCost = costFor(stridedType, /*useStrided=*/false);
    EXPECT_GT(compactContiguousCost, 0u);
    EXPECT_EQ(compactContiguousCost, stridedContiguousCost) << "the contiguous interface must be blind to striding";

    if (!vpux::usesStrideAwareDMACostModel(archKind)) {
        // Forcing useStrided=true is still allowed here and is the point of the override, but VPUNN has no
        // stride penalty tuned for this platform, so the resulting numbers are not comparable and asserting
        // on them would only pin down unvalidated data. Check the default selection instead.
        EXPECT_EQ(costFor(stridedType, std::nullopt), stridedContiguousCost);
        return;
    }

    // Stride-aware descriptor interface, forced on through the explicit override.
    const auto compactStridedCost = costFor(compactType, /*useStrided=*/true);
    const auto stridedStridedCost = costFor(stridedType, /*useStrided=*/true);
    EXPECT_GT(compactStridedCost, 0u);
    EXPECT_GT(stridedStridedCost, compactStridedCost) << "striding must be charged by the descriptor interface";
    EXPECT_GT(stridedStridedCost, stridedContiguousCost);

    // Without an explicit override the platform default is followed.
    EXPECT_EQ(costFor(stridedType, std::nullopt), stridedStridedCost);
    EXPECT_EQ(costFor(compactType, std::nullopt), compactStridedCost);
}

// ---------------------------------------------------------------------------
// Distributed DMA descriptor shape and stride tests
// ---------------------------------------------------------------------------

namespace {

// A distributed DMA is priced one cluster at a time, so what reaches VPUNN are the per-cluster shapes and
// strides, never the full buffer. Both sides of such a transfer are described that way, whichever direction
// it runs in: the DDR side of a DDR->CMX distributed copy is the DDR region backing one cluster, not the
// whole tensor. A full-buffer descriptor is only produced for a plain, non-distributed buffer.
//
// The buffer below is 1x64x48x16 f16 NHWC, distributed over H across 3 clusters. 48 divides evenly by 3, so
// every expectation stays exact.
constexpr llvm::StringLiteral distributedDdrIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

    module @main {
        func.func @main(%arg0: memref<1x64x48x16xf16, {order = #NHWC}, @DDR>)
                -> memref<1x64x48x16xf16, {order = #NHWC}, @DDR> {
        return %arg0 : memref<1x64x48x16xf16, {order = #NHWC}, @DDR>
    }
    }
    )";

constexpr int64_t distributedNumClusters = 3;
constexpr int32_t distributedNumDims = 4;
constexpr int64_t distributedTotalElements = 1 * 64 * 48 * 16;

// Shape and byte strides of one VPUDMATensor, in VPUDMATensor storage order, i.e. outermost memory dimension
// first. For NHWC that is N, H, W, C.
struct ExpectedDMATensor {
    std::array<int32_t, distributedNumDims> shape;
    std::array<int32_t, distributedNumDims> byteStrides;
    int32_t contiguousBytes;
};

// The whole 1x64x48x16 f16 NHWC buffer, densely packed: C is innermost at 2 bytes, then W at 64*2=128,
// H at 16*128=2048 and N at 48*2048=98304.
constexpr ExpectedDMATensor expectedFullBufferTensor{{{1, 48, 16, 64}}, {{98304, 2048, 128, 2}}, 98304};

// SEGMENTED splits the 48 rows into 3 equal slices. Each cluster owns a compact 16-row block, so only the
// outermost stride shrinks, from 48 to 16 rows; the inner strides are those of the parent buffer.
constexpr std::array<ExpectedDMATensor, distributedNumClusters> expectedSegmentedClusterTensors{{
        {{{1, 16, 16, 64}}, {{32768, 2048, 128, 2}}, 32768},
        {{{1, 16, 16, 64}}, {{32768, 2048, 128, 2}}, 32768},
        {{{1, 16, 16, 64}}, {{32768, 2048, 128, 2}}, 32768},
}};

// OVERLAPPED with k3x3 s1 pad(1, 1, 1, 1) adds the halo rows a cluster needs from its neighbours: the first
// and the last cluster gain one row each, the middle one gains two. The halo rows are adjacent to the owned
// ones, so every cluster view stays compact and only the outermost stride grows.
constexpr std::array<ExpectedDMATensor, distributedNumClusters> expectedOverlappedClusterTensors{{
        {{{1, 17, 16, 64}}, {{34816, 2048, 128, 2}}, 34816},
        {{{1, 18, 16, 64}}, {{36864, 2048, 128, 2}}, 36864},
        {{{1, 17, 16, 64}}, {{34816, 2048, 128, 2}}, 34816},
}};

VPU::DistributionInfoAttr getSegmentedOverHAttr(mlir::MLIRContext* ctx) {
    const auto numClusters = getIntAttr(ctx, distributedNumClusters);
    const auto numTiles = getIntArrayAttr(ctx, SmallVector<int64_t>({1, 1, distributedNumClusters, 1}));
    return VPU::DistributionInfoAttr::get(ctx, VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::SEGMENTED),
                                          numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr, nullptr,
                                          nullptr, nullptr, nullptr, nullptr, nullptr);
}

VPU::DistributionInfoAttr getOverlappedOverHAttr(mlir::MLIRContext* ctx) {
    const auto numClusters = getIntAttr(ctx, distributedNumClusters);
    const auto numTiles = getIntArrayAttr(ctx, SmallVector<int64_t>({1, 1, distributedNumClusters, 1}));
    const auto kernel = getIntArrayAttr(ctx, SmallVector<int64_t>({3, 3}));
    const auto pads =
            VPU::PaddingAttr::get(ctx, getIntAttr(ctx, 1), getIntAttr(ctx, 1), getIntAttr(ctx, 1), getIntAttr(ctx, 1));
    const auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>({1, 1}));
    return VPU::DistributionInfoAttr::get(ctx, VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::OVERLAPPED),
                                          numTiles, kernel, pads, strides, numClusters, nullptr, nullptr, nullptr,
                                          nullptr, nullptr, nullptr, nullptr, nullptr);
}

VPU::DistributedTensorType getDistributedCmxType(mlir::MLIRContext* ctx, VPU::DistributionInfoAttr distAttr) {
    const auto shape = SmallVector<int64_t>({1, 64, 48, 16});
    const auto elemType = mlir::Float16Type::get(ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(ctx, "CMX_NN");
    return VPU::DistributedTensorType::get(ctx, shape, elemType, dimsOrder, dimsSpace, distAttr);
}

// Rebuilds the per-cluster VPUDMATensors the stride-aware cost path hands to VPUNN, from the same two
// ingredients it uses: the per-cluster memory strided shapes of the distributed type and the public tensor
// builder. The cost entry points only return a cycle count, so this is the way to observe the descriptors.
SmallVector<VPUNN::VPUDMATensor> getPerClusterDMATensors(VPU::DistributedTensorType distType,
                                                         VPUNN::VPUDevice vpuDevice) {
    const auto stridedShapes = distType.getPerClusterMemoryStridedShapes();
    SmallVector<VPUNN::VPUDMATensor> tensors;
    tensors.reserve(stridedShapes.size());
    for (const auto& stridedShape : stridedShapes) {
        tensors.push_back(vpux::getVPUDMATensor(stridedShape.shape, stridedShape.strides, distType.getDimsOrder(),
                                                distType.getElementType(), vpuDevice));
    }
    return tensors;
}

void expectDMATensor(const VPUNN::VPUDMATensor& tensor, const ExpectedDMATensor& expected, const std::string& label) {
    ASSERT_EQ(tensor.num_dims, distributedNumDims) << label;
    for (int32_t dim = 0; dim < distributedNumDims; ++dim) {
        EXPECT_EQ(tensor.shape[dim], expected.shape[dim]) << label << ", shape dim " << dim;
        EXPECT_EQ(tensor.byte_strides[dim], expected.byteStrides[dim]) << label << ", byte stride dim " << dim;
    }
    // A populated layout is what lets VPUNN detect a permuting transfer.
    EXPECT_TRUE(tensor.layout.has_value()) << label;
    EXPECT_EQ(tensor.getContiguousBytes(), expected.contiguousBytes) << label;
}

// Pins the geometry of one distributed transfer: shape, byte strides and contiguous bytes of every
// per-cluster descriptor, on both sides, together with the memory locations the pair is queried with.
void expectPerClusterDescriptors(ArrayRef<VPUNN::VPUDMATensor> clusterTensors, ArrayRef<ExpectedDMATensor> expected,
                                 VPUNN::VPUDevice vpuDevice, VPUNN::MemoryLocation srcLoc, VPUNN::MemoryLocation dstLoc,
                                 StringRef caseName) {
    ASSERT_EQ(clusterTensors.size(), expected.size()) << caseName.str();

    for (size_t idx = 0; idx < clusterTensors.size(); ++idx) {
        const auto label = caseName.str() + " cluster " + std::to_string(idx);
        expectDMATensor(clusterTensors[idx], expected[idx], label);

        // Both descriptor sides carry the same per-cluster geometry; only the memory locations differ.
        const auto descriptor =
                vpux::buildDMADescriptor(vpuDevice, clusterTensors[idx], clusterTensors[idx], srcLoc, dstLoc);
        EXPECT_EQ(descriptor.src_location, srcLoc) << label;
        EXPECT_EQ(descriptor.dst_location, dstLoc) << label;
        EXPECT_EQ(descriptor.getContiguousBytesSrc(), expected[idx].contiguousBytes) << label;
        EXPECT_EQ(descriptor.getContiguousBytesDst(), expected[idx].contiguousBytes) << label;
    }
}

int64_t getTotalElements(ArrayRef<VPUNN::VPUDMATensor> tensors) {
    int64_t total = 0;
    for (const auto& tensor : tensors) {
        int64_t elements = 1;
        for (int32_t dim = 0; dim < tensor.num_dims; ++dim) {
            elements *= tensor.shape[dim];
        }
        total += elements;
    }
    return total;
}

}  // namespace

// DDR->CMX with a SEGMENTED distribution.
TEST_P(MLIR_VPU_DMACost_MultiPlatform, DMA_Descriptor_DdrToCmx_Segmented) {
    auto module = parse(distributedDdrIR);
    ASSERT_TRUE(module.get() != nullptr);

    const auto vpuDevice = VPU::getVPUDeviceType(*module);
    const auto distType = getDistributedCmxType(ctx.get(), getSegmentedOverHAttr(ctx.get()));
    ASSERT_TRUE(distType != nullptr);

    const auto clusterTensors = getPerClusterDMATensors(distType, vpuDevice);
    expectPerClusterDescriptors(clusterTensors, expectedSegmentedClusterTensors, vpuDevice, VPUNN::MemoryLocation::DRAM,
                                VPUNN::MemoryLocation::CMX, "DDR_TO_CMX SEGMENTED");

    // SEGMENTED partitions the buffer: no element is moved twice.
    EXPECT_EQ(getTotalElements(clusterTensors), distributedTotalElements);
}

// DDR->CMX with an OVERLAPPED distribution: same geometry pinning as the SEGMENTED case, plus the halo rows
// that make the per-cluster shapes uneven and the moved volume larger than the buffer.
TEST_P(MLIR_VPU_DMACost_MultiPlatform, DMA_Descriptor_DdrToCmx_Overlapped) {
    auto module = parse(distributedDdrIR);
    ASSERT_TRUE(module.get() != nullptr);

    const auto vpuDevice = VPU::getVPUDeviceType(*module);
    const auto distType = getDistributedCmxType(ctx.get(), getOverlappedOverHAttr(ctx.get()));
    ASSERT_TRUE(distType != nullptr);

    const auto clusterTensors = getPerClusterDMATensors(distType, vpuDevice);
    expectPerClusterDescriptors(clusterTensors, expectedOverlappedClusterTensors, vpuDevice,
                                VPUNN::MemoryLocation::DRAM, VPUNN::MemoryLocation::CMX, "DDR_TO_CMX OVERLAPPED");

    // The 4 halo rows are moved twice, which is what the cost has to reflect. See E#66557: once the halo is
    // transferred only once, this expectation and the cost it guards change together.
    constexpr int64_t haloRows = 4;
    EXPECT_EQ(getTotalElements(clusterTensors), distributedTotalElements + 1 * 64 * haloRows * 16);
}

// CMX->DDR: the descriptors are geometry only, so reversing the transfer leaves every shape and stride
// untouched and swaps the memory locations alone. The non-distributed DDR buffer is pinned here as well,
// since that is the only case in which a full-buffer shape reaches VPUNN.
TEST_P(MLIR_VPU_DMACost_MultiPlatform, DMA_Descriptor_CmxToDdr) {
    auto module = parse(distributedDdrIR);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);
    ASSERT_EQ(func.getNumArguments(), 1u);

    const auto vpuDevice = VPU::getVPUDeviceType(*module);

    const auto ddrType = mlir::cast<vpux::NDTypeInterface>(func.getArgument(0).getType());
    expectDMATensor(vpux::getVPUDMATensor(ddrType, vpuDevice), expectedFullBufferTensor, "full DDR buffer");

    const auto segmentedType = getDistributedCmxType(ctx.get(), getSegmentedOverHAttr(ctx.get()));
    ASSERT_TRUE(segmentedType != nullptr);
    expectPerClusterDescriptors(getPerClusterDMATensors(segmentedType, vpuDevice), expectedSegmentedClusterTensors,
                                vpuDevice, VPUNN::MemoryLocation::CMX, VPUNN::MemoryLocation::DRAM,
                                "CMX_TO_DDR SEGMENTED");

    const auto overlappedType = getDistributedCmxType(ctx.get(), getOverlappedOverHAttr(ctx.get()));
    ASSERT_TRUE(overlappedType != nullptr);
    expectPerClusterDescriptors(getPerClusterDMATensors(overlappedType, vpuDevice), expectedOverlappedClusterTensors,
                                vpuDevice, VPUNN::MemoryLocation::CMX, VPUNN::MemoryLocation::DRAM,
                                "CMX_TO_DDR OVERLAPPED");
}

// End-to-end counterpart of the descriptor tests above: the per-cluster geometry they pin must be what the
// cost API actually prices. OVERLAPPED moves the halo rows on top of the segmented volume, so it can never
// come out cheaper than SEGMENTED, in either direction.
TEST_P(MLIR_VPU_DMACost_MultiPlatform, DMA_Cost_Distributed_OverlappedPaysForHalo) {
    auto module = parse(distributedDdrIR);
    ASSERT_TRUE(module.get() != nullptr);

    const auto vpuDevice = VPU::getVPUDeviceType(*module);
    const auto archKind = config::getArch(*module);
    const auto costModel = VPU::CostModelConfig::createCostModel(module.get());
    const auto dmaPorts = config::getAvailableExecutor(module.get(), config::ExecutorKind::DMA_NN).getCount();

    const auto segmentedType =
            mlir::cast<vpux::NDTypeInterface>(getDistributedCmxType(ctx.get(), getSegmentedOverHAttr(ctx.get())));
    const auto overlappedType =
            mlir::cast<vpux::NDTypeInterface>(getDistributedCmxType(ctx.get(), getOverlappedOverHAttr(ctx.get())));

    const auto costFor = [&](vpux::NDTypeInterface type, vpux::DmaDirectionType direction) {
        return getDMACost(type, archKind, vpuDevice, costModel, dmaPorts, direction);
    };

    for (const auto direction : {vpux::DmaDirectionType::DDR_TO_CMX, vpux::DmaDirectionType::CMX_TO_DDR}) {
        const auto label = direction == vpux::DmaDirectionType::DDR_TO_CMX ? "DDR_TO_CMX" : "CMX_TO_DDR";
        const auto segmentedCost = costFor(segmentedType, direction);
        const auto overlappedCost = costFor(overlappedType, direction);

        EXPECT_GT(segmentedCost, 0u) << label;
        EXPECT_GE(overlappedCost, segmentedCost) << label << ": the halo rows must not be free";
    }
}

namespace {

// Redirects the global logger into a buffer for the duration of the scope. The level is raised because a
// release build keeps the global logger silent by default, and restored on exit together with the stream.
class ScopedLogCapture final {
public:
    ScopedLogCapture(): _stream(_buffer), _previousLevel(Logger::global().level()) {
        Logger::setBaseStream(_stream);
        Logger::global().setLevel(LogLevel::Warning);
    }

    ScopedLogCapture(const ScopedLogCapture&) = delete;
    ScopedLogCapture& operator=(const ScopedLogCapture&) = delete;

    ~ScopedLogCapture() {
        Logger::global().setLevel(_previousLevel);
        Logger::setBaseStream(llvm::outs());
    }

    std::string str() {
        _stream.flush();
        return _buffer;
    }

private:
    std::string _buffer;
    llvm::raw_string_ostream _stream;
    LogLevel _previousLevel;
};

}  // namespace

// A stride that is not a whole number of bytes describes a buffer the DMA engine cannot walk, since it has no
// fractional byte handling. The descriptor builder rounds such a stride up so the cost query still returns a
// usable number, and reports it so the malformed buffer is not silently priced as a well-formed one.
TEST_P(MLIR_VPU_DMACost_MultiPlatform, DMA_Descriptor_ReportsNonByteAlignedStride) {
    auto module = parse(distributedDdrIR);
    ASSERT_TRUE(module.get() != nullptr);

    const auto vpuDevice = VPU::getVPUDeviceType(*module);
    const auto elemType = mlir::Float16Type::get(ctx.get());

    // 1x4x2x2 f16 NHWC. Strides are indexed by logical dimension, so the order below is N, C, H, W.
    // Compact bit strides: C = 16 (one f16), W = 4 * 16 = 64, H = 2 * 64 = 128, N = 2 * 128 = 256.
    const Shape shape({1, 4, 2, 2});
    const Strides compactStrides({Bit(256), Bit(16), Bit(128), Bit(64)});
    // The same buffer with the W stride pushed off a byte boundary: 68 bits is 8.5 bytes.
    const Strides misalignedStrides({Bit(256), Bit(16), Bit(128), Bit(68)});

    // VPUDMATensor stores dimensions outermost-first, which for NHWC is N, H, W, C, so W sits at index 2.
    constexpr int32_t wIdx = 2;

    {
        ScopedLogCapture capture;
        const auto tensor = vpux::getVPUDMATensor(shape, compactStrides, DimsOrder::NHWC, elemType, vpuDevice);
        EXPECT_EQ(tensor.byte_strides[wIdx], 8);
        EXPECT_TRUE(capture.str().empty()) << "a byte aligned buffer must not be reported";
    }

    {
        ScopedLogCapture capture;
        const auto tensor = vpux::getVPUDMATensor(shape, misalignedStrides, DimsOrder::NHWC, elemType, vpuDevice);
        // Rounded up, never down: 68 bits still spans 9 bytes.
        EXPECT_EQ(tensor.byte_strides[wIdx], 9);

        const auto logged = capture.str();
        EXPECT_NE(logged.find("not byte aligned"), std::string::npos) << logged;
        EXPECT_NE(logged.find("68 bits"), std::string::npos) << logged;
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
INSTANTIATE_TEST_SUITE_P(NPU60XX, MLIR_VPU_DMACost_MultiPlatform, ::testing::Values(Platform::NPU6010),
                         dmaCostPlatformName);

// ---------------------------------------------------------------------------
// LayerCostModel::hasSpilling tests
// ---------------------------------------------------------------------------

namespace {

static mlir::OwningOpRef<mlir::ModuleOp> parseAndInitHasSpillingModule(mlir::MLIRContext* ctx, llvm::StringRef ir) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, ctx);
    if (!module) {
        return {};
    }
    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initOptions = VPU::InitCompilerOptions(Platform::NPU6010, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initOptions, Logger::global());
    if (mlir::failed(pm.run(module.get()))) {
        return {};
    }
    return module;
}

}  // namespace

// NCEConvolution (SOH) → Subtract (SOH): no spilling for same strategy;
// has spilling when parent or user strategy changes.
TEST_F(MLIR_VPU_LayerVPUNNCost, HasSpilling_ConvSubtract_SOH_SOH) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @nceconv_subtract_direct {
  func.func @main(%arg0: tensor<1x32x16x16xf16, {order = #NHWC}>,
                  %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
                  -> tensor<1x32x16x16xf16, {order = #NHWC}> {
    %out, %rmin = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [32, 32, 1, 1] {
        axes_value           = [1],
        kernel_size          = [1, 1],
        pad                  = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe                  = #VPU.PPEStub<>,
        strides              = [1, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        resultSegmentSizes  = array<i32: 1, 0, 1, 0>
    } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
       -> tensor<1x32x16x16xf16, {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>
    %0 = VPU.Subtract(%out, %rmin) {
        auto_broadcast       = #IE.auto_broadcast_type<NUMPY>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<1x1x16x16xf16, {order = #NHWC}>
      -> tensor<1x32x16x16xf16, {order = #NHWC}>
    return %0 : tensor<1x32x16x16xf16, {order = #NHWC}>
  }
}
    )";
    auto module = parseAndInitHasSpillingModule(&ctx, inputIR);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    VPU::SiblingOpsAnalysis siblingsAnalysis(func);
    VPU::LayerCostModel costModel(func, /*enablePrefetchTiling=*/false, Logger::global(), siblingsAnalysis);

    VPU::NCEConvolutionOp convOp;
    VPU::SubtractOp subtractOp;
    func->walk([&](mlir::Operation* op) {
        if (auto conv = mlir::dyn_cast_if_present<VPU::NCEConvolutionOp>(op)) {
            convOp = conv;
        }
        if (auto sub = mlir::dyn_cast_if_present<VPU::SubtractOp>(op)) {
            subtractOp = sub;
        }

        if (convOp != nullptr && subtractOp != nullptr) {
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });

    ASSERT_TRUE(convOp != nullptr);
    ASSERT_TRUE(subtractOp != nullptr);

    auto conv = mlir::cast<VPU::ClusteredOpInterface>(convOp.getOperation());
    auto subtract = mlir::cast<VPU::ClusteredOpInterface>(subtractOp.getOperation());

    // Same strategy on both sides: no spilling.
    EXPECT_FALSE(costModel.hasSpilling(conv, MultiClusterStrategy::SplitOverHeight, subtract,
                                       MultiClusterStrategy::SplitOverHeight));
    // Parent switches to HKSwitch: fused reduce outputs keep the parent output segmented, so no spilling.
    // Track E#233730. Support halo-sharing for reduce buffer.
    EXPECT_FALSE(costModel.hasSpilling(conv, MultiClusterStrategy::HKSwitch, subtract,
                                       MultiClusterStrategy::SplitOverHeight));
    // User switches to SOK: different split dimension, has spilling.
    EXPECT_TRUE(costModel.hasSpilling(conv, MultiClusterStrategy::SplitOverHeight, subtract,
                                      MultiClusterStrategy::SplitOverKernel));
}

// NCEConvolution (SOH) → Subtract (SOK): has spilling
// no spilling when both use the same strategy explicitly.
TEST_F(MLIR_VPU_LayerVPUNNCost, HasSpilling_ConvSubtract_SOH_SOK) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @nceconv_subtract_direct_spill {
  func.func @main(%arg0: tensor<1x32x16x16xf16, {order = #NHWC}>,
                  %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
                  -> tensor<1x32x16x16xf16, {order = #NHWC}> {
    %out, %rmin = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [32, 32, 1, 1] {
        axes_value           = [1],
        kernel_size          = [1, 1],
        pad                  = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe                  = #VPU.PPEStub<>,
        strides              = [1, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        resultSegmentSizes   = array<i32: 1, 0, 1, 0>
    } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
       -> tensor<1x32x16x16xf16, {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>
    %0 = VPU.Subtract(%out, %rmin) {
        auto_broadcast       = #IE.auto_broadcast_type<NUMPY>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<1x1x16x16xf16, {order = #NHWC}>
      -> tensor<1x32x16x16xf16, {order = #NHWC}>
    return %0 : tensor<1x32x16x16xf16, {order = #NHWC}>
  }
}
    )";
    auto module = parseAndInitHasSpillingModule(&ctx, inputIR);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    VPU::SiblingOpsAnalysis siblingsAnalysis(func);
    VPU::LayerCostModel costModel(func, /*enablePrefetchTiling=*/false, Logger::global(), siblingsAnalysis);

    VPU::NCEConvolutionOp convOp;
    VPU::SubtractOp subtractOp;
    func->walk([&](mlir::Operation* op) {
        if (auto conv = mlir::dyn_cast_if_present<VPU::NCEConvolutionOp>(op)) {
            convOp = conv;
        }
        if (auto sub = mlir::dyn_cast_if_present<VPU::SubtractOp>(op)) {
            subtractOp = sub;
        }

        if (convOp != nullptr && subtractOp != nullptr) {
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    ASSERT_TRUE(convOp != nullptr);
    ASSERT_TRUE(subtractOp != nullptr);

    auto conv = mlir::cast<VPU::ClusteredOpInterface>(convOp.getOperation());
    auto subtract = mlir::cast<VPU::ClusteredOpInterface>(subtractOp.getOperation());

    // SOH → SOK: different split dimensions, has spilling.
    EXPECT_TRUE(costModel.hasSpilling(conv, MultiClusterStrategy::SplitOverHeight, subtract,
                                      MultiClusterStrategy::SplitOverKernel));
    // HKSwitch → SOK: different distributions, has spilling.
    EXPECT_TRUE(costModel.hasSpilling(conv, MultiClusterStrategy::HKSwitch, subtract,
                                      MultiClusterStrategy::SplitOverKernel));
    // Both SOH: same distribution, no spilling.
    EXPECT_FALSE(costModel.hasSpilling(conv, MultiClusterStrategy::SplitOverHeight, subtract,
                                       MultiClusterStrategy::SplitOverHeight));
}

// NCEConvolution (SOH) → PermuteCast → Convert (SOK): the PermuteCast
// changes the shape (NHWC [1,32,16,16] → NCHW [1,16,16,32]), so
// isDistributedCastCompatible always returns false due to shape mismatch.
// All strategy combinations therefore have spillings.
TEST_F(MLIR_VPU_LayerVPUNNCost, HasSpilling_ConvPermuteCastConvert) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @nceconv_convert_view {
  func.func @main(%arg0: tensor<1x32x16x16xf16, {order = #NHWC}>,
                  %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
                  -> tensor<1x16x16x32xf32> {
    %out = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [32, 32, 1, 1] {
        kernel_size          = [1, 1],
        pad                  = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe                  = #VPU.PPEStub<>,
        strides              = [1, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        resultSegmentSizes   = array<i32: 1, 0, 0, 0>
    } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
       -> tensor<1x32x16x16xf16, {order = #NHWC}>
    %permCast = VPU.PermuteCast(%out) {dst_order = #NCHW, mem_perm = #NCHW}
        : tensor<1x32x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x32xf16>
    %0 = VPU.Convert(%permCast) {
        dstElemType          = f32,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    } : tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf32>
    return %0 : tensor<1x16x16x32xf32>
  }
}
    )";
    auto module = parseAndInitHasSpillingModule(&ctx, inputIR);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    VPU::SiblingOpsAnalysis siblingsAnalysis(func);
    VPU::LayerCostModel costModel(func, /*enablePrefetchTiling=*/false, Logger::global(), siblingsAnalysis);

    VPU::NCEConvolutionOp convOp;
    VPU::ConvertOp convertOp;
    func->walk([&](mlir::Operation* op) {
        if (auto conv = mlir::dyn_cast_if_present<VPU::NCEConvolutionOp>(op)) {
            convOp = conv;
        }
        if (auto convert = mlir::dyn_cast_if_present<VPU::ConvertOp>(op)) {
            convertOp = convert;
        }

        if (convOp != nullptr && convertOp != nullptr) {
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    ASSERT_TRUE(convOp != nullptr);
    ASSERT_TRUE(convertOp != nullptr);

    auto conv = mlir::cast<VPU::ClusteredOpInterface>(convOp.getOperation());
    auto convert = mlir::cast<VPU::ClusteredOpInterface>(convertOp.getOperation());

    // PermuteCast changes shape (NHWC [1,32,16,16] → NCHW [1,16,16,32]).
    // isDistributedCastCompatible fails at shape equality for all combinations.
    EXPECT_TRUE(costModel.hasSpilling(conv, MultiClusterStrategy::SplitOverHeight, convert,
                                      MultiClusterStrategy::SplitOverKernel));
    EXPECT_TRUE(costModel.hasSpilling(conv, MultiClusterStrategy::SplitOverKernel, convert,
                                      MultiClusterStrategy::SplitOverKernel));
    EXPECT_TRUE(costModel.hasSpilling(conv, MultiClusterStrategy::SplitOverHeight, convert,
                                      MultiClusterStrategy::SplitOverHeight));
}

// NCEConvolution (SOH) → PermuteCast on both %out and %rmin → Subtract (SOH):
// PermuteCast changes shape on both paths (NHWC→NCHW), so
// isDistributedCastCompatible fails for all combinations.
TEST_F(MLIR_VPU_LayerVPUNNCost, HasSpilling_ConvPermuteCastSubtract) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @nceconv_subtract_view {
  func.func @main(%arg0: tensor<1x32x16x16xf16, {order = #NHWC}>,
                  %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
                  -> tensor<1x16x16x32xf16> {
    %out, %rmin = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [32, 32, 1, 1] {
        axes_value           = [1],
        kernel_size          = [1, 1],
        pad                  = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe                  = #VPU.PPEStub<>,
        strides              = [1, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        resultSegmentSizes   = array<i32: 1, 0, 1, 0>
    } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
       -> tensor<1x32x16x16xf16, {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>
    %permCastAct = VPU.PermuteCast(%out) {dst_order = #NCHW, mem_perm = #NCHW}
        : tensor<1x32x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x32xf16>
    %permCastRed = VPU.PermuteCast(%rmin) {dst_order = #NCHW, mem_perm = #NCHW}
        : tensor<1x1x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x1xf16>
    %0 = VPU.Subtract(%permCastAct, %permCastRed) {
        auto_broadcast       = #IE.auto_broadcast_type<NUMPY>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x16x16x32xf16>, tensor<1x16x16x1xf16> -> tensor<1x16x16x32xf16>
    return %0 : tensor<1x16x16x32xf16>
  }
}
    )";
    auto module = parseAndInitHasSpillingModule(&ctx, inputIR);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    VPU::SiblingOpsAnalysis siblingsAnalysis(func);
    VPU::LayerCostModel costModel(func, /*enablePrefetchTiling=*/false, Logger::global(), siblingsAnalysis);

    VPU::NCEConvolutionOp convOp;
    VPU::SubtractOp subtractOp;
    func->walk([&](mlir::Operation* op) {
        if (auto conv = mlir::dyn_cast_if_present<VPU::NCEConvolutionOp>(op)) {
            convOp = conv;
        }
        if (auto sub = mlir::dyn_cast_if_present<VPU::SubtractOp>(op)) {
            subtractOp = sub;
        }

        if (convOp != nullptr && subtractOp != nullptr) {
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    ASSERT_TRUE(convOp != nullptr);
    ASSERT_TRUE(subtractOp != nullptr);

    auto conv = mlir::cast<VPU::ClusteredOpInterface>(convOp.getOperation());
    auto subtract = mlir::cast<VPU::ClusteredOpInterface>(subtractOp.getOperation());

    // PermuteCast changes shapes on both paths (NHWC [1,32,16,16]→NCHW [1,16,16,32]
    // and NHWC [1,1,16,16]→NCHW [1,16,16,1]), so isDistributedCastCompatible
    // fails at the shape check for all strategy combinations.
    EXPECT_TRUE(costModel.hasSpilling(conv, MultiClusterStrategy::SplitOverHeight, subtract,
                                      MultiClusterStrategy::SplitOverHeight));
    EXPECT_TRUE(costModel.hasSpilling(conv, MultiClusterStrategy::HKSwitch, subtract,
                                      MultiClusterStrategy::SplitOverKernel));
    EXPECT_TRUE(costModel.hasSpilling(conv, MultiClusterStrategy::SplitOverKernel, subtract,
                                      MultiClusterStrategy::SplitOverWidth));
}
