//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/init/hw_strategy_registry.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

#include <numeric>
#include <optional>

using namespace vpux;

using MLIR_CostModelUtilsTest = MLIR_UnitBase;

namespace {

constexpr llvm::StringLiteral CONV_WITH_C_TILING_IR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @test {
    func.func @main(%input: tensor<1x64x16x16xf16, {order = #NHWC}>)
            -> tensor<1x64x16x16xf16, {order = #NHWC}> {
        %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
        %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
        %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) rawFilterShape [64, 64, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1]
        } : tensor<1x64x16x16xf16, {order = #NHWC}>,
            tensor<64x64x1x1xf16, {order = #NHWC}>,
            tensor<64x1x1x4xsi32>
            -> tensor<1x64x16x16xf16, {order = #NHWC}>
        return %conv : tensor<1x64x16x16xf16, {order = #NHWC}>
    }
}
)";

bool initModuleForPlatform(mlir::ModuleOp module, config::Platform platform) {
    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    const auto initCompilerOptions = VPU::InitCompilerOptions(platform, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, Logger::global());
    return mlir::succeeded(pm.run(module));
}

uint32_t sumCosts(ArrayRef<uint32_t> costs) {
    return std::accumulate(costs.begin(), costs.end(), uint32_t(0));
}

std::optional<VPU::ComputeAndDMATimeCost> getComputeAndDataCostsForPlatform(mlir::DialectRegistry& registry,
                                                                            config::Platform platform) {
    auto interfacesRegistry = vpux::createInterfacesRegistry(platform);
    interfacesRegistry->registerInterfaces(registry);
    VPU::initializeSingletons(registry, platform);
    VPU::registerStrategies(registry, platform);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(CONV_WITH_C_TILING_IR, &ctx);
    if (module.get() == nullptr || !initModuleForPlatform(module.get(), platform)) {
        return std::nullopt;
    }

    auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
    if (func == nullptr) {
        return std::nullopt;
    }

    auto convOps = to_small_vector(func.getOps<VPU::NCEConvolutionOp>());
    if (convOps.size() != 1u) {
        return std::nullopt;
    }
    auto convOp = convOps.front();

    const auto outputType = mlir::cast<NDTypeInterface>(convOp->getResult(0).getType());
    const auto outputShape = getBoundedShape(outputType);
    const Shape cTiling({1, 2, 1, 1});
    auto outTiles = fillDividedTiles(convOp, cTiling, outputShape);
    if (mlir::failed(outTiles) || outTiles.value().size() != 2u) {
        return std::nullopt;
    }

    VPU::SiblingOpsAnalysis siblingsOpsAnalysis(func);
    auto layerCostModel = VPU::CostModelConfig::createLayerCostModel(module.get());
    VPU::LayerCostModel costModel(func, /*enablePrefetchTiling=*/true, siblingsOpsAnalysis, std::move(layerCostModel),
                                  Logger::global());

    return costModel.getComputeAndDataCosts(convOp.getOperation(), VPU::MultiClusterStrategy::SplitOverHeight,
                                            outTiles.value());
}

}  // namespace

TEST_F(MLIR_CostModelUtilsTest, ApplyStrideDMACorrectionUsesFullSearchThresholdForNPU50XXAndAbove) {
    mlir::MLIRContext ctx(registry);

    const Shape shape{1, 1, 1, 64};
    const auto tensorType = mlir::RankedTensorType::get(shape.raw(), mlir::Float16Type::get(&ctx));
    const auto ndType = mlir::dyn_cast<NDTypeInterface>(tensorType);
    ASSERT_TRUE(ndType != nullptr);
    ASSERT_EQ(ndType.getShape()[Dims4D::Act::W] * ndType.getElemTypeSize(), 1024_Bit);

    // Stride DMA correction threshold is 512 for NPU40XX
    // 1024 stride doesn't trigger cost correction
    uint32_t fullSearchCostForNPU40XX = 100;
    EXPECT_FALSE(applyStrideDMACorrectionForTile(ndType, /*isStridedDMA=*/true, fullSearchCostForNPU40XX,
                                                 config::ArchKind::NPU40XX, /*isFullSearchVersion=*/true));
    EXPECT_EQ(fullSearchCostForNPU40XX, 100u);

    // Stride DMA correction threshold is 1024 for NPU50XX with isFullSearchVersion=false
    // continuousBitsOnLowestDim = 1024 doesn't trigger cost correction
    uint32_t defaultSearchCostForNPU50XX = 100;
    EXPECT_FALSE(applyStrideDMACorrectionForTile(ndType, /*isStridedDMA=*/true, defaultSearchCostForNPU50XX,
                                                 config::ArchKind::NPU50XX, /*isFullSearchVersion=*/false));
    EXPECT_EQ(defaultSearchCostForNPU50XX, 100u);

    // Stride DMA correction threshold is 2048 for NPU50XX with isFullSearchVersion=true
    // continuousBitsOnLowestDim = 1024 triggers cost correction
    uint32_t fullSearchCostForNPU50XX = 100;
    EXPECT_TRUE(applyStrideDMACorrectionForTile(ndType, /*isStridedDMA=*/true, fullSearchCostForNPU50XX,
                                                config::ArchKind::NPU50XX, /*isFullSearchVersion=*/true));
    EXPECT_EQ(fullSearchCostForNPU50XX, 200u);
}

TEST_F(MLIR_CostModelUtilsTest, GetComputeAndDataCostsUsesFullSearchStrideDMACorrectionForNPU50XX) {
    const auto npu40XXCosts = getComputeAndDataCostsForPlatform(registry, config::Platform::NPU4000);
    const auto npu50XXCosts = getComputeAndDataCostsForPlatform(registry, config::Platform::NPU5010);

    ASSERT_TRUE(npu40XXCosts.has_value());
    ASSERT_TRUE(npu50XXCosts.has_value());
    ASSERT_EQ(npu40XXCosts->outputCosts.size(), npu50XXCosts->outputCosts.size());
    ASSERT_FALSE(npu40XXCosts->outputCosts.empty());
    ASSERT_FALSE(npu50XXCosts->outputCosts.empty());

    EXPECT_NE(sumCosts(npu40XXCosts->outputCosts), sumCosts(npu50XXCosts->outputCosts));
}
