//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/isolated_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/pipeline_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/prefetch_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_driver.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/init/hw_strategy_registry.hpp"

#include <gtest/gtest.h>

using namespace vpux;

//
// Test fixture verifying that the last-scenario termination logic is correctly configured in the temporal tiling
// driver. NPU5010 is used intentionally because this regression depends on PTL CMX/search-space behavior.
//
// Run cmd: npuUnitTests --gtest_filter="MLIR_InefficientTilingInlined.*"
//

using MLIR_InefficientTilingInlined = vpux::VPU::arch50xx::UnitTest;

constexpr auto kTestPlatform = config::Platform::NPU5010;

static void initModuleForTestArch(mlir::ModuleOp module, mlir::DialectRegistry& registry) {
    VPU::registerStrategies(registry, kTestPlatform);

    module->removeAttr("config.arch");

    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(kTestPlatform, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module)));
}

// IR: a 1x1 conv with moderate weight size that requires tiling.
// [1, 1152, 1024, 4] conv [1152, 1152, 1, 1] -> [1, 1152, 1024, 4]
// SOH multi-cluster. This model is known to select PIPELINE scenario.
static constexpr llvm::StringLiteral kConvIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU50XX>} {
        func.func @main(%arg0: tensor<1x1152x1024x4xf16, {order = #NHWC}>)
            -> tensor<1x1152x1024x4xf16, {order = #NHWC}> {
            %cst_wt = const.Declare tensor<1152x1x1x4xsi32> = dense<1> : tensor<1152x1x1x4xsi32>
            %cst_w = const.Declare tensor<1152x1152x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1152x1152x1x1xf16>, [#const.Reorder<#NHWC>]
            %0 = VPU.NCE.Convolution(%arg0, %cst_w, %cst_wt) rawFilterShape [1152, 1152, 1, 1] {
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]
            } : tensor<1x1152x1024x4xf16, {order = #NHWC}>, tensor<1152x1152x1x1xf16, {order = #NHWC}>, tensor<1152x1x1x4xsi32>
              -> tensor<1x1152x1024x4xf16, {order = #NHWC}>
            return %0 : tensor<1x1152x1024x4xf16, {order = #NHWC}>
        }
    }
)";

// ============================================================================
// Golden reference: getBestTilingStrategy must produce the same result after
// inlining the inefficient-tiling termination logic.
// ============================================================================
TEST_F(MLIR_InefficientTilingInlined, GetBestTilingStrategyGoldenReference) {
    VPU::registerStrategies(registry, kTestPlatform);
    mlir::MLIRContext localCtx(registry);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(kConvIR, &localCtx);
    ASSERT_TRUE(module.get() != nullptr);
    initModuleForTestArch(module.get(), registry);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto layerCostModel = VPU::CostModelConfig::createLayerCostModel(&localCtx);
    auto siblingsOpsAnalysis = vpux::VPU::SiblingOpsAnalysis(func);
    auto costModel = std::make_shared<vpux::VPU::LayerCostModel>(
            func, /*enablePrefetchTiling=*/true, siblingsOpsAnalysis, layerCostModel, vpux::Logger::global());

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(convOp.getOperation());
        ASSERT_TRUE(tilingOp != nullptr);

        // Verify search termination logic through the configured last-scenario stop predicate.
        std::vector<std::unique_ptr<VPU::TemporalTilingScenarioBase>> scenarios;
        scenarios.push_back(std::make_unique<VPU::IsolatedTiling>());
        scenarios.push_back(std::make_unique<VPU::PrefetchTiling>());
        scenarios.push_back(std::make_unique<VPU::PipelineTiling>());

        const auto& searchConfig = VPU::getDefaultTemporalTilingSearchSpaceConfig();
        const auto validStrategies = VPU::TemporalTilingDriver::getAllValidTilingStrategies(
                tilingOp.getOperation(), scenarios, *costModel, vpux::Logger::global(), searchConfig);

        // Extract PIPELINE scenario results
        auto pipelineIt = validStrategies.find("PIPELINE");
        ASSERT_TRUE(pipelineIt != validStrategies.end()) << "PIPELINE scenario not found";
        const auto& pipelineOptions = pipelineIt->second;

        ASSERT_GT(pipelineOptions.size(), 0) << "PIPELINE should have at least one valid strategy";

        // Verify termination condition: the candidate that satisfies the stop predicate is not collected.
        const auto& lastStrategy = pipelineOptions.back();
        const auto outputShape = vpux::getBoundedShape(tilingOp->getResult(0));
        auto tilingResult = vpux::fillDividedTiles(tilingOp.getOperation(), lastStrategy, outputShape);
        ASSERT_TRUE(mlir::succeeded(tilingResult) && !tilingResult.value().empty());

        auto module = tilingOp->getParentOfType<mlir::ModuleOp>();
        const auto cmxSize = vpux::VPU::getTotalCMXSize(module);
        const auto pipelineScenario = scenarios.back().get();
        const auto peakMemory =
                pipelineScenario->calculatePeakMemory(tilingOp.getOperation(), tilingResult.value(), *costModel);

        ASSERT_NE(searchConfig.lastScenarioSearchStopPredicate, nullptr);
        EXPECT_FALSE(searchConfig.lastScenarioSearchStopPredicate(peakMemory, cmxSize, searchConfig))
                << "The PIPELINE strategy that satisfies the configured stop predicate should terminate search "
                   "before it is added";
    });
}
