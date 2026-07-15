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
// Test fixture for temporal tiling on large weight 1x1 convolutions (matmul-like).
//
// The tests verify that TemporalTilingDriver::getBestTilingStrategy returns
// the expected tiling shape, scenario, and rough cost for each model,
// ensuring regressions are caught during changing large-convolution temporal tiling behavior.
//
// Run cmd: npuUnitTests --gtest_filter="MLIR_TemporalTilingLargeConv.*"
//

using MLIR_TemporalTilingLargeConv = vpux::VPU::arch50xx::UnitTest;

constexpr auto kTestPlatform = config::Platform::NPU5010;

static void initModuleForTestArch(mlir::ModuleOp module, mlir::DialectRegistry& registry) {
    VPU::registerStrategies(registry, kTestPlatform);

    module->removeAttr("config.arch");

    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(kTestPlatform, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module)));
}

struct TemporalTilingExpected {
    llvm::StringRef scenario;
    int64_t baselineCost;       // current baseline: new result should stay <= this cost
    size_t validStrategyCount;  // total temporal tiling search space size
};

enum class NoTemporalTilingMode { NoStrategy, UntiledStrategy };

static std::vector<std::unique_ptr<VPU::TemporalTilingScenarioBase>> createTemporalTilingScenarios() {
    std::vector<std::unique_ptr<VPU::TemporalTilingScenarioBase>> scenarios;
    scenarios.push_back(std::make_unique<VPU::IsolatedTiling>());
    scenarios.push_back(std::make_unique<VPU::PrefetchTiling>());
    scenarios.push_back(std::make_unique<VPU::PipelineTiling>());
    return scenarios;
}

static void verifyTemporalTiling(mlir::func::FuncOp func, mlir::MLIRContext& ctx,
                                 const TemporalTilingExpected& expected) {
    auto layerCostModel = VPU::CostModelConfig::createLayerCostModel(&ctx);
    auto siblingsOpsAnalysis = vpux::VPU::SiblingOpsAnalysis(func);
    auto costModel = std::make_shared<vpux::VPU::LayerCostModel>(
            func, /*enablePrefetchTiling=*/true, siblingsOpsAnalysis, layerCostModel, vpux::Logger::global());

    bool found = false;
    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(convOp.getOperation());
        ASSERT_TRUE(tilingOp != nullptr);

        auto temporalTilingScenarios = createTemporalTilingScenarios();
        const auto validStrategies = VPU::TemporalTilingDriver::getAllValidTilingStrategies(
                tilingOp.getOperation(), temporalTilingScenarios, *costModel, vpux::Logger::global());
        size_t validStrategyCount = 0;
        for (const auto& [scenarioName, options] : validStrategies) {
            validStrategyCount += options.size();
        }

        auto result = VPU::TemporalTilingDriver::getBestTilingStrategy(tilingOp, *costModel);
        ASSERT_TRUE(result.has_value()) << "getBestTilingStrategy returned nullopt";

        auto [tilingShape, costInfo, scenarioName] = result.value();

        const std::string scenarioNameStr(scenarioName.data(), scenarioName.size());
        const std::string expectedScenarioStr(expected.scenario.data(), expected.scenario.size());

        EXPECT_EQ(scenarioNameStr, expectedScenarioStr)
                << "Scenario mismatch: got " << scenarioNameStr << " expected " << expectedScenarioStr;
        bool hasTiledDim = false;
        for (const auto tiles : tilingShape) {
            hasTiledDim |= tiles > 1;
        }
        EXPECT_TRUE(hasTiledDim) << "Expected temporal tiling shape to have at least one tiled dimension";
        EXPECT_LE(costInfo.overallCost, expected.baselineCost)
                << "Cost regression: " << costInfo.overallCost << " > " << expected.baselineCost;
        EXPECT_EQ(validStrategyCount, expected.validStrategyCount) << "Valid strategy count mismatch";
        EXPECT_GT(costInfo.overallCost, 0) << "Cost should be positive";
        EXPECT_GT(costInfo.dmaCost, 0) << "DMA cost should be positive";
        EXPECT_GT(costInfo.computeCost, 0) << "Compute cost should be positive";
        found = true;
    });
    ASSERT_TRUE(found) << "No NCE.Convolution op found in the module";
}

static void verifyNoTemporalTiling(mlir::func::FuncOp func, mlir::MLIRContext& ctx, NoTemporalTilingMode mode) {
    auto layerCostModel = VPU::CostModelConfig::createLayerCostModel(&ctx);
    auto siblingsOpsAnalysis = vpux::VPU::SiblingOpsAnalysis(func);
    auto costModel = std::make_shared<vpux::VPU::LayerCostModel>(
            func, /*enablePrefetchTiling=*/true, siblingsOpsAnalysis, layerCostModel, vpux::Logger::global());

    bool found = false;
    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(convOp.getOperation());
        ASSERT_TRUE(tilingOp != nullptr);

        auto temporalTilingScenarios = createTemporalTilingScenarios();
        const auto validStrategies = VPU::TemporalTilingDriver::getAllValidTilingStrategies(
                tilingOp.getOperation(), temporalTilingScenarios, *costModel, vpux::Logger::global());
        size_t validStrategyCount = 0;
        for (const auto& [scenarioName, options] : validStrategies) {
            validStrategyCount += options.size();
        }

        auto result = VPU::TemporalTilingDriver::getBestTilingStrategy(tilingOp, *costModel);
        if (mode == NoTemporalTilingMode::NoStrategy) {
            EXPECT_FALSE(result.has_value()) << "Expected nullopt for unsupported temporal tiling shape";
            EXPECT_EQ(validStrategyCount, 0) << "Expected no valid temporal tiling strategies";
        } else {
            ASSERT_TRUE(result.has_value()) << "Expected the unique untiled strategy";
            auto [tilingShape, costInfo, scenarioName] = result.value();
            (void)costInfo;

            const std::string scenarioNameStr(scenarioName.data(), scenarioName.size());
            EXPECT_EQ(scenarioNameStr, "ISOLATED") << "Expected untiled strategy to use ISOLATED scenario";
            EXPECT_EQ(validStrategyCount, 1) << "Expected only the unique untiled strategy";
            for (const auto tiles : tilingShape) {
                EXPECT_EQ(tiles, 1) << "Expected untiled strategy [1, 1, 1, 1]";
            }
        }
        found = true;
    });
    ASSERT_TRUE(found) << "No NCE.Convolution op found in the module";
}

struct TemporalTilingTestCase {
    llvm::StringRef funcName;
    TemporalTilingExpected expected;
};

// Wrap test cases in one moduleOp to speed up unit test run time
// Avoid calling initCompiler multiple times with same configuration
TEST_F(MLIR_TemporalTilingLargeConv, LargeConvCases) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU50XX>} {
        func.func @conv_3072x1152_to_1152x1152_SOH(%arg0: tensor<1x1152x768x4xf16, {order = #NHWC}>)
            -> tensor<1x1152x768x4xf16, {order = #NHWC}> {
            %cst_wt = const.Declare tensor<1152x1x1x4xsi32> = dense<1> : tensor<1152x1x1x4xsi32>
            %cst_w = const.Declare tensor<1152x1152x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1152x1152x1x1xf16>, [#const.Reorder<#NHWC>]
            %0 = VPU.NCE.Convolution(%arg0, %cst_w, %cst_wt) rawFilterShape [1152, 1152, 1, 1] {
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]
                        } : tensor<1x1152x768x4xf16, {order = #NHWC}>, tensor<1152x1152x1x1xf16, {order = #NHWC}>, tensor<1152x1x1x4xsi32>
                            -> tensor<1x1152x768x4xf16, {order = #NHWC}>
                        return %0 : tensor<1x1152x768x4xf16, {order = #NHWC}>
        }

        func.func @conv_1d_spatial_tiling_SOH(%arg0: tensor<1x256x2048x4xf16, {order = #NHWC}>)
            -> tensor<1x256x2048x4xf16, {order = #NHWC}> {
            %cst_wt = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
            %cst_w = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16>, [#const.Reorder<#NHWC>]
            %0 = VPU.NCE.Convolution(%arg0, %cst_w, %cst_wt) rawFilterShape [256, 256, 1, 1] {
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]
                        } : tensor<1x256x2048x4xf16, {order = #NHWC}>, tensor<256x256x1x1xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32>
                            -> tensor<1x256x2048x4xf16, {order = #NHWC}>
                        return %0 : tensor<1x256x2048x4xf16, {order = #NHWC}>
        }

        func.func @conv_fits_without_tiling_SOH(%arg0: tensor<1x64x1024x4xf16, {order = #NHWC}>)
            -> tensor<1x64x1024x4xf16, {order = #NHWC}> {
            %cst_wt = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
            %cst_w = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
            %0 = VPU.NCE.Convolution(%arg0, %cst_w, %cst_wt) rawFilterShape [64, 64, 1, 1] {
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]
                        } : tensor<1x64x1024x4xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32>
                            -> tensor<1x64x1024x4xf16, {order = #NHWC}>
                        return %0 : tensor<1x64x1024x4xf16, {order = #NHWC}>
        }

        func.func @conv_2048x16384_to_16384x2048_SOK_NoTiling(%arg0: tensor<1x16384x512x4xf16, {order = #NHWC}>)
            -> tensor<1x2048x512x4xf16, {order = #NHWC}> {
            %cst_wt = const.Declare tensor<2048x1x1x4xsi32> = dense<1> : tensor<2048x1x1x4xsi32>
            %cst_w = const.Declare tensor<2048x16384x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<2048x16384x1x1xf16>, [#const.Reorder<#NHWC>]
            %0 = VPU.NCE.Convolution(%arg0, %cst_w, %cst_wt) rawFilterShape [2048, 16384, 1, 1] {
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]
                        } : tensor<1x16384x512x4xf16, {order = #NHWC}>, tensor<2048x16384x1x1xf16, {order = #NHWC}>, tensor<2048x1x1x4xsi32>
                            -> tensor<1x2048x512x4xf16, {order = #NHWC}>
                        return %0 : tensor<1x2048x512x4xf16, {order = #NHWC}>
        }
    }
    )";

    VPU::registerStrategies(registry, kTestPlatform);
    mlir::MLIRContext localCtx(registry);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &localCtx);
    ASSERT_TRUE(module.get() != nullptr);
    initModuleForTestArch(module.get(), registry);

    const SmallVector<TemporalTilingTestCase> testCases = {
            {"conv_3072x1152_to_1152x1152_SOH", {"PIPELINE", /*baselineCost=*/800000, 21}},
            {"conv_1d_spatial_tiling_SOH", {"PIPELINE", /*baselineCost=*/120000, 11}},
    };

    for (const auto& testCase : testCases) {
        SCOPED_TRACE(testCase.funcName.str());
        auto func = module.get().lookupSymbol<mlir::func::FuncOp>(testCase.funcName);
        ASSERT_TRUE(func != nullptr);
        verifyTemporalTiling(func, localCtx, testCase.expected);
    }

    auto untiledFunc = module.get().lookupSymbol<mlir::func::FuncOp>("conv_fits_without_tiling_SOH");
    ASSERT_TRUE(untiledFunc != nullptr);
    verifyNoTemporalTiling(untiledFunc, localCtx, NoTemporalTilingMode::UntiledStrategy);

    auto unsupportedFunc = module.get().lookupSymbol<mlir::func::FuncOp>("conv_2048x16384_to_16384x2048_SOK_NoTiling");
    ASSERT_TRUE(unsupportedFunc != nullptr);
    verifyNoTemporalTiling(unsupportedFunc, localCtx, NoTemporalTilingMode::NoStrategy);
}
