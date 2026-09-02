//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include "common/utils.hpp"
#include "vpux/compiler/core/tiling.hpp"
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
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/init/hw_strategy_registry.hpp"
#include "vpux/utils/core/numeric.hpp"

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
    auto log = vpux::Logger::global();
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log);
    // Added CMX memory reservation passes so that correct size of available memory is calculated before tiling
    pm.addPass(VPU::createCMXStackFramesReserveMemPass(log));
    pm.addPass(VPU::createCMXMetadataReserveMemPass(log));
    pm.addPass(VPU::createCMXAdditionalStackFramesReserveMemPass(log));
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

TEST_F(MLIR_TemporalTilingLargeConv, EfficientWorkloadAlignChangesDividedTilesSOH) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU50XX>} {
        func.func @main(%arg0: tensor<1x64x128x4xf16, {order = #NHWC}>)
            -> tensor<1x64x128x4xf16, {order = #NHWC}> {
            %cst_wt = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
            %cst_w = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
            %0 = VPU.NCE.Convolution(%arg0, %cst_w, %cst_wt) rawFilterShape [64, 64, 1, 1] {
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]
                        } : tensor<1x64x128x4xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32>
                            -> tensor<1x64x128x4xf16, {order = #NHWC}>
                        return %0 : tensor<1x64x128x4xf16, {order = #NHWC}>
        }
    }
    )";

    VPU::registerStrategies(registry, kTestPlatform);
    mlir::MLIRContext localCtx(registry);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &localCtx);
    ASSERT_TRUE(module.get() != nullptr);
    initModuleForTestArch(module.get(), registry);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    bool found = false;
    func->walk([&](VPU::NCEConvolutionOp convOp) {
        const auto outputShape = getBoundedShape(convOp->getResult(0));
        const Shape hTiling({1, 1, 6, 1});

        const auto legacyTiles = fillDividedTiles(convOp.getOperation(), hTiling, outputShape,
                                                  /*efficientWorkloadAlign=*/false);
        const auto efficientTiles = fillDividedTiles(convOp.getOperation(), hTiling, outputShape,
                                                     /*efficientWorkloadAlign=*/true);
        ASSERT_TRUE(mlir::succeeded(legacyTiles));
        ASSERT_TRUE(mlir::succeeded(efficientTiles));
        ASSERT_EQ(legacyTiles.value().size(), efficientTiles.value().size());
        ASSERT_EQ(legacyTiles.value().size(), 6u);

        const auto moduleOp = convOp->getParentOfType<mlir::ModuleOp>();
        const auto preferredSpatialAlignment = config::getPreferredSpatialAlignment(moduleOp);
        const int64_t hwAlignment =
                static_cast<int64_t>(config::getTileExecutor(moduleOp).getCount()) * preferredSpatialAlignment;

        const auto hDim = Dims4D::Act::H;
        // H = 128 tiled over 6 tiles reproduces the i4 channelwise MatMul root-cause case.
        //
        // Without efficient workload alignment (legacy/develop path) the remainder is spread
        // evenly across the leading tiles, producing uneven 22/21 spatial tiles:
        //   offsets [0, 22, 44, 65, 86, 107] -> sizes [22, 22, 21, 21, 21, 21]
        // Each of these rounds up to 2 DPU spatial quanta (~16 granularity) -> no size-8 fast tile.
        EXPECT_EQ(legacyTiles.value()[0].shape[hDim], 22);
        EXPECT_EQ(legacyTiles.value()[1].shape[hDim], 22);
        EXPECT_EQ(legacyTiles.value()[2].shape[hDim], 21);
        EXPECT_EQ(legacyTiles.value()[3].shape[hDim], 21);
        EXPECT_EQ(legacyTiles.value()[4].shape[hDim], 21);
        EXPECT_EQ(legacyTiles.value()[5].shape[hDim], 21);

        // With efficient workload alignment (orig path) the leading tiles are aligned up to multiples of the
        // configured HW alignment, leaving the remainder as a small size-8 tail tile that only costs a single DPU
        // spatial quantum:
        //   offsets [0, 24, 48, 72, 96, 120] -> sizes [24, 24, 24, 24, 24, 8]
        EXPECT_EQ(efficientTiles.value()[0].shape[hDim], 24);
        EXPECT_EQ(efficientTiles.value()[1].shape[hDim], 24);
        EXPECT_EQ(efficientTiles.value()[2].shape[hDim], 24);
        EXPECT_EQ(efficientTiles.value()[3].shape[hDim], 24);
        EXPECT_EQ(efficientTiles.value()[4].shape[hDim], 24);
        EXPECT_EQ(efficientTiles.value()[5].shape[hDim], 8);

        // The leading (non-remainder) efficient tiles must be multiples of the HW alignment.
        for (size_t tileIndex = 0; tileIndex + 1 < efficientTiles.value().size(); ++tileIndex) {
            EXPECT_EQ(efficientTiles.value()[tileIndex].shape[hDim] % hwAlignment, 0);
        }
        // The two tilings must actually differ on the leading tiles.
        EXPECT_NE(legacyTiles.value()[0].shape[hDim], efficientTiles.value()[0].shape[hDim]);

        for (auto tileIndex : irange(legacyTiles.value().size())) {
            EXPECT_EQ(legacyTiles.value()[tileIndex].axis[hDim], hTiling[hDim]);
            EXPECT_EQ(efficientTiles.value()[tileIndex].axis[hDim], hTiling[hDim]);
        }

        found = true;
    });
    ASSERT_TRUE(found) << "No NCE.Convolution op found in the module";
}

// Regression guard for efficientWorkloadAlign H-alignment selection by MC strategy.
// We use H=31 with H-tiling=2 so ceil(31/2)=16:
//   - SOH aligns H by numClusters * preferredSpatialAlignment (3 * 8 = 24) -> [24, 7]
//   - SOK aligns H by preferredSpatialAlignment only (8) -> [16, 15]
// This shape/divisor pair makes SOH and SOK behavior observably different.
TEST_F(MLIR_TemporalTilingLargeConv, EfficientWorkloadAlignWithCorrectNumCluster) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU50XX>} {
        func.func @main_soh(%arg0: tensor<1x64x31x4xf16, {order = #NHWC}>)
            -> tensor<1x64x31x4xf16, {order = #NHWC}> {
            %cst_wt = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
            %cst_w = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
            %0 = VPU.NCE.Convolution(%arg0, %cst_w, %cst_wt) rawFilterShape [64, 64, 1, 1] {
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]
                        } : tensor<1x64x31x4xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32>
                            -> tensor<1x64x31x4xf16, {order = #NHWC}>
                        return %0 : tensor<1x64x31x4xf16, {order = #NHWC}>
        }

        func.func @main_sok(%arg0: tensor<1x64x31x4xf16, {order = #NHWC}>)
            -> tensor<1x64x31x4xf16, {order = #NHWC}> {
            %cst_wt = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
            %cst_w = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
            %0 = VPU.NCE.Convolution(%arg0, %cst_w, %cst_wt) rawFilterShape [64, 64, 1, 1] {
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]
                        } : tensor<1x64x31x4xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32>
                            -> tensor<1x64x31x4xf16, {order = #NHWC}>
                        return %0 : tensor<1x64x31x4xf16, {order = #NHWC}>
        }
    }
    )";

    VPU::registerStrategies(registry, kTestPlatform);
    mlir::MLIRContext localCtx(registry);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &localCtx);
    ASSERT_TRUE(module.get() != nullptr);
    initModuleForTestArch(module.get(), registry);

    const Shape hTiling({1, 1, 2, 1});
    const auto hDim = Dims4D::Act::H;

    const auto verify = [&](llvm::StringRef funcName, VPU::MultiClusterStrategy expectedStrategy,
                            int64_t expectedSpatialAlignment, int64_t expectedTile0, int64_t expectedTile1) {
        auto func = module.get().lookupSymbol<mlir::func::FuncOp>(funcName);
        ASSERT_TRUE(func != nullptr);

        bool found = false;
        func->walk([&](VPU::NCEConvolutionOp convOp) {
            ASSERT_EQ(VPU::getMultiClusterStrategyFromOp(convOp.getOperation()), expectedStrategy);

            const auto outputShape = getBoundedShape(convOp->getResult(0));
            const auto efficientTiles =
                    fillDividedTiles(convOp.getOperation(), hTiling, outputShape, /*efficientWorkloadAlign=*/true);
            ASSERT_TRUE(mlir::succeeded(efficientTiles));
            ASSERT_EQ(efficientTiles.value().size(), 2u);

            EXPECT_EQ(efficientTiles.value()[0].shape[hDim], expectedTile0);
            EXPECT_EQ(efficientTiles.value()[1].shape[hDim], expectedTile1);
            EXPECT_EQ(efficientTiles.value()[0].shape[hDim] % expectedSpatialAlignment, 0);
            EXPECT_EQ(efficientTiles.value()[0].shape[hDim],
                      alignValUp(divUp(outputShape[hDim], hTiling[hDim]), expectedSpatialAlignment));
            EXPECT_EQ(efficientTiles.value()[0].shape[hDim] + efficientTiles.value()[1].shape[hDim], outputShape[hDim]);

            found = true;
        });
        ASSERT_TRUE(found) << "No NCE.Convolution op found in function: " << funcName.str();
    };

    const auto numClusters = static_cast<int64_t>(config::getTileExecutor(module.get()).getCount());
    const auto preferredSpatialAlignment = config::getPreferredSpatialAlignment(module.get());

    // For SOH the H alignment is numClusters * preferredSpatialAlignment.
    verify("main_soh", VPU::MultiClusterStrategy::SplitOverHeight, numClusters * preferredSpatialAlignment,
           /*expectedTile0=*/24, /*expectedTile1=*/7);

    // For SOK the H alignment stays at preferredSpatialAlignment.
    verify("main_sok", VPU::MultiClusterStrategy::SplitOverKernel, preferredSpatialAlignment, /*expectedTile0=*/16,
           /*expectedTile1=*/15);
}
