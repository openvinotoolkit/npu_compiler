//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/isolated_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/pipeline_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/prefetch_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/init/interfaces_registry.hpp"
#include "vpux/compiler/init/singleton_initializer.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

// Run cmd: npuUnitTests --gtest_filter="TemporalTilingScenario/MLIR_TemporalTilingScenario.*"

using namespace vpux;
using vpux::config::Platform;

namespace {

// Shared IR snippets for convolution-based tests
// Small convolution, no MCS: 1x16x16x16 -> 1x16x16x16, kernel 1x1
constexpr llvm::StringLiteral CONV_1x1_16x16x16_TO_16x16x16_NONE_IR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @main {
    func.func @main(%input: tensor<1x16x16x16xf16, {order = #NHWC}>)
                    -> tensor<1x16x16x16xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16, {order = #NHWC}>
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
        %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) rawFilterShape [16, 16, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>,
            tensor<16x16x1x1xf16, {order = #NHWC}>,
            tensor<16x1x1x4xsi32>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>
        return %conv : tensor<1x16x16x16xf16, {order = #NHWC}>
    }
}
)";

// Convolution: 1x64x112x112 -> 1x128x56x56, kernel 3x3, stride 2, SOH
constexpr llvm::StringLiteral CONV_3x3_64x112x112_TO_128x56x56_SOH_IR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @main {
    func.func @main(%input: tensor<1x64x112x112xf16, {order = #NHWC}>)
                    -> tensor<1x128x56x56xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<128x64x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<128x64x3x3xf16, {order = #NHWC}>
        %weights_table = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
        %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) rawFilterShape [128, 64, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            strides = [2, 2]
        } : tensor<1x64x112x112xf16, {order = #NHWC}>,
            tensor<128x64x3x3xf16, {order = #NHWC}>,
            tensor<128x1x1x4xsi32>
            -> tensor<1x128x56x56xf16, {order = #NHWC}>
        return %conv : tensor<1x128x56x56xf16, {order = #NHWC}>
    }
}
)";

// MaxPool: 1x64x112x112 -> 1x64x56x56, kernel 3x3, stride 2, SOH
// No weights operand — only activation + weight_table
constexpr llvm::StringLiteral MAXPOOL_3x3_64x112x112_TO_64x56x56_SOH_IR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @main {
    func.func @main(%input: tensor<1x64x112x112xf16, {order = #NHWC}>)
                    -> tensor<1x64x56x56xf16, {order = #NHWC}> {
        %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
        %pool = VPU.NCE.MaxPool(%input, %weights_table) {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            kernel_size = [3, 3],
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            strides = [2, 2]
        } -> tensor<1x64x56x56xf16, {order = #NHWC}>
        return %pool : tensor<1x64x56x56xf16, {order = #NHWC}>
    }
}
)";

// DepthConvolution: 1x64x112x112 -> 1x64x56x56, kernel 3x3, stride 2, SOH
// Physical filter shape is [OC, KY*KX+pad, 1, 1] = [64, 16, 1, 1] for 3x3 kernel aligned to 16
constexpr llvm::StringLiteral DEPTHCONV_3x3_64x112x112_TO_64x56x56_SOH_IR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @main {
    func.func @main(%input: tensor<1x64x112x112xf16, {order = #NHWC}>)
                    -> tensor<1x64x56x56xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x1x1xf16, {order = #NHWC}>
        %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
        %dconv = VPU.NCE.DepthConvolution(%input, %weights, %weights_table : tensor<64x1x1x4xsi32>) rawFilterShape [64, 1, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            strides = [2, 2]
        } : tensor<1x64x112x112xf16, {order = #NHWC}>,
            tensor<64x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x56x56xf16, {order = #NHWC}>
        return %dconv : tensor<1x64x56x56xf16, {order = #NHWC}>
    }
}
)";

// Large 1x1 conv with SOH: 1x256x128x128 -> 1x256x128x128
// Large spatial dims (128x128) make SOH effective; pipeline beneficial
constexpr llvm::StringLiteral CONV_1x1_256x128x128_TO_256x128x128_SOH_IR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @main {
    func.func @main(%input: tensor<1x256x128x128xf16, {order = #NHWC}>)
                    -> tensor<1x256x128x128xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
        %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) rawFilterShape [256, 256, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1]
        } : tensor<1x256x128x128xf16, {order = #NHWC}>,
            tensor<256x256x1x1xf16, {order = #NHWC}>,
            tensor<256x1x1x4xsi32>
            -> tensor<1x256x128x128xf16, {order = #NHWC}>
        return %conv : tensor<1x256x128x128xf16, {order = #NHWC}>
    }
}
)";

// 3x3 conv with SOK: 1x64x56x56 -> 1x128x56x56
// SplitOverKernel distributes output channels across clusters.
// Activations are broadcast (shared), weights are split.
constexpr llvm::StringLiteral CONV_3x3_64x56x56_TO_128x56x56_SOK_IR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @main {
    func.func @main(%input: tensor<1x64x56x56xf16, {order = #NHWC}>)
                    -> tensor<1x128x56x56xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<128x64x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<128x64x3x3xf16, {order = #NHWC}>
        %weights_table = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
        %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) rawFilterShape [128, 64, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            strides = [1, 1]
        } : tensor<1x64x56x56xf16, {order = #NHWC}>,
            tensor<128x64x3x3xf16, {order = #NHWC}>,
            tensor<128x1x1x4xsi32>
            -> tensor<1x128x56x56xf16, {order = #NHWC}>
        return %conv : tensor<1x128x56x56xf16, {order = #NHWC}>
    }
}
)";

// Large 3x3 conv with SOK: 1x128x112x112 -> 1x768x56x56, stride 2
// SplitOverKernel with OC=768 ensures full cluster utilization after C-tiling.
// Bidimensional tiling (C=8, H=4) yields per-tile OC=96, H=14.
constexpr llvm::StringLiteral CONV_3x3_128x112x112_TO_768x56x56_SOK_IR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @main {
    func.func @main(%input: tensor<1x128x112x112xf16, {order = #NHWC}>)
                    -> tensor<1x768x56x56xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<768x128x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<768x128x3x3xf16, {order = #NHWC}>
        %weights_table = const.Declare tensor<768x1x1x4xsi32> = dense<1> : tensor<768x1x1x4xsi32>
        %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) rawFilterShape [768, 128, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            strides = [2, 2]
        } : tensor<1x128x112x112xf16, {order = #NHWC}>,
            tensor<768x128x3x3xf16, {order = #NHWC}>,
            tensor<768x1x1x4xsi32>
            -> tensor<1x768x56x56xf16, {order = #NHWC}>
        return %conv : tensor<1x768x56x56xf16, {order = #NHWC}>
    }
}
)";

// Helper: parse IR, run init compiler pipeline, return module
mlir::OwningOpRef<mlir::ModuleOp> parseAndInitModule(llvm::StringLiteral ir, mlir::MLIRContext& ctx,
                                                     Platform platform = Platform::NPU3720) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, &ctx);
    if (!module.get()) {
        return {};
    }

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initOptions = VPU::InitCompilerOptions(platform, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initOptions, Logger::global());

    if (mlir::failed(pm.run(module.get()))) {
        return {};
    }
    return module;
}

// Bundle that keeps SiblingOpsAnalysis alive alongside the LayerCostModel it feeds.
struct CostModelBundle {
    VPU::SiblingOpsAnalysis siblingsAnalysis;
    std::shared_ptr<VPU::LayerCostModel> costModel;

    CostModelBundle(mlir::ModuleOp moduleOp, mlir::func::FuncOp func): siblingsAnalysis(func) {
        auto vpunnCostModel = VPU::CostModelConfig::createLayerCostModel(moduleOp);
        bool enablePrefetchTiling = true;
        costModel = std::make_shared<VPU::LayerCostModel>(func, enablePrefetchTiling, siblingsAnalysis,
                                                          std::move(vpunnCostModel), Logger::global());
    }
};

std::string platformName(const testing::TestParamInfo<Platform>& info) {
    return stringifyEnum(info.param).str();
}

}  // namespace

class MLIR_TemporalTilingScenario : public testing::TestWithParam<Platform> {
public:
    void SetUp() override {
        registry = vpux::createDialectRegistry();
        auto interfacesRegistry = vpux::createInterfacesRegistry(GetParam());
        interfacesRegistry->registerInterfaces(registry);
        VPU::initializeSingletons(registry, GetParam());

        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<VPU::VPUDialect>();
    }

protected:
    mlir::DialectRegistry registry;
    mlir::MLIRContext ctx;
};

// ============================================================================
// Small 1x1 conv: fits CMX in a single tile
// IsolatedTiling should satisfy with [1,1,1,1] (no tiling); Pipeline/Prefetch should NOT
// (minTileCount=2)
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, SmallConv_NoTilingNeeded_SatisfyMemory) {
    auto module = parseAndInitModule(CONV_1x1_16x16x16_TO_16x16x16_NONE_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        Shape noTiling({1, 1, 1, 1});

        // IsolatedTiling: minTileCount=1, single tile should fit
        VPU::IsolatedTiling isolated;
        EXPECT_TRUE(isolated.satisfyMemoryConstraint(convOp, noTiling, *bundle.costModel, Logger::global()));

        // Pipeline/Prefetch: minTileCount=2, single tile should NOT satisfy (needs >=2 tiles)
        VPU::PrefetchTiling prefetch;
        EXPECT_FALSE(prefetch.satisfyMemoryConstraint(convOp, noTiling, *bundle.costModel, Logger::global()));

        VPU::PipelineTiling pipeline;
        EXPECT_FALSE(pipeline.satisfyMemoryConstraint(convOp, noTiling, *bundle.costModel, Logger::global()));
    });
}

// ============================================================================
// Small 1x1 conv: cost calculation with single tile
// IsolatedTiling should produce a valid cost (skips DMA for single tile)
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, SmallConv_SingleTile_CostValid) {
    auto module = parseAndInitModule(CONV_1x1_16x16x16_TO_16x16x16_NONE_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(convOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        Shape noTiling({1, 1, 1, 1});
        auto tiles = fillDividedTiles(convOp, noTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(tiles));
        ASSERT_FALSE(tiles.value().empty());

        VPU::IsolatedTiling isolated;
        auto cost = isolated.calculateCost(convOp, tiles.value(), *bundle.costModel);

        // Cost should be valid
        EXPECT_GT(cost.computeCost, 0);
        EXPECT_GT(cost.overallCost, 0);
        // IsolatedTiling skips activation and output DMA for single tile,
        // but weight DMA is still counted, so dmaCost >= 0
        EXPECT_GE(cost.dmaCost, 0);
    });
}

// ============================================================================
// Small 1x1 conv: peak memory with single tile
// IsolatedTiling should report non-zero peak memory
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, SmallConv_SingleTile_PeakMemory) {
    auto module = parseAndInitModule(CONV_1x1_16x16x16_TO_16x16x16_NONE_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(convOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        Shape noTiling({1, 1, 1, 1});
        auto tiles = fillDividedTiles(convOp, noTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(tiles));

        VPU::IsolatedTiling isolated;
        auto peakMem = isolated.calculatePeakMemory(convOp, tiles.value(), *bundle.costModel);
        EXPECT_GT(peakMem.count(), 0);

        // Pipeline/Prefetch return 0 for single tile (minTileCount=2)
        VPU::PrefetchTiling prefetch;
        EXPECT_EQ(prefetch.calculatePeakMemory(convOp, tiles.value(), *bundle.costModel), Byte(0));

        VPU::PipelineTiling pipeline;
        EXPECT_EQ(pipeline.calculatePeakMemory(convOp, tiles.value(), *bundle.costModel), Byte(0));
    });
}

// ============================================================================
// Large 3x3 conv: tiling on H dimension should satisfy memory constraints for all scenarios
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, LargeConv_HTiling_SatisfyMemory) {
    auto module = parseAndInitModule(CONV_3x3_64x112x112_TO_128x56x56_SOH_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        // Tile on H should fit even for pipeline/prefetch
        Shape hTiling({1, 1, 8, 1});

        VPU::IsolatedTiling isolated;
        VPU::PrefetchTiling prefetch;
        VPU::PipelineTiling pipeline;

        EXPECT_TRUE(isolated.satisfyMemoryConstraint(convOp, hTiling, *bundle.costModel, Logger::global()));
        EXPECT_TRUE(prefetch.satisfyMemoryConstraint(convOp, hTiling, *bundle.costModel, Logger::global()));
        EXPECT_TRUE(pipeline.satisfyMemoryConstraint(convOp, hTiling, *bundle.costModel, Logger::global()));
    });
}

// ============================================================================
// Large 3x3 conv: cost comparison across scenarios with H tiling
// Isolated >= Prefetch >= Pipeline (for tile overall cost formula)
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, LargeConv_HTiling_CostOrdering) {
    auto module = parseAndInitModule(CONV_3x3_64x112x112_TO_128x56x56_SOH_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(convOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        Shape hTiling({1, 1, 4, 1});
        auto tiles = fillDividedTiles(convOp, hTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(tiles));
        ASSERT_GE(tiles.value().size(), 2u);

        VPU::IsolatedTiling isolated;
        VPU::PrefetchTiling prefetch;
        VPU::PipelineTiling pipeline;

        auto isolatedCost = isolated.calculateCost(convOp, tiles.value(), *bundle.costModel);
        auto prefetchCost = prefetch.calculateCost(convOp, tiles.value(), *bundle.costModel);
        auto pipelineCost = pipeline.calculateCost(convOp, tiles.value(), *bundle.costModel);

        // Due to the cost calculations:
        // Isolated (serial sum) >= Prefetch (max(in,compute)+out) >= Pipeline (max(max(in,compute),out)) > 0
        EXPECT_GE(isolatedCost.overallCost, prefetchCost.overallCost);
        EXPECT_GE(prefetchCost.overallCost, pipelineCost.overallCost);
        EXPECT_GT(pipelineCost.overallCost, 0);
    });
}

// ============================================================================
// Large 3x3 conv: peak memory ordering
// Pipeline (double-buffered) >= Prefetch (prefetch bucket) >= Isolated (single)
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, LargeConv_HTiling_PeakMemoryOrdering) {
    auto module = parseAndInitModule(CONV_3x3_64x112x112_TO_128x56x56_SOH_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(convOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        Shape hTiling({1, 1, 4, 1});
        auto tiles = fillDividedTiles(convOp, hTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(tiles));
        ASSERT_GE(tiles.value().size(), 2u);

        VPU::IsolatedTiling isolated;
        VPU::PrefetchTiling prefetch;
        VPU::PipelineTiling pipeline;

        auto isolatedPeak = isolated.calculatePeakMemory(convOp, tiles.value(), *bundle.costModel);
        auto prefetchPeak = prefetch.calculatePeakMemory(convOp, tiles.value(), *bundle.costModel);
        auto pipelinePeak = pipeline.calculatePeakMemory(convOp, tiles.value(), *bundle.costModel);

        // Pipeline (2x individual + shared) should use the most memory
        EXPECT_GE(pipelinePeak, prefetchPeak);
        // Prefetch (individual + shared + prefetch) >= Isolated (individual only) > 0
        EXPECT_GE(prefetchPeak, isolatedPeak);
        // All peaks should be non-zero
        EXPECT_GT(isolatedPeak.count(), 0);
    });
}

// ============================================================================
// Large 3x3 conv: compute cost and DMA cost are consistent across scenarios
// Same VPUNN workloads -> same compute cost; DMA costs may differ due to sharing
// Conv efficiency: for NCE.Convolution the efficiency field should be positive
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, LargeConv_HTiling_ComputeCostConsistent) {
    auto module = parseAndInitModule(CONV_3x3_64x112x112_TO_128x56x56_SOH_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(convOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        Shape hTiling({1, 1, 4, 1});
        auto tiles = fillDividedTiles(convOp, hTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(tiles));

        VPU::IsolatedTiling isolated;
        VPU::PrefetchTiling prefetch;
        VPU::PipelineTiling pipeline;

        auto isolatedCost = isolated.calculateCost(convOp, tiles.value(), *bundle.costModel);
        auto prefetchCost = prefetch.calculateCost(convOp, tiles.value(), *bundle.costModel);
        auto pipelineCost = pipeline.calculateCost(convOp, tiles.value(), *bundle.costModel);

        // Compute cost is determined by VPUNN and is the same regardless of scheduling scenario
        EXPECT_EQ(isolatedCost.computeCost, prefetchCost.computeCost);
        EXPECT_EQ(pipelineCost.computeCost, prefetchCost.computeCost);

        // Convolutions should report positive efficiency
        EXPECT_GT(isolatedCost.efficiency, 0.0);
        EXPECT_LE(isolatedCost.efficiency, 1.0);
    });
}

// ============================================================================
// MaxPool: non-conv op — efficiency should be -1.0, cost still valid
// Only 2 operands (activation + weight_table), no weights operand
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, MaxPool_CostEfficiencyIsNegative) {
    auto module = parseAndInitModule(MAXPOOL_3x3_64x112x112_TO_64x56x56_SOH_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEMaxPoolOp poolOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(poolOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        Shape hTiling({1, 1, 4, 1});
        auto tiles = fillDividedTiles(poolOp, hTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(tiles));
        ASSERT_GE(tiles.value().size(), 2u);

        VPU::IsolatedTiling isolated;
        auto cost = isolated.calculateCost(poolOp, tiles.value(), *bundle.costModel);

        // Non-convolution ops report efficiency = -1.0
        EXPECT_DOUBLE_EQ(cost.efficiency, -1.0);
        // Cost is still valid
        EXPECT_GT(cost.computeCost, 0);
        EXPECT_GT(cost.overallCost, 0);
    });
}

// ============================================================================
// MaxPool: peak memory and cost ordering should hold just like convolution
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, MaxPool_HTiling_CostAndPeakOrdering) {
    auto module = parseAndInitModule(MAXPOOL_3x3_64x112x112_TO_64x56x56_SOH_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEMaxPoolOp poolOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(poolOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        Shape hTiling({1, 1, 4, 1});
        auto tiles = fillDividedTiles(poolOp, hTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(tiles));
        ASSERT_GE(tiles.value().size(), 2u);

        VPU::IsolatedTiling isolated;
        VPU::PrefetchTiling prefetch;
        VPU::PipelineTiling pipeline;

        auto isolatedCost = isolated.calculateCost(poolOp, tiles.value(), *bundle.costModel);
        auto prefetchCost = prefetch.calculateCost(poolOp, tiles.value(), *bundle.costModel);
        auto pipelineCost = pipeline.calculateCost(poolOp, tiles.value(), *bundle.costModel);

        // Cost ordering: Isolated >= Prefetch >= Pipeline
        EXPECT_GE(isolatedCost.overallCost, prefetchCost.overallCost);
        EXPECT_GE(prefetchCost.overallCost, pipelineCost.overallCost);

        // Peak memory ordering: Pipeline >= Prefetch >= Isolated
        auto isolatedPeak = isolated.calculatePeakMemory(poolOp, tiles.value(), *bundle.costModel);
        auto prefetchPeak = prefetch.calculatePeakMemory(poolOp, tiles.value(), *bundle.costModel);
        auto pipelinePeak = pipeline.calculatePeakMemory(poolOp, tiles.value(), *bundle.costModel);

        EXPECT_GE(pipelinePeak, prefetchPeak);
        EXPECT_GE(prefetchPeak, isolatedPeak);
    });
}

// ============================================================================
// DepthConvolution: verifies cost and peak memory work for depthwise ops
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, DepthConv_HTiling_CostAndPeakValid) {
    auto module = parseAndInitModule(DEPTHCONV_3x3_64x112x112_TO_64x56x56_SOH_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEDepthConvolutionOp dconvOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(dconvOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        Shape hTiling({1, 1, 4, 1});
        auto tiles = fillDividedTiles(dconvOp, hTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(tiles));
        ASSERT_GE(tiles.value().size(), 2u);

        VPU::IsolatedTiling isolated;
        VPU::PrefetchTiling prefetch;
        VPU::PipelineTiling pipeline;

        // Cost should be valid for all scenarios
        auto isolatedCost = isolated.calculateCost(dconvOp, tiles.value(), *bundle.costModel);
        auto prefetchCost = prefetch.calculateCost(dconvOp, tiles.value(), *bundle.costModel);
        auto pipelineCost = pipeline.calculateCost(dconvOp, tiles.value(), *bundle.costModel);

        EXPECT_GT(isolatedCost.overallCost, 0);
        EXPECT_GT(prefetchCost.overallCost, 0);
        EXPECT_GT(pipelineCost.overallCost, 0);

        // DepthConv is NCEDepthConvolutionOp (not NCEConvolutionOp): efficiency = -1.0
        EXPECT_DOUBLE_EQ(isolatedCost.efficiency, -1.0);

        // Peak memory ordering holds
        auto isolatedPeak = isolated.calculatePeakMemory(dconvOp, tiles.value(), *bundle.costModel);
        auto prefetchPeak = prefetch.calculatePeakMemory(dconvOp, tiles.value(), *bundle.costModel);
        auto pipelinePeak = pipeline.calculatePeakMemory(dconvOp, tiles.value(), *bundle.costModel);

        EXPECT_GE(pipelinePeak, prefetchPeak);
        EXPECT_GE(prefetchPeak, isolatedPeak);
        EXPECT_GT(isolatedPeak.count(), 0);
    });
}

// ============================================================================
// C-tiling vs H-tiling: different dimensions produce different cost/peak profiles
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, LargeConv_CTiling_vs_HTiling) {
    auto module = parseAndInitModule(CONV_3x3_64x112x112_TO_128x56x56_SOH_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(convOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        // C-only tiling: each C-tile uses a different slice of weights (splits output channels),
        // so weights are not shared (unlike H-tiling where all tiles use the full filter).
        // Use enough tiles so that tiles fit CMX.
        Shape cTiling({1, 8, 1, 1});
        auto cTiles = fillDividedTiles(convOp, cTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(cTiles));
        ASSERT_GE(cTiles.value().size(), 2u);

        // H-only tiling for comparison with the same number of tiles
        Shape hTiling({1, 1, 8, 1});
        auto hTiles = fillDividedTiles(convOp, hTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(hTiles));
        ASSERT_GE(hTiles.value().size(), 2u);

        VPU::PipelineTiling pipeline;

        // Peak memory with C-tiling vs H-tiling may differ due to different shared/individual classification
        auto cPeak = pipeline.calculatePeakMemory(convOp, cTiles.value(), *bundle.costModel);
        auto hPeak = pipeline.calculatePeakMemory(convOp, hTiles.value(), *bundle.costModel);

        EXPECT_GT(cPeak.count(), 0);
        EXPECT_GT(hPeak.count(), 0);

        // Costs should be valid for both
        auto cCost = pipeline.calculateCost(convOp, cTiles.value(), *bundle.costModel);
        auto hCost = pipeline.calculateCost(convOp, hTiles.value(), *bundle.costModel);

        EXPECT_GT(cCost.overallCost, 0);
        EXPECT_GT(hCost.overallCost, 0);

        // C-tiling and H-tiling should generally produce different costs/peaks
        // because they split different tensor dimensions
        EXPECT_TRUE(cPeak != hPeak || cCost.overallCost != hCost.overallCost)
                << "C-tiling and H-tiling should differ in cost or peak memory";
    });
}

// ============================================================================
// Peak memory with different tile counts: more tiles = smaller per-tile buffers
// Verifies peak memory is valid for both coarse and fine tiling granularities
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, LargeConv_DifferentTileCounts_PeakMemoryValid) {
    auto module = parseAndInitModule(CONV_3x3_64x112x112_TO_128x56x56_SOH_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(convOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        // Output H=56. Use 4-way tiling (H=14 per tile) and 7-way tiling (H=8 per tile).
        // Both keep per-tile H large enough for SOH across all target archs
        Shape fewerTiles({1, 1, 4, 1});
        auto tiles4 = fillDividedTiles(convOp, fewerTiles, outputShape);
        ASSERT_TRUE(mlir::succeeded(tiles4));
        ASSERT_GE(tiles4.value().size(), 4u);

        Shape moreTiles({1, 1, 7, 1});
        auto tiles7 = fillDividedTiles(convOp, moreTiles, outputShape);
        ASSERT_TRUE(mlir::succeeded(tiles7));
        ASSERT_GE(tiles7.value().size(), 7u);

        VPU::IsolatedTiling isolated;
        auto peak4 = isolated.calculatePeakMemory(convOp, tiles4.value(), *bundle.costModel);
        auto peak7 = isolated.calculatePeakMemory(convOp, tiles7.value(), *bundle.costModel);

        // Both should be valid
        EXPECT_GT(peak4.count(), 0);
        EXPECT_GT(peak7.count(), 0);
    });
}

// ============================================================================
// Large SOH conv (256x256x1x1): pipeline should produce lower cost than isolated
// Large weights (~128KB) and large activation make DMA/compute overlap beneficial
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, LargeConv_SOH_PipelineCostLowerThanIsolated) {
    auto module = parseAndInitModule(CONV_1x1_256x128x128_TO_256x128x128_SOH_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(convOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        // Fine H-tiling: 128/8 = 16 lines per tile
        Shape hTiling({1, 1, 8, 1});
        auto tiles = fillDividedTiles(convOp, hTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(tiles));
        ASSERT_GE(tiles.value().size(), 2u);

        VPU::IsolatedTiling isolated;
        VPU::PrefetchTiling prefetch;
        VPU::PipelineTiling pipeline;

        auto isolatedCost = isolated.calculateCost(convOp, tiles.value(), *bundle.costModel);
        auto prefetchCost = prefetch.calculateCost(convOp, tiles.value(), *bundle.costModel);
        auto pipelineCost = pipeline.calculateCost(convOp, tiles.value(), *bundle.costModel);

        // Expected cost ordering: INT64_MAX > Isolated >= Prefetch >= Pipeline > 0
        // All costs should be valid (not sentinel INT64_MAX)
        EXPECT_NE(isolatedCost.overallCost, INT64_MAX);
        EXPECT_GE(isolatedCost.overallCost, prefetchCost.overallCost);
        EXPECT_GE(prefetchCost.overallCost, pipelineCost.overallCost);
        EXPECT_GT(pipelineCost.overallCost, 0);
    });
}

// ============================================================================
// SplitOverKernel: C-tiling with SOK strategy — costs and peaks should be valid
// SOK distributes output channels across clusters (activations shared, weights split)
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, Conv_SOK_CTiling_CostAndPeakValid) {
    auto module = parseAndInitModule(CONV_3x3_64x56x56_TO_128x56x56_SOK_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(convOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        // C-tiling
        Shape cTiling({1, 4, 1, 1});
        auto cTiles = fillDividedTiles(convOp, cTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(cTiles));
        ASSERT_GE(cTiles.value().size(), 2u);

        VPU::IsolatedTiling isolated;
        VPU::PrefetchTiling prefetch;
        VPU::PipelineTiling pipeline;

        auto isolatedCost = isolated.calculateCost(convOp, cTiles.value(), *bundle.costModel);
        auto prefetchCost = prefetch.calculateCost(convOp, cTiles.value(), *bundle.costModel);
        auto pipelineCost = pipeline.calculateCost(convOp, cTiles.value(), *bundle.costModel);

        // Expected cost ordering: INT64_MAX > Isolated >= Prefetch >= Pipeline > 0
        EXPECT_NE(isolatedCost.overallCost, INT64_MAX);
        EXPECT_GE(isolatedCost.overallCost, prefetchCost.overallCost);
        EXPECT_GE(prefetchCost.overallCost, pipelineCost.overallCost);
        EXPECT_GT(pipelineCost.overallCost, 0);

        // Peak memory should be valid for all scenarios
        auto isolatedPeak = isolated.calculatePeakMemory(convOp, cTiles.value(), *bundle.costModel);
        auto prefetchPeak = prefetch.calculatePeakMemory(convOp, cTiles.value(), *bundle.costModel);
        auto pipelinePeak = pipeline.calculatePeakMemory(convOp, cTiles.value(), *bundle.costModel);

        // Peak ordering: Pipeline >= Prefetch >= Isolated > 0
        EXPECT_GE(pipelinePeak, prefetchPeak);
        EXPECT_GE(prefetchPeak, isolatedPeak);
        EXPECT_GT(isolatedPeak.count(), 0);

        // Efficiency should be positive for NCE.Convolution
        EXPECT_GT(isolatedCost.efficiency, 0.0);
    });
}

// ============================================================================
// Large SOK conv with bidimensional tiling (C+H): verifies cost and peak memory
// with nested tiling across both output channels and height.
// OC=768, tiling {1,8,4,1} -> per tile OC=96, H=14.
// SOK distributes 96 channels per tile across clusters
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, LargeConv_SOK_BiDimTiling_CostAndPeakValid) {
    auto module = parseAndInitModule(CONV_3x3_128x112x112_TO_768x56x56_SOK_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(convOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        VPU::IsolatedTiling isolated;
        VPU::PrefetchTiling prefetch;
        VPU::PipelineTiling pipeline;

        // No tiling —> 768x128x3x3 weights (~3.4MB) won't fit CMX
        Shape noTiling({1, 1, 1, 1});
        EXPECT_FALSE(isolated.satisfyMemoryConstraint(convOp, noTiling, *bundle.costModel, Logger::global()));
        EXPECT_FALSE(prefetch.satisfyMemoryConstraint(convOp, noTiling, *bundle.costModel, Logger::global()));
        EXPECT_FALSE(pipeline.satisfyMemoryConstraint(convOp, noTiling, *bundle.costModel, Logger::global()));

        // Insufficient C-only tiling: OC=768/2=384 per tile, weights still ~1.7MB
        Shape smallCTiling({1, 2, 1, 1});
        EXPECT_FALSE(isolated.satisfyMemoryConstraint(convOp, smallCTiling, *bundle.costModel, Logger::global()));
        EXPECT_FALSE(prefetch.satisfyMemoryConstraint(convOp, smallCTiling, *bundle.costModel, Logger::global()));
        EXPECT_FALSE(pipeline.satisfyMemoryConstraint(convOp, smallCTiling, *bundle.costModel, Logger::global()));

        // Adequate tiling (2D): C=8, H=4 -> per tile OC=96, H=14
        Shape biDimTiling({1, 8, 4, 1});
        EXPECT_TRUE(isolated.satisfyMemoryConstraint(convOp, biDimTiling, *bundle.costModel, Logger::global()));
        EXPECT_TRUE(prefetch.satisfyMemoryConstraint(convOp, biDimTiling, *bundle.costModel, Logger::global()));
        EXPECT_TRUE(pipeline.satisfyMemoryConstraint(convOp, biDimTiling, *bundle.costModel, Logger::global()));

        auto tiles = fillDividedTiles(convOp, biDimTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(tiles));
        ASSERT_GE(tiles.value().size(), 16u);

        auto isolatedCost = isolated.calculateCost(convOp, tiles.value(), *bundle.costModel);
        auto prefetchCost = prefetch.calculateCost(convOp, tiles.value(), *bundle.costModel);
        auto pipelineCost = pipeline.calculateCost(convOp, tiles.value(), *bundle.costModel);

        // Expected cost ordering: Isolated >= Prefetch >= Pipeline > 0
        EXPECT_NE(isolatedCost.overallCost, INT64_MAX);
        EXPECT_GE(isolatedCost.overallCost, prefetchCost.overallCost);
        EXPECT_GE(prefetchCost.overallCost, pipelineCost.overallCost);
        EXPECT_GT(pipelineCost.overallCost, 0);

        // Peak memory should be valid for all scenarios
        auto isolatedPeak = isolated.calculatePeakMemory(convOp, tiles.value(), *bundle.costModel);
        auto prefetchPeak = prefetch.calculatePeakMemory(convOp, tiles.value(), *bundle.costModel);
        auto pipelinePeak = pipeline.calculatePeakMemory(convOp, tiles.value(), *bundle.costModel);

        // Peak ordering: Pipeline >= Prefetch >= Isolated > 0
        EXPECT_GE(pipelinePeak, prefetchPeak);
        EXPECT_GE(prefetchPeak, isolatedPeak);
        EXPECT_GT(isolatedPeak.count(), 0);

        // Efficiency should be positive for NCE.Convolution
        EXPECT_GT(isolatedCost.efficiency, 0.0);
    });
}

// ============================================================================
// Shared buffer detection: H-tiling, weights offsets stay constant across tiles,
// so weights DMA is counted only on tile 0 (stall) and skipped for subsequent tiles.
// This means dmaCost < numTiles * (perTileActCost + perTileWgtCost).
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, SharedBuffer_HTiling_WeightsAliveAcrossTiles) {
    auto module = parseAndInitModule(CONV_3x3_64x112x112_TO_128x56x56_SOH_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(convOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        // H-tiling: weights are shared across all tiles (same offsets)
        Shape hTiling({1, 1, 4, 1});
        auto tiles = fillDividedTiles(convOp, hTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(tiles));
        const auto& outTiles = tiles.value();
        const auto numTiles = outTiles.size();
        ASSERT_EQ(numTiles, 4u);

        // Use IsolatedTiling (purely additive cost formula) for clear verification
        VPU::IsolatedTiling isolated;
        auto cost = isolated.calculateCost(convOp, outTiles, *bundle.costModel);

        // Verify cost is valid
        EXPECT_GT(cost.dmaCost, 0);
        EXPECT_GT(cost.overallCost, 0);

        // Verify the back-inferred input offsets confirm weight sharing:
        // For H-tiling on convolution, all tiles should have the same weight offsets
        auto tilingOp = mlir::cast<VPU::TilingBuilderOpInterface>(convOp.getOperation());
        const auto inputTiles0 = tilingOp.backInferTileInfo(outTiles[0], Logger::global()).tiles;
        ASSERT_GE(inputTiles0.size(), 2u);
        const auto& weightOffsetsForTile0 = inputTiles0[1].offsets;

        for (size_t i = 1; i < numTiles; ++i) {
            const auto inputTilesI = tilingOp.backInferTileInfo(outTiles[i], Logger::global()).tiles;
            ASSERT_GE(inputTilesI.size(), 2u);
            // Weight offsets should be identical for all tiles in H-tiling
            EXPECT_EQ(inputTilesI[1].offsets, weightOffsetsForTile0)
                    << "H-tiling: weight offsets should be constant across tiles (tile " << i << ")";
        }

        // --- Verify the algorithm correctly skips DMA for shared weights ---
        // Strategy: compute cost for all 4 tiles together vs two separate halves.
        // When computed together, weight DMA is only counted at tile 0 (offset doesn't change).
        // When computed in halves, each half resets offset tracking => weight DMA counted in both.
        // Therefore: fullCost.dmaCost < firstHalf.dmaCost + secondHalf.dmaCost
        VPU::PipelineTiling pipeline;
        auto fullCost = pipeline.calculateCost(convOp, tiles.value(), *bundle.costModel);
        EXPECT_GT(fullCost.dmaCost, 0);

        const auto halfSize = numTiles / 2;
        OutputTiling firstHalf(tiles.value().begin(), tiles.value().begin() + halfSize);
        OutputTiling secondHalf(tiles.value().begin() + halfSize, tiles.value().end());
        auto firstHalfCost = pipeline.calculateCost(convOp, firstHalf, *bundle.costModel);
        auto secondHalfCost = pipeline.calculateCost(convOp, secondHalf, *bundle.costModel);

        // Full tiling has less DMA because weights are counted once; split halves count them twice.
        EXPECT_LT(fullCost.dmaCost, vpux::addSaturating(firstHalfCost.dmaCost, secondHalfCost.dmaCost))
                << "Shared weights: full tiling dmaCost < sum of split halves (weights DMA'd once vs twice)";

        // Pipeline benefits from overlapping activation DMA with compute on middle tiles.
        EXPECT_LE(fullCost.overallCost, cost.overallCost)
                << "Pipeline should be cheaper than isolated due to DMA/compute overlap on middle tiles";
    });
}

// ============================================================================
// Shared buffer detection: C-tiling, activation offsets stay constant across tiles,
// so activation DMA is counted only on tile 0 (stall) and skipped for subsequent tiles.
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, SharedBuffer_CTiling_ActivationAliveAcrossTiles) {
    auto module = parseAndInitModule(CONV_3x3_64x56x56_TO_128x56x56_SOK_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(convOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        // C-tiling: splits output channels, activations are shared
        Shape cTiling({1, 4, 1, 1});
        auto tiles = fillDividedTiles(convOp, cTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(tiles));
        const auto& outTiles = tiles.value();
        const auto numTiles = outTiles.size();
        ASSERT_EQ(numTiles, 4u);

        // Verify the back-inferred input offsets confirm activation sharing:
        // For C-tiling on convolution, all tiles should have the same activation offsets
        auto tilingOp = mlir::cast<VPU::TilingBuilderOpInterface>(convOp.getOperation());
        const auto inputTiles0 = tilingOp.backInferTileInfo(outTiles[0], Logger::global()).tiles;
        ASSERT_GE(inputTiles0.size(), 2u);
        const auto& actOffsetsForTile0 = inputTiles0[0].offsets;
        const auto& wgtOffsetsForTile0 = inputTiles0[1].offsets;

        bool allActSame = true;
        bool allWgtSame = true;
        for (size_t i = 1; i < numTiles; ++i) {
            const auto inputTilesI = tilingOp.backInferTileInfo(outTiles[i], Logger::global()).tiles;
            ASSERT_GE(inputTilesI.size(), 2u);
            if (inputTilesI[0].offsets != actOffsetsForTile0) {
                allActSame = false;
            }
            if (inputTilesI[1].offsets != wgtOffsetsForTile0) {
                allWgtSame = false;
            }
        }

        // C-tiling: activation offsets should be the same (shared), weights should differ
        EXPECT_TRUE(allActSame) << "C-tiling: activation offsets should be constant across tiles";
        EXPECT_FALSE(allWgtSame) << "C-tiling: weight offsets should vary across tiles";

        // Cost should be valid
        VPU::IsolatedTiling isolated;
        auto cost = isolated.calculateCost(convOp, outTiles, *bundle.costModel);
        EXPECT_GT(cost.dmaCost, 0);
        EXPECT_GT(cost.overallCost, 0);

        // --- Verify the algorithm correctly skips DMA for shared activation ---
        // Strategy: compute cost for all 4 tiles together vs two separate halves.
        // When computed together, activation DMA is only counted at tile 0 (offset doesn't change).
        // When computed in halves, each half resets offset tracking => activation DMA counted in both.
        // Therefore: fullCost.dmaCost < firstHalf.dmaCost + secondHalf.dmaCost
        VPU::PipelineTiling pipeline;
        auto fullCost = pipeline.calculateCost(convOp, outTiles, *bundle.costModel);
        EXPECT_GT(fullCost.dmaCost, 0);

        const auto halfSize = numTiles / 2;
        OutputTiling firstHalf(outTiles.begin(), outTiles.begin() + halfSize);
        OutputTiling secondHalf(outTiles.begin() + halfSize, outTiles.end());
        auto firstHalfCost = pipeline.calculateCost(convOp, firstHalf, *bundle.costModel);
        auto secondHalfCost = pipeline.calculateCost(convOp, secondHalf, *bundle.costModel);

        // Full tiling has less DMA because activation is counted once; split halves count it twice.
        EXPECT_LT(fullCost.dmaCost, vpux::addSaturating(firstHalfCost.dmaCost, secondHalfCost.dmaCost))
                << "Shared activation: full tiling dmaCost < sum of split halves (act DMA'd once vs twice)";

        // PipelineTiling: weights change every tile (pipelineable after tile 0) so
        // middle tiles benefit from max(dmaIn, compute) overlap on weight DMA.
        EXPECT_LE(fullCost.overallCost, cost.overallCost)
                << "Pipeline should be cheaper than isolated due to DMA/compute overlap on weight DMA";
    });
}

// ============================================================================
// Shared buffer detection: nested tiling (C+H), activation is shared (stall) at boundaries.
// For this model, isSpatialFirstNestedTiling returns true (inputMem*C > filterMem*H*W),
// so fillDividedTiles produces tiles in NHWC order: H is outer, C is inner.
// - Weights change every tile (C varies in inner loop) => pipelineable (dmaIn)
// - Activations stay constant within each H group, change at H boundaries => stall
// With offset-based computeSharedFlags (comparing input offsets of tile 0 vs tile 1),
// the classification is determined once: activation=shared (stall), weights=non-shared (dmaIn).
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, SharedBuffer_NestedTiling_BoundaryTransitionIsStall) {
    auto module = parseAndInitModule(CONV_3x3_128x112x112_TO_768x56x56_SOK_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(convOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        // Nested tiling: C=4, H=4 => 16 tiles
        // isSpatialFirstNestedTiling => true for this model, so tile order is NHWC:
        // H is outer (varies slowest among non-trivial dims), C is inner (varies fastest).
        Shape nestedTiling({1, 4, 4, 1});
        auto nestedTiles = fillDividedTiles(convOp, nestedTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(nestedTiles));
        const auto& outTiles = nestedTiles.value();
        ASSERT_EQ(outTiles.size(), 16u);

        // Verify structural property:
        // - Weight offsets change every tile (C is inner loop => 15 transitions)
        // - Activation offsets change at H boundaries (every 4 tiles => 3 transitions)
        auto tilingOp = mlir::cast<VPU::TilingBuilderOpInterface>(convOp.getOperation());
        size_t weightOffsetChanges = 0;
        size_t actOffsetChanges = 0;
        Shape prevWgtOffset;
        Shape prevActOffset;
        for (size_t i = 0; i < outTiles.size(); ++i) {
            const auto inputTilesI = tilingOp.backInferTileInfo(outTiles[i], Logger::global()).tiles;
            ASSERT_GE(inputTilesI.size(), 2u);
            if (!prevWgtOffset.empty() && inputTilesI[1].offsets != prevWgtOffset) {
                ++weightOffsetChanges;
            }
            if (!prevActOffset.empty() && inputTilesI[0].offsets != prevActOffset) {
                ++actOffsetChanges;
            }
            prevWgtOffset = inputTilesI[1].offsets;
            prevActOffset = inputTilesI[0].offsets;
        }
        // C is inner loop: weight offsets change every tile (numTiles - 1 = 15)
        EXPECT_EQ(weightOffsetChanges, 15u)
                << "Nested tiling (spatial-first): weights change every tile since C is inner loop";
        // H is outer loop: activation offsets change at H boundaries (H=4 => 3 transitions)
        EXPECT_EQ(actOffsetChanges, 3u) << "Nested tiling (spatial-first): activation offsets change at H boundaries";

        // --- Verify that offset-based computeSharedFlags produces consistent results across H-boundaries ---
        // At H boundaries (tiles 4, 8, 12), activation gets stallCost
        // but weights remain classified as dmaInCost (pipelineable).
        //
        // If we compute each H-group of 4 tiles independently, each group's "tile 0" resets
        // the tracking: both activation AND weights won't be pipelined in tile 0.
        // This adds extra weight stall cost at each group boundary that the full sequence avoids.
        //
        // Therefore: full_cost < sum(independent_group_costs)
        VPU::PipelineTiling pipeline;
        auto nestedCost = pipeline.calculateCost(convOp, outTiles, *bundle.costModel);
        EXPECT_GT(nestedCost.overallCost, 0);
        EXPECT_GT(nestedCost.dmaCost, 0);

        // Split into 4 independent H-groups (4 tiles each) and sum their costs
        const size_t groupSize = 4;
        const size_t numGroups = outTiles.size() / groupSize;
        uint64_t sumOfGroupCosts = 0;
        for (size_t g = 0; g < numGroups; ++g) {
            OutputTiling group(outTiles.begin() + g * groupSize, outTiles.begin() + (g + 1) * groupSize);
            auto groupCost = pipeline.calculateCost(convOp, group, *bundle.costModel);
            sumOfGroupCosts = vpux::addSaturating(sumOfGroupCosts, groupCost.overallCost);
        }
        // Static shared flags: full sequence and independent groups have the same stall/dmaIn
        // classification. At H boundaries, only activation stalls (not weights).
        // Independent groups pay extra weight stall at each group's tile 0.
        EXPECT_LT(nestedCost.overallCost, sumOfGroupCosts)
                << "Full sequence cost < sum of independent H-groups: continuous tracking avoids weight stalls";

        // Verify pipeline is cheaper than isolated (overlap helps in steady-state)
        VPU::IsolatedTiling isolated;
        auto nestedIsolatedCost = isolated.calculateCost(convOp, outTiles, *bundle.costModel);
        EXPECT_GT(nestedIsolatedCost.overallCost, 0);
        EXPECT_LE(nestedCost.overallCost, nestedIsolatedCost.overallCost);
    });
}

// ============================================================================
// Peak memory verification: offset-based shared buffer detection correctly
// classifies operands into shared vs individual buckets for double-buffering.
//
// PipelineTiling formula: individual*2 + shared
// IsolatedTiling formula: individual + shared
//
// When sharing IS detected: pipelinePeak < isolatedPeak * 2
//   (shared operands are NOT double-buffered, reducing peak CMX demand)
// When no sharing: pipelinePeak == isolatedPeak * 2 (everything double-buffered)
//
// The gap (pipelinePeak - isolatedPeak) equals the individual portion size,
// which differs between H-tiling (weights shared) and C-tiling (activation shared).
// ============================================================================
TEST_P(MLIR_TemporalTilingScenario, SharedBuffer_PeakMemory_SharingReducesDoublebuffering) {
    auto module = parseAndInitModule(CONV_3x3_64x112x112_TO_128x56x56_SOH_IR, ctx, GetParam());
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    CostModelBundle bundle(module.get(), func);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(convOp->getResult(0).getType());
        auto outputShape = getBoundedShape(outputType);

        VPU::IsolatedTiling isolated;
        VPU::PipelineTiling pipeline;

        // --- H-tiling: weights are shared (same offsets across tiles) ---
        Shape hTiling({1, 1, 4, 1});
        auto hTiles = fillDividedTiles(convOp, hTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(hTiles));
        ASSERT_GE(hTiles.value().size(), 2u);

        auto hIsolatedPeak = isolated.calculatePeakMemory(convOp, hTiles.value(), *bundle.costModel);
        auto hPipelinePeak = pipeline.calculatePeakMemory(convOp, hTiles.value(), *bundle.costModel);
        EXPECT_GT(hIsolatedPeak.count(), 0);
        EXPECT_GT(hPipelinePeak.count(), 0);

        // Core assertion: shared weights are NOT double-buffered.
        // If there were no sharing: pipelinePeak == isolatedPeak * 2.
        // With sharing: pipelinePeak < isolatedPeak * 2.
        EXPECT_LT(hPipelinePeak, hIsolatedPeak * 2) << "H-tiling: weights shared => pipeline peak < 2x isolated peak "
                                                    << "(shared operands not double-buffered)";

        // --- C-tiling: activation is shared (same offsets across tiles) ---
        Shape cTiling({1, 8, 1, 1});
        auto cTiles = fillDividedTiles(convOp, cTiling, outputShape);
        ASSERT_TRUE(mlir::succeeded(cTiles));
        ASSERT_GE(cTiles.value().size(), 2u);

        auto cIsolatedPeak = isolated.calculatePeakMemory(convOp, cTiles.value(), *bundle.costModel);
        auto cPipelinePeak = pipeline.calculatePeakMemory(convOp, cTiles.value(), *bundle.costModel);
        EXPECT_GT(cIsolatedPeak.count(), 0);
        EXPECT_GT(cPipelinePeak.count(), 0);

        // Same property for C-tiling: shared activation not double-buffered
        EXPECT_LT(cPipelinePeak, cIsolatedPeak * 2)
                << "C-tiling: activation shared => pipeline peak < 2x isolated peak "
                << "(shared operands not double-buffered)";
    });
}

INSTANTIATE_TEST_SUITE_P(TemporalTilingScenario, MLIR_TemporalTilingScenario,
                         testing::Values(Platform::NPU4000, Platform::NPU5010), platformName);
