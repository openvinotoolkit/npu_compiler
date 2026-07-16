//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "temporal_tiling_test_utils.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/undefined_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/init/interfaces_registry.hpp"
#include "vpux/compiler/init/singleton_initializer.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

// Run cmd: npuUnitTests --gtest_filter="MLIR_TemporalTilingCommon.*"

using namespace vpux;
using namespace vpux::VPU;
using namespace vpux::VPU::test;
using vpux::config::Platform;

namespace {

constexpr llvm::StringLiteral DUMMY_FUNC_IR = R"(
module @main {
    func.func @main() {
        return
    }
}
)";

mlir::OwningOpRef<mlir::ModuleOp> parseAndInitModule(llvm::StringLiteral ir, mlir::MLIRContext& ctx) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, &ctx);
    if (!module.get()) {
        return {};
    }

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initOptions = VPU::InitCompilerOptions(Platform::NPU3720, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initOptions, Logger::global());

    if (mlir::failed(pm.run(module.get()))) {
        return {};
    }
    return module;
}

}  // namespace

// ============================================================================
// CostInfo equality
// ============================================================================
TEST(MLIR_TemporalTilingCommon, CostInfoEquality) {
    CostInfo a{100, 200, 300, 0.5};
    CostInfo b{100, 200, 300, 0.5};
    CostInfo c{100, 200, 301, 0.5};
    CostInfo d{100, 200, 300, 0.6};
    EXPECT_EQ(a, b);
    EXPECT_FALSE(a == c);
    EXPECT_FALSE(a == d);
}

// ============================================================================
// UndefinedTiling: all cost/memory methods throw
// ============================================================================
TEST(MLIR_TemporalTilingCommon, UndefinedTilingMethodsThrow) {
    auto registry = vpux::createDialectRegistry();
    auto interfacesRegistry = vpux::createInterfacesRegistry(Platform::NPU3720);
    interfacesRegistry->registerInterfaces(registry);
    VPU::initializeSingletons(registry, Platform::NPU3720);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    auto module = parseAndInitModule(DUMMY_FUNC_IR, ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    VPU::SiblingOpsAnalysis siblingsAnalysis(func);
    auto vpunnCostModel = VPU::CostModelConfig::createLayerCostModel(module.get());
    VPU::LayerCostModel costModel(func, true, siblingsAnalysis, std::move(vpunnCostModel), Logger::global());

    UndefinedTiling scenario;
    Shape dummyShape({1, 1, 1, 1});
    OutputTiling dummyTiling = {TileInfo(ShapeRef({1, 16, 32, 32}))};

    EXPECT_ANY_THROW(scenario.satisfyMemoryConstraint(nullptr, dummyShape, costModel, Logger::global()));
    EXPECT_ANY_THROW(scenario.calculateCost(nullptr, dummyTiling, costModel));
    EXPECT_ANY_THROW(scenario.calculatePeakMemory(nullptr, dummyTiling, costModel));
}

// ============================================================================
// Cost formula comparison: same inputs produce different costs across scenarios
// ============================================================================
TEST(MLIR_TemporalTilingCommon, CostFormulasAreDifferent) {
    TestableIsolatedTiling isolated;
    TestablePrefetchTiling prefetch;
    TestablePipelineTiling pipeline;

    // With these values, all three formulas should produce different results:
    // Isolated:  10 + 200 + 300 + 100 = 610
    // Pipeline:  10 + max(max(200,300),100) = 10 + 300 = 310
    // Prefetch:  10 + max(200,300) + 100 = 10 + 300 + 100 = 410
    constexpr uint64_t stallCost = 10;
    constexpr uint64_t dmaInCost = 200;
    constexpr uint64_t computeCost = 300;
    constexpr uint64_t dmaOutCost = 100;
    const auto isolatedCost =
            isolated.computeTileOverallCost(stallCost, dmaInCost, computeCost, dmaOutCost, false, false);
    const auto prefetchCost =
            prefetch.computeTileOverallCost(stallCost, dmaInCost, computeCost, dmaOutCost, false, false);
    const auto pipelineCost =
            pipeline.computeTileOverallCost(stallCost, dmaInCost, computeCost, dmaOutCost, false, false);

    // Ordering: isolated >= prefetch >= pipeline
    EXPECT_GT(isolatedCost, prefetchCost);
    EXPECT_GT(prefetchCost, pipelineCost);
}

// ============================================================================
// Cost formula with UINT64_MAX inputs — verify no overflow/UB
// ============================================================================
TEST(MLIR_TemporalTilingCommon, CostFormula_SaturatingArithmetic_IsolatedTiling) {
    TestableIsolatedTiling isolated;

    // Isolated: stall + dmaIn + compute + dmaOut (plain addition, no saturation)
    constexpr auto maxVal = UINT64_MAX;
    const auto result = isolated.computeTileOverallCost(maxVal, 0, 0, 0, false, false);
    EXPECT_EQ(result, maxVal);

    // Verify identity: adding zero to any component doesn't change result
    const auto result2 = isolated.computeTileOverallCost(0, maxVal, 0, 0, false, false);
    EXPECT_EQ(result2, maxVal);

    // Verify linearity
    constexpr uint64_t a = 100, b = 200, c = 300, d = 400;
    EXPECT_EQ(isolated.computeTileOverallCost(a, b, c, d, false, false), a + b + c + d);
}

TEST(MLIR_TemporalTilingCommon, CostFormula_SaturatingArithmetic_PrefetchTiling) {
    TestablePrefetchTiling prefetch;

    // Prefetch: stall + max(dmaIn, compute) + dmaOut
    // stall=0, dmaOut=0 -> result is just max(dmaIn, compute)
    constexpr auto maxVal = UINT64_MAX;
    const auto result = prefetch.computeTileOverallCost(0, maxVal, maxVal, 0, false, false);
    EXPECT_EQ(result, maxVal);

    // stall at max, rest zero -> just max
    const auto result2 = prefetch.computeTileOverallCost(maxVal, 0, 0, 0, false, false);
    EXPECT_EQ(result2, maxVal);

    // Verify formula: stall + max(dmaIn, compute) + dmaOut
    const auto result3 = prefetch.computeTileOverallCost(10, 200, 500, 30, false, false);
    EXPECT_EQ(result3, 540u);  // 10 + max(200,500) + 30 = 540
}

// ============================================================================
// PipelineTiling: all three tile-position variants produce distinct costs
// ============================================================================
TEST(MLIR_TemporalTilingCommon, PipelineTiling_AllThreePositionFormulas) {
    TestablePipelineTiling pipeline;
    // dmaIn=200, compute=300, dmaOut=400: all different, so all three formulas produce distinct results
    constexpr uint64_t stall = 0, dmaIn = 200, compute = 300, dmaOut = 400;
    const auto firstTileCost = pipeline.computeTileOverallCost(stall, dmaIn, compute, dmaOut, true, false);
    const auto lastTileCost = pipeline.computeTileOverallCost(stall, dmaIn, compute, dmaOut, false, true);
    const auto middleTileCost = pipeline.computeTileOverallCost(stall, dmaIn, compute, dmaOut, false, false);
    // first:  0 + 200 + max(300,400) = 600
    // last:   0 + max(200,300) + 400 = 700
    // middle: 0 + max(200,300,400)   = 400
    EXPECT_EQ(firstTileCost, 600u);
    EXPECT_EQ(lastTileCost, 700u);
    EXPECT_EQ(middleTileCost, 400u);
    // Middle achieves full overlap (cheapest); last serializes DMA-out (costliest)
    EXPECT_LT(middleTileCost, firstTileCost);
    EXPECT_LT(firstTileCost, lastTileCost);
}

// ============================================================================
// PrefetchTiling: isFirstTile=true collapses to the same serial sum as Isolated
// ============================================================================
TEST(MLIR_TemporalTilingCommon, PrefetchTiling_FirstTileEqualsIsolatedFormula) {
    TestableIsolatedTiling isolated;
    TestablePrefetchTiling prefetch;
    constexpr uint64_t stall = 10, dmaIn = 200, compute = 300, dmaOut = 100;
    const auto isolatedCost = isolated.computeTileOverallCost(stall, dmaIn, compute, dmaOut, false, false);
    const auto prefetchFirstTile = prefetch.computeTileOverallCost(stall, dmaIn, compute, dmaOut, true, false);
    const auto prefetchMiddle = prefetch.computeTileOverallCost(stall, dmaIn, compute, dmaOut, false, false);
    // First tile: no previous compute to overlap with, so falls back to serial sum
    EXPECT_EQ(prefetchFirstTile, isolatedCost);
    // Middle tile benefits from DMA/compute overlap: cheaper
    EXPECT_LT(prefetchMiddle, prefetchFirstTile);
}

// ============================================================================
// PrefetchTiling: isLastTile flag has no effect
// ============================================================================
TEST(MLIR_TemporalTilingCommon, PrefetchTiling_LastTileFlagIgnored) {
    TestablePrefetchTiling prefetch;
    constexpr uint64_t stall = 10, dmaIn = 200, compute = 300, dmaOut = 100;
    const auto costLastFalse = prefetch.computeTileOverallCost(stall, dmaIn, compute, dmaOut, false, false);
    const auto costLastTrue = prefetch.computeTileOverallCost(stall, dmaIn, compute, dmaOut, false, true);
    EXPECT_EQ(costLastFalse, costLastTrue);
}

TEST(MLIR_TemporalTilingCommon, CostFormula_SaturatingArithmetic_PipelineTiling) {
    TestablePipelineTiling pipeline;

    // Pipeline: stall + max(max(dmaIn, compute), dmaOut)
    // stall=0 -> result is just max(max(dmaIn, compute), dmaOut)
    constexpr auto maxVal = UINT64_MAX;
    const auto result = pipeline.computeTileOverallCost(0, maxVal, maxVal, maxVal, false, false);
    EXPECT_EQ(result, maxVal);

    // stall=0, one large component -> the max of the three
    constexpr uint64_t dmaIn = 100, compute = 500, dmaOut = 300;
    const auto result2 = pipeline.computeTileOverallCost(/*stall=*/0, dmaIn, compute, dmaOut, false, false);
    EXPECT_EQ(result2, 500u);  // max(max(100,500),300) = 500

    // Non-zero stall adds to the max
    const auto result3 = pipeline.computeTileOverallCost(/*stall=*/50, dmaIn, compute, dmaOut, false, false);
    EXPECT_EQ(result3, 550u);  // 50 + max(max(100,500),300) = 550
}
