//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scheduler_test_utils.hpp"

#include "common/nce_utils.hpp"
#include "common/utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/loop_schedule_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

// Run cmd: npuUnitTests --gtest_filter="MLIR_SchedulerVFLoopCreationTest.*"

using namespace vpux;

class MLIR_SchedulerVFLoopCreationTest : public MLIR_SchedulerLoopCreationTestBase {};

namespace {

// Helper to create a VF model where each iteration contains multiple COMPUTE ops and can contain multiple vf regions
mlir::OwningOpRef<mlir::ModuleOp> createMultiRegionMultiComputeVfModule(mlir::MLIRContext* ctx, int numTilesC,
                                                                        int numVfRegions, int numChainedOps,
                                                                        config::Platform platform) {
    VPUX_THROW_UNLESS(numChainedOps >= 1, "numChainedOps must be at least 1, got {0}", numChainedOps);
    auto loc = mlir::UnknownLoc::get(ctx);
    auto module = mlir::ModuleOp::create(loc);
    auto builder = mlir::OpBuilder(module.getBody(), module.getBody()->begin());

    const DimsOrder orderNHWC = DimsOrder::NHWC;
    const auto cmxSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::CMX_NN), 0);
    const auto ddrSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::DDR), 0);
    const auto f16Type = mlir::Float16Type::get(ctx);

    const int64_t inputSizeH = 160;
    const int64_t inputSizeW = 64;
    const int64_t inputChannels = 16;
    const int64_t outputChannels = 192;
    const int64_t kernelSize = 3;
    const int64_t stride = 1;
    const int64_t padding = 1;
    VPUX_THROW_UNLESS(numTilesC >= 1, "numTilesC must be at least 1, got {0}", numTilesC);
    VPUX_THROW_UNLESS(outputChannels % numTilesC == 0,
                      "outputChannels must be divisible by numTilesC, got outputChannels={0}, numTilesC={1}",
                      outputChannels, numTilesC);
    const int64_t tileC = outputChannels / numTilesC;

    auto inputTypeDDR = vpux::getMemRefType({1, inputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, ddrSpace);
    auto outputTypeDDR = vpux::getMemRefType({1, outputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, ddrSpace);
    auto inputTypeCMX = vpux::getMemRefType({1, inputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, cmxSpace);
    auto outputTypeCMX = vpux::getMemRefType({1, outputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, cmxSpace);

    auto funcType = builder.getFunctionType({inputTypeDDR, outputTypeDDR}, {outputTypeDDR});
    auto func = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
    func.setPublic();

    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto inputArg = entryBlock->getArgument(0);
    auto outputArg = entryBlock->getArgument(1);

    auto input = builder.create<mlir::memref::AllocOp>(loc, inputTypeCMX);
    auto output = builder.create<mlir::memref::AllocOp>(loc, outputTypeCMX);
    auto copyIn = builder.create<VPUIP::NNDMAOp>(loc, inputArg, input);

    auto outputTileType = vpux::getMemRefType({1, tileC, inputSizeH, inputSizeW}, f16Type, orderNHWC, cmxSpace);
    auto intermediateTileType = vpux::getMemRefType({1, tileC, inputSizeH, inputSizeW}, f16Type, orderNHWC, cmxSpace);

    mlir::Value currentInput = copyIn.getOutput();

    for (int vfRegion = 0; vfRegion < numVfRegions; vfRegion++) {
        const int64_t regionInputChannels = (vfRegion == 0) ? inputChannels : outputChannels;

        auto regionOutputCMX =
                (vfRegion == numVfRegions - 1) ? output : builder.create<mlir::memref::AllocOp>(loc, outputTypeCMX);

        llvm::SmallVector<mlir::Value> allTiles;

        for (int c = 0; c < numTilesC; c++) {
            auto mpeEngineAttr = createMPEEngineAttr(ctx, platform);
            mlir::Value chainInput;

            for (int opIdx = 0; opIdx < numChainedOps; opIdx++) {
                const bool isFirst = (opIdx == 0);
                const bool isLast = (opIdx == numChainedOps - 1);
                const int64_t wInChannels = isFirst ? regionInputChannels : tileC;

                const Shape weightsShape = {tileC, wInChannels, kernelSize, kernelSize};
                auto weight = createWeights(builder, loc, f16Type, weightsShape, ddrSpace, cmxSpace);
                auto weightTable = createWeightsTable(builder, loc, tileC, ddrSpace, cmxSpace);

                mlir::Value opInput;
                if (isFirst) {
                    opInput = builder.create<vpux::VPUIP::SubViewOp>(
                            loc, currentInput, mlir::ArrayRef<int64_t>{0, 0, 0, 0},
                            mlir::ArrayRef<int64_t>{1, regionInputChannels, inputSizeH, inputSizeW});
                } else {
                    opInput = chainInput;
                }

                auto outType = isLast ? outputTileType : intermediateTileType;
                auto outBuf = builder.create<mlir::memref::AllocOp>(loc, outType);

                auto nceOp = createNCEClusterTaskOp(builder, ctx, loc, kernelSize, padding, stride, opInput, weight,
                                                    weightTable, outBuf.getResult(), mpeEngineAttr);

                nceOp->setAttr(VF_LOOP_INDEX_ATTR_NAME, builder.getI64IntegerAttr(vfRegion));
                nceOp->setAttr(VF_LOOP_TILE_INDEX_ATTR_NAME, builder.getI64IntegerAttr(c));

                auto& dpuRegion = nceOp.getVariants();
                builder.setInsertionPointToStart(&dpuRegion.front());
                createDPUTaskOp(builder, {0, c * tileC, 0}, {1, (c + 1) * tileC, inputSizeH});
                builder.setInsertionPointAfter(nceOp);

                chainInput = nceOp.getOutput();
            }

            allTiles.push_back(chainInput);
        }

        auto concatOp = builder.create<vpux::VPUIP::ConcatViewOp>(loc, allTiles, regionOutputCMX);
        currentInput = concatOp.getOutput();
    }

    auto copyOut = builder.create<VPUIP::NNDMAOp>(loc, currentInput, outputArg);
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{copyOut.getOutput()});

    printDebugIR(module, Logger::global());

    mlir::PassManager pm(ctx);
    VPUIP::buildAsyncSchedulingPipeline(pm);
    EXPECT_TRUE(mlir::succeeded(pm.run(func)));

    func.walk([&](mlir::async::ExecuteOp execOp) {
        execOp->setAttr(cycleCostAttrName, builder.getI64IntegerAttr(1000));
    });

    return module;
}

// Helper to create a model with both tiling-attributed ops and VF-attributed ops
// First conv group uses TILING_LOOP_INDEX_ATTR_NAME, second uses VF_LOOP_INDEX/VF_LOOP_TILE_INDEX
mlir::OwningOpRef<mlir::ModuleOp> createMixedTilingAndVfModule(mlir::MLIRContext* ctx, int numTilesC, int numTilesC_vf,
                                                               int numChainedVfOps, config::Platform platform) {
    VPUX_THROW_UNLESS(numChainedVfOps >= 1, "numChainedVfOps must be at least 1, got {0}", numChainedVfOps);
    auto loc = mlir::UnknownLoc::get(ctx);
    auto module = mlir::ModuleOp::create(loc);
    auto builder = mlir::OpBuilder(module.getBody(), module.getBody()->begin());

    const DimsOrder orderNHWC = DimsOrder::NHWC;
    const auto cmxSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::CMX_NN), 0);
    const auto ddrSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::DDR), 0);
    const auto f16Type = mlir::Float16Type::get(ctx);

    const int64_t inputSizeH = 160;
    const int64_t inputSizeW = 64;
    const int64_t inputChannels = 16;
    const int64_t outputChannels = 384;
    const int64_t kernelSize = 3;
    const int64_t stride = 1;
    const int64_t padding = 1;

    const int64_t tileC = outputChannels / numTilesC;
    VPUX_THROW_UNLESS(outputChannels % numTilesC == 0, "outputChannels not divisible by numTilesC");
    const int64_t tileCVf = outputChannels / numTilesC_vf;
    VPUX_THROW_UNLESS(outputChannels % numTilesC_vf == 0, "outputChannels not divisible by numTilesC_vf");

    auto inputTypeDDR = vpux::getMemRefType({1, inputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, ddrSpace);
    auto outputTypeDDR = vpux::getMemRefType({1, outputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, ddrSpace);
    auto inputTypeCMX = vpux::getMemRefType({1, inputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, cmxSpace);
    auto outputTypeCMX = vpux::getMemRefType({1, outputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, cmxSpace);

    auto funcType = builder.getFunctionType({inputTypeDDR, outputTypeDDR}, {outputTypeDDR});
    auto func = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
    func.setPublic();

    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto inputArg = entryBlock->getArgument(0);
    auto outputArg = entryBlock->getArgument(1);

    auto input = builder.create<mlir::memref::AllocOp>(loc, inputTypeCMX);
    auto output = builder.create<mlir::memref::AllocOp>(loc, outputTypeCMX);
    auto copyIn = builder.create<VPUIP::NNDMAOp>(loc, inputArg, input);

    // --- Part 1: Tiling conv (TILING_LOOP_INDEX_ATTR_NAME) ---
    llvm::SmallVector<mlir::Value> tilingWeightOps;
    llvm::SmallVector<mlir::Value> tilingWeightTableOps;
    for (int c = 0; c < numTilesC; c++) {
        const Shape weightsShape = {tileC, inputChannels, kernelSize, kernelSize};
        tilingWeightOps.push_back(createWeights(builder, loc, f16Type, weightsShape, ddrSpace, cmxSpace));
        tilingWeightTableOps.push_back(createWeightsTable(builder, loc, tileC, ddrSpace, cmxSpace));
    }

    auto intermediateTypeCMX = outputTypeCMX;
    auto intermediateBuf = builder.create<mlir::memref::AllocOp>(loc, intermediateTypeCMX);
    llvm::SmallVector<mlir::Value> tilingTiles;

    for (int c = 0; c < numTilesC; c++) {
        auto inputTile = builder.create<vpux::VPUIP::SubViewOp>(
                loc, copyIn.getOutput(), mlir::ArrayRef<int64_t>{0, 0, 0, 0},
                mlir::ArrayRef<int64_t>{1, inputChannels, inputSizeH, inputSizeW});
        auto outputTileType = vpux::getMemRefType({1, tileC, inputSizeH, inputSizeW}, f16Type, orderNHWC, cmxSpace);
        auto outputTile = builder.create<mlir::memref::AllocOp>(loc, outputTileType);

        auto mpeEngineAttr = createMPEEngineAttr(ctx, platform);
        auto nceOp =
                createNCEClusterTaskOp(builder, ctx, loc, kernelSize, padding, stride, inputTile, tilingWeightOps[c],
                                       tilingWeightTableOps[c], outputTile.getResult(), mpeEngineAttr);

        // Set tiling attribute: all tiles share tiling_loop_index=0
        nceOp->setAttr(TILING_LOOP_INDEX_ATTR_NAME, builder.getI64IntegerAttr(0));

        auto& dpuTaskRegion = nceOp.getVariants();
        builder.setInsertionPointToStart(&dpuTaskRegion.front());
        createDPUTaskOp(builder, {0, c * tileC, 0}, {1, (c + 1) * tileC, inputSizeH});
        builder.setInsertionPointAfter(nceOp);

        tilingTiles.push_back(nceOp.getOutput());
    }

    auto tilingConcat = builder.create<vpux::VPUIP::ConcatViewOp>(loc, tilingTiles, intermediateBuf);
    mlir::Value vfInput = tilingConcat.getOutput();

    // --- Part 2: VF conv (VF_LOOP_INDEX_ATTR_NAME + VF_LOOP_TILE_INDEX_ATTR_NAME) ---
    auto outputTileType = vpux::getMemRefType({1, tileCVf, inputSizeH, inputSizeW}, f16Type, orderNHWC, cmxSpace);
    auto intermediateTileType = vpux::getMemRefType({1, tileCVf, inputSizeH, inputSizeW}, f16Type, orderNHWC, cmxSpace);

    llvm::SmallVector<mlir::Value> vfTiles;
    for (int c = 0; c < numTilesC_vf; c++) {
        auto mpeEngineAttr = createMPEEngineAttr(ctx, platform);
        mlir::Value chainInput;

        for (int opIdx = 0; opIdx < numChainedVfOps; opIdx++) {
            const bool isFirst = (opIdx == 0);
            const bool isLast = (opIdx == numChainedVfOps - 1);
            const int64_t wInChannels = isFirst ? outputChannels : tileCVf;

            const Shape weightsShape = {tileCVf, wInChannels, kernelSize, kernelSize};
            auto weight = createWeights(builder, loc, f16Type, weightsShape, ddrSpace, cmxSpace);
            auto weightTable = createWeightsTable(builder, loc, tileCVf, ddrSpace, cmxSpace);

            mlir::Value opInput;
            if (isFirst) {
                opInput = builder.create<vpux::VPUIP::SubViewOp>(
                        loc, vfInput, mlir::ArrayRef<int64_t>{0, 0, 0, 0},
                        mlir::ArrayRef<int64_t>{1, outputChannels, inputSizeH, inputSizeW});
            } else {
                opInput = chainInput;
            }

            auto outType = isLast ? outputTileType : intermediateTileType;
            auto outBuf = builder.create<mlir::memref::AllocOp>(loc, outType);

            auto nceOp = createNCEClusterTaskOp(builder, ctx, loc, kernelSize, padding, stride, opInput, weight,
                                                weightTable, outBuf.getResult(), mpeEngineAttr);

            // Set VF attributes: vf_loop_index=0, vf_loop_tile_index=c
            nceOp->setAttr(VF_LOOP_INDEX_ATTR_NAME, builder.getI64IntegerAttr(0));
            nceOp->setAttr(VF_LOOP_TILE_INDEX_ATTR_NAME, builder.getI64IntegerAttr(c));

            auto& dpuTaskRegion = nceOp.getVariants();
            builder.setInsertionPointToStart(&dpuTaskRegion.front());
            createDPUTaskOp(builder, {0, c * tileCVf, 0}, {1, (c + 1) * tileCVf, inputSizeH});
            builder.setInsertionPointAfter(nceOp);

            chainInput = nceOp.getOutput();
        }

        vfTiles.push_back(chainInput);
    }

    auto vfConcat = builder.create<vpux::VPUIP::ConcatViewOp>(loc, vfTiles, output);
    auto copyOut = builder.create<VPUIP::NNDMAOp>(loc, vfConcat.getOutput(), outputArg);
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{copyOut.getOutput()});

    // Debug IR creation
    printDebugIR(module, Logger::global());

    mlir::PassManager pm(ctx);
    VPUIP::buildAsyncSchedulingPipeline(pm);
    EXPECT_TRUE(mlir::succeeded(pm.run(func)));

    func.walk([&](mlir::async::ExecuteOp execOp) {
        execOp->setAttr(cycleCostAttrName, builder.getI64IntegerAttr(1000));
    });

    return module;
}

}  // namespace

TEST_P(MLIR_SchedulerVFLoopCreationTest, MultiComputeVfIterationMatching) {
    /*
    VF iterations with multiple COMPUTE ops (3 chained NCE convs per tile)
    should be correctly matched by the iteration matching algorithm.
    The matching compares: number of COMPUTE ops and deduplicated local buffer counts
    per iteration

    Model: 4 C-tiles, each has conv1 -> conv2 -> conv3 (all share same vfLoopTileIndex).
    All 4 tiles match and form a VF loop (last iteration absorbs DATA_OUT).

    ComputeRegionVec
    |---ComputeRegion 0:
    |   \---SchedulingLoop: type: None, iterations: 1
    |       \---Alloc: DATA_IN <-- input copy from DDR (shared)
    |
    \---ComputeRegion 1: <-- all 4 tiles match (3 COMPUTEs per iter, last absorbs DATA_OUT)
        \---SchedulingLoop: type: VF, iterations: 4
            \---loopBody:
                |---Iteration 0..3: each has
                    DATA_IN (conv1 weights), DATA_IN (conv1 wt), COMPUTE (conv1 NCE),
                    DATA_IN (conv2 weights), DATA_IN (conv2 wt), COMPUTE (conv2 NCE),
                    DATA_IN (conv3 weights), DATA_IN (conv3 wt), COMPUTE (conv3 NCE)
                \---Last iteration also has DATA_OUT

    Summary: 1 VF loop (4 iters, 3 COMPUTEs each) + 1 non-loop = 2 regions
    */
    const int tilesC = 4;
    auto module = createMultiRegionMultiComputeVfModule(getCtx(), tilesC, /*numVfRegions=*/1, /*numChainedOps=*/3,
                                                        GetParam());
    ASSERT_TRUE(module);
    EXPECT_TRUE(module->verify().succeeded());

    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);
    printComputeRegions(regions, Logger::global());

    size_t numVfLoops = 0;
    size_t totalVfIterations = 0;
    for (const auto& region : regions) {
        if (region.getLoopType() == LoopType::VF) {
            ++numVfLoops;
            totalVfIterations += region.schedulingLoop->loopBodies.size();
        }
    }

    EXPECT_GE(numVfLoops, 1) << "At least one VF loop must be created from multi-compute iterations";
    EXPECT_GE(totalVfIterations, 2) << "VF loop should have at least 2 matching multi-compute iterations";

    for (const auto& region : regions) {
        if (region.getLoopType() != LoopType::VF) {
            continue;
        }
        for (const auto& iteration : region.schedulingLoop->loopBodies) {
            size_t computeCount = 0;
            for (const auto& op : iteration) {
                if (op.allocationType == AllocationType::COMPUTE) {
                    ++computeCount;
                }
            }
            EXPECT_EQ(computeCount, 3) << "Each multi-compute VF iteration should have exactly 3 COMPUTE ops";
        }
    }
}

TEST_P(MLIR_SchedulerVFLoopCreationTest, MultiRegionMultiComputeVf) {
    /*
    Multiple VF regions, each with 4 chained NCE convs per C-tile.
    Tests that the algorithm handles multiple independent VF loops each containing
    multi-compute iterations.

    Model: 3 VF regions, 4 C-tiles each, 4 chained convs per tile.
    Region 0 -> Region 1 -> Region 2 (output feeds into next input).

    All regions: all N tiles match. The last iteration of the last region absorbs
    DATA_OUT. Result: VF loop with N iterations for each region.

    Expected: 3 VF loops total, each with N=4 iterations, 4 COMPUTEs per iteration.
    */
    const unsigned tilesC = 4;
    const unsigned vfRegions = 3;
    const unsigned chainedOps = 4;
    auto module = createMultiRegionMultiComputeVfModule(getCtx(), tilesC, vfRegions, chainedOps, GetParam());
    ASSERT_TRUE(module);
    EXPECT_TRUE(module->verify().succeeded());

    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);
    printComputeRegions(regions, Logger::global());

    size_t numVfLoops = 0;
    size_t totalVfIterations = 0;
    for (const auto& region : regions) {
        if (region.getLoopType() == LoopType::VF) {
            ++numVfLoops;
            totalVfIterations += region.schedulingLoop->loopBodies.size();
        }
    }

    EXPECT_EQ(numVfLoops, vfRegions) << "Should have one VF loop per VF region";
    // All regions: N tiles match (last iteration of last region absorbs DATA_OUT)
    const size_t expectedTotalIters = vfRegions * tilesC;
    EXPECT_EQ(totalVfIterations, expectedTotalIters) << "All regions: N iters each (DATA_OUT absorbed in last)";

    size_t vfLoopIdx = 0;
    for (const auto& region : regions) {
        if (region.getLoopType() != LoopType::VF) {
            continue;
        }
        const size_t expectedIters = tilesC;
        EXPECT_EQ(region.schedulingLoop->loopBodies.size(), expectedIters)
                << "VF loop " << vfLoopIdx << ": N iterations expected (DATA_OUT absorbed in last)";
        for (const auto& iteration : region.schedulingLoop->loopBodies) {
            size_t computeCount = 0;
            for (const auto& op : iteration) {
                if (op.allocationType == AllocationType::COMPUTE) {
                    ++computeCount;
                }
            }
            EXPECT_EQ(computeCount, chainedOps) << "Each iteration should have " << chainedOps << " COMPUTE ops";
        }
        ++vfLoopIdx;
    }
}

TEST_P(MLIR_SchedulerVFLoopCreationTest, ConvolutionTileOnC_ComputeOpsMinThresholdSkipVfLoopCreation) {
    // Create module with 1 VF region with one compute op each, 2 C-tiles.
    // Each VF iteration has only 1 compute op (below MIN_VF_LOOP_BODY_COMPUTE_OPS), so the VF loop should be skipped
    // and all ops become individual non-loop regions.
    const int belowMinChainedComputeOps = 1;
    const int tilesC = 2;
    auto module = createMultiRegionMultiComputeVfModule(getCtx(), tilesC, /*numVfRegions=*/1, belowMinChainedComputeOps,
                                                        GetParam());
    ASSERT_TRUE(module);
    EXPECT_TRUE(module->verify().succeeded());
    // Function contains:
    // 2 VF tiles over C, each tile has 2 DMAs: weights in, weights table in and 1 DPU CONV
    // Also there are copies to/from DDR for input and output

    // get alias info and async deps info
    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    // test start
    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);
    /*
    Expected structure with 2 C-tiles:
    Each VF iteration has only 1 compute op (below MIN_VF_LOOP_BODY_COMPUTE_OPS),
    so the VF loop is skipped and all ops become individual non-loop regions.

    8 regions expected: 1 input copy + 2 x (weights DMA + weight table DMA + DPU CONV) + 1 output copy
    */

    printComputeRegions(regions, Logger::global());

    // all ops are individual non-loop regions (VF loop skipped due to insufficient compute ops per iteration)
    EXPECT_EQ(regions.size(), 8);

    // count VF vs non-loop regions
    size_t numVfRegions = 0;
    size_t numNonLoopRegions = 0;
    for (const auto& region : regions) {
        if (region.getLoopType() == LoopType::VF) {
            ++numVfRegions;
        } else {
            ++numNonLoopRegions;
        }
    }
    EXPECT_EQ(numVfRegions, 0);
    EXPECT_EQ(numNonLoopRegions, 8);
}

TEST_P(MLIR_SchedulerVFLoopCreationTest, VfMinimumThreshold_TwoIterationsFormLoop) {
    /*
    VF loops require only MIN_VF_LOOP_OPS=2 matching iterations (vs MIN_TILING_LOOP_OPS=5 for Tiling)
    and a min of MIN_VF_LOOP_BODY_COMPUTE_OPS=2 compute ops per iteration to be formed.
    This test creates a model with 2 chained VF regions, each with 2 C-tiles and 2 chained compute ops
    per iteration (numMinChainedOps=2), satisfying the MIN_VF_LOOP_BODY_COMPUTE_OPS threshold:
      - VF Region 0: 2 tiles, both matching (output chains to Region 1, no DATA_OUT)
                      -> VF loop of 2 iterations
      - VF Region 1: 2 tiles, both matching (last absorbs DATA_OUT)
                      -> VF loop of 2 iterations

    This verifies that VF loop creation succeeds at the minimum threshold of both
    MIN_VF_LOOP_OPS=2 iterations and MIN_VF_LOOP_BODY_COMPUTE_OPS=2 compute ops per iteration.
    A boundary that would fail for Tiling (which requires MIN_TILING_LOOP_OPS=5),
    and would also fail if each iteration had fewer than MIN_VF_LOOP_BODY_COMPUTE_OPS compute ops.

    ComputeRegionVec
    |---ComputeRegion 0:
    |   \---SchedulingLoop: type: None, iterations: 1
    |       \---loopBody:
    |           \---Alloc: DATA_IN <-- input copy from DDR (shared)
    |
    |---ComputeRegion 1: <-- VF Region 0: both C-tiles match (no DATA_OUT)
    |   \---SchedulingLoop: type: VF, iterations: 2
    |       \---loopBody:
    |           |---Iteration 0: DATA_IN (weights), DATA_IN (wt), COMPUTE (NCE), COMPUTE (NCE)
    |           \---Iteration 1: DATA_IN (weights), DATA_IN (wt), COMPUTE (NCE), COMPUTE (NCE)
    |
    \---ComputeRegion 2: <-- VF Region 1: both C-tiles match (last absorbs DATA_OUT)
        \---SchedulingLoop: type: VF, iterations: 2
            \---loopBody:
                |---Iteration 0: DATA_IN (weights), DATA_IN (wt), COMPUTE (NCE), COMPUTE (NCE)
                \---Iteration 1: DATA_IN (weights), DATA_IN (wt), COMPUTE (NCE), COMPUTE (NCE), DATA_OUT

    Summary: 2 VF loops (2 iters each) + 1 non-loop = 3 regions
    */
    const int tilesC = 2;
    const int numMinChainedOps = 2;
    const int numVfRegions = 2;

    auto module = createMultiRegionMultiComputeVfModule(getCtx(), tilesC, numVfRegions, numMinChainedOps, GetParam());
    ASSERT_TRUE(module);
    EXPECT_TRUE(module->verify().succeeded());

    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);

    printComputeRegions(regions, Logger::global());

    // Count VF loops and their sizes
    size_t numVfLoops = 0;
    bool foundLoopWithMinIterations = false;
    for (const auto& region : regions) {
        if (region.getLoopType() == LoopType::VF) {
            ++numVfLoops;
            if (region.schedulingLoop->loopBodies.size() == MIN_VF_LOOP_OPS) {
                foundLoopWithMinIterations = true;
            }
        }
    }

    // Region 0 should form a VF loop of exactly MIN_VF_LOOP_OPS iterations (the minimum threshold)
    EXPECT_TRUE(foundLoopWithMinIterations)
            << "Expected at least one VF loop with exactly " << MIN_VF_LOOP_OPS << " iterations (minimum VF threshold)";
    EXPECT_GE(numVfLoops, 1) << "Expected at least one VF loop";

    // Verify all VF iterations contain a DPU COMPUTE op
    for (const auto& region : regions) {
        if (region.getLoopType() != LoopType::VF) {
            continue;
        }
        for (const auto& iteration : region.schedulingLoop->loopBodies) {
            bool hasDpu = false;
            for (const auto& op : iteration) {
                if (op.allocationType == AllocationType::COMPUTE) {
                    hasDpu = true;
                }
            }
            EXPECT_TRUE(hasDpu);
        }
    }
}

TEST_P(MLIR_SchedulerVFLoopCreationTest, MixedTilingAndVf_DeterministicOrdering) {
    /*
    Tests that when both Tiling and VF loops exist in the same model,
    the LoopType ordering is deterministic: all Tiling-type ComputeRegions
    appear before all VF-type ComputeRegions in the output vector.

    This is due to the use of std::map<LoopType, ...> used in
    getComputeRegionsFromAsyncExec, since LoopType::Tiling (1) < LoopType::VF (2).

    Model structure:
      Copy DDR->CMX
      -> Tiling conv: 8 C-tiles, tiling_loop_index=0 -> intermediate buffer
      -> VF conv: 4 C-tiles, 2 chained compute ops per tile,
         vf_loop_index=0, vf_loop_tile_index=0..3 -> output
      Copy CMX->DDR

    Each VF iteration has 2 chained NCE ops (≥ MIN_VF_LOOP_BODY_COMPUTE_OPS) so the
    VF loop is formed.

    ComputeRegionVec
    |---ComputeRegion 0:
    |   \---SchedulingLoop: type: None, iterations: 1
    |       \---Alloc: DATA_IN <-- input copy from DDR (shared)
    |
    |---ComputeRegion 1: <-- Tiling conv: all 8 tiles match (last absorbs DATA_OUT of tiling stage)
    |   \---SchedulingLoop: type: Tiling, iterations: 8
    |       \---loopBody:
    |           |---Iteration 0..7: each has DATA_IN (weights), DATA_IN (wt), COMPUTE (NCE)
    |
    \---ComputeRegion 2: <-- VF conv: all 4 tiles match (last absorbs DATA_OUT)
        \---SchedulingLoop: type: VF, iterations: 4
            \---loopBody:
                |---Iteration 0..3: each has DATA_IN (w1), DATA_IN (wt1), COMPUTE (NCE),
                |                   DATA_IN (w2), DATA_IN (wt2), COMPUTE (NCE)
                \---Last iteration also has DATA_OUT

    Summary: 1 Tiling loop (8 iters) + 1 VF loop (4 iters) + 1 non-loop = 3 regions
    Expected ordering: Tiling regions before VF regions (std::map<LoopType> guarantee).
    */
    const int numTilesC = 8;        // 8 tiles -> all matching -> Tiling loop created
    const int numTilesC_vf = 4;     // 4 tiles -> all matching ≥ MIN_VF_LOOP_OPS(2) -> VF loop created
    const int numChainedVfOps = 2;  // 2 chained compute ops per VF iteration ≥ MIN_VF_LOOP_BODY_COMPUTE_OPS
    auto module = createMixedTilingAndVfModule(getCtx(), numTilesC, numTilesC_vf, numChainedVfOps, GetParam());
    ASSERT_TRUE(module);
    EXPECT_TRUE(module->verify().succeeded());

    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);

    printComputeRegions(regions, Logger::global());

    // Count regions by type
    size_t numTilingRegions = 0;
    size_t numVfRegions = 0;
    for (const auto& region : regions) {
        if (region.getLoopType() == LoopType::Tiling) {
            ++numTilingRegions;
        } else if (region.getLoopType() == LoopType::VF) {
            ++numVfRegions;
        }
    }

    // Both loop types should be detected
    EXPECT_GE(numTilingRegions, 1) << "Expected at least one Tiling loop region";
    EXPECT_GE(numVfRegions, 1) << "Expected at least one VF loop region";

    // Verify deterministic ordering: all Tiling regions appear before all VF regions.
    // Track the last index of a Tiling region and the first index of a VF region.
    int64_t lastTilingIndex = -1;
    int64_t firstVfIndex = -1;
    for (int64_t i = 0; i < static_cast<int64_t>(regions.size()); ++i) {
        if (regions[i].getLoopType() == LoopType::Tiling) {
            lastTilingIndex = i;
        }
        if (regions[i].getLoopType() == LoopType::VF && firstVfIndex == -1) {
            firstVfIndex = i;
        }
    }

    ASSERT_NE(lastTilingIndex, -1) << "No Tiling region found";
    ASSERT_NE(firstVfIndex, -1) << "No VF region found";
    EXPECT_LT(lastTilingIndex, firstVfIndex)
            << "Tiling regions (last at index " << lastTilingIndex << ") must appear before VF regions (first at index "
            << firstVfIndex << ") due to std::map<LoopType> ordering";
}

TEST_P(MLIR_SchedulerVFLoopCreationTest, GenerateLoopSchedules_VfRegionWithUndefinedSchedulerFallback) {
    /*
    Integration test: calls generateLoopSchedules with enableVfUndefinedScheduler=true on a model
    containing VF regions. Verifies that:
    1. The VF -> UNDEFINED_VF scenario mapping is exercised (no crash, no throw).
    2. UndefinedVF returns an empty result (Phase 1 stub), triggering the fallback path.
    3. The fallback produces no scheduleResults, loopRegionInd, or loopPrefetchInd for VF regions.
    */
    const int tilesC = 4;
    auto module = createMultiRegionMultiComputeVfModule(getCtx(), tilesC, /*numVfRegions=*/1, /*numChainedOps=*/3,
                                                        GetParam());
    ASSERT_TRUE(module);
    EXPECT_TRUE(module->verify().succeeded());

    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);

    // Confirm VF regions exist in the input
    bool hasVfRegion = false;
    for (const auto& region : regions) {
        if (region.getLoopType() == LoopType::VF) {
            hasVfRegion = true;
            break;
        }
    }
    ASSERT_TRUE(hasVfRegion) << "Test precondition: model must produce at least one VF region";

    vpux::AddressType memorySize = 2 * 1024 * 1024;

    // Call with enableVfUndefinedScheduler=true: exercises VF mapping + empty-result fallback
    auto result = vpux::VPU::generateLoopSchedules(regions, memorySize, /*enableVfUndefinedScheduler=*/true,
                                                   Logger::global());

    // UndefinedVF currently returns empty result -> fallback: no schedules produced for VF regions.
    // No operations should be categorized into loop scheduling sets from VF regions.
    EXPECT_TRUE(result.scheduleResults.empty()) << "UndefinedVF empty-result fallback must produce no schedule entries";
    EXPECT_TRUE(result.loopRegionInd.empty())
            << "Fallback must not populate loopRegionInd when VF scenario returns empty";
    EXPECT_TRUE(result.loopPrefetchInd.empty())
            << "Fallback must not populate loopPrefetchInd when VF scenario returns empty";
}

INSTANTIATE_TEST_SUITE_P(SchedulerVFLoopCreation, MLIR_SchedulerVFLoopCreationTest,
                         testing::Values(config::Platform::NPU4000, config::Platform::NPU5010), schedulerTestParamName);
