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
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <memory>

// Run cmd: npuUnitTests --gtest_filter="SchedulerLoopCreation/MLIR_SchedulerLoopCreationTest.*"

using namespace vpux;

class MLIR_SchedulerLoopCreationTest : public MLIR_SchedulerLoopCreationTestBase {};

namespace {

std::pair<mlir::Value, mlir::Value> createPackedWeightsAndTable(mlir::OpBuilder& builder, mlir::Location loc,
                                                                int64_t tileC, int64_t inputChannels,
                                                                int64_t kernelSize,
                                                                const vpux::IndexedSymbolAttr& ddrSpace,
                                                                const vpux::IndexedSymbolAttr& cmxSpace) {
    const auto i8Type = builder.getIntegerType(8, false);
    const auto f16Type = mlir::Float16Type::get(builder.getContext());
    const auto i32Type = builder.getIntegerType(32, true);

    const int64_t weightsBytes = tileC * inputChannels * kernelSize * kernelSize * sizeof(uint16_t);
    const int64_t tableBytes = tileC * 1 * 1 * 4 * sizeof(int32_t);
    const int64_t packedBytes = weightsBytes + tableBytes;

    auto packedTypeDDR = vpux::getMemRefType({1, 1, 1, packedBytes}, i8Type, DimsOrder::NHWC, ddrSpace);
    auto packedTypeCMX = vpux::getMemRefType({1, 1, 1, packedBytes}, i8Type, DimsOrder::NHWC, cmxSpace);
    auto packedTensorType = mlir::RankedTensorType::get({1, 1, 1, packedBytes}, i8Type);

    auto packedAttr = mlir::DenseElementsAttr::get(packedTensorType, mlir::APInt(8, 1, false));
    Const::ContentSetup packedContentSetup(packedAttr, packedTensorType);
    packedContentSetup = packedContentSetup.castElemType(i8Type);
    packedContentSetup = packedContentSetup.reorder(DimsOrder::NHWC);
    auto packedContentAttr = Const::ContentAttr::get(packedAttr, packedContentSetup);
    auto packedDDR = builder.create<Const::DeclareOp>(loc, packedTypeDDR, packedContentAttr);
    auto packedCMX = builder.create<mlir::memref::AllocOp>(loc, packedTypeCMX);
    auto copyPacked = builder.create<VPUIP::NNDMAOp>(loc, packedDDR, packedCMX);

    auto weightBytesSubview =
            builder.create<VPUIP::SubViewOp>(loc, copyPacked.getOutput(), mlir::ArrayRef<int64_t>{0, 0, 0, 0},
                                             mlir::ArrayRef<int64_t>{1, 1, 1, weightsBytes});
    auto tableBytesSubview = builder.create<VPUIP::SubViewOp>(loc, copyPacked.getOutput(),
                                                              mlir::ArrayRef<int64_t>{0, 0, 0, weightsBytes},
                                                              mlir::ArrayRef<int64_t>{1, 1, 1, tableBytes});

    auto weightsTypeCMX =
            vpux::getMemRefType({tileC, inputChannels, kernelSize, kernelSize}, f16Type, DimsOrder::NHWC, cmxSpace);
    auto weightTableTypeCMX = vpux::getMemRefType({tileC, 1, 1, 4}, i32Type, DimsOrder::OIYX, cmxSpace);

    auto weightsView = builder.create<VPUIP::ViewOp>(loc, weightsTypeCMX, weightBytesSubview.getResult());
    auto tableView = builder.create<VPUIP::ViewOp>(loc, weightTableTypeCMX, tableBytesSubview.getResult());

    return {weightsView.getResult(), tableView.getResult()};
}

mlir::OwningOpRef<mlir::ModuleOp> createHeterogeneousLoopPatternModule(mlir::MLIRContext* ctx,
                                                                       config::Platform platform) {
    auto loc = mlir::UnknownLoc::get(ctx);
    auto module = mlir::ModuleOp::create(loc);
    auto builder = mlir::OpBuilder(module.getBody(), module.getBody()->begin());

    const DimsOrder orderNHWC = DimsOrder::NHWC;
    const auto cmxSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::CMX_NN), 0);
    const auto ddrSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::DDR), 0);
    const auto f16Type = mlir::Float16Type::get(ctx);

    const int64_t numTilesH = 16;
    const int64_t inputSizeH = 32;
    const int64_t inputSizeW = 16;
    const int64_t inputChannels = 4;
    const int64_t outputChannels = 16;
    const int64_t kernelSize = 3;
    const int64_t stride = 1;
    const int64_t padding = 1;
    const int64_t tileH = inputSizeH / numTilesH;
    const int64_t tileC = outputChannels;

    auto inputTypeDDR = vpux::getMemRefType({1, inputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, ddrSpace);
    auto outputTypeDDR = vpux::getMemRefType({1, outputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, ddrSpace);
    auto inputTypeCMX = vpux::getMemRefType({1, inputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, cmxSpace);

    auto funcType = builder.getFunctionType({inputTypeDDR, outputTypeDDR}, {outputTypeDDR});
    auto func = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
    func.setPublic();

    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto inputArg = entryBlock->getArgument(0);
    auto outputArg = entryBlock->getArgument(1);

    auto input = builder.create<mlir::memref::AllocOp>(loc, inputTypeCMX);
    auto copyIn = builder.create<VPUIP::NNDMAOp>(loc, inputArg, input);

    // Pattern A iterations: packed single DMA for weights + weight table.
    const std::set<int64_t> patternAIterations = {0, 4, 5, 8, 13};

    llvm::SmallVector<mlir::Value> allTiles;
    allTiles.reserve(numTilesH);
    for (int64_t h = 0; h < numTilesH; ++h) {
        auto inputTile =
                builder.create<VPUIP::SubViewOp>(loc, copyIn.getOutput(), mlir::ArrayRef<int64_t>{0, 0, h * tileH, 0},
                                                 mlir::ArrayRef<int64_t>{1, inputChannels, tileH, inputSizeW});

        mlir::Value weightOp = nullptr;
        mlir::Value weightTableOp = nullptr;
        if (patternAIterations.count(h) > 0) {
            auto packedOperands =
                    createPackedWeightsAndTable(builder, loc, tileC, inputChannels, kernelSize, ddrSpace, cmxSpace);
            weightOp = packedOperands.first;
            weightTableOp = packedOperands.second;
        } else {
            Shape weightsShape = {tileC, inputChannels, kernelSize, kernelSize};
            weightOp = createWeights(builder, loc, f16Type, weightsShape, ddrSpace, cmxSpace);
            weightTableOp = createWeightsTable(builder, loc, tileC, ddrSpace, cmxSpace);
        }

        // Per-tile CMX output buffer (small, tile-sized)
        auto outputTileTypeCMX = vpux::getMemRefType({1, tileC, tileH, inputSizeW}, f16Type, orderNHWC, cmxSpace);
        auto outputTileCMX = builder.create<mlir::memref::AllocOp>(loc, outputTileTypeCMX);

        auto mpeEngineAttr = createMPEEngineAttr(ctx, platform);
        auto nceOp = createNCEClusterTaskOp(builder, ctx, loc, kernelSize, padding, stride, inputTile, weightOp,
                                            weightTableOp, outputTileCMX.getResult(), mpeEngineAttr);
        nceOp->setAttr(TILING_LOOP_INDEX_ATTR_NAME, builder.getI64IntegerAttr(0));

        auto& dpuTaskRegion = nceOp.getVariants();
        builder.setInsertionPointToStart(&dpuTaskRegion.front());
        createDPUTaskOp(builder, {0, 0, h * tileH}, {1, tileC, (h + 1) * tileH});
        builder.setInsertionPointAfter(nceOp);

        // Per-tile DATA_OUT: copy from per-tile CMX buffer to DDR subview
        auto outputTileDDR =
                builder.create<vpux::VPUIP::SubViewOp>(loc, outputArg, mlir::ArrayRef<int64_t>{0, 0, h * tileH, 0},
                                                       mlir::ArrayRef<int64_t>{1, tileC, tileH, inputSizeW});
        auto copyOut = builder.create<VPUIP::NNDMAOp>(loc, nceOp.getOutput(), outputTileDDR);
        allTiles.push_back(copyOut.getOutput());
    }

    auto concatOp = builder.create<VPUIP::ConcatViewOp>(loc, allTiles, outputArg);
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{concatOp.getOutput()});

    mlir::PassManager pm(ctx);
    VPUIP::buildAsyncSchedulingPipeline(pm);
    EXPECT_TRUE(mlir::succeeded(pm.run(func)));

    func.walk([&](mlir::async::ExecuteOp execOp) {
        execOp->setAttr(cycleCostAttrName, builder.getI64IntegerAttr(1000));
    });

    return module;
}

// Helper function to create a tiled convolution test
mlir::OwningOpRef<mlir::ModuleOp> createTiledConvolutionModule(mlir::MLIRContext* ctx, int numTilesH, int numTilesC,
                                                               config::Platform platform) {
    auto loc = mlir::UnknownLoc::get(ctx);
    auto module = mlir::ModuleOp::create(loc);
    auto builder = mlir::OpBuilder(module.getBody(), module.getBody()->begin());

    // setup
    const DimsOrder orderNHWC = DimsOrder::NHWC;
    const auto cmxSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::CMX_NN), 0);
    const auto ddrSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::DDR), 0);
    const auto f16Type = mlir::Float16Type::get(ctx);

    // Hardcoded test parameters for simplicity. Move to parameters later if needed.
    // They should be divisible by numTilesH and numTilesC, otherwise test logic will be more complicated.
    const int64_t inputSizeH = 160;
    const int64_t inputSizeW = 64;
    const int64_t inputChannels = 16;
    const int64_t outputChannels = 160;
    const int64_t kernelSize = 3;
    const int64_t stride = 1;
    const int64_t padding = 1;

    // Actual test parameters are number of tiles in H and C dimensions
    const int64_t tileH = inputSizeH / numTilesH;
    const int64_t remH = inputSizeH % numTilesH;
    VPUX_THROW_UNLESS(remH == 0, "Input height {0} is not divisible by numTilesH {1}", inputSizeH, numTilesH);
    const int64_t tileC = outputChannels / numTilesC;
    const int64_t remC = outputChannels % numTilesC;
    VPUX_THROW_UNLESS(remC == 0, "Output channels {0} is not divisible by numTilesC {1}", outputChannels, numTilesC);

    // Create function
    auto inputTypeDDR = vpux::getMemRefType({1, inputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, ddrSpace);
    auto outputTypeDDR = vpux::getMemRefType({1, outputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, ddrSpace);
    auto inputTypeCMX = vpux::getMemRefType({1, inputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, cmxSpace);

    auto funcType = builder.getFunctionType({inputTypeDDR, outputTypeDDR}, {outputTypeDDR});
    auto func = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
    func.setPublic();

    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto inputArg = entryBlock->getArgument(0);
    auto outputArg = entryBlock->getArgument(1);

    // Allocate CMX buffer for input
    auto input = builder.create<mlir::memref::AllocOp>(loc, inputTypeCMX);

    // Copy input argument from DDR to CMX
    auto copyIn = builder.create<VPUIP::NNDMAOp>(loc, inputArg, input);

    // Create weights and weight tables
    llvm::SmallVector<mlir::Value> weightOps;
    llvm::SmallVector<mlir::Value> weightTableOps;
    for (int c = 0; c < numTilesC; c++) {
        Shape weightsShape = {tileC, inputChannels, kernelSize, kernelSize};
        auto copyWeights = createWeights(builder, loc, f16Type, weightsShape, ddrSpace, cmxSpace);
        auto copyWeightsTable = createWeightsTable(builder, loc, tileC, ddrSpace, cmxSpace);
        weightOps.push_back(copyWeights);
        weightTableOps.push_back(copyWeightsTable);
    }

    llvm::SmallVector<mlir::Value> allCopyOuts;

    for (int h = 0; h < numTilesH; h++) {
        for (int c = 0; c < numTilesC; c++) {
            // Create input tile view
            auto inputTile = builder.create<vpux::VPUIP::SubViewOp>(
                    loc, copyIn.getOutput(), mlir::ArrayRef<int64_t>{0, 0, h * tileH, 0},
                    mlir::ArrayRef<int64_t>{1, inputChannels, tileH, inputSizeW});

            // Per-tile CMX output buffer (small, tile-sized)
            auto outputTileTypeCMX = vpux::getMemRefType({1, tileC, tileH, inputSizeW}, f16Type, orderNHWC, cmxSpace);
            auto outputTileCMX = builder.create<mlir::memref::AllocOp>(loc, outputTileTypeCMX);

            // Create NCE task
            auto mpeEngineAttr = createMPEEngineAttr(ctx, platform);
            auto nceOp = createNCEClusterTaskOp(builder, ctx, loc, kernelSize, padding, stride, inputTile, weightOps[c],
                                                weightTableOps[c], outputTileCMX.getResult(), mpeEngineAttr);

            // Set tilingIndex attribute
            auto tilingIndexAttr = builder.getI64IntegerAttr(0);
            nceOp->setAttr(TILING_LOOP_INDEX_ATTR_NAME, tilingIndexAttr);

            // Add DPU task
            auto& dpuTaskRegion = nceOp.getVariants();
            builder.setInsertionPointToStart(&dpuTaskRegion.front());

            createDPUTaskOp(builder, {0, c * tileC, h * tileH}, {1, (c + 1) * tileC, (h + 1) * tileH});
            builder.setInsertionPointAfter(nceOp);

            // Per-tile DATA_OUT: copy from per-tile CMX buffer to DDR subview
            auto outputTileDDR = builder.create<vpux::VPUIP::SubViewOp>(
                    loc, outputArg, mlir::ArrayRef<int64_t>{0, c * tileC, h * tileH, 0},
                    mlir::ArrayRef<int64_t>{1, tileC, tileH, inputSizeW});
            auto copyOut = builder.create<VPUIP::NNDMAOp>(loc, nceOp.getOutput(), outputTileDDR);
            allCopyOuts.push_back(copyOut.getOutput());
        }
    }

    // Concatenate all per-tile DDR copies into the DDR output argument
    auto concatOp = builder.create<vpux::VPUIP::ConcatViewOp>(loc, allCopyOuts, outputArg);

    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{concatOp.getOutput()});

    // Wrap into async regions
    mlir::PassManager pm(ctx);
    VPUIP::buildAsyncSchedulingPipeline(pm);
    EXPECT_TRUE(mlir::succeeded(pm.run(func)));

    // Assign dummy cost
    func.walk([&](mlir::async::ExecuteOp execOp) {
        execOp->setAttr(cycleCostAttrName, builder.getI64IntegerAttr(1000));
    });

    return module;
}

// Module builder with a shared CMX output buffer, ConcatView, and a single DATA_OUT copy.
// Each NCE tile writes to a SubView of the shared CMX buffer.  A ConcatViewOp combines the
// SubViews, and one NNDMAOp copies the full result from CMX to DDR.
mlir::OwningOpRef<mlir::ModuleOp> createSharedOutputTiledModule(mlir::MLIRContext* ctx, int numTilesC,
                                                                config::Platform platform) {
    auto loc = mlir::UnknownLoc::get(ctx);
    auto module = mlir::ModuleOp::create(loc);
    auto builder = mlir::OpBuilder(module.getBody(), module.getBody()->begin());

    const DimsOrder orderNHWC = DimsOrder::NHWC;
    const auto cmxSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::CMX_NN), 0);
    const auto ddrSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::DDR), 0);
    const auto f16Type = mlir::Float16Type::get(ctx);

    // Small dimensions so the shared CMX output buffer stays well under 2 MB.
    const int64_t inputSizeH = 8;
    const int64_t inputSizeW = 8;
    const int64_t inputChannels = 16;
    const int64_t outputChannels = 512;
    const int64_t kernelSize = 3;
    const int64_t stride = 1;
    const int64_t padding = 1;

    const int64_t tileC = outputChannels / numTilesC;
    VPUX_THROW_UNLESS(outputChannels % numTilesC == 0, "outputChannels {0} not divisible by numTilesC {1}",
                      outputChannels, numTilesC);

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

    // Copy input from DDR to CMX.
    auto inputCMX = builder.create<mlir::memref::AllocOp>(loc, inputTypeCMX);
    auto copyIn = builder.create<VPUIP::NNDMAOp>(loc, inputArg, inputCMX);

    // Shared CMX output buffer -- all NCE tiles write SubViews into this single allocation.
    auto outputCMX = builder.create<mlir::memref::AllocOp>(loc, outputTypeCMX);

    // Per-tile weights (private, not shared).
    llvm::SmallVector<mlir::Value> weightOps;
    llvm::SmallVector<mlir::Value> weightTableOps;
    for (int c = 0; c < numTilesC; ++c) {
        Shape weightsShape = {tileC, inputChannels, kernelSize, kernelSize};
        weightOps.push_back(createWeights(builder, loc, f16Type, weightsShape, ddrSpace, cmxSpace));
        weightTableOps.push_back(createWeightsTable(builder, loc, tileC, ddrSpace, cmxSpace));
    }

    llvm::SmallVector<mlir::Value> allTiles;
    for (int c = 0; c < numTilesC; ++c) {
        auto inputTile =
                builder.create<VPUIP::SubViewOp>(loc, copyIn.getOutput(), mlir::ArrayRef<int64_t>{0, 0, 0, 0},
                                                 mlir::ArrayRef<int64_t>{1, inputChannels, inputSizeH, inputSizeW});

        // Output tile is a SubView of the shared CMX root buffer.
        // ConcatView inputs and output share the same root buffer.
        auto outputTile = builder.create<VPUIP::SubViewOp>(loc, outputCMX.getResult(),
                                                           mlir::ArrayRef<int64_t>{0, c * tileC, 0, 0},
                                                           mlir::ArrayRef<int64_t>{1, tileC, inputSizeH, inputSizeW});

        auto mpeEngineAttr = createMPEEngineAttr(ctx, platform);
        auto nceOp = createNCEClusterTaskOp(builder, ctx, loc, kernelSize, padding, stride, inputTile, weightOps[c],
                                            weightTableOps[c], outputTile.getResult(), mpeEngineAttr);
        nceOp->setAttr(TILING_LOOP_INDEX_ATTR_NAME, builder.getI64IntegerAttr(0));

        auto& dpuTaskRegion = nceOp.getVariants();
        builder.setInsertionPointToStart(&dpuTaskRegion.front());
        createDPUTaskOp(builder, {0, c * tileC, 0}, {inputSizeW, (c + 1) * tileC, inputSizeH});
        builder.setInsertionPointAfter(nceOp);

        allTiles.push_back(nceOp.getOutput());
    }

    // ConcatView combines all SubViews into the shared CMX output buffer (same root).
    auto concatOp = builder.create<VPUIP::ConcatViewOp>(loc, allTiles, outputCMX);

    // Single DATA_OUT: copy the entire shared CMX buffer to DDR.
    auto copyOut = builder.create<VPUIP::NNDMAOp>(loc, concatOp.getOutput(), outputArg);
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{copyOut.getOutput()});

    mlir::PassManager pm(ctx);
    VPUIP::buildAsyncSchedulingPipeline(pm);
    EXPECT_TRUE(mlir::succeeded(pm.run(func)));

    func.walk([&](mlir::async::ExecuteOp execOp) {
        execOp->setAttr(cycleCostAttrName, builder.getI64IntegerAttr(1000));
    });

    return module;
}

mlir::OwningOpRef<mlir::ModuleOp> createChainedTiledModule(mlir::MLIRContext* ctx, int numTiles,
                                                           config::Platform platform) {
    auto loc = mlir::UnknownLoc::get(ctx);
    auto module = mlir::ModuleOp::create(loc);
    auto builder = mlir::OpBuilder(module.getBody(), module.getBody()->begin());

    const DimsOrder orderNHWC = DimsOrder::NHWC;
    const auto cmxSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::CMX_NN), 0);
    const auto ddrSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::DDR), 0);
    const auto f16Type = mlir::Float16Type::get(ctx);

    // 1x1 conv: output shape == input shape, so tiles can be chained arbitrarily.
    const int64_t channels = 16;
    const int64_t inputSizeH = 8;
    const int64_t inputSizeW = 8;
    const int64_t kernelSize = 1;
    const int64_t stride = 1;
    const int64_t padding = 0;

    auto tensorTypeDDR = vpux::getMemRefType({1, channels, inputSizeH, inputSizeW}, f16Type, orderNHWC, ddrSpace);
    auto tensorTypeCMX = vpux::getMemRefType({1, channels, inputSizeH, inputSizeW}, f16Type, orderNHWC, cmxSpace);

    auto funcType = builder.getFunctionType({tensorTypeDDR, tensorTypeDDR}, {tensorTypeDDR});
    auto func = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
    func.setPublic();

    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto inputArg = entryBlock->getArgument(0);
    auto outputArg = entryBlock->getArgument(1);

    // Copy input from DDR to CMX (shared only for tile_0; tiles 1..N-1 depend on the previous NCE output).
    auto inputCMX = builder.create<mlir::memref::AllocOp>(loc, tensorTypeCMX);
    auto copyIn = builder.create<VPUIP::NNDMAOp>(loc, inputArg, inputCMX);

    // Build the chain.  currentInput is updated to each tile's NCE output for the next tile.
    mlir::Value currentInput = copyIn.getOutput();
    for (int i = 0; i < numTiles; ++i) {
        // Private weights and weight table for each tile (not shared across iterations).
        Shape weightsShape = {channels, channels, kernelSize, kernelSize};
        auto weightOp = createWeights(builder, loc, f16Type, weightsShape, ddrSpace, cmxSpace);
        auto weightTableOp = createWeightsTable(builder, loc, channels, ddrSpace, cmxSpace);

        auto outputTile = builder.create<mlir::memref::AllocOp>(loc, tensorTypeCMX);

        auto mpeEngineAttr = createMPEEngineAttr(ctx, platform);
        auto nceOp = createNCEClusterTaskOp(builder, ctx, loc, kernelSize, padding, stride, currentInput, weightOp,
                                            weightTableOp, outputTile.getResult(), mpeEngineAttr);
        nceOp->setAttr(TILING_LOOP_INDEX_ATTR_NAME, builder.getI64IntegerAttr(0));

        auto& dpuTaskRegion = nceOp.getVariants();
        builder.setInsertionPointToStart(&dpuTaskRegion.front());
        // All tiles cover the full spatial region (no spatial tiling).
        createDPUTaskOp(builder, {0, 0, 0}, {inputSizeW, channels, inputSizeH});
        builder.setInsertionPointAfter(nceOp);

        // Pass this tile's CMX output as the next tile's input (inner dependency).
        currentInput = nceOp.getOutput();
    }

    // Copy final output from CMX to DDR.
    auto copyOut = builder.create<VPUIP::NNDMAOp>(loc, currentInput, outputArg);
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{copyOut.getOutput()});

    mlir::PassManager pm(ctx);
    VPUIP::buildAsyncSchedulingPipeline(pm);
    EXPECT_TRUE(mlir::succeeded(pm.run(func)));

    func.walk([&](mlir::async::ExecuteOp execOp) {
        execOp->setAttr(cycleCostAttrName, builder.getI64IntegerAttr(1000));
    });

    return module;
}

}  // namespace

TEST_P(MLIR_SchedulerLoopCreationTest, ConvolutionTiledOnC) {
    // create module
    const int tilesH = 1;
    const int tilesC = 10;
    auto module = createTiledConvolutionModule(getCtx(), tilesH, tilesC, GetParam());
    ASSERT_TRUE(module);
    EXPECT_TRUE(module->verify().succeeded());
    // Function contains:
    // 10 tiles over C, each tile has 2 DMAs: weights in, weights table in and 1 DPU CONV
    // Also there are copies to/from DDR for input and output

    // get alias info, live range info and async deps info
    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    // test start
    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);
    /*
    Expected structure with 10 C-tiles:
    All 10 iterations match: 2 DATA_IN + 1 COMPUTE + 1 DATA_OUT.

    ComputeRegionVec
    |---ComputeRegion 0:
    |   \---SchedulingLoop: type: None, iterations: 1
    |       \---loopBody:
    |           \---Alloc: opIdx_: 0, DATA_IN <-- input copy from DDR
    |
    \---ComputeRegion 1:
        \---SchedulingLoop: type: Tiling, iterations: 10
            \---loopBody:
                |---Iteration 0:
                |        Alloc: DATA_IN <-- weights in
                |        Alloc: DATA_IN <-- weights table in
                |        Alloc: COMPUTE
                |        Alloc: DATA_OUT --> per-tile copy to DDR
                |---...
                \---Iteration 9: (same structure)
    */

    // 2 compute regions expected: 1 non-loop DATA_IN + 1 tiling loop
    EXPECT_EQ(regions.size(), 2);

    // Find the tiling loop region
    size_t tiledRegionIndex = regions.size();
    for (size_t i = 0; i < regions.size(); ++i) {
        if (regions[i].getLoopType() == LoopType::Tiling) {
            tiledRegionIndex = i;
            break;
        }
    }
    ASSERT_LT(tiledRegionIndex, regions.size()) << "No tiling region found";
    const auto loopSize = regions[tiledRegionIndex].schedulingLoop->loopBodies.size();
    EXPECT_EQ(loopSize, 10);

    // check that the first iteration contains expected ops: 2 DATA_IN + 1 COMPUTE + 1 DATA_OUT
    const auto& firstIteration = regions[tiledRegionIndex].schedulingLoop->loopBodies[0];
    EXPECT_EQ(firstIteration.size(), 4);
    // check that the first op is weights data in
    const auto firstOp = depsInfo.getExecuteOpAtIndex(firstIteration[0].opIdx);
    const auto firstExecKind = VPUIP::VPUIPDialect::getExecutorKind(firstOp);
    EXPECT_EQ(firstExecKind, config::ExecutorKind::DMA_NN);
    // check that the second op is weights table data in
    const auto secondOp = depsInfo.getExecuteOpAtIndex(firstIteration[1].opIdx);
    const auto secondExecKind = VPUIP::VPUIPDialect::getExecutorKind(secondOp);
    EXPECT_EQ(secondExecKind, config::ExecutorKind::DMA_NN);
    // check that the third op is DPU
    const auto dpuOp = depsInfo.getExecuteOpAtIndex(firstIteration[2].opIdx);
    const auto dpuExecKind = VPUIP::VPUIPDialect::getExecutorKind(dpuOp);
    EXPECT_EQ(dpuExecKind, config::ExecutorKind::DPU);
    // check that the fourth op is DATA_OUT DMA
    const auto dataOutOp = depsInfo.getExecuteOpAtIndex(firstIteration[3].opIdx);
    const auto dataOutExecKind = VPUIP::VPUIPDialect::getExecutorKind(dataOutOp);
    EXPECT_EQ(dataOutExecKind, config::ExecutorKind::DMA_NN);
}

TEST_P(MLIR_SchedulerLoopCreationTest, ConvolutionTiledOnH) {
    // create module
    const int tilesH = 8;
    const int tilesC = 1;
    auto module = createTiledConvolutionModule(getCtx(), tilesH, tilesC, GetParam());
    ASSERT_TRUE(module);
    EXPECT_TRUE(module->verify().succeeded());

    // get alias info, live range info and async deps info
    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    // test start
    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);
    /*
    Expected structure with 8 H-tiles:
    Weights and weight table are shared across all H-tiles and filtered out of the loop.
    Each iteration has its own per-tile DATA_OUT (per-tile CMX buffer -> DDR subview).
    All 8 iterations match: 1 COMPUTE + 1 DATA_OUT.

    ComputeRegionVec
    |---ComputeRegion 0:
    |   \---SchedulingLoop: type: None, iterations: 1
    |       \---loopBody:
    |           \---Alloc: opIdx_: 0, DATA_IN <-- input copy from DDR
    |
    |---ComputeRegion 1:
    |   \---SchedulingLoop: type: None, iterations: 1
    |       \---loopBody:
    |           \---Alloc: opIdx_: 1, DATA_IN <-- weights (shared across all H-tiles)
    |
    |---ComputeRegion 2:
    |   \---SchedulingLoop: type: None, iterations: 1
    |       \---loopBody:
    |           \---Alloc: opIdx_: 2, DATA_IN <-- weight table (shared across all H-tiles)
    |
    \---ComputeRegion 3:
        \---SchedulingLoop: type: Tiling, iterations: 8
            \---loopBody:
                |---Iteration 0:
                |   \---Alloc: COMPUTE
                |   \---Alloc: DATA_OUT --> per-tile copy to DDR
                |---...
                \---Iteration 7: (same structure)
    */

    // 4 compute regions expected: 3 non-loop DATA_IN + 1 tiling loop
    EXPECT_EQ(regions.size(), 4);

    const size_t tiledRegionIndex = 3;
    const auto loopSize = regions[tiledRegionIndex].schedulingLoop->loopBodies.size();
    EXPECT_EQ(loopSize, 8);

    // check that iterations contain expected ops: COMPUTE + DATA_OUT (weights are shared)
    const auto& firstIteration = regions[tiledRegionIndex].schedulingLoop->loopBodies[0];
    EXPECT_EQ(firstIteration.size(), 2);  // 1 DPU CONV + 1 DATA_OUT DMA
    // check that the first op is DPU
    const auto conv = depsInfo.getExecuteOpAtIndex(firstIteration[0].opIdx);
    const auto convExecKind = VPUIP::VPUIPDialect::getExecutorKind(conv);
    EXPECT_EQ(convExecKind, config::ExecutorKind::DPU);
}

TEST_P(MLIR_SchedulerLoopCreationTest, ConvolutionTiledOnCAndH) {
    // create module
    const int tilesH = 8;
    const int tilesC = 10;
    auto module = createTiledConvolutionModule(getCtx(), tilesH, tilesC, GetParam());
    ASSERT_TRUE(module);
    EXPECT_TRUE(module->verify().succeeded());

    // get alias info, live range info and async deps info
    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    // test start
    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);
    /*
    Expected structure with 8 H-tiles x 10 C-tiles = 80 total tiles:
    Each iteration has per-tile DATA_OUT (per-tile CMX buffer -> DDR subview).
    All 80 iterations match: 2 DATA_IN + 1 COMPUTE + 1 DATA_OUT.

    ComputeRegionVec
    |---ComputeRegion 0:
    |   \---SchedulingLoop: type: None, iterations: 1
    |       \---loopBody:
    |           \---Alloc: opIdx_: 0, DATA_IN <-- input copy from DDR
    |
    \---ComputeRegion 1:
        \---SchedulingLoop: type: Tiling, iterations: 80
            \---loopBody:
                |---Iteration 0: DATA_IN, DATA_IN, COMPUTE, DATA_OUT
                |---...
                \---Iteration 79: (same structure)
    */

    // 2 compute regions expected: 1 non-loop DATA_IN + 1 tiling loop
    EXPECT_EQ(regions.size(), 2);

    size_t numTiledRegions = 0;
    size_t numNonLoopRegions = 0;
    for (const auto& region : regions) {
        if (region.getLoopType() == LoopType::Tiling) {
            ++numTiledRegions;
        } else {
            ++numNonLoopRegions;
        }
    }
    EXPECT_EQ(numTiledRegions, 1);
    EXPECT_EQ(numNonLoopRegions, 1);
}

TEST_P(MLIR_SchedulerLoopCreationTest, HeterogeneousIterationsAreNotMergedIntoSingleRegion) {
    /*
    Heterogeneous tiled loop pattern (same tiling index, mixed iteration structure):

    Pattern A (5 iterations): 0, 4, 5, 8, 13
        - Weight DMA: 1 DMA (packed fused constant)
            shape: 1x1x1x147712xf16
        - Iteration ops: 3 ops = 1 DATA_IN + 1 COMPUTE + 1 DATA_OUT

    Pattern B (11 iterations): 1-3, 6-7, 9-12, 14-15
        - Weight DMA: 2 DMA (separate weight + weight_table)
            weight shape:       16x4x3x3xf16
            weight_table shape: 16x1x1x4xsi32
        - Iteration ops: 4 ops = 2 DATA_IN + 1 COMPUTE + 1 DATA_OUT

    Expected behavior for current policy:
        - keep loop creation enabled for homogeneous sub-groups
        - split this heterogeneous set into 2 tiling regions (A-group and B-group)
        - each iteration includes its own DATA_OUT copy
    */
    auto module = createHeterogeneousLoopPatternModule(getCtx(), GetParam());
    ASSERT_TRUE(module);
    EXPECT_TRUE(module->verify().succeeded());

    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);

    size_t numTiledRegions = llvm::count_if(regions, [](const auto& region) {
        return region.schedulingLoop != nullptr && region.schedulingLoop->type == LoopType::Tiling;
    });

    EXPECT_EQ(numTiledRegions, 2);

    bool foundPatternA = false;
    bool foundPatternB = false;
    for (const auto& region : regions) {
        if (region.schedulingLoop == nullptr || region.schedulingLoop->type != LoopType::Tiling) {
            continue;
        }

        const auto& loopBodies = region.schedulingLoop->loopBodies;
        ASSERT_FALSE(loopBodies.empty());

        const auto iterations = loopBodies.size();
        const auto opsPerIteration = loopBodies.front().size();
        if (iterations == 5 && opsPerIteration == 3) {
            foundPatternA = true;  // 1 DATA_IN + 1 COMPUTE + 1 DATA_OUT
        }
        if (iterations == 11 && opsPerIteration == 4) {
            foundPatternB = true;  // 2 DATA_IN + 1 COMPUTE + 1 DATA_OUT
        }
    }

    EXPECT_TRUE(foundPatternA);
    EXPECT_TRUE(foundPatternB);
}

TEST_P(MLIR_SchedulerLoopCreationTest, GenerateLoopSchedules_EmptyRegions) {
    // generateLoopSchedules with empty input produces empty output
    ComputeRegionVec emptyRegions;
    vpux::AddressType memorySize = 1024 * 1024;  // 1 MB
    auto result = VPU::generateLoopSchedules(emptyRegions, memorySize, /*enableVfUndefinedScheduler=*/false,
                                             Logger::global());

    EXPECT_TRUE(result.scheduleResults.empty()) << "No schedules should be generated for empty regions";
    EXPECT_TRUE(result.loopRegionInd.empty()) << "No loop region indices for empty regions";
    EXPECT_TRUE(result.loopPrefetchInd.empty()) << "No prefetch indices for empty regions";
}

TEST_P(MLIR_SchedulerLoopCreationTest, GenerateLoopSchedules_NonLoopRegionsIgnored) {
    // Regions with LoopType::None should be skipped by generateLoopSchedules
    const int tilesH = 8;
    const int tilesC = 1;
    auto module = createTiledConvolutionModule(getCtx(), tilesH, tilesC, GetParam());
    ASSERT_TRUE(module);

    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);

    // Extract only non-loop regions
    ComputeRegionVec nonLoopRegions;
    for (auto& region : regions) {
        if (region.getLoopType() == LoopType::None) {
            nonLoopRegions.push_back(std::move(region));
        }
    }
    ASSERT_FALSE(nonLoopRegions.empty()) << "Should have non-loop regions";

    vpux::AddressType memorySize = 2 * 1024 * 1024;
    auto result = VPU::generateLoopSchedules(nonLoopRegions, memorySize, /*enableVfUndefinedScheduler=*/false,
                                             Logger::global());

    EXPECT_TRUE(result.scheduleResults.empty()) << "Non-loop regions should produce no schedules";
    EXPECT_TRUE(result.loopRegionInd.empty());
    EXPECT_TRUE(result.loopPrefetchInd.empty());
}

TEST_P(MLIR_SchedulerLoopCreationTest, GenerateLoopSchedules_TilingRegionProducesSchedule) {
    // Tiling regions should produce non-empty predefined schedules
    const int tilesH = 1;
    const int tilesC = 10;
    auto module = createTiledConvolutionModule(getCtx(), tilesH, tilesC, GetParam());
    ASSERT_TRUE(module);

    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);

    vpux::AddressType memorySize = 2 * 1024 * 1024;
    auto result =
            VPU::generateLoopSchedules(regions, memorySize, /*enableVfUndefinedScheduler=*/false, Logger::global());

    // At least one tiling region should generate a schedule
    EXPECT_FALSE(result.scheduleResults.empty()) << "Tiling regions should produce at least one schedule";

    // Verify that operations are categorized into loop region or prefetch sets
    EXPECT_FALSE(result.loopRegionInd.empty() && result.loopPrefetchInd.empty())
            << "At least some operations should be categorized in loop region or prefetch sets";
}

TEST_P(MLIR_SchedulerLoopCreationTest, GenerateLoopSchedules_OperationIndexSetsAreDisjoint) {
    // loopRegionInd and loopPrefetchInd should have no overlap
    const int tilesH = 1;
    const int tilesC = 10;
    auto module = createTiledConvolutionModule(getCtx(), tilesH, tilesC, GetParam());
    ASSERT_TRUE(module);

    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);

    vpux::AddressType memorySize = 2 * 1024 * 1024;
    auto result =
            VPU::generateLoopSchedules(regions, memorySize, /*enableVfUndefinedScheduler=*/false, Logger::global());

    // Check no overlap between loopRegionInd and loopPrefetchInd
    for (auto idx : result.loopRegionInd) {
        EXPECT_FALSE(result.loopPrefetchInd.contains(idx))
                << "Operation index " << idx << " is in both loopRegionInd and loopPrefetchInd";
    }
}

TEST_P(MLIR_SchedulerLoopCreationTest, GenerateLoopSchedules_ChainedTiledIterationsHaveInnerDependencies) {
    /*
    A chain of numTiles 1x1 NCE ops where tile_i's CMX output is tile_{i+1}'s CMX input.
    This creates cross-iteration dependencies (inner deps) within the same tiling loop:

        tile_0: input_CMX            -> NCE_0 -> output_0_CMX
        tile_1: output_0_CMX (inner dep on tile_0) -> NCE_1 -> output_1_CMX
        tile_2: output_1_CMX (inner dep on tile_1) -> NCE_2 -> output_2_CMX
        ...
        tile_5: output_4_CMX (inner dep on tile_4) -> NCE_5 -> output_5_CMX -> DATA_OUT_DMA

    Each tile has private weight and weight-table DMAs (count=1 per tile, never filtered).
    Actual behavior with current implementation (no global deps/cons comparison):
        createTiledOpDepsConsDescriptor:
            tile_0 body: {input_DMA, weight_DMA_0, weight_table_DMA_0, NCE_0}     (4 ops)
            tile_i body (0<i<5): {weight_DMA_i, weight_table_DMA_i, NCE_i}        (3 ops)
            tile_5 body: {weight_DMA_5, weight_table_DMA_5, NCE_5, DATA_OUT_DMA}  (4 ops)
            No op is shared across all 6 iterations -> no filtering.

        createInnerLoopsFromIterations:
            All NCE ops have 3 unique inBuffers and 1 unique outBuffer -> all 6 match.
            Cross-tile deps (NCE_i <- NCE_{i+1}) are NOT compared -> all 6 are merged.
            allLoopOperations contains all 20 ops -> 0 non-loop regions.

    Result: 1 tiling loop with 6 iterations (no non-loop regions):
        iteration 0: 4 ops (input_DMA absorbed as private dep of tile_0)
        iterations 1..4: 3 ops each
        iteration 5: 4 ops (DATA_OUT_DMA absorbed as consumer of last tile)
    */
    const int numTiles = 6;  // > MIN_TILING_LOOP_OPS (5)
    auto module = createChainedTiledModule(getCtx(), numTiles, GetParam());
    ASSERT_TRUE(module);
    EXPECT_TRUE(module->verify().succeeded());

    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);

    // All ops (including input/output DMAs) are absorbed into the tiling loop's iteration bodies.
    // No non-loop regions are produced because every op appears in exactly one iteration body.
    ASSERT_EQ(regions.size(), 1u);
    EXPECT_EQ(regions[0].getLoopType(), LoopType::Tiling);

    const auto& loopBodies = regions[0].schedulingLoop->loopBodies;
    ASSERT_EQ(loopBodies.size(), static_cast<size_t>(numTiles));

    // Iteration 0: input_DMA is private to tile_0 (count=1 != numTiles -> not filtered),
    // so it remains in the first iteration's body alongside the weight DMAs and COMPUTE.
    EXPECT_EQ(loopBodies.front().size(), 4u) << "iter 0: input_DMA + weight + weight_table + COMPUTE";
    EXPECT_EQ(loopBodies.front()[0].allocationType, AllocationType::DATA_IN);  // input_DMA
    EXPECT_EQ(loopBodies.front()[3].allocationType, AllocationType::COMPUTE);  // NCE_0

    // Middle iterations: only private weight DMAs + COMPUTE (no absorbed input/output DMAs).
    for (int i = 1; i < numTiles - 1; ++i) {
        EXPECT_EQ(loopBodies[i].size(), 3u) << "iter " << i << ": weight + weight_table + COMPUTE";
        EXPECT_EQ(loopBodies[i][2].allocationType, AllocationType::COMPUTE);
    }

    // Last iteration: DATA_OUT_DMA is absorbed because it is the sole consumer of NCE_5
    // and its count (1) != numTiles -> not filtered.
    EXPECT_EQ(loopBodies.back().size(), 4u) << "iter 5: weight + weight_table + COMPUTE + DATA_OUT_DMA";
    EXPECT_EQ(loopBodies.back()[2].allocationType, AllocationType::COMPUTE);   // NCE_5
    EXPECT_EQ(loopBodies.back()[3].allocationType, AllocationType::DATA_OUT);  // DATA_OUT_DMA

    // The chained tiling loop should also produce a non-empty schedule
    vpux::AddressType memorySize = 2 * 1024 * 1024;
    auto result =
            VPU::generateLoopSchedules(regions, memorySize, /*enableVfUndefinedScheduler=*/false, Logger::global());
    EXPECT_FALSE(result.scheduleResults.empty()) << "Chained tiling loop should produce at least one schedule";
    EXPECT_FALSE(result.loopRegionInd.empty() && result.loopPrefetchInd.empty())
            << "At least some operations should be categorized in loop region or prefetch sets";
}

TEST_P(MLIR_SchedulerLoopCreationTest, GenerateLoopSchedules_CMXConcatDataOutAbsorbedByLastIteration) {
    /*
    Shared CMX output buffer pattern: all NCE tiles write SubViews of one shared CMX AllocOp.
    A ConcatViewOp combines the SubViews, and a single NNDMAOp copies to DDR (one DATA_OUT).

        DDR input -> CMX input (DATA_IN, shared across all tiles)
        tile_0: SubView(CMX_input) -> NCE_0 -> SubView(CMX_output, [0..tileC))
        tile_1: SubView(CMX_input) -> NCE_1 -> SubView(CMX_output, [tileC..2*tileC))
        ...
        tile_7: SubView(CMX_input) -> NCE_7 -> SubView(CMX_output, [7*tileC..64))
        ConcatView(all SubViews) -> CMX_output
        CMX_output -> DDR output (single DATA_OUT)

    With 8 C-tiles (outputChannels=64, tileC=8), each tile body has:
        2 DATA_IN (weights DMA + weight_table DMA) + 1 COMPUTE (NCE)
    The input copy is shared across all iterations and filtered out.
    The single DATA_OUT (ConcatView -> NNDMAOp) is absorbed into the last iteration
    because its dependency count (1) != numTiles -> not filtered.

    Result:
        ComputeRegion 0: non-loop (input copy from DDR)
        ComputeRegion 1: tiling loop with 8 iterations
            iterations 0..6: 3 ops each (weight + weight_table + COMPUTE)
            iteration 7: 4 ops (weight + weight_table + COMPUTE + DATA_OUT)
    */
    const int numTilesC = 8;
    auto module = createSharedOutputTiledModule(getCtx(), numTilesC, GetParam());
    ASSERT_TRUE(module);
    EXPECT_TRUE(module->verify().succeeded());

    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);

    // Expect 2 regions: non-loop input copy + tiling loop.
    ASSERT_EQ(regions.size(), 2u);
    EXPECT_EQ(regions[0].getLoopType(), LoopType::None);
    EXPECT_EQ(regions[1].getLoopType(), LoopType::Tiling);

    const auto& loopBodies = regions[1].schedulingLoop->loopBodies;
    ASSERT_EQ(loopBodies.size(), static_cast<size_t>(numTilesC));

    // Middle iterations: 3 ops each (2 DATA_IN + COMPUTE).
    for (int i = 0; i < numTilesC - 1; ++i) {
        EXPECT_EQ(loopBodies[i].size(), 3u) << "iter " << i << ": weight + weight_table + COMPUTE";
        EXPECT_EQ(loopBodies[i][0].allocationType, AllocationType::DATA_IN);
        EXPECT_EQ(loopBodies[i][1].allocationType, AllocationType::DATA_IN);
        EXPECT_EQ(loopBodies[i][2].allocationType, AllocationType::COMPUTE);
    }

    // Last iteration absorbs the single DATA_OUT (ConcatView -> DDR copy).
    EXPECT_EQ(loopBodies.back().size(), 4u) << "last iter: weight + weight_table + COMPUTE + DATA_OUT";
    EXPECT_EQ(loopBodies.back()[2].allocationType, AllocationType::COMPUTE);
    EXPECT_EQ(loopBodies.back()[3].allocationType, AllocationType::DATA_OUT);

    // Schedule generation should succeed -- the shared CMX output buffer is tracked via SubViews.
    vpux::AddressType memorySize = 2 * 1024 * 1024;
    auto result =
            VPU::generateLoopSchedules(regions, memorySize, /*enableVfUndefinedScheduler=*/false, Logger::global());
    EXPECT_FALSE(result.scheduleResults.empty()) << "Shared-output tiling loop should produce a schedule";
}

INSTANTIATE_TEST_SUITE_P(SchedulerLoopCreation, MLIR_SchedulerLoopCreationTest,
                         testing::Values(config::Platform::NPU4000, config::Platform::NPU5010), schedulerTestParamName);
