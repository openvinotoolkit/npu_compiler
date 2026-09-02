//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scheduler_test_utils.hpp"

#include "common/nce_utils.hpp"

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/init/interfaces_registry.hpp"
#include "vpux/compiler/init/singleton_initializer.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/PassManager.h>

namespace vpux {

void MLIR_SchedulerLoopCreationTestBase::SetUp() {
    registry = vpux::createDialectRegistry();
    auto interfacesRegistry = vpux::createInterfacesRegistry(GetParam());
    interfacesRegistry->registerInterfaces(registry);
    VPU::initializeSingletons(registry, GetParam());

    ctx = std::make_unique<mlir::MLIRContext>(registry);
    ctx->appendDialectRegistry(registry);
    ctx->loadDialect<VPUIP::VPUIPDialect>();
    ctx->loadDialect<vpux::VPU::VPUDialect>();
}

mlir::MLIRContext* MLIR_SchedulerLoopCreationTestBase::getCtx() {
    return ctx.get();
}

std::string schedulerTestParamName(const testing::TestParamInfo<config::Platform>& info) {
    return config::stringifyEnum(info.param).str();
}

void printComputeRegions(const ComputeRegionVec& regions, vpux::Logger log) {
    log.debug("Compute Regions:");
    for (const auto& region : regions) {
        std::string buf;
        llvm::raw_string_ostream os(buf);
        region.printFormat(os);
        log.nest().debug("{0}", buf);
    }
}

void printDebugIR(mlir::ModuleOp module, vpux::Logger log) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    module.print(os);
    log.debug("IR Dump:\n{0}", buf);
}

VPU::MPEEngineAttr createMPEEngineAttr(mlir::MLIRContext* ctx, [[maybe_unused]] config::Platform platform) {
    return VPU::MPEEngine37XXAttr::get(ctx, VPU::MPEEngine37XXModeAttr::get(ctx, VPU::MPEEngine37XXMode::SCL),
                                       /*weightZp=*/nullptr, /*activationZp=*/nullptr);
}

VPUIP::NCEClusterTaskOp createNCEClusterTaskOp(mlir::OpBuilder& builder, mlir::MLIRContext* ctx, mlir::Location loc,
                                               int64_t kernel, int64_t padding, int64_t stride, mlir::Value inputTile,
                                               mlir::Value weightOp, mlir::Value weightTableOp, mlir::Value outputTile,
                                               VPU::MPEEngineAttr mpeEngineAttr) {
    auto paddingAttr =
            vpux::VPU::PaddingAttr::get(ctx, builder.getI64IntegerAttr(padding), builder.getI64IntegerAttr(padding),
                                        builder.getI64IntegerAttr(padding), builder.getI64IntegerAttr(padding));
    auto kernelSize = builder.getI64ArrayAttr({kernel, kernel});
    auto kernelStrides = builder.getI64ArrayAttr({stride, stride});
    auto nceOp = builder.create<VPUIP::NCEClusterTaskOp>(
            loc, inputTile, nullptr, nullptr, weightOp, nullptr, weightTableOp, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, inputTile, nullptr, nullptr, outputTile, nullptr, mlir::ValueRange(), outputTile,
            nullptr, nullptr, nullptr, nullptr, nullptr, mlir::ValueRange(), VPUIP::NCETaskType::CONV, kernelSize,
            kernelStrides, paddingAttr, false, nullptr, false, nullptr, false, false, false, nullptr, nullptr, nullptr,
            false, false, mpeEngineAttr, nullptr, nullptr, nullptr, nullptr);
    return nceOp;
}

mlir::Value createWeightsTable(mlir::OpBuilder& builder, mlir::Location loc, int64_t tileC,
                               const vpux::IndexedSymbolAttr& ddrSpace, const vpux::IndexedSymbolAttr& cmxSpace) {
    auto weightTableTypeDDR =
            vpux::getMemRefType({tileC, 1, 1, 4}, builder.getIntegerType(32, true), DimsOrder::OIYX, ddrSpace);
    auto weightTableTypeCMX =
            vpux::getMemRefType({tileC, 1, 1, 4}, builder.getIntegerType(32, true), DimsOrder::OIYX, cmxSpace);
    auto weightTableTensorType = mlir::RankedTensorType::get({tileC, 1, 1, 4}, builder.getIntegerType(32, true));

    auto weightTableAttr = mlir::DenseElementsAttr::get(weightTableTensorType, mlir::APInt(32, 1, true));
    auto weightTableDDR = builder.create<vpux::Const::DeclareOp>(loc, weightTableTypeDDR,
                                                                 vpux::Const::ContentAttr::get(weightTableAttr));
    auto weightTableCMX = builder.create<mlir::memref::AllocOp>(loc, weightTableTypeCMX);
    auto copyWeightsTable = builder.create<VPUIP::NNDMAOp>(loc, weightTableDDR, weightTableCMX);

    return copyWeightsTable.getOutput();
}

mlir::Value createWeights(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type elemType, ShapeRef weightsShape,
                          const vpux::IndexedSymbolAttr& ddrSpace, const vpux::IndexedSymbolAttr& cmxSpace) {
    auto weightsTypeDDR = vpux::getMemRefType(weightsShape, elemType, DimsOrder::NHWC, ddrSpace);
    auto weightsTypeCMX = vpux::getMemRefType(weightsShape, elemType, DimsOrder::NHWC, cmxSpace);

    auto weightsTensorType = mlir::RankedTensorType::get(weightsShape, elemType);

    Const::ContentSetup contentAttrSetup(nullptr, weightsTensorType);
    contentAttrSetup = contentAttrSetup.castElemType(elemType);
    contentAttrSetup = contentAttrSetup.reorder(DimsOrder::NHWC);
    auto weightsAttr = mlir::DenseElementsAttr::get(weightsTensorType, llvm::APFloat(mlir::APFloat::IEEEhalf(), "1.0"));
    auto contentAttr = Const::ContentAttr::get(weightsAttr, contentAttrSetup);

    auto weightsDDR = builder.create<Const::DeclareOp>(loc, weightsTypeDDR, contentAttr).getOutput();
    auto weightsCMX = builder.create<mlir::memref::AllocOp>(loc, weightsTypeCMX);
    auto copyWeights = builder.create<VPUIP::NNDMAOp>(loc, weightsDDR, weightsCMX);

    return copyWeights.getOutput();
}

mlir::OwningOpRef<mlir::ModuleOp> createTiledConvolutionModule(mlir::MLIRContext* ctx, int numTilesH, int numTilesC,
                                                               config::Platform platform, bool addDdr2DdrConsumers) {
    auto loc = mlir::UnknownLoc::get(ctx);
    auto module = mlir::ModuleOp::create(loc);
    module->setAttr(config::PlatformAttr::name, config::PlatformAttr::get(ctx, platform));
    auto builder = mlir::OpBuilder(module.getBody(), module.getBody()->begin());

    const DimsOrder orderNHWC = DimsOrder::NHWC;
    const auto cmxSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::CMX_NN), 0);
    const auto ddrSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::DDR), 0);
    const auto f16Type = mlir::Float16Type::get(ctx);

    const int64_t inputSizeH = 160;
    const int64_t inputSizeW = 64;
    const int64_t inputChannels = 16;
    const int64_t outputChannels = 160;
    const int64_t kernelSize = 3;
    const int64_t stride = 1;
    const int64_t padding = 1;

    const int64_t tileH = inputSizeH / numTilesH;
    const int64_t remH = inputSizeH % numTilesH;
    VPUX_THROW_UNLESS(remH == 0, "Input height {0} is not divisible by numTilesH {1}", inputSizeH, numTilesH);
    const int64_t tileC = outputChannels / numTilesC;
    const int64_t remC = outputChannels % numTilesC;
    VPUX_THROW_UNLESS(remC == 0, "Output channels {0} is not divisible by numTilesC {1}", outputChannels, numTilesC);

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

    mlir::Value intermediateOutputDDR;
    if (addDdr2DdrConsumers) {
        intermediateOutputDDR = builder.create<mlir::memref::AllocOp>(loc, outputTypeDDR).getResult();
    }

    auto input = builder.create<mlir::memref::AllocOp>(loc, inputTypeCMX);
    auto copyIn = builder.create<VPUIP::NNDMAOp>(loc, inputArg, input);

    llvm::SmallVector<mlir::Value> weightOps;
    llvm::SmallVector<mlir::Value> weightTableOps;
    for (int c = 0; c < numTilesC; c++) {
        Shape weightsShape = {tileC, inputChannels, kernelSize, kernelSize};
        weightOps.push_back(createWeights(builder, loc, f16Type, weightsShape, ddrSpace, cmxSpace));
        weightTableOps.push_back(createWeightsTable(builder, loc, tileC, ddrSpace, cmxSpace));
    }

    llvm::SmallVector<mlir::Value> allCopyOuts;
    for (int h = 0; h < numTilesH; h++) {
        for (int c = 0; c < numTilesC; c++) {
            auto inputTile = builder.create<vpux::VPUIP::SubViewOp>(
                    loc, copyIn.getOutput(), mlir::ArrayRef<int64_t>{0, 0, h * tileH, 0},
                    mlir::ArrayRef<int64_t>{1, inputChannels, tileH, inputSizeW});

            auto outputTileTypeCMX = vpux::getMemRefType({1, tileC, tileH, inputSizeW}, f16Type, orderNHWC, cmxSpace);
            auto outputTileCMX = builder.create<mlir::memref::AllocOp>(loc, outputTileTypeCMX);

            auto mpeEngineAttr = createMPEEngineAttr(ctx, platform);
            auto nceOp = createNCEClusterTaskOp(builder, ctx, loc, kernelSize, padding, stride, inputTile, weightOps[c],
                                                weightTableOps[c], outputTileCMX.getResult(), mpeEngineAttr);
            nceOp->setAttr(TILING_LOOP_INDEX_ATTR_NAME, builder.getI64IntegerAttr(0));

            auto& dpuTaskRegion = nceOp.getVariants();
            builder.setInsertionPointToStart(&dpuTaskRegion.front());
            createDPUTaskOp(builder, {0, c * tileC, h * tileH}, {1, (c + 1) * tileC, (h + 1) * tileH});
            builder.setInsertionPointAfter(nceOp);

            const auto outputTarget = addDdr2DdrConsumers ? intermediateOutputDDR : outputArg;
            auto outputTileDDR = builder.create<vpux::VPUIP::SubViewOp>(
                    loc, outputTarget, mlir::ArrayRef<int64_t>{0, c * tileC, h * tileH, 0},
                    mlir::ArrayRef<int64_t>{1, tileC, tileH, inputSizeW});
            auto copyOut = builder.create<VPUIP::NNDMAOp>(loc, nceOp.getOutput(), outputTileDDR);
            allCopyOuts.push_back(copyOut.getOutput());
        }
    }

    mlir::Value finalOutput;
    if (addDdr2DdrConsumers) {
        auto copyToOutput = builder.create<VPUIP::NNDMAOp>(loc, intermediateOutputDDR, outputArg);
        finalOutput = copyToOutput.getOutput();
    } else {
        auto concatOp = builder.create<vpux::VPUIP::ConcatViewOp>(loc, allCopyOuts, outputArg);
        finalOutput = concatOp.getOutput();
    }

    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{finalOutput});

    mlir::PassManager pm(ctx);
    VPUIP::buildAsyncSchedulingPipeline(pm);
    EXPECT_TRUE(mlir::succeeded(pm.run(func)));

    func.walk([&](mlir::async::ExecuteOp execOp) {
        execOp->setAttr(cycleCostAttrName, builder.getI64IntegerAttr(1000));
    });

    return module;
}

}  // namespace vpux
