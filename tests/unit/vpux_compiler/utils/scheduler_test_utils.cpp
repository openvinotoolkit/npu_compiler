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
    return VPU::MPEEngine37XXAttr::get(ctx, VPU::MPEEngine37XXModeAttr::get(ctx, VPU::MPEEngine37XXMode::SCL));
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

}  // namespace vpux
