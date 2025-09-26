//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <mlir/IR/Builders.h>

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_SETUPPROFILINGVPUMI40XX
#define GEN_PASS_DEF_SETUPPROFILINGVPUMI40XX
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

//
// SetupProfilingVPUMI40XXPass
//

class SetupProfilingVPUMI40XXPass final :
        public VPUMI40XX::impl::SetupProfilingVPUMI40XXBase<SetupProfilingVPUMI40XXPass> {
public:
    explicit SetupProfilingVPUMI40XXPass(const std::string& enableDmaProfiling, Logger log)
            : SetupProfilingVPUMI40XXBase({enableDmaProfiling}) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

    mlir::Value createDmaHwpBaseStatic(mlir::OpBuilder builderFunc, VPUIP::ProfilingSectionOp dmaSection) {
        _log.trace("createDmaHwpBase");
        const auto ctx = builderFunc.getContext();
        VPUX_THROW_UNLESS((dmaSection.getOffset() % VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX) == 0,
                          "Unaligned HWP base");
        VPUX_THROW_UNLESS((dmaSection.getSize() % VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX) == 0,
                          "Bad DMA section size");

        const auto outputType = mlir::cast<vpux::NDTypeInterface>(getMemRefType(
                {dmaSection.getSize() / 4}, getUInt32Type(ctx), DimsOrder::C, VPURT::BufferSection::ProfilingOutput));
        const auto profilingOutputType =
                mlir::MemRefType::get(outputType.getShape().raw(), outputType.getElementType());

        auto dmaHwpBase = builderFunc.create<VPURT::DeclareBufferOp>(
                mlir::NameLoc::get(mlir::StringAttr::get(ctx, "dmaHwpBase")), profilingOutputType,
                VPURT::BufferSection::ProfilingOutput, 0, dmaSection.getOffset());

        return dmaHwpBase.getResult();
    }

    mlir::Value createDmaHwpBaseDynamic(mlir::OpBuilder builderFunc, mlir::ModuleOp moduleOp) {
        _log.trace("createDmaHwpBase");
        const auto ctx = builderFunc.getContext();
        auto dmaProfMem = config::getDmaProfilingReservedMemory(moduleOp, VPU::MemoryKind::DDR);
        VPUX_THROW_WHEN(dmaProfMem == nullptr, "Missing DMA HWP reserved buffer");
        auto dmaProfMemOffset = dmaProfMem.getOffset();
        VPUX_THROW_WHEN(dmaProfMemOffset == std::nullopt, "DMA HWP has no allocated address");
        VPUX_THROW_UNLESS((dmaProfMemOffset.value() % VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX) == 0,
                          "Unaligned HWP reserved base address");

        const auto memKind = IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR));
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(
                getMemRefType({dmaProfMem.getByteSize() / 4}, getUInt32Type(ctx), DimsOrder::C, memKind));

        auto dmaHwpBase = builderFunc.create<VPURT::DeclareBufferOp>(
                mlir::NameLoc::get(mlir::StringAttr::get(ctx, "dmaHwpBase")), outputType, VPURT::BufferSection::DDR,
                dmaProfMemOffset.value());

        return dmaHwpBase.getResult();
    }

    // Note on 40xx DMA Scratch buffer is mandatory for non-profiling blobs, see E#101929
    mlir::Value createDmaHwpScratch(mlir::OpBuilder builderFunc, mlir::ModuleOp moduleOp) {
        _log.trace("createDmaHwpScratch");
        auto dmaProfMem = config::getDmaProfilingReservedMemory(moduleOp, VPU::MemoryKind::CMX_NN);
        VPUX_THROW_WHEN(dmaProfMem == nullptr, "Missing DMA HWP scratch buffer");
        auto dmaProfMemOffset = dmaProfMem.getOffset();
        VPUX_THROW_WHEN(dmaProfMemOffset == std::nullopt, "No address allocated.");

        const auto ctx = builderFunc.getContext();
        const auto memKind = IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(getMemRefType(
                {VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX / 4}, getUInt32Type(ctx), DimsOrder::C, memKind));

        auto dmaHwpScratch = builderFunc.create<VPURT::DeclareBufferOp>(
                mlir::NameLoc::get(mlir::StringAttr::get(ctx, "dmaHwpScratch")), outputType,
                VPURT::BufferSection::CMX_NN, 0, dmaProfMemOffset.value());
        return dmaHwpScratch.getResult();
    }

    void addDmaHwpBase(DMAProfilingMode dmaProfilingMode, mlir::OpBuilder builderFunc, mlir::ModuleOp moduleOp,
                       VPUMI40XX::MappedInferenceOp mpi) {
        _log.trace("addDmaHwpBase");

        mlir::Value dmaHwpBase = nullptr;
        switch (dmaProfilingMode) {
        case DMAProfilingMode::DYNAMIC_HWP: {
            dmaHwpBase = createDmaHwpBaseDynamic(builderFunc, moduleOp);
            break;
        }
        case DMAProfilingMode::STATIC_HWP: {
            auto dmaSection = vpux::getProfilingSection(moduleOp, profiling::ExecutorType::DMA_HW);
            VPUX_THROW_UNLESS(dmaSection.has_value(), "Can't find DMA_HW profiling output section");
            dmaHwpBase = createDmaHwpBaseStatic(builderFunc, dmaSection.value());
            break;
        }
        case DMAProfilingMode::SCRATCH: {
            dmaHwpBase = createDmaHwpScratch(builderFunc, moduleOp);
            break;
        }
        case DMAProfilingMode::SW:
        case DMAProfilingMode::DISABLED:
            break;
        }

        if (dmaHwpBase != nullptr) {
            auto dmaHwpBaseOperand = mpi.getDmaHwpBaseMutable();
            dmaHwpBaseOperand.assign(dmaHwpBase);
        }
    }

    void addWorkpointCapture(mlir::OpBuilder builderFunc, mlir::ModuleOp moduleOp, VPUMI40XX::MappedInferenceOp mpi) {
        _log.trace("addWorkpointCapture");

        auto maybeCaptureSection = vpux::getProfilingSection(moduleOp, profiling::ExecutorType::WORKPOINT);
        if (!maybeCaptureSection) {
            _log.trace("No workpoint section");
            return;
        }

        const auto ctx = builderFunc.getContext();
        auto captureSection = maybeCaptureSection.value();
        unsigned pllSizeBytes = captureSection.getSize();
        VPUX_THROW_UNLESS(pllSizeBytes == profiling::WORKPOINT_BUFFER_SIZE, "Bad PLL section size: {0}", pllSizeBytes);
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(getMemRefType(
                {pllSizeBytes / 4}, getUInt32Type(ctx), DimsOrder::C, VPURT::BufferSection::ProfilingOutput));
        const auto profilingOutputType =
                mlir::MemRefType::get(outputType.getShape().raw(), outputType.getElementType());

        auto workpointBase = builderFunc.create<VPURT::DeclareBufferOp>(
                mlir::NameLoc::get(mlir::StringAttr::get(ctx, "workpointBase")), profilingOutputType,
                VPURT::BufferSection::ProfilingOutput, 0, captureSection.getOffset());
        auto hwpWorkpointCfg = workpointBase.getResult();

        auto hwpWorkpointOperand = mpi.getHwpWorkpointCfgMutable();
        hwpWorkpointOperand.assign(hwpWorkpointCfg);
    }
};

void SetupProfilingVPUMI40XXPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto arch = config::getArch(moduleOp);

    VPUX_THROW_UNLESS(enableDMAProfiling.hasValue(), "No option");
    auto dmaProfilingMode = getDMAProfilingMode(arch, enableDMAProfiling);

    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp funcOp;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, funcOp);
    mlir::OpBuilder builderFunc(&(funcOp.getFunctionBody()));

    auto mpi = VPUMI40XX::getMPI(funcOp);

    // create DMA hardware profiling base ref in MI
    addDmaHwpBase(dmaProfilingMode, builderFunc, moduleOp, mpi);

    // create workpoint cfg ref in MI for hardware profiling
    addWorkpointCapture(builderFunc, moduleOp, mpi);
}

}  // namespace

//
// createSetupProfilingVPUMI40XXPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createSetupProfilingVPUMI40XXPass(const std::string& enableDmaProfiling,
                                                                               Logger log) {
    return std::make_unique<SetupProfilingVPUMI40XXPass>(enableDmaProfiling, log);
}
