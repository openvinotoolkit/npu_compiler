//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include "vpux/compiler/core/profiling.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_RESERVEPROFILINGMEM
#define GEN_PASS_DEF_RESERVEPROFILINGMEM
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
//  ReserveProfilingMemPass
//

class ReserveProfilingMemPass final : public VPU::impl::ReserveProfilingMemBase<ReserveProfilingMemPass> {
public:
    explicit ReserveProfilingMemPass(const std::string& enableDMAProfiling, Logger log)
            : ReserveProfilingMemBase({enableDMAProfiling}) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    void reserveDMAProfilingMem();
    void reserveDPUProfilingMem();
    void reserveActShaveProfilingMem();
};

void ReserveProfilingMemPass::safeRunOnModule() {
    reserveDMAProfilingMem();
    reserveDPUProfilingMem();
    reserveActShaveProfilingMem();
}

void ReserveProfilingMemPass::reserveDMAProfilingMem() {
    auto module = getOperation();
    auto* ctx = module->getContext();
    auto arch = config::getArch(module);

    VPUX_THROW_UNLESS(enableDMAProfiling.hasValue(), "No option");
    auto dmaProfilingMode = getDMAProfilingMode(arch, enableDMAProfiling);

    auto dmaOp = config::getAvailableExecutor(module, config::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();
    VPUX_THROW_UNLESS(dmaPortCount > 0, "No DMA ports");
    VPUX_THROW_UNLESS((VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE % dmaPortCount) == 0,
                      "Reserved memory for DMA profiling cannot be equally split between ports");

    // Small chunk of CMX memory is always reserved in NPU37xx and NPU40xx
    if (arch == config::ArchKind::NPU37XX || arch == config::ArchKind::NPU40XX) {
        auto memSpaceAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
        _log.trace("DMA profiling reserved CMX memory - size: '{0}'", VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE);
        config::setDmaProfilingReservedMemory(module, memSpaceAttr, VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE);
    }

    // Chunk of DDR is reserved if profiling is enabled
    if (dmaProfilingMode == DMAProfilingMode::DYNAMIC_HWP) {
        _log.trace("DMA HW profiling reserved DDR memory - size: '{0}'",
                   VPUIP::HW_DMA_PROFILING_ID_LIMIT * VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX);
        auto memSpaceAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR));
        config::setDmaProfilingReservedMemory(
                module, memSpaceAttr, VPUIP::HW_DMA_PROFILING_ID_LIMIT * VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX);
    }
}

void ReserveProfilingMemPass::reserveActShaveProfilingMem() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    unsigned actShaveProfilingBufferSize = VPUIP::HW_ACT_SHAVE_PROFILING_BUFFER_SIZE;
    auto cmxMemSpaceAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    _log.trace("ActShave profiling reserved CMX memory - size: '{0}'", actShaveProfilingBufferSize);
    config::setActShaveProfilingReservedMemory(module, cmxMemSpaceAttr, actShaveProfilingBufferSize);
}

void ReserveProfilingMemPass::reserveDPUProfilingMem() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    unsigned dpuProfilingBufferSize = VPUIP::HW_DPU_PROFILING_BUFFER_SIZE;
    auto cmxMemSpaceAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    _log.trace("DPU profiling reserved CMX memory - size: '{0}'", dpuProfilingBufferSize);
    config::setDpuProfilingReservedMemory(module, cmxMemSpaceAttr, dpuProfilingBufferSize);
}

}  // namespace

//
// createReserveProfilingMemPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createReserveProfilingMemPass(const std::string& enableDMAProfiling,
                                                                     Logger log) {
    return std::make_unique<ReserveProfilingMemPass>(enableDMAProfiling, log);
}
