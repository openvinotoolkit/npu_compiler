//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include "vpux/compiler/core/profiling.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_DMATASKPROFILINGRESERVEMEM
#define GEN_PASS_DEF_DMATASKPROFILINGRESERVEMEM
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
//  DMATaskProfilingReserveMemPass
//

class DMATaskProfilingReserveMemPass final :
        public VPUIP::impl::DMATaskProfilingReserveMemBase<DMATaskProfilingReserveMemPass> {
public:
    explicit DMATaskProfilingReserveMemPass(DMAProfilingMode profilingMode, Logger log): _profilingMode(profilingMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    DMAProfilingMode _profilingMode;
    void safeRunOnModule() final;
};

void DMATaskProfilingReserveMemPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();
    auto arch = VPU::getArch(module);

    if (enableDMAProfiling.hasValue()) {
        _profilingMode = getDMAProfilingMode(arch, enableDMAProfiling.getValue());
    }

    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();
    VPUX_THROW_UNLESS((VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE % dmaPortCount) == 0,
                      "Reserved memory for DMA profiling cannot be equally split between ports");

    if (_profilingMode == DMAProfilingMode::DISABLED) {
        return;
    }

    // Small chunk of CMX memory is always reserved
    auto memSpaceAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    _log.trace("DMA profiling reserved CMX memory - size: '{0}'", VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE);
    IE::setDmaProfilingReservedMemory(module, memSpaceAttr, VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE);

    // Chunk of DDR is reserved if profiling is enabled
    if (_profilingMode == DMAProfilingMode::DYNAMIC_HWP) {
        _log.trace("DMA HW profiling reserved DDR memory - size: '{0}'",
                   VPUIP::HW_DMA_PROFILING_ID_LIMIT * VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX);
        auto memSpaceAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR));
        IE::setDmaProfilingReservedMemory(module, memSpaceAttr,
                                          VPUIP::HW_DMA_PROFILING_ID_LIMIT * VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX);
    }
}

}  // namespace

//
// createDMATaskProfilingReserveMemPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDMATaskProfilingReserveMemPass(DMAProfilingMode dmaProfilingMode,
                                                                              Logger log) {
    return std::make_unique<DMATaskProfilingReserveMemPass>(dmaProfilingMode, log);
}
