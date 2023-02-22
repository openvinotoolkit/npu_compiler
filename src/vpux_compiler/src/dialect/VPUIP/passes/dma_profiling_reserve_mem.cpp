//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"

using namespace vpux;

namespace {

//
//  DMATaskProfilingReserveMemPass
//

class DMATaskProfilingReserveMemPass final :
        public VPUIP::DMATaskProfilingReserveMemBase<DMATaskProfilingReserveMemPass> {
public:
    explicit DMATaskProfilingReserveMemPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void DMATaskProfilingReserveMemPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.count();

    VPUX_THROW_UNLESS((VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE % dmaPortCount) == 0,
                      "Reserved memory for DMA profiling cannot be equally split between ports");

    auto memSpaceAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    _log.trace("DMA profiling reserved memory - offset: '{0}', size: '{1}'",
               VPUIP::HW_DMA_PROFILING_RESERVED_MEM_OFFSET, VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE);

    IE::setDmaProfilingReservedMemory(module, memSpaceAttr, VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE,
                                      VPUIP::HW_DMA_PROFILING_RESERVED_MEM_OFFSET);
}

}  // namespace

//
// createDMATaskProfilingReserveMemPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDMATaskProfilingReserveMemPass(Logger log) {
    return std::make_unique<DMATaskProfilingReserveMemPass>(log);
}
