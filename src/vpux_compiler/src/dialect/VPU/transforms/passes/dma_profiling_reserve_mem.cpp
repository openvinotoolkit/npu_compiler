//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include "vpux/compiler/core/profiling.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_DMATASKPROFILINGRESERVEMEM
#define GEN_PASS_DEF_DMATASKPROFILINGRESERVEMEM
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
//  DMATaskProfilingReserveMemPass
//

class DMATaskProfilingReserveMemPass final :
        public VPU::impl::DMATaskProfilingReserveMemBase<DMATaskProfilingReserveMemPass> {
public:
    explicit DMATaskProfilingReserveMemPass(const std::string& enableDMAProfiling, Logger log)
            : DMATaskProfilingReserveMemBase({enableDMAProfiling}) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void DMATaskProfilingReserveMemPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();
    auto arch = config::getArch(module);

    VPUX_THROW_UNLESS(enableDMAProfiling.hasValue(), "No option");
    auto dmaProfilingMode = getDMAProfilingMode(arch, enableDMAProfiling);

    auto dmaOp = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
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

}  // namespace

//
// createDMATaskProfilingReserveMemPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createDMATaskProfilingReserveMemPass(const std::string& enableDMAProfiling,
                                                                            Logger log) {
    return std::make_unique<DMATaskProfilingReserveMemPass>(enableDMAProfiling, log);
}
