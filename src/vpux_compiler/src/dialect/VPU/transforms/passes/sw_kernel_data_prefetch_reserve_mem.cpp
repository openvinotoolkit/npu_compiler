//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_SWKERNELDATAPREFETCHRESERVEMEM
#define GEN_PASS_DEF_SWKERNELDATAPREFETCHRESERVEMEM
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
//  SWKernelDataPrefetchReserveMemPass
//

class SWKernelDataPrefetchReserveMemPass final :
        public VPU::impl::SWKernelDataPrefetchReserveMemBase<SWKernelDataPrefetchReserveMemPass> {
public:
    explicit SWKernelDataPrefetchReserveMemPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

bool checkSWKernelOp(mlir::ModuleOp& func) {
    bool hasSWKernelOp = false;
    func->walk([&](VPU::SWOpInterface) {
        hasSWKernelOp = true;
        return;
    });

    return hasSWKernelOp;
}

void SWKernelDataPrefetchReserveMemPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    auto hasSWKernelOp = checkSWKernelOp(module);
    if (!hasSWKernelOp) {
        return;
    }

    auto maxPrefetchDataSize = VPUIP::getMaximalSWKernelPrefetchDataSize(module);
    auto memSpaceAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    int64_t reservedMemTotalSize = 0;
    for (auto& resMem : config::getReservedMemoryResources(module, memSpaceAttr)) {
        reservedMemTotalSize += resMem.getByteSize();
    }

    // Enlarge the original reserved memory range when total reserved memory is not safe for SW Kernel data
    // prefetching
    if (reservedMemTotalSize < maxPrefetchDataSize) {
        _log.trace("Enlarge the original reserved memory range for SW Kernel prefetching - size: '{0}'",
                   maxPrefetchDataSize - reservedMemTotalSize);
        config::setSWKernelPrefetchingReservedMemory(module, memSpaceAttr, maxPrefetchDataSize - reservedMemTotalSize);
    }
}

}  // namespace

//
// createSWKernelDataPrefetchReserveMemPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSWKernelDataPrefetchReserveMemPass(Logger log) {
    return std::make_unique<SWKernelDataPrefetchReserveMemPass>(log);
}
