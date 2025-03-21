//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_SWKERNELPREFETCHINGRESERVEMEM
#define GEN_PASS_DEF_SWKERNELPREFETCHINGRESERVEMEM
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
//  SWKernelPrefetchingReserveMemPass
//

class SWKernelPrefetchingReserveMemPass final :
        public VPUIP::impl::SWKernelPrefetchingReserveMemBase<SWKernelPrefetchingReserveMemPass> {
public:
    explicit SWKernelPrefetchingReserveMemPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

bool checkSWKernelOp(mlir::ModuleOp& func) {
    bool hasSWKernelOp = false;
    func->walk([&](VPUIP::SwKernelOp) {
        hasSWKernelOp = true;
        return;
    });

    return hasSWKernelOp;
}

void SWKernelPrefetchingReserveMemPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    auto hasSWKernelOp = checkSWKernelOp(module);
    if (!hasSWKernelOp) {
        return;
    }

    auto maxPrefetchDataSize = VPUIP::getMaximalSWKernelPrefetchDataSize(module);

    auto memSpaceAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    auto available = IE::getAvailableMemory(module, memSpaceAttr);
    const auto maxSize = available.size();
    auto reservedMemoryResources = IE::getReservedMemoryResources(module, memSpaceAttr);
    if (reservedMemoryResources.empty()) {
        // Insert a dummy reserved memory when there's no reserved memory
        _log.trace("Reserve dummy memory for SW Kernel prefetching - size: '{0}'", maxPrefetchDataSize);
        IE::setSWKernelPrefetchingReservedMemory(module, memSpaceAttr, maxPrefetchDataSize);
    } else {
        // Calculate reserved memory total size
        int64_t reservedMemTotalSize = 0;
        for (auto& resMem : reservedMemoryResources) {
            reservedMemTotalSize += resMem.getByteSize();
        }

        // Enlarge the original reserved memory range when total reserved memory is not safe for SW Kernel data
        // prefetching
        if (reservedMemTotalSize < maxPrefetchDataSize) {
            _log.trace("Enlarge the original reserved memory range for SW Kernel prefetching - size: '{0}'",
                       maxPrefetchDataSize - reservedMemTotalSize);

            auto lastResMem = reservedMemoryResources.back();
            auto lastResMemSize = lastResMem.getByteSize();
            auto newResMemSize = lastResMemSize + maxPrefetchDataSize - reservedMemTotalSize;
            lastResMem.setByteSizeAttr(getIntAttr(module->getContext(), newResMemSize));
        }
    }

    // Put all reserved memory at the end of CMX
    auto newReservedMemoryResources = IE::getReservedMemoryResources(module, memSpaceAttr);
    size_t resMemOffset = maxSize.count();
    for (auto& resMem : newReservedMemoryResources) {
        auto currResMemSize = resMem.getByteSize();
        resMemOffset -= currResMemSize;
        auto currResMemOffset = resMemOffset;
        resMem.setOffsetAttr(getIntAttr(module->getContext(), currResMemOffset));
    }
}

}  // namespace

//
// createSWKernelPrefetchingReserveMemPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSWKernelPrefetchingReserveMemPass(Logger log) {
    return std::make_unique<SWKernelPrefetchingReserveMemPass>(log);
}
