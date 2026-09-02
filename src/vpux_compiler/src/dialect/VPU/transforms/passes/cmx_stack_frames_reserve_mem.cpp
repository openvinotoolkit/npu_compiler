//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_CMXSTACKFRAMESRESERVEMEM
#define GEN_PASS_DEF_CMXSTACKFRAMESRESERVEMEM
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
//  CMXStackFramesReserveMemPass
//

class CMXStackFramesReserveMemPass final :
        public VPU::impl::CMXStackFramesReserveMemBase<CMXStackFramesReserveMemPass> {
public:
    explicit CMXStackFramesReserveMemPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void CMXStackFramesReserveMemPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    if (config::getArch(moduleOp) < config::ArchKind::NPU40XX) {
        return;
    }

    auto scratchSize = static_cast<uint32_t>(HW_RESERVED_CMX.count());
    auto stackSize = static_cast<uint32_t>(CMX_SHAVE_STACK_SIZE.count());
    const size_t defaultStacksNum = 2;
    auto totalStacksSize = defaultStacksNum * stackSize + scratchSize;

    _log.trace("Shave stack frames reserved CMX memory - size: '{0}'", totalStacksSize);
    config::setCMXStackFramesReservedMemory(moduleOp, totalStacksSize, ELF::VPUX_SHAVE_ALIGNMENT);
}

}  // namespace

//
// createCMXStackFramesReserveMemPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createCMXStackFramesReserveMemPass(Logger log) {
    return std::make_unique<CMXStackFramesReserveMemPass>(log);
}
