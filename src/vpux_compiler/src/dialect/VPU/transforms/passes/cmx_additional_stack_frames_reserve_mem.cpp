//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_CMXADDITIONALSTACKFRAMESRESERVEMEM
#define GEN_PASS_DEF_CMXADDITIONALSTACKFRAMESRESERVEMEM
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
//  CMXAdditionalStackFramesReserveMemPass
//

class CMXAdditionalStackFramesReserveMemPass final :
        public VPU::impl::CMXAdditionalStackFramesReserveMemBase<CMXAdditionalStackFramesReserveMemPass> {
public:
    explicit CMXAdditionalStackFramesReserveMemPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

}  // namespace

void CMXAdditionalStackFramesReserveMemPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    if (config::getArch(moduleOp) < config::ArchKind::NPU40XX) {
        return;
    }

    const size_t defaultStacksNum = 2;
    auto stackSize = static_cast<uint32_t>(CMX_SHAVE_STACK_SIZE.count());
    auto shvPerTile = checked_cast<uint32_t>(
            config::getTileExecutor(moduleOp).getSubExecutor(config::ExecutorKind::SHAVE_ACT).getCount());

    // Two stack frames already reserved at the beginning of the CMX space, reserve resources only for additional stack
    // frames if needed
    if (shvPerTile > defaultStacksNum) {
        auto stacksToReserve = shvPerTile - defaultStacksNum;
        _log.trace("Additional shave stack frames reserved CMX memory - size: '{0}'", stacksToReserve * stackSize);
        config::setCMXAdditionalStackFramesReservedMemory(moduleOp, stacksToReserve * stackSize,
                                                          ELF::VPUX_SHAVE_ALIGNMENT);
    }
}

//
// createCMXAdditionalStackFramesReserveMemPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createCMXAdditionalStackFramesReserveMemPass(Logger log) {
    return std::make_unique<CMXAdditionalStackFramesReserveMemPass>(log);
}
