//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/undefined_vf.hpp"

using namespace vpux;
using namespace vpux::VPU;

UndefinedVF::UndefinedVF(): _log(Logger::global()) {
    _log.setName("UndefinedVF");
}

llvm::StringRef UndefinedVF::getName() const {
    return "UndefinedVF";
}

LoopScheduleResult UndefinedVF::getScheduleStrategy(const ComputeRegion& loopRegion,
                                                    vpux::AddressType memorySize) const {
    VPUX_THROW_UNLESS(loopRegion.schedulingLoop != nullptr, "ComputeRegion has no scheduling loop");
    VPUX_THROW_UNLESS(loopRegion.schedulingLoop->type == LoopType::VF, "UndefinedVF invoked on non-VF loop type: {0}",
                      toString(loopRegion.schedulingLoop->type));

    // Phase 1 (TODO: E#202070): no allocation logic yet. Returning an empty result
    // makes the dispatcher fall back to the legacy per-op allocation path.
    _log.trace("UndefinedVF: empty result for region (iterations={0}, memorySize={1}), falling back to legacy path",
               loopRegion.schedulingLoop->loopBodies.size(), memorySize);

    return {/*schedule=*/{}, /*reservedSize=*/0, /*sharedExternalBuffers=*/{},
            /*baseAlignment=*/vpux::DEFAULT_CMX_ALIGNMENT};
}
