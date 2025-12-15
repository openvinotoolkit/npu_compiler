//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

using namespace vpux;

//
// ConfigureBarrierOp
//

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ConfigureBarrierOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::ConfigureBarrierOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "barrier"), ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::ConfigureBarrierOp::hasMemoryFootprint() {
    return true;
}

mlir::LogicalResult vpux::VPUASM::ConfigureBarrierOp::verify() {
    const auto nextSameId = getNextSameId();
    const auto currentIndex = getTaskIndex().getValue();
    if (nextSameId != -1) {
        const auto uNextSameId = static_cast<uint32_t>(nextSameId);
        if (currentIndex > uNextSameId) {
            return errorAt(getLoc(),
                           "Operation {0}: barrier next_same_id {1} value is smaller than current index value {2}",
                           getOperationName(), uNextSameId, currentIndex);
        }
    }

    const auto id = getId();
    const auto noOfAvailableBarriers = VPUIP::getNumAvailableBarriers(getOperation());
    if (id >= noOfAvailableBarriers) {
        return errorAt(getLoc(), "Operation {0}: barrier id {0} value is higher than available barriers {1}",
                       getOperationName(), id, noOfAvailableBarriers);
    }

    return ::mlir::success();
}
