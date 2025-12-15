//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

#include <npu_37xx_nnrt.hpp>

using namespace vpux;
using namespace npu37xx;

//
// ConfigureBarrierOp
//

void vpux::VPUMI37XX::ConfigureBarrierOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuBarrierCountConfig barrier{};

    barrier.next_same_id_ = getNextSameId();
    barrier.consumer_count_ = getConsumerCount().value_or(0);
    barrier.producer_count_ = getProducerCount().value_or(0);
    barrier.real_id_ = getId();

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&barrier);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUMI37XX::ConfigureBarrierOp::getBinarySize() {
    return sizeof(nn_public::VpuBarrierCountConfig);
}

size_t vpux::VPUMI37XX::ConfigureBarrierOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuBarrierCountConfig);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::ConfigureBarrierOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::ConfigureBarrierOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_EXECINSTR);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::ConfigureBarrierOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}
