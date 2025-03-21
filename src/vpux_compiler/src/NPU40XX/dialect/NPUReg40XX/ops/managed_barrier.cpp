//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

//
// ManagedBarrierOp
//

void NPUReg40XX::ManagedBarrierOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto barrierDescriptor = getDescriptor().getRegMapped();

    VPUX_THROW_UNLESS(sizeof(nn_public::VpuTaskBarrierMap) == barrierDescriptor.size(),
                      "HW VpuTaskBarrierMap size {0} != regMapped representation size {1}.",
                      sizeof(nn_public::VpuTaskBarrierMap), barrierDescriptor.size());

    auto serializedBarrierDescriptor = barrierDescriptor.getStorage();
    binDataSection.appendData(serializedBarrierDescriptor.data(), getBinarySize(VPU::ArchKind::NPU40XX));
}

size_t NPUReg40XX::ManagedBarrierOp::getBinarySize(VPU::ArchKind) {
    return sizeof(nn_public::VpuTaskBarrierMap);
}

size_t vpux::NPUReg40XX::ManagedBarrierOp::getAlignmentRequirements(VPU::ArchKind) {
    return alignof(nn_public::VpuTaskBarrierMap);
}
