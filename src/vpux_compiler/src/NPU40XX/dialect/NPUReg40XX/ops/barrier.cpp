//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

//
// ConfigureBarrierOp
//

void vpux::NPUReg40XX::ConfigureBarrierOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto barrierDescriptor = getDescriptor().getRegMapped();

    VPUX_THROW_UNLESS(sizeof(nn_public::VpuBarrierCountConfig) == barrierDescriptor.size(),
                      "HW VpuBarrierCountConfig size {0} != regMapped representation size {1}.",
                      sizeof(nn_public::VpuBarrierCountConfig), barrierDescriptor.size());

    auto serializedBarrierDescriptor = barrierDescriptor.getStorage();
    binDataSection.appendData(serializedBarrierDescriptor.data(), getBinarySize(VPU::ArchKind::NPU40XX));
}

size_t vpux::NPUReg40XX::ConfigureBarrierOp::getBinarySize(VPU::ArchKind) {
    return sizeof(nn_public::VpuBarrierCountConfig);
}

size_t vpux::NPUReg40XX::ConfigureBarrierOp::getAlignmentRequirements(VPU::ArchKind) {
    return alignof(nn_public::VpuBarrierCountConfig);
}
