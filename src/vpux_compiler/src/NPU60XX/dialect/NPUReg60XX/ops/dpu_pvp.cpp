//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"

#include <npu_60xx_nnrt.hpp>
#include <vpux/compiler/NPU60XX/dialect/NPUReg60XX/npu_reg_types.hpp.inc>

using namespace vpux;
using namespace npu60xx;

//
// DpuPVPOp
//

std::optional<ELF::SectionSignature> vpux::NPUReg60XX::DpuPVPOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("dpu", "pvp"), ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::NPUReg60XX::DpuPVPOp::hasMemoryFootprint() {
    return true;
}

void NPUReg60XX::DpuPVPOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto dpuPVPDescriptor = getProperties().getDescriptor();

    VPUX_THROW_UNLESS(sizeof(nn_public::VpuDpuPVP) == dpuPVPDescriptor.size(),
                      "HW VpuDpuPVP size {0} != regMapped representation size {1}.", sizeof(nn_public::VpuDpuPVP),
                      dpuPVPDescriptor.size());

    auto serializedDpuPVP = dpuPVPDescriptor.getStorage();
    binDataSection.appendData(serializedDpuPVP.data(), getBinarySize(config::ArchKind::NPU60XX));
}

size_t NPUReg60XX::DpuPVPOp::getBinarySize(config::ArchKind) {
    return sizeof(nn_public::VpuDpuPVP);
}

size_t vpux::NPUReg60XX::DpuPVPOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(nn_public::VpuDpuPVP);
}

void vpux::NPUReg60XX::DpuPVPOp::build(mlir::OpBuilder&, mlir::OperationState& state, mlir::StringAttr symName,
                                       vpux::NPUReg60XX::Descriptors::VpuDpuPVP&& descriptor) {
    auto& props = state.getOrAddProperties<Properties>();

    props.sym_name = symName;
    props.descriptor = std::move(descriptor);
}
