//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;

//
// MappedInferenceOp
//

vpux::ELF::SectionFlagsAttr vpux::VPUASM::MappedInferenceOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::MappedInferenceOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "mapped_inference"),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::MappedInferenceOp::hasMemoryFootprint() {
    return true;
}

//
// MappedInferenceOp_37XX
//

size_t vpux::VPUASM::MappedInferenceOp_37XX::getBinarySize(config::ArchKind) {
    return sizeof(npu40xx::nn_public::VpuMappedInference);
}

size_t vpux::VPUASM::MappedInferenceOp_37XX::getAlignmentRequirements(config::ArchKind) {
    return alignof(npu40xx::nn_public::VpuMappedInference);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::MappedInferenceOp_37XX::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::MappedInferenceOp_37XX::getSectionSignature() {
    return ELF::SectionSignature("text.mappedInference", ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::MappedInferenceOp_37XX::hasMemoryFootprint() {
    return true;
}
