//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/performance_metrics_utils.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"

// Including the header file for the NPU 40xx NNRT is fine, VpuHostParsedInference definition is the same.
#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;
using MappedInferenceFormat = config::NPUConstraints::MappedInferenceFormat;

//
// HostParsedInferenceOp
//

vpux::ELF::SectionFlagsAttr vpux::VPUASM::HostParsedInferenceOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::HostParsedInferenceOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "host_parsed_inference"),
                                 ELF::SectionFlagsAttr::SHF_ALLOC, ELF::SectionTypeAttr::VPU_SHT_HPI);
}

bool vpux::VPUASM::HostParsedInferenceOp::hasMemoryFootprint() {
    return true;
}

void vpux::VPUASM::HostParsedInferenceOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuHostParsedInference hpi = {};

    hpi.resource_requirements_.nn_slice_length_ = checked_cast<uint32_t>(getNnSliceLength());
    hpi.resource_requirements_.nn_slice_count_ = checked_cast<uint8_t>(getNnSliceCount());
    hpi.resource_requirements_.nn_barriers_ = checked_cast<uint8_t>(getNnBarriers());
    auto useDirectMmi = (config::getNPUConstraints(getOperation()->getContext()).mappedInferenceFormat ==
                         MappedInferenceFormat::ManagedMappedInference);
    hpi.mmi_access_ = useDirectMmi ? nn_public::VpuHostParsedInference::VpuMmiAccessMode::DIRECT
                                   : nn_public::VpuHostParsedInference::VpuMmiAccessMode::INDIRECT;

    // mapped_ points to a single (managed) mapped inference; the address is resolved via relocation.
    hpi.mapped_.count = 1;
    VPUASM::populatePerformanceMetrics(hpi.performance_metrics_, getOperation()->getParentOfType<mlir::ModuleOp>());

    binDataSection.appendData(reinterpret_cast<uint8_t*>(&hpi), getBinarySize(config::ArchKind::UNKNOWN));
}

size_t vpux::VPUASM::HostParsedInferenceOp::getBinarySize(config::ArchKind) {
    return sizeof(nn_public::VpuHostParsedInference);
}

size_t vpux::VPUASM::HostParsedInferenceOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(nn_public::VpuHostParsedInference);
}

std::vector<ELF::RelocationInfo> VPUASM::HostParsedInferenceOp::getRelocationInfo(ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    auto mapped = getMapped();
    const auto mappedOffset =
            offsetof(nn_public::VpuHostParsedInference, mapped_) + offsetof(nn_public::VpuTaskReference<void>, address);
    relocs.emplace_back(mapped, targetSection, mappedOffset, ELF::RelocationType::R_VPU_64,
                        ELF::getOffsetOfSymRef(symRefMap, mapped), "mapped inference for host parsed inference reloc");

    return relocs;
}
