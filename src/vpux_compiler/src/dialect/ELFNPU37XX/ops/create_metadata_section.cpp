//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_elf/types/vpu_extensions.hpp>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/attributes.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"

void vpux::ELFNPU37XX::CreateMetadataSectionOp::serialize(elf::Writer& writer,
                                                          vpux::ELFNPU37XX::SectionMapType& sectionMap,
                                                          vpux::ELFNPU37XX::SymbolMapType& symbolMap) {
    VPUX_UNUSED(writer);
    VPUX_UNUSED(sectionMap);
    VPUX_UNUSED(symbolMap);
    VPUX_THROW("Can't Serialize a Metadata Section with no NetworkMetadata information.");
}

void vpux::ELFNPU37XX::CreateMetadataSectionOp::serialize(elf::Writer& writer,
                                                          vpux::ELFNPU37XX::SectionMapType& sectionMap,
                                                          vpux::ELFNPU37XX::SymbolMapType& symbolMap,
                                                          elf::NetworkMetadata& metadata) {
    VPUX_UNUSED(writer);
    VPUX_UNUSED(symbolMap);

    const auto section = sectionMap.find(getOperation());
    VPUX_THROW_WHEN(section == sectionMap.end(), "ELF section not found: {0}", getSecName().str());
    auto binDataSection = dynamic_cast<elf::writer::BinaryDataSection<uint8_t>*>(section->second);
    VPUX_THROW_WHEN(binDataSection == nullptr, "Invalid binary section in ELF writer");

    bool isMetadataSerialized = false;
    auto block = getBody();
    for (auto& op : block->getOperations()) {
        VPUX_THROW_WHEN(isMetadataSerialized, "There should be only 1 metadata op in an ELF metadata section");
        if (auto metadata_op = mlir::dyn_cast<vpux::VPUMI37XX::NetworkMetadataOp>(op)) {
            isMetadataSerialized = true;
            metadata_op.serialize(*binDataSection, metadata);
        }
    }
    VPUX_THROW_UNLESS(isMetadataSerialized, "No metadata defined in the ELF metadata section");
}

void vpux::ELFNPU37XX::CreateMetadataSectionOp::preserialize(elf::Writer& writer,
                                                             vpux::ELFNPU37XX::SectionMapType& sectionMap) {
    const auto name = getSecName().str();
    auto section = writer.addBinaryDataSection<uint8_t>(
            name, static_cast<elf::Elf_Word>(vpux::ELFNPU37XX::SectionTypeAttr::VPU_SHT_NETDESC));
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));
    section->setAddrAlign(getSecAddrAlign());

    bool isMetadataSerialized = false;
    auto block = getBody();
    size_t sectionSize = 0;
    for (auto& op : block->getOperations()) {
        VPUX_THROW_WHEN(isMetadataSerialized, "There should be only 1 metadata op in an ELF metadata section");
        if (auto metadata_op = mlir::dyn_cast<vpux::VPUMI37XX::NetworkMetadataOp>(op)) {
            isMetadataSerialized = true;
            sectionSize = metadata_op.getBinarySize();
        }
    }
    VPUX_THROW_UNLESS(isMetadataSerialized, "No metadata defined in the ELF metadata section");
    section->setSize(sectionSize);

    sectionMap[getOperation()] = section;
}
