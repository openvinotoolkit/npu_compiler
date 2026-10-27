//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_elf/types/vpu_extensions.hpp>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/IR/attributes.hpp"
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

void vpux::ELF::CreateMetadataSectionOp::serialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                                   vpux::ELF::SymbolMapType& symbolMap,
                                                   elf::NetworkMetadata& metadata) {
    VPUX_UNUSED(symbolMap);
    VPUX_UNUSED(writer);

    const auto section = sectionMap.find(getOperation());
    VPUX_THROW_WHEN(section == sectionMap.end(), "ELF section not found: {0}", getSymName().str());
    auto binDataSection = dynamic_cast<elf::writer::BinaryDataSection<uint8_t>*>(section->second);
    VPUX_THROW_WHEN(binDataSection == nullptr, "Invalid binary section in ELF writer");

    bool isMetadataSerialized = false;
    auto block = getBody();
    for (auto& op : block->getOperations()) {
        VPUX_THROW_WHEN(isMetadataSerialized, "There should be only 1 metadata op in an ELF metadata section");

        if (auto metadata_op = mlir::dyn_cast<vpux::VPUASM::NetworkMetadataOp>(op)) {
            isMetadataSerialized = true;
            metadata_op.serialize(*binDataSection, metadata);
        }
    }
    VPUX_THROW_UNLESS(isMetadataSerialized, "No metadata defined in the ELF metadata section");
}

void vpux::ELF::CreateMetadataSectionOp::preserialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                                      vpux::ELF::SymbolReferenceMap&) {
    const auto name = getSymName().str();
    auto section = writer.addBinaryDataSection<uint8_t>(
            name, static_cast<elf::Elf_Word>(vpux::ELF::SectionTypeAttr::VPU_SHT_NETDESC));
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));
    section->setAddrAlign(getSecAddrAlign());

    bool isMetadataSerialized = false;
    auto block = getBody();
    size_t sectionSize = 0;
    for (auto& op : block->getOperations()) {
        VPUX_THROW_WHEN(isMetadataSerialized, "There should be only 1 metadata op in an ELF metadata section");
        if (auto metadata_op = mlir::dyn_cast<vpux::VPUASM::NetworkMetadataOp>(op)) {
            isMetadataSerialized = true;
            sectionSize = metadata_op.getBinarySize(config::ArchKind::UNKNOWN);
        }
    }
    VPUX_THROW_UNLESS(isMetadataSerialized, "No metadata defined in the ELF metadata section");
    section->setSize(sectionSize);

    sectionMap[getOperation()] = section;
}
