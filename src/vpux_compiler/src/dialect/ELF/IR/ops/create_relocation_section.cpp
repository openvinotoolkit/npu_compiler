//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_elf/types/vpu_extensions.hpp>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"

#include <mlir/IR/SymbolTable.h>

using namespace vpux;

void ELF::CreateRelocationSectionOp::serialize(elf::Writer& writer, ELF::SectionMapType& sectionMap,
                                               ELF::SymbolMapType& symbolMap, ELF::DmaSymbolMapType& dmaSymbolMap,
                                               ELF::SymbolReferenceMap& symRefMap) {
    VPUX_UNUSED(symRefMap);
    const auto name = getSymName().str();
    auto section = writer.addRelocationSection(name);

    // Look up dependent sections
    auto targetRef = ELF::lookupNearestSymbolFrom(getOperation(), getTargetSectionAttr());
    auto target = mlir::dyn_cast_or_null<ELF::ElfSectionInterface>(targetRef);

    auto symTabRef = ELF::lookupNearestSymbolFrom(getOperation(), getSourceSymbolTableSectionAttr());

    VPUX_THROW_UNLESS(target, "Reloc section expected to refer at a valid target section");

    auto targetMapEntry = sectionMap.find(target.getOperation());
    VPUX_THROW_WHEN(targetMapEntry == sectionMap.end(),
                    "Can't serialize a reloc section that doesn't have its dependent target section");

    auto targetSection = targetMapEntry->second;
    section->setSectionToPatch(targetSection);
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));

    auto symTabMapEntry = sectionMap.find(symTabRef);
    VPUX_THROW_WHEN(symTabMapEntry == sectionMap.end(),
                    "Can't serialize a reloc section that doesn't have its dependent symbol table section");

    auto symTabSection = symTabMapEntry->second;
    section->setSpecialSymbolTable(symTabSection->getIndex());

    auto block = getBody();
    for (auto& op : block->getOperations()) {
        auto relocation = section->addRelocationEntry();

        auto relocOp = llvm::dyn_cast<ELF::ElfRelocationObjectInterface>(op);

        VPUX_THROW_UNLESS(relocOp,
                          "CreateRelocationSection op is expected to have only RelocOps or RelocImmOfsetOps. Got {0}",
                          op);

        relocOp.serialize(relocation, symbolMap, dmaSymbolMap);
    }

    sectionMap[getOperation()] = section;
}

void vpux::ELF::CreateRelocationSectionOp::preserialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                                        vpux::ELF::SymbolReferenceMap&) {
    VPUX_UNUSED(writer);
    VPUX_UNUSED(sectionMap);

    // don't implement and go into elf::Writer internal state first
    // as there're relocation entries to be updated first
    // E#136375
}
