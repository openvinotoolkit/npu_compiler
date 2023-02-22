//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vpux_elf/types/vpu_extensions.hpp>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/ops.hpp"

void vpux::ELF::CreateRelocationSectionOp::serialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                                     vpux::ELF::SymbolMapType& symbolMap) {
    const auto name = secName().str();
    auto section = writer.addRelocationSection(name);

    // Look up dependent sections
    auto symTab = llvm::dyn_cast<vpux::ELF::CreateSymbolTableSectionOp>(sourceSymbolTableSection().getDefiningOp());
    VPUX_THROW_UNLESS(symTab != nullptr, "Reloc section expected to refer to a symbol table section");

    auto target = llvm::dyn_cast_or_null<vpux::ELF::CreateSectionOp>(targetSection().getDefiningOp());
    VPUX_THROW_UNLESS(target != nullptr, "Reloc section expected to refer at a valid target section");

    auto targetMapEntry = sectionMap.find(target.getOperation());
    VPUX_THROW_UNLESS(targetMapEntry != sectionMap.end(),
                      "Can't serialize a reloc section that doesn't have its dependent target section");

    auto targetSection = targetMapEntry->second;
    section->setSectionToPatch(targetSection);
    section->maskFlags(static_cast<elf::Elf_Xword>(secFlags()));

    if (symTab.isBuiltin()) {
        auto symTab_value = elf::VPU_RT_SYMTAB;
        section->setSpecialSymbolTable(symTab_value);
    } else {
        auto symTabMapEntry = sectionMap.find(symTab.getOperation());
        VPUX_THROW_UNLESS(symTabMapEntry != sectionMap.end(),
                          "Can't serialize a reloc section that doesn't have its dependent symbol table section");

        auto symTabSection = symTabMapEntry->second;
        section->setSymbolTable(dynamic_cast<elf::writer::SymbolSection*>(symTabSection));
    }

    auto block = getBody();
    for (auto& op : block->getOperations()) {
        auto relocation = section->addRelocationEntry();

        if (auto relocOp = llvm::dyn_cast<vpux::ELF::RelocOp>(op)) {
            relocOp.serialize(relocation, symbolMap);
        } else if (auto relocOp = llvm::dyn_cast<vpux::ELF::RelocImmOffsetOp>(op)) {
            relocOp.serialize(relocation, symbolMap);
        } else if (auto placeholder = llvm::dyn_cast<vpux::ELF::PutOpInSectionOp>(op)) {
            auto actualOp = placeholder.inputArg().getDefiningOp();

            if (auto relocOp = llvm::dyn_cast<vpux::ELF::RelocOp>(actualOp)) {
                relocOp.serialize(relocation, symbolMap);
            } else if (auto relocOp = llvm::dyn_cast<vpux::ELF::RelocImmOffsetOp>(actualOp)) {
                relocOp.serialize(relocation, symbolMap);
            }
        } else {
            VPUX_THROW(
                    "CreateRelocationSection op is expected to have either RelocOps or PutOpInSectionOp that refer to "
                    "RelocOps");
        }
    }

    sectionMap[getOperation()] = section;
}
