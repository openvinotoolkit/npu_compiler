//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_elf/types/vpu_extensions.hpp>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"

void vpux::ELFNPU37XX::CreateRelocationSectionOp::serialize(elf::Writer& writer,
                                                            vpux::ELFNPU37XX::SectionMapType& sectionMap,
                                                            vpux::ELFNPU37XX::SymbolMapType& symbolMap) {
    const auto name = getSecName().str();
    auto section = writer.addRelocationSection(name);
    section->setAddrAlign(alignof(elf::RelocationAEntry));

    // Look up dependent sections
    auto symTab =
            llvm::dyn_cast<vpux::ELFNPU37XX::CreateSymbolTableSectionOp>(getSourceSymbolTableSection().getDefiningOp());
    VPUX_THROW_WHEN(symTab == nullptr, "Reloc section expected to refer to a symbol table section");

    auto target = llvm::dyn_cast_or_null<vpux::ELFNPU37XX::CreateSectionOp>(getTargetSection().getDefiningOp());
    VPUX_THROW_WHEN(target == nullptr, "Reloc section expected to refer at a valid target section");

    auto targetMapEntry = sectionMap.find(target.getOperation());
    VPUX_THROW_WHEN(targetMapEntry == sectionMap.end(),
                    "Can't serialize a reloc section that doesn't have its dependent target section");

    auto targetSection = targetMapEntry->second;
    section->setSectionToPatch(targetSection);
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));

    if (symTab.getIsBuiltin()) {
        auto symTab_value = elf::VPU_RT_SYMTAB;
        section->setSpecialSymbolTable(symTab_value);
    } else {
        auto symTabMapEntry = sectionMap.find(symTab.getOperation());
        VPUX_THROW_WHEN(symTabMapEntry == sectionMap.end(),
                        "Can't serialize a reloc section that doesn't have its dependent symbol table section");

        auto symTabSection = symTabMapEntry->second;
        section->setSymbolTable(dynamic_cast<elf::writer::SymbolSection*>(symTabSection));
    }

    vpux::ELFNPU37XX::OffsetCache cache;
    auto block = getBody();
    for (auto& op : block->getOperations()) {
        auto relocation = section->addRelocationEntry();

        if (auto relocOp = llvm::dyn_cast<vpux::ELFNPU37XX::RelocOp>(op)) {
            relocOp.serialize(relocation, symbolMap, cache);
        } else if (auto placeholder = llvm::dyn_cast<vpux::ELFNPU37XX::PutOpInSectionOp>(op)) {
            auto actualOp = placeholder.getInputArg().getDefiningOp();

            if (auto relocOp = llvm::dyn_cast<vpux::ELFNPU37XX::RelocOp>(actualOp)) {
                relocOp.serialize(relocation, symbolMap, cache);
            }
        } else {
            VPUX_THROW(
                    "CreateRelocationSection op is expected to have either RelocOps or PutOpInSectionOp that refer to "
                    "RelocOps");
        }
    }

    sectionMap[getOperation()] = section;
}

void vpux::ELFNPU37XX::CreateRelocationSectionOp::preserialize(elf::Writer&, vpux::ELFNPU37XX::SectionMapType&) {
    // don't implement and go into elf::Writer internal state first
    // as there're relocation entries to be updated first
    // E#136375
}
