//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/reloc_manager.hpp"

using namespace vpux;

ELF::CreateRelocationSectionOp ELF::RelocManager::getRelocationSection(ELF::ElfSectionInterface targetSection,
                                                                       ELF::CreateSymbolTableSectionOp symbolTable) {
    auto key = std::make_pair(targetSection.getOperation(), symbolTable);

    auto relocSectionIt = relocMap_.find(key);

    if (relocSectionIt != relocMap_.end()) {
        return relocSectionIt->getSecond();
    }

    auto targetSectionSymbolIface = mlir::cast<mlir::SymbolOpInterface>(targetSection.getOperation());
    auto symtabSymbolIface = mlir::cast<mlir::SymbolOpInterface>(symbolTable.getOperation());

    mlir::StringAttr nameAttr =
            mlir::StringAttr::get(builder_.getContext(), llvm::Twine("rela.") + targetSectionSymbolIface.getName() +
                                                                 "." + symtabSymbolIface.getName());

    auto targetSectionRef = mlir::FlatSymbolRefAttr::get(targetSectionSymbolIface.getNameAttr());
    auto symTabRef = mlir::FlatSymbolRefAttr::get(symtabSymbolIface.getNameAttr());

    auto symTabFlags = symbolTable.getSecFlags();

    auto relaSectionFlags = symTabFlags;
    auto isJITRelaSection = (static_cast<uint32_t>(relaSectionFlags & ELF::SectionFlagsAttr::VPU_SHF_USERINPUT) ||
                             static_cast<uint32_t>(relaSectionFlags & ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT) ||
                             static_cast<uint32_t>(relaSectionFlags & ELF::SectionFlagsAttr::VPU_SHF_PROFOUTPUT));

    VPUX_THROW_WHEN(isJITRelaSection && !ELF::bitEnumContainsAll(relaSectionFlags, ELF::SectionFlagsAttr::VPU_SHF_JIT),
                    "Reloc Section for JIT symbols must have VPU_SHF_JIT Flag");

    auto flags = ELF::SectionFlagsAttrAttr::get(builder_.getContext(), relaSectionFlags);
    auto newRelocSection = builder_.create<ELF::CreateRelocationSectionOp>(symbolTable.getLoc(), nameAttr,
                                                                           targetSectionRef, symTabRef, flags);

    relocMap_[key] = newRelocSection;

    return newRelocSection;
}

ELF::SymbolOp ELF::RelocManager::getSymbolOfBinOpOrEncapsulatingSection(mlir::Operation* binOp) {
    auto sectionOp = mlir::isa<ELF::ElfSectionInterface>(binOp) ? mlir::cast<ELF::ElfSectionInterface>(binOp)
                                                                : binOp->getParentOfType<ELF::ElfSectionInterface>();

    auto symbolMapIt = symbolMap_.find(sectionOp.getOperation());

    if (symbolMapIt != symbolMap_.end()) {
        return symbolMapIt->getSecond();
    }

    VPUX_THROW("No ELF Symbol found for the provided operation");
}

ELF::SymbolOp ELF::RelocManager::getCMXBaseAddressSym() {
    auto logicalSections = elfMain_.getOps<ELF::LogicalSectionOp>();
    for (auto section : logicalSections) {
        if (section.getSecType() == ELF::SectionTypeAttr::VPU_SHT_CMX_WORKSPACE) {
            return getSymbolOfBinOpOrEncapsulatingSection(section.getOperation());
        }
    }
    VPUX_THROW("Can't find any CMX Logical Sections");
}

void ELF::RelocManager::createRelocations(mlir::Operation* op, ELF::RelocationInfo& relocInfo) {
    auto sourceOp = symRefMap_.lookupSymbol(relocInfo.source);

    ELF::SymbolOp sourceSym = getSymbolOfBinOpOrEncapsulatingSection(sourceOp);
    ELF::CreateSymbolTableSectionOp symTab = mlir::dyn_cast<ELF::CreateSymbolTableSectionOp>(sourceSym->getParentOp());
    ELF::CreateRelocationSectionOp relocSection = getRelocationSection(relocInfo.targetSection, symTab);
    auto symForReloc = ELF::composeSectionObjectSymRef(symTab, sourceSym.getOperation());

    auto relocBuilder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

    auto offset = relocInfo.offset;

    if (relocInfo.isOffsetRelative) {
        auto baseBinaryOp = mlir::cast<ELF::BinaryOpInterface>(op);
        offset += baseBinaryOp.getMemoryOffset();
    }

    relocBuilder.create<ELF::RelocOp>(relocSection.getLoc(), offset, symForReloc, relocInfo.relocType, relocInfo.addend,
                                      relocInfo.description);
}

void ELF::RelocManager::createRelocations(mlir::Operation* op, std::vector<ELF::RelocationInfo>& relocInfo) {
    for (auto& reloc : relocInfo) {
        createRelocations(op, reloc);
    }
}

void ELF::RelocManager::createRelocations(ELF::RelocatableOpInterface relocatableOp) {
    auto relocsInfo = relocatableOp.getRelocationInfo(symRefMap_);
    createRelocations(relocatableOp.getOperation(), relocsInfo);
}

void ELF::RelocManager::constructSymbolMap(ELF::MainOp elfMain) {
    auto symbolTables = elfMain.getOps<ELF::CreateSymbolTableSectionOp>();

    for (auto symbolTable : symbolTables) {
        auto elfSymbols = symbolTable.getOps<ELF::SymbolOp>();
        for (auto elfSymbol : elfSymbols) {
            auto reference = symRefMap_.lookupSymbol(elfSymbol.getReference());
            symbolMap_[reference] = elfSymbol;
        }
    }
}
