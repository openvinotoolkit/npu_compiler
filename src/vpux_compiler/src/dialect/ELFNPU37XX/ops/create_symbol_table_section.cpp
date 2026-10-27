//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"

void vpux::ELFNPU37XX::CreateSymbolTableSectionOp::serialize(elf::Writer& writer,
                                                             vpux::ELFNPU37XX::SectionMapType& sectionMap,
                                                             vpux::ELFNPU37XX::SymbolMapType& symbolMap) {
    if (getIsBuiltin()) {
        return;
    }

    const auto name = getSecName().str();
    auto section = writer.addSymbolSection(name);
    section->setAddrAlign(alignof(elf::SymbolEntry));

    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));

    auto block = getBody();
    for (auto& op : block->getOperations()) {
        auto symbol = section->addSymbolEntry();

        if (auto symOp = llvm::dyn_cast<vpux::ELFNPU37XX::SymbolOp>(op)) {
            symOp.serialize(symbol, sectionMap);
            symbolMap[symOp.getOperation()] = symbol;
        } else if (auto placeholder = llvm::dyn_cast<vpux::ELFNPU37XX::PutOpInSectionOp>(op)) {
            auto actualOp = placeholder.getInputArg().getDefiningOp();
            auto symOp = llvm::dyn_cast<vpux::ELFNPU37XX::SymbolOp>(actualOp);

            VPUX_THROW_WHEN(
                    symOp == nullptr,
                    "Symbol table section op is expected to have PutOpInSectionOps that point to SymbolOps only."
                    " Got *actualOp {0}.",
                    *actualOp);

            symOp.serialize(symbol, sectionMap);
            symbolMap[symOp.getOperation()] = symbol;
        } else {
            VPUX_THROW("Symbol table section op is expected to have either SymbolOps or PutOpInSectionOps that have as "
                       "params "
                       "SymbolOps. Got {0}.",
                       op);
        }
    }

    sectionMap[getOperation()] = section;
}

void vpux::ELFNPU37XX::CreateSymbolTableSectionOp::preserialize(elf::Writer& writer,
                                                                vpux::ELFNPU37XX::SectionMapType& sectionMap) {
    VPUX_UNUSED(writer);
    VPUX_UNUSED(sectionMap);

    // don't implement and go into elf::Writer internal state first
    // as there're symbol entries to be updated first
    // E#136375
}
