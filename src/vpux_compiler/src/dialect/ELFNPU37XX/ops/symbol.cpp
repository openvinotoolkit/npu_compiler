//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace {
mlir::Operation* getParentSectionOp(mlir::Value val) {
    // If one of the users of the value is a PutOpInSection op, then we are interested in its encapsulating section.
    // Based on IR validity, any PutOpInSection op is always in a CreateSymbolTableSection (or CreateSection).
    mlir::Operation* op = nullptr;
    for (auto user : val.getUsers()) {
        if (mlir::dyn_cast_or_null<vpux::ELFNPU37XX::PutOpInSectionOp>(user)) {
            op = user;
            // In a section we should have only 1 PutOpInSectionOp user for each object put in it.
            break;
        }
    }

    // This happens normally in case val is a BlockArgument (e.g., "arg0" in the .mlir file).
    //  OR in the case of a value assigned in the FuncOp scope.
    if (op == nullptr) {
        op = val.getDefiningOp();

        return val.getParentRegion()->getParentOp();
    }
    VPUX_THROW_WHEN(op == nullptr, "Both user and producer operation can't be found.");

    auto region = op->getParentRegion();
    VPUX_THROW_WHEN(region == nullptr, "Unlinked ops are unsupported");

    return region->getParentOp();
}

}  // namespace

void vpux::ELFNPU37XX::SymbolOp::serialize(elf::writer::Symbol* symbol, vpux::ELFNPU37XX::SectionMapType& sectionMap) {
    if (getIsBuiltin()) {
        return;
    }

    auto symName = getName().value_or("");
    auto symType = getSymType().value_or(vpux::ELFNPU37XX::SymbolTypeEnum::STT_NOTYPE);
    auto symSize = getSize().value_or(0);
    auto symVal = getValue().value_or(0);

    /* From the serialization perspective the symbols can be of 5 types:
        - Section symbols: in this case the parentSection is the defining op itself;
        - Generic symbols: Symbols representing an OP inside the IR. In this case we need the parent section of either
       the OP or its placeholder;
        - I/O symbols: symbols that represent function arguments. In this case we will not have a parentSection, and no
       relatedSection;
        - Symbols referring to the "Special Symbol Table";
        - Standalone symbols: symbols that do not relate to any entity inside the IR (nor the ELF itself).
      The ticket E#29144 plans to handle these last 2 types of sections.
    */

    // We initialize parentSection to nullptr, since getInputArg() can be a BlockArgument,
    //   which has getDefiningOp() equal to nullptr.
    mlir::Operation* parentSection = nullptr;

    if (auto inputArgDefOp = getInputArg().getDefiningOp()) {
        if (mlir::isa<ELFNPU37XX::ElfSectionInterface>(inputArgDefOp)) {
            parentSection = inputArgDefOp;
        } else {
            parentSection = getParentSectionOp(getInputArg());

            if (mlir::isa<mlir::func::FuncOp>(parentSection)) {
                parentSection = nullptr;
            } else {
                VPUX_THROW_WHEN(parentSection == nullptr, "Could not find valid parent section for op");
            }
        }
    }

    symbol->setName(symName.str());
    symbol->setType(static_cast<elf::Elf_Word>(symType));
    symbol->setSize(symSize);
    symbol->setValue(symVal);

    if (parentSection != nullptr) {
        auto sectionMapEntry = sectionMap.find(parentSection);
        VPUX_THROW_WHEN(sectionMapEntry == sectionMap.end(), "Unable to find section entry for SymbolOp");
        auto sectionEntry = sectionMapEntry->second;

        symbol->setRelatedSection(sectionEntry);
    }
}
