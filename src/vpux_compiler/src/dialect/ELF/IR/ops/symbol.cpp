//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"

#include "vpux/compiler/utils/types.hpp"

#include <vpux_elf/writer.hpp>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/SymbolTable.h>

using namespace vpux;

void ELF::SymbolOp::serialize(elf::writer::Symbol* symbol, ELF::SectionMapType& sectionMap) {
    auto symName = getSymName();
    auto symType = getType();
    // contract with loader requires that symbol value and size must be 0 if non existent
    auto symSize = getSize().value_or(0);
    auto symVal = getValue().value_or(0);

    /* From the serialization perspective the symbols can be of 5 types:
        - Section symbols: in this case the parentSection is the defining op itself;
        - Generic symbols: Symbols representing an OP inside the IR. In this case we need the parent section of either
       the OP or its placeholder;
        - Standalone symbols: symbols that do not relate to any entity inside the IR (nor the ELF itself).
      The ticket E#29144 plans to handle Standalone symbols.
    */

    auto referenceOp = ELF::lookupNearestSymbolFrom(getOperation()->getParentOp(), getReferenceAttr());
    auto parentSection = referenceOp;
    if (!mlir::isa<ELF::ElfSectionInterface>(referenceOp)) {
        parentSection = referenceOp->getParentOp();
        VPUX_THROW_UNLESS(mlir::isa<ELF::ElfSectionInterface>(parentSection),
                          "Symbol op referencing and OP not in a section {0}", this);
    }

    symbol->setName(symName.str());
    symbol->setType(static_cast<elf::Elf_Word>(symType));
    symbol->setSize(symSize);
    symbol->setValue(symVal);

    auto sectionMapEntry = sectionMap.find(parentSection);
    VPUX_THROW_WHEN(sectionMapEntry == sectionMap.end(), "Unable to find section entry for SymbolOp");
    auto sectionEntry = sectionMapEntry->second;

    symbol->setRelatedSection(sectionEntry);
}

void ELF::SymbolOp::build(mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                          ELF::SymbolSignature& signature) {
    auto ctx = odsState.getContext();
    auto signatureSize = signature.size ? mlir::IntegerAttr::get(getUInt64Type(ctx), signature.size) : nullptr;
    auto signatureValue = signature.value ? mlir::IntegerAttr::get(getUInt64Type(ctx), signature.value) : nullptr;

    build(odsBuilder, odsState, signature.name, signature.reference, signature.type, signatureSize, signatureValue);
}

void ELF::SymbolOp::build(mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::llvm::StringRef symName,
                          ::mlir::SymbolRefAttr reference, vpux::ELF::SymbolType type) {
    build(odsBuilder, odsState, symName, reference, type, nullptr, nullptr);
}
