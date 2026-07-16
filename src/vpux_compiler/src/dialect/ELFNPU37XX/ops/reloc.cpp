//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/types.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>

namespace {
mlir::Value getTargetSectionOfRelocOp(mlir::Operation* op) {
    auto parentOp = op->getParentOp();
    auto relocSectionOp = mlir::dyn_cast<vpux::ELFNPU37XX::CreateRelocationSectionOp>(parentOp);
    VPUX_THROW_UNLESS(relocSectionOp, "Parent Op of RelocOp must be an ELFNPU37XX::CreateRelocationSectionOp");

    return relocSectionOp.getTargetSection();
}
}  // namespace

void vpux::ELFNPU37XX::RelocOp::serialize(elf::writer::Relocation* relocation,
                                          vpux::ELFNPU37XX::SymbolMapType& symbolMap,
                                          vpux::ELFNPU37XX::OffsetCache& cache) {
    auto symbolOp = getSourceSymbol().getDefiningOp();
    auto actualSymbolOp = llvm::cast<vpux::ELFNPU37XX::SymbolOp>(symbolOp);

    if (actualSymbolOp.getIsBuiltin()) {
        auto symInputValue = actualSymbolOp.getInputArg();
        auto const_val = llvm::cast<mlir::arith::ConstantOp>(symInputValue.getDefiningOp());
        auto symValue = mlir::cast<mlir::IntegerAttr>(const_val.getValue()).getInt();

        relocation->setSpecialSymbol(static_cast<elf::Elf_Word>(symValue));
    } else {
        auto symbolMapEntry = symbolMap.find(symbolOp);
        VPUX_THROW_UNLESS(symbolMapEntry != symbolMap.end(), "Unable to locate symbol entry for relocation");
        auto symbolEntry = symbolMapEntry->second;
        relocation->setSymbol(symbolEntry);
    }

    auto baseOpVal = getBaseOp();
    auto relocType = getRelocationType();
    auto relocAddend = getAddend();
    auto relocOffset = getOffset();

    if (baseOpVal) {
        auto targetSection = getTargetSectionOfRelocOp(getOperation());
        relocOffset += ELFNPU37XX::getOffsetOfOpInSection(baseOpVal, targetSection, cache);
    }

    relocation->setType(static_cast<elf::Elf_Word>(relocType));
    relocation->setOffset(relocOffset);
    relocation->setAddend(relocAddend);
}

void vpux::ELFNPU37XX::RelocOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                      mlir::Value baseOpOffset, int64_t offset,
                                      vpux::ELFNPU37XX::RelocationType relocationType, mlir::Value sourceSymbol,
                                      int64_t addend, llvm::StringRef description) {
    build(odsBuilder, odsState, baseOpOffset, offset, relocationType, sourceSymbol, addend,
          odsBuilder.getStringAttr(description));
}

void vpux::ELFNPU37XX::RelocOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                      mlir::Value baseOpOffset, int64_t offset,
                                      vpux::ELFNPU37XX::RelocationType relocationType, ::mlir::Value sourceSymbol,
                                      int64_t addend) {
    build(odsBuilder, odsState, baseOpOffset, offset, relocationType, sourceSymbol, addend, "");
}
