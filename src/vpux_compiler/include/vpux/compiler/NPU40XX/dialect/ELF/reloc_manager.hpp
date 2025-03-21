//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

namespace vpux {
namespace ELF {

class RelocManager {
public:
    RelocManager(vpux::ELF::MainOp mainOp)
            : builder_(mlir::OpBuilder::atBlockEnd(&mainOp.getContent().front())),
              relocMap_(),
              symbolMap_(),
              symRefMap_(mainOp),
              elfMain_(mainOp) {
        constructSymbolMap(mainOp);
    }

    RelocManager() = delete;
    ~RelocManager() = default;
    RelocManager(RelocManager& other) = delete;
    RelocManager& operator=(const RelocManager&) = delete;

    void createRelocations(ELF::RelocatableOpInterface relocatableOp);

private:
    void createRelocations(mlir::Operation* op, ELF::RelocationInfo& relocInfo);
    void createRelocations(mlir::Operation* op, std::vector<ELF::RelocationInfo>& relocInfo);

    void constructSymbolMap(ELF::MainOp elfMain);

    ELF::SymbolOp getSymbolOfBinOpOrEncapsulatingSection(mlir::Operation* binOp);
    ELF::SymbolOp getCMXBaseAddressSym();

    ELF::CreateRelocationSectionOp getRelocationSection(ELF::ElfSectionInterface targetSection,
                                                        ELF::CreateSymbolTableSectionOp symbolTable);

private:
    mlir::OpBuilder builder_;
    llvm::DenseMap<std::pair<mlir::Operation*, ELF::CreateSymbolTableSectionOp>, ELF::CreateRelocationSectionOp>
            relocMap_;
    llvm::DenseMap<mlir::Operation*, ELF::SymbolOp>
            symbolMap_;                  // maps ops to their attached ELF symbol (currently only section symbols)
    ELF::SymbolReferenceMap symRefMap_;  // maps mlir::SymbolRefAttrs to the op that they reference
    vpux::ELF::MainOp elfMain_;
};

}  // namespace ELF
}  // namespace vpux
