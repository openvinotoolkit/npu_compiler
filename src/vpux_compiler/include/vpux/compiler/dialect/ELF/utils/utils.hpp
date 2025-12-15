//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux_headers/platform.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

using namespace vpux;

namespace llvm {
using RelocKey = std::pair<ELF::ElfSectionInterface, ELF::CreateSymbolTableSectionOp>;
template <>
struct DenseMapInfo<RelocKey> {
    static RelocKey getEmptyKey() {
        void* pointer = llvm::DenseMapInfo<void*>::getEmptyKey();
        return RelocKey(RelocKey::first_type::getFromOpaquePointer(pointer),
                        RelocKey::second_type::getFromOpaquePointer(pointer));
    }

    static RelocKey getTombstoneKey() {
        void* pointer = llvm::DenseMapInfo<void*>::getTombstoneKey();
        return RelocKey(RelocKey::first_type::getFromOpaquePointer(pointer),
                        RelocKey::second_type::getFromOpaquePointer(pointer));
    }

    static unsigned getHashValue(RelocKey val) {
        auto h1 = hash_value(val.first.getAsOpaquePointer());
        auto h2 = hash_value(val.second.getAsOpaquePointer());

        return checked_cast<unsigned>(h1 * h2);
    }

    static bool isEqual(RelocKey lhs, RelocKey rhs) {
        auto l1 = DenseMapInfo<mlir::Operation*>::isEqual(lhs.first.getOperation(), rhs.first.getOperation());
        auto l2 = DenseMapInfo<mlir::Operation*>::isEqual(lhs.second.getOperation(), rhs.second.getOperation());

        return l1 && l2;
    }
};
}  // namespace llvm

namespace vpux {
namespace ELF {
namespace math {

size_t gcd(size_t a, size_t b);
size_t lcm(size_t a, size_t b);

}  // namespace math

//
// Platform Information
//
elf::platform::ArchKind getElfArchKind(mlir::Operation* op);

ArrayRef<uint8_t> getKernelELF(mlir::Operation* operation, StringRef kernelPath, ArrayRef<StringRef> sectionNames = {});
ArrayRef<uint8_t> getDataAndSizeOfElfSection(ArrayRef<uint8_t> elfBlob, ArrayRef<StringRef> possibleSecNames);

class SymbolReferenceMap {
public:
    SymbolReferenceMap(vpux::ELF::MainOp elfMain, bool preLoadSymbols = false)
            : _elfMainSymbolTable(elfMain.getOperation()) {
        if (preLoadSymbols) {
            walkAllSymbols();
        }
    }

    mlir::Operation* lookupSymbol(mlir::SymbolRefAttr symRef);

private:
    void walkAllSymbols();
    mlir::SymbolTable _elfMainSymbolTable;
    mlir::DenseMap<mlir::StringAttr, mlir::SymbolTable> _sectionSymbolContainers;
};

ELF::MainOp getElfMainOp(mlir::ModuleOp moduleOp);
ELF::MainOp getElfMainOp(mlir::func::FuncOp funcOp);

size_t getOffsetOfSymRef(ELF::SymbolReferenceMap& symRefMap, mlir::SymbolRefAttr symRef);

mlir::SymbolRefAttr composeSectionObjectSymRef(ELF::ElfSectionInterface sectionIface, mlir::Operation* op);

template <class LHS>
std::string generateSignatureImpl(LHS&& lhs, const std::string& rhs) {
    return std::string(std::forward<LHS>(lhs)) + "." + rhs;
}

template <class LHS, class RHS>
decltype(std::to_string(RHS{}), void(), std::string()) generateSignatureImpl(LHS&& lhs, RHS&& rhs) {
    return generateSignatureImpl(std::forward<LHS>(lhs), std::to_string(std::forward<RHS>(rhs)));
}

template <class LHS, class RHS>
decltype(stringifyEnum(RHS{}), void(), std::string()) generateSignatureImpl(LHS&& lhs, RHS&& rhs) {
    return generateSignatureImpl(std::forward<LHS>(lhs), stringifyEnum(std::forward<RHS>(rhs)).str());
}

template <class LHS>
std::string generateSignatureImpl(LHS&& lhs, VPURegMapped::IndexType index) {
    const auto tileIndex = index.getTileIdx();
    const auto listIndex = index.getListIdx();
    auto signature = generateSignatureImpl(std::forward<LHS>(lhs), tileIndex);
    return generateSignatureImpl(std::move(signature), listIndex);
}

template <class LHS>
std::string generateSignatureImpl(LHS&& lhs, VPUASM::BufferType bufferType) {
    const auto location = bufferType.getLocation();
    const auto section = location.getSection();
    const auto sectionIndex = location.getSectionIndex();
    auto signature = generateSignatureImpl(std::forward<LHS>(lhs), section);
    return generateSignatureImpl(std::move(signature), sectionIndex);
}

template <class T>
std::string generateSignature(T&& signature) {
    return std::forward<T>(signature);
}

template <class LHS, class RHS, class... Rest>
std::string generateSignature(LHS&& lhs, RHS&& rhs, Rest&&... rest) {
    return generateSignature(generateSignatureImpl(std::forward<LHS>(lhs), std::forward<RHS>(rhs)),
                             std::forward<Rest>(rest)...);
}

std::pair<uint8_t, uint8_t> reduceWaitMaskTo8bit(uint64_t waitMask);

// creates a linear (1D) MemrefType of dimension (memrefSize x dataType)
mlir::MemRefType getLinearMemrefType(mlir::MLIRContext* ctx, int64_t memrefSize, mlir::Type dataType,
                                     VPU::MemoryKind memKind);

using SectionMapper = typename std::unordered_map<ELF::SectionSignature, ELF::ElfSectionInterface>;
mlir::SymbolRefAttr moveOpToSection(mlir::Operation* op, SectionMapper& sectionMap, mlir::OpBuilder& builder);
mlir::SymbolRefAttr cloneSectionSymbol(mlir::SymbolRefAttr from, mlir::SymbolRefAttr to);

void insertELFMain(mlir::func::FuncOp netFunc);
size_t getOpBinarySize(vpux::NDTypeInterface type);

}  // namespace ELF
}  // namespace vpux
