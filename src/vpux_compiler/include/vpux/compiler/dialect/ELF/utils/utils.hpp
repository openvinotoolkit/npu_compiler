//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux_headers/platform.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/SymbolTable.h>

namespace llvm {
using RelocKey = std::pair<vpux::ELF::ElfSectionInterface, vpux::ELF::CreateSymbolTableSectionOp>;
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

        return vpux::checked_cast<unsigned>(h1 * h2);
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
            : _elfMainSymbolTable(elfMain.getOperation()), _elfMain(elfMain) {
        if (preLoadSymbols) {
            walkAllSymbols();
        }
    }

    mlir::Operation* lookupSymbol(mlir::SymbolRefAttr symRef);

private:
    void walkAllSymbols();
    mlir::SymbolTable _elfMainSymbolTable;
    ELF::MainOp _elfMain;
    mlir::DenseMap<mlir::StringAttr, mlir::SymbolTable> _sectionSymbolContainers;
};

ELF::MainOp getElfMainOp(mlir::ModuleOp moduleOp);
ELF::MainOp getElfMainOp(mlir::func::FuncOp funcOp);

int64_t getOffsetOfSymRef(ELF::SymbolReferenceMap& symRefMap, mlir::SymbolRefAttr symRef);

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

// if `op` does not implement ELF::WrappableOpInterface - do nothing and return nullptr
// otherwise, moves `op` into a corresponding ELF section, that is created if necessary
//
// if `op` is not a `Symbol`, given ELF section ops can contain only operations of the same type
// the `op` cannot reference other `Symbol` ops in the same section (there are no `Symbol` ops there)
// then there are no references to be updated after original op symbolization (that produced the `op`)
// returns nullptr in this case
//
// otherwise, returns symbol reference in format of @section_name::@symbol_name
// to allow caller to update `op`'s symbol references to other `Symbol` ops in the section
mlir::SymbolRefAttr moveOpToSection(mlir::Operation* op, mlir::OpBuilder& builder);

// moveOpToSection overload with optimized ELF section lookup via `sectionMap`
using SectionMapper = typename std::unordered_map<ELF::SectionSignature, ELF::ElfSectionInterface>;
mlir::SymbolRefAttr moveOpToSection(mlir::Operation* op, SectionMapper& sectionMap, mlir::OpBuilder& builder);

mlir::SymbolRefAttr cloneSectionSymbol(mlir::SymbolRefAttr from, mlir::SymbolRefAttr to);

void insertELFMain(mlir::func::FuncOp netFunc);
size_t getOpBinarySize(vpux::NDTypeInterface type);
size_t getBufferBinarySize(vpux::NDTypeInterface type, mlir::Operation* contextOp, VPURT::BufferSection section,
                           int64_t sectionIndex);

// lookup operation that defines `symbol` in scope of ELF.Main that defines `from`
// note: mlir::SymbolTable::lookupNearestSymbolFrom limits the scope at the closest `SymbolTable` parent
mlir::Operation* lookupNearestSymbolFrom(mlir::Operation* from, mlir::StringAttr symbol);
mlir::Operation* lookupNearestSymbolFrom(mlir::Operation* from, mlir::SymbolRefAttr symbol);

// collect all uses of `symbol` inside `from`, including regions of nested `SymbolTable` operations
// assumes nested `SymbolTable` operations have no nested regions on their own
// note: mlir::SymbolTable::getSymbolUses will not recurse into nested `SymbolTable` operations
mlir::SmallVector<mlir::SymbolTable::SymbolUse> getSymbolUses(mlir::Operation* symbol, ELF::MainOp from);

/*
    Canonical form is a form in which we preserve strides of the argument. Legalization pass
    allows for reshapes that expand/contract original shape with unit dimensions. Below function
    will go over DMA buffer strides and recover compatible DMA shapes and strides from them by
    comparing them to argument type. For example consider below case
    argumentShape 5x10x1xf16
    argumentStrides [10, 1, 1] note: final stride of 50 is not returned by getStrides
    dmaShape 1x5x1x5xf16
    dmaStrides [50, 10, 10, 1] note: here we suddenly get stride of 50 since we have one extra dimension

    Note the tiling on first dim of dmaShape(10 got tiled to 5). Canonical form of dmaShape and DmaStrides would be
    canonicalDmaShape 5x5x1xf16
    canonicalDmaStrides [50, 10, 1, 1]
    note that in canonical form final stride of 50 is present. It's a quirk of the algorithm and isn't really needed
    but also doesn't hurt. Also note how canonical dma strides match the strides of the argument which is exactly the
    point.
*/
void getCanonicalDmaForm(MemShape& dmaBufferShape, MemStrides& dmaBufferStrides, ShapeRef argumentShape,
                         llvm::SmallVector<int64_t>& tileOffsets, llvm::SmallVector<int64_t>& canonicalDmaShapes,
                         llvm::SmallVector<int64_t>& canonicalDmaStrides, llvm::SmallVector<int64_t>& canonicalOffsets);

}  // namespace ELF
}  // namespace vpux
