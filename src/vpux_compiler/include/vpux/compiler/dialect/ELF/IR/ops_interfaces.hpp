//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/ELF/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <vpux_elf/writer.hpp>

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>

namespace vpux {
namespace ELF {

typedef DenseMap<mlir::Operation*, elf::writer::Section*> SectionMapType;
// Note that this works since in our case the IR is immutable troughout the life-time of the map.
typedef DenseMap<mlir::Operation*, elf::writer::Symbol*> SymbolMapType;

typedef DenseMap<mlir::Operation*, elf::writer::DmaSymbol*> DmaSymbolMapType;

constexpr auto VPUX_SHAVE_ALIGNMENT = static_cast<size_t>(Byte(1_KB).count());
constexpr auto VPUX_METADATA_ALIGNMENT = static_cast<size_t>((32_Byte).count());
constexpr auto VPUX_DEFAULT_ALIGNMENT = static_cast<size_t>((64_Byte).count());
constexpr auto VPUX_NO_ALIGNMENT = static_cast<size_t>((1_Byte).count());

//
// ElfSectionInterface
//

template <typename ConcreteOp>
mlir::Block* getSectionBlock(ConcreteOp op) {
    mlir::Operation* operation = op.getOperation();
    auto& region = operation->getRegion(0);

    if (region.empty()) {
        region.emplaceBlock();
    }

    return &region.front();
}

//
// SectionSignature
//

class SectionSignature {
public:
    SectionSignature(std::string name, SectionFlagsAttr flags, SectionTypeAttr type = SectionTypeAttr::SHT_PROGBITS);

    std::string_view getName() const;
    StringRef getNameRef() const;
    SectionFlagsAttr getFlags() const;
    SectionTypeAttr getType() const;

private:
    std::string _name;
    SectionFlagsAttr _flags;
    SectionTypeAttr _type;
};

bool operator==(const SectionSignature& lhs, const SectionSignature& rhs);

//
// SymbolSignature
//

struct SymbolSignature {
    SymbolSignature(mlir::SymbolRefAttr reference, std::string name, ELF::SymbolType type = ELF::SymbolType::STT_OBJECT,
                    size_t size = 0, size_t value = 0);

    mlir::SymbolRefAttr reference;
    std::string name;
    ELF::SymbolType type;
    size_t size;
    size_t value;
};

bool operator==(const SymbolSignature& lhs, const SymbolSignature& rhs);

struct RelocationInfo;

class SymbolReferenceMap;

}  // namespace ELF
}  // namespace vpux

//
// Hash
//

namespace std {
template <>
struct hash<vpux::ELF::SectionSignature> final {
    std::size_t operator()(const vpux::ELF::SectionSignature& signature) const {
        return llvm::hash_combine(llvm::hash_value(signature.getName()),
                                  llvm::hash_value(static_cast<uint64_t>(signature.getFlags())),
                                  llvm::hash_value(static_cast<uint64_t>(signature.getType())));
    }
};
}  // namespace std

//
// Generated
//

#include <vpux/compiler/dialect/ELF/ops_interfaces.hpp.inc>

struct vpux::ELF::RelocationInfo {
    RelocationInfo(mlir::SymbolRefAttr source, vpux::ELF::ElfSectionInterface targetSection, size_t offset,
                   vpux::ELF::RelocationType relocType, int64_t addend, std::string description = "",
                   bool isOffsetRelative = true)
            : source{source},
              targetSection{targetSection},
              offset{offset},
              relocType{relocType},
              addend{addend},
              description(std::move(description)),
              isOffsetRelative{isOffsetRelative} {};

    mlir::SymbolRefAttr source;
    vpux::ELF::ElfSectionInterface targetSection;
    size_t offset;
    vpux::ELF::RelocationType relocType;
    int64_t addend;
    std::string description;
    bool isOffsetRelative;
};
