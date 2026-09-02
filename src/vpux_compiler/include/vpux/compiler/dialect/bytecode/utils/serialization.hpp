//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/type/element_type.hpp>
#include "vpux/compiler/dialect/bytecode/IR/attributes.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>

#include <cstdint>
#include <optional>

namespace vpux::bytecode {

template <typename SectionOp, typename EntryOp>
std::optional<uint64_t> getIndex(mlir::SymbolRefAttr symbol, mlir::ModuleOp moduleOp) {
    if (symbol.getNestedReferences().empty()) {
        auto sections = moduleOp.template getOps<SectionOp>();
        auto it = sections.begin();
        if (it == sections.end()) {
            return std::nullopt;
        }
        auto nextIt = it;
        ++nextIt;
        if (nextIt != sections.end()) {
            return std::nullopt;
        }
        auto* ctx = moduleOp.getContext();
        symbol = mlir::SymbolRefAttr::get(ctx, (*it).getSymName(),
                                          {mlir::FlatSymbolRefAttr::get(symbol.getRootReference())});
    }
    auto* resolved = mlir::SymbolTable::lookupSymbolIn(moduleOp, symbol);
    if (resolved == nullptr) {
        return std::nullopt;
    }
    auto parentSection = resolved->getParentOfType<SectionOp>();
    if (parentSection == nullptr) {
        return std::nullopt;
    }
    auto ops = parentSection.getContent().template getOps<EntryOp>();
    for (const auto& [i, op] : llvm::enumerate(ops)) {
        if (op.getOperation() == resolved) {
            return static_cast<uint64_t>(i);
        }
    }
    return std::nullopt;
}

// Overload that constructs the nested SymbolRefAttr internally from an operation
// and a leaf symbol name, looking up the section by type in the parent module.
template <typename SectionOp, typename EntryOp>
std::optional<uint64_t> getIndex(mlir::Operation* from, llvm::StringRef leafSymbol) {
    auto moduleOp = mlir::isa<mlir::ModuleOp>(from) ? mlir::cast<mlir::ModuleOp>(from)
                                                    : from->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
        return std::nullopt;
    }
    auto sections = moduleOp.template getOps<SectionOp>();
    auto it = sections.begin();
    if (it == sections.end()) {
        return std::nullopt;
    }
    auto nextIt = it;
    ++nextIt;
    if (nextIt != sections.end()) {
        return std::nullopt;
    }
    auto* ctx = moduleOp.getContext();
    auto symbol = mlir::SymbolRefAttr::get(ctx, (*it).getSymName(), {mlir::FlatSymbolRefAttr::get(ctx, leafSymbol)});
    return getIndex<SectionOp, EntryOp>(symbol, moduleOp);
}

inline uint64_t getStringIndex(llvm::StringRef leafSymbol, mlir::ModuleOp moduleOp) {
    auto result = getIndex<StringSectionOp, StringOp>(moduleOp.getOperation(), leafSymbol);
    VPUX_THROW_UNLESS(result.has_value(), "Failed to resolve string symbol '{0}' in string section", leafSymbol);
    return result.value();
}

inline uint64_t getConstantIndex(llvm::StringRef leafSymbol, mlir::ModuleOp moduleOp) {
    auto result = getIndex<ConstantSectionOp, ConstantOp>(moduleOp.getOperation(), leafSymbol);
    VPUX_THROW_UNLESS(result.has_value(), "Failed to resolve constant symbol '{0}' in constant section", leafSymbol);
    return result.value();
}

// Get register number for the given operand
// This util is intended to be used only for instructions that have their registers explicitly defined via
// bytecode::GeneralRegisterOp
int16_t getRegisterNumber(mlir::Value operand);

// Build a map from type symbol name to its positional index in the type section.
// Use this to resolve type references in O(1) after an O(n) build step.
llvm::StringMap<uint64_t> buildTypeIndexMap(bytecode::TypeSectionOp typeSection);

// Map an MLIR FloatType to its corresponding bytecode FloatFormat enum value.
bytecode::FloatFormat getFloatFormat(mlir::FloatType floatType);

// Map an OpenVINO element type to its corresponding bytecode FloatFormat enum value.
bytecode::FloatFormat getFloatFormat(const ov::element::Type& type);

// Lookup the operation that defines `symbol` starting from the top-level SymbolTable (ModuleOp)
// that contains `from`.
mlir::Operation* lookupNearestSymbolFrom(mlir::Operation* from, mlir::SymbolRefAttr symbol);
mlir::Operation* lookupNearestSymbolFrom(mlir::Operation* from, mlir::StringAttr symbol);

}  // namespace vpux::bytecode
