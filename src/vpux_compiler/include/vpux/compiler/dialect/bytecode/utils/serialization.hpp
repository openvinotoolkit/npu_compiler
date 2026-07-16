//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/type/element_type.hpp>
#include "vpux/compiler/dialect/bytecode/IR/attributes.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <llvm/ADT/StringMap.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>

#include <cstdint>

namespace vpux::bytecode {

// Get register number for the given operand
// This util is intended to be used only for instructions that have their registers explicitly defined via
// bytecode::GeneralRegisterOp
int16_t getRegisterNumber(mlir::Value operand);

// Get the index of the given string from the string section. The string is identified by the symbol name
uint64_t getStringIndex(StringRef symName, mlir::ModuleOp moduleOp);

// Get the index of the given constant from the constant section. The constant is identified by the symbol name.
uint64_t getConstantIndex(StringRef symName, mlir::ModuleOp moduleOp);

// Get the index of the given type from the type section. The type is identified by the symbol name.
uint64_t getTypeIndex(StringRef symName, mlir::ModuleOp moduleOp);

// Build a map from type symbol name to its positional index in the type section.
// Use this to resolve type references in O(1) after an O(n) build step.
llvm::StringMap<uint64_t> buildTypeIndexMap(bytecode::TypeSectionOp typeSection);

// Map an MLIR FloatType to its corresponding bytecode FloatFormat enum value.
bytecode::FloatFormat getFloatFormat(mlir::FloatType floatType);

// Map an OpenVINO element type to its corresponding bytecode FloatFormat enum value.
bytecode::FloatFormat getFloatFormat(const ov::element::Type& type);

}  // namespace vpux::bytecode
