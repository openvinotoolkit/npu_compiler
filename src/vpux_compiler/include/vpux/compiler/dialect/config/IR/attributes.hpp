//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpux/compiler/dialect/config/version.hpp>
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>

#include <optional>

//
// Generated
//

#include <vpux/compiler/dialect/config/enums.hpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/config/attributes.hpp.inc>

namespace vpux::config {

config::Platform getPlatform(mlir::Operation* op);
config::ArchKind getArch(config::Platform platform);

//
// CompilationMode
//

void setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode);
bool hasCompilationMode(mlir::ModuleOp module);
CompilationMode getCompilationMode(mlir::Operation* op);

// Returns true for HostCompile, HostCompile_JIT, and HostCompile_Interpreter modes.
bool isHostCompileMode(CompilationMode mode);
bool isHostCompileMode(mlir::Operation* op);
bool isHostCompileInterpreterMode(CompilationMode mode);
bool isHostCompileInterpreterMode(mlir::Operation* op);

//
// ELF ABI Version
//

void setElfAbiVersion(mlir::ModuleOp module, const Version& version);
std::optional<Version> getElfAbiVersion(mlir::Operation* op);

//
// Resource kind value getter
//

template <typename ConcreteKind, typename ResourceOp>
ConcreteKind getKindValue(ResourceOp op) {
    VPUX_THROW_WHEN(!op.getKind(), "Can't find attributes for Operation");
    const auto maybeKind = vpux::config::symbolizeEnum<ConcreteKind>(op.getKind());
    VPUX_THROW_WHEN(!maybeKind.has_value(), "Unsupported attribute kind");
    return maybeKind.value();
}

//
// CompileMethodDebatch
//

void setCompileMethodDebatch(mlir::ModuleOp module);
bool hasCompileMethodDebatch(mlir::ModuleOp module);
void removeCompileMethodDebatch(mlir::ModuleOp module);

//
// PureHostCompileFunc
//

// `PureHostCompileFunc` attribute is used to mark a host function that contains only operations from MLIR
// dialects (tensor, scf, async, memref and arith) and which code is compiled into CPU code
void setPureHostCompileFuncAttribute(mlir::func::FuncOp func);
bool isPureHostCompileFunc(mlir::func::FuncOp func);

// Sets the FunctionToPack attribute on a function with a required target module name.
// The target module name must not be empty.
void setFunctionToPackAttribute(mlir::func::FuncOp func, llvm::StringRef targetModuleName);
// Returns the target module name, or empty string if function should go to top cluster
std::string getFunctionToPackTargetModule(mlir::func::FuncOp func);

// Removes the FunctionToPack attribute from a function.
void removeFunctionToPackAttribute(mlir::func::FuncOp func);

//
// FunctionToPackEntryPoint
//

// Marks a function as the designated entry point when packed into a module.
// This function will be promoted to have an EntryPoint when the module is created.
void setFunctionToPackEntryPointAttribute(mlir::func::FuncOp func);
bool hasFunctionToPackEntryPointAttribute(mlir::func::FuncOp func);
void removeFunctionToPackEntryPointAttribute(mlir::func::FuncOp func);

//
// PackedModule
//

// Sets the PackedModule attribute on a module to indicate it was created via FunctionToPack.
void setPackedModuleAttribute(mlir::ModuleOp module);

// Returns true if the module has the PackedModule attribute.
bool hasPackedModuleAttribute(mlir::ModuleOp module);

/// Removes the PackedModule attribute from a module.
void removePackedModuleAttribute(mlir::ModuleOp module);

}  // namespace vpux::config
