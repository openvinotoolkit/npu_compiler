//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Bytecode/BytecodeImplementation.h>
#include <mlir/IR/Attributes.h>

#include <cstdint>
#include <optional>

namespace vpux {

mlir::LogicalResult convertFromAttribute(std::optional<int64_t>& prop, mlir::Attribute attr,
                                         llvm::function_ref<mlir::InFlightDiagnostic()> emitError);
mlir::Attribute convertToAttribute(mlir::MLIRContext* ctx, const std::optional<int64_t>& prop);
mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader&, std::optional<int64_t>& prop);
void writeToMlirBytecode(mlir::DialectBytecodeWriter&, const std::optional<int64_t>& prop);

mlir::LogicalResult convertFromAttribute(std::optional<bool>& prop, mlir::Attribute attr,
                                         llvm::function_ref<mlir::InFlightDiagnostic()> emitError);
mlir::Attribute convertToAttribute(mlir::MLIRContext* ctx, const std::optional<bool>& prop);
mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader&, std::optional<bool>& prop);
void writeToMlirBytecode(mlir::DialectBytecodeWriter&, const std::optional<bool>& prop);

mlir::LogicalResult convertFromAttribute(SmallVector<uint8_t>& storage, mlir::Attribute attr,
                                         llvm::function_ref<mlir::InFlightDiagnostic()>);
mlir::Attribute convertToAttribute(mlir::MLIRContext* ctx, ArrayRef<uint8_t> storage);
llvm::hash_code hash_value(ArrayRef<uint8_t> storage);

}  // namespace vpux
