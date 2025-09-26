//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <vpux_elf/writer.hpp>
#include <vpux_headers/metadata.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/Timing.h>

#include <transformations/utils/utils.hpp>

namespace vpux {
namespace ELFNPU37XX {

std::unique_ptr<elf::NetworkMetadata> constructMetadata(mlir::ModuleOp module, Logger log);
void setResourceRequirement(mlir::ModuleOp moduleOp, elf::NetworkMetadata& metadata);

elf::TensorRef createTensorRef(mlir::Value val, StringRef name);
elf::TensorRef createTensorRef(vpux::NDTypeInterface type, StringRef name);

elf::DType createDType(mlir::Type type);
elf::OVNodeType createOVNodeType(mlir::Type type);

}  // namespace ELFNPU37XX
}  // namespace vpux
