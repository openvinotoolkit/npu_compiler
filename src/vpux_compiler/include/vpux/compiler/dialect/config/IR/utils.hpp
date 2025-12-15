//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/StringExtras.h>
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <optional>

//
// Run-time resources
//
namespace vpux {

namespace config {
llvm::StringLiteral getMemoryDerateAttrName();
llvm::StringLiteral getMemoryBandwidthAttrName();

//
// ArchKind
//
void setArch(mlir::ModuleOp module, std::optional<config::Platform> platform, config::ArchKind kind, int numOfDPUGroups,
             std::optional<int> numOfDMAPorts = std::nullopt,
             std::optional<vpux::Byte> availableCMXMemory = std::nullopt, bool allowCustomValues = false);

config::ArchKind getArch(mlir::Operation* op);
bool isArchVPUX3XXX(config::ArchKind arch);
bool isArchVPUX5XXX(config::ArchKind arch);

//
// RevisionID
//

void setRevisionID(mlir::ModuleOp module, config::RevisionID revisionID);
bool hasRevisionID(mlir::ModuleOp module);
config::RevisionID getRevisionID(mlir::Operation* op);

}  // namespace config
}  // namespace vpux
