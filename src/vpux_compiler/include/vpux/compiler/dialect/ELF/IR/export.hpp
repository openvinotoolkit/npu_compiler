//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/compiler.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <vpux_elf/writer.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/Timing.h>

#include <string>

namespace vpux::ELF {

std::vector<uint8_t> exportToELF(mlir::ModuleOp module, Logger log = Logger::global());

BlobView exportToELF(mlir::ModuleOp module, BlobAllocator& allocator, std::string& compatibilityString,
                     Logger log = Logger::global());

}  // namespace vpux::ELF
