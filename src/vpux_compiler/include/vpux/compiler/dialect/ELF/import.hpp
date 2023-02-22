//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/preprocessing.hpp"
#include "vpux_compiler.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/Timing.h>

namespace vpux {
namespace ELF {

mlir::OwningOpRef<mlir::ModuleOp> importELF(mlir::MLIRContext* ctx, const std::string& elfFileName,
                                            Logger log = Logger::global());

}  // namespace ELF
}  // namespace vpux
