//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/core/IR/attributes.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

void setCompileMethodDebatch(mlir::ModuleOp module);
bool hasCompileMethodDebatch(mlir::ModuleOp module);

//
// Generated
//

#include <vpux/compiler/dialect/IE/enums.hpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/IE/attributes.hpp.inc>
