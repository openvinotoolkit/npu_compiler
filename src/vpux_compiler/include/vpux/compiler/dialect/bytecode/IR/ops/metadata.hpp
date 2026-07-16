//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/StringExtras.h>
#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/bytecode/IR/ops/section_fwd.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/bytecode/IR/types.hpp"

namespace vpux::bytecode {

constexpr auto BYTECODE_IO_DESC_OUTPUT_TENSOR_NAMES = "outputTensorNames";
constexpr auto BYTECODE_IO_DESC_NODE_FRIENDLY_NAME = "nodeFriendlyName";
constexpr auto BYTECODE_IO_DESC_SHAPE_FROM_IR_MODEL = "shapeFromIRModel";

}  // namespace vpux::bytecode

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/bytecode/ops/metadata.hpp.inc>
