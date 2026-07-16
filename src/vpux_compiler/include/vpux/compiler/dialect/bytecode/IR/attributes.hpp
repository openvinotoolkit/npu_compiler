//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "npu_bytecode_utils/type_section.hpp"

#include <mlir/IR/BuiltinAttributes.h>

//
// Generated
//

#include <vpux/compiler/dialect/bytecode/enums.hpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/bytecode/attributes.hpp.inc>

namespace vpux::bytecode {

intel_npu::vm::FloatTypeFormat convertToFloatTypeFormat(bytecode::FloatFormat format);

}  // namespace vpux::bytecode
