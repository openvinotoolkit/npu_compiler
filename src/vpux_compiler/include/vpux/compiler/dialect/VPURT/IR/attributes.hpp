//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/IR/BuiltinAttributes.h>

//
// Generated
//

#include <vpux/compiler/dialect/VPURT/enums.hpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPURT/attributes.hpp.inc>
#undef GET_ATTRDEF_CLASSES

//
// BufferSection/MemoryKind conversion
//

namespace vpux {
namespace VPURT {

VPU::MemoryKind getMemoryKind(BufferSection section);
BufferSection getBufferSection(VPU::MemoryKind memKind);

bool isMemoryCompatible(BufferSection section, vpux::NDTypeInterface ndType);

}  // namespace VPURT
}  // namespace vpux
