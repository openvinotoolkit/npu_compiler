//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/types.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/NPU60XX/dialect/NPUReg60XX/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// register Types
//

void vpux::NPUReg60XX::NPUReg60XXDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/NPU60XX/dialect/NPUReg60XX/types.cpp.inc>
#undef GET_TYPEDEF_LIST
            >();
}
