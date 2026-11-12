//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/dialect.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/dialect.hpp"

//
// initialize
//

void vpux::NPUReg60XX::NPUReg60XXDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.cpp.inc>
            >();
    registerTypes();
    registerAttributes();
}

//
// Generated
//

#include <vpux/compiler/NPU60XX/dialect/NPUReg60XX/dialect.cpp.inc>
