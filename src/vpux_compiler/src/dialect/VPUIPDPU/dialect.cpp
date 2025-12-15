//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"

#include "vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"

using namespace vpux;

//
// initialize
//

// TODO: E120294 - remove arch specific references
void vpux::VPUIPDPU::VPUIPDPUDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUIPDPU/ops.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.cpp.inc>
            >();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIPDPU/dialect.cpp.inc>
