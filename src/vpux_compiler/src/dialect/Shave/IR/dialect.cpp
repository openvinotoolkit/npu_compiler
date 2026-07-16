//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/dialect/Shave/IR/dialect.hpp>
#include <vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp>
#include "vpux/compiler/dialect/Shave/IR/ops_interfaces.hpp"

using namespace vpux;

void vpux::Shave::ShaveDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/Shave/ops/meta-ops.cpp.inc>
            >();

    registerAttributes();
}

//
// Generated
//

#include <vpux/compiler/dialect/Shave/dialect.cpp.inc>
