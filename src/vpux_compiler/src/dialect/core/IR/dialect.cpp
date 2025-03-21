//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/dialect/core/IR/dialect.hpp>
#include <vpux/compiler/dialect/core/IR/ops.hpp>

using namespace vpux;

//
// initialize
//

void vpux::Core::CoreDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/core/ops.cpp.inc>
            >();
}

//
// Generated
//

#include <vpux/compiler/dialect/core/dialect.cpp.inc>
