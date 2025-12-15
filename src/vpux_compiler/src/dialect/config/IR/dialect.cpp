//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/dialect/config/IR/dialect.hpp>
#include <vpux/compiler/dialect/config/IR/ops.hpp>
#include <vpux/compiler/dialect/config/constraints.hpp>
#include <vpux/compiler/dialect/core/IR/dialect.hpp>

using namespace vpux;

void vpux::config::ConfigDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/config/ops.cpp.inc>
            >();

    registerAttributes();
    addInterfaces<ConfigCache>();
}

//
// Generated
//

#include <vpux/compiler/dialect/config/dialect.cpp.inc>
