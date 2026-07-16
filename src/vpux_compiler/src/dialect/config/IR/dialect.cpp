//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/dialect/config/IR/dialect.hpp>
#include <vpux/compiler/dialect/config/IR/ops.hpp>
#include <vpux/compiler/dialect/config/constraints.hpp>
#include <vpux/compiler/dialect/core/IR/dialect.hpp>
#include <vpux/compiler/utils/npu_action_handler.hpp>

using namespace vpux;

void vpux::config::ConfigDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/config/ops.cpp.inc>
            >();

    registerAttributes();
    addInterfaces<ConfigCache>();

#ifdef VPUX_DEVELOPER_BUILD
    auto ctx = getContext();
    ctx->registerActionHandler(NpuActionHandler());
#endif  // VPUX_DEVELOPER_BUILD
}

//
// Generated
//

#include <vpux/compiler/dialect/config/dialect.cpp.inc>
