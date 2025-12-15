//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/IR/ops.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/unified_func_inliner_interface.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Transforms/InliningUtils.h>

using namespace vpux;

//
// initialize
//

void vpux::HostExec::HostExecDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/HostExec/ops.cpp.inc>
            >();

    registerAttributes();
}

//
// Generated
//

#include <vpux/compiler/dialect/HostExec/dialect.cpp.inc>
