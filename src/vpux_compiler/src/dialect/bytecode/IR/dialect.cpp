//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"
#include "vpux/compiler/dialect/bytecode/IR/attributes.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/buffer.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/conditional.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/conversion.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/external.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/kernel_submission.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/metadata.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/register.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/IR/types.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"

using namespace vpux;

void vpux::bytecode::BytecodeDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/arithmetic.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/comparison.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/conversion.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/conditional.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/bitwise.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/buffer.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/control_flow.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/external.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/kernel_submission.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/metadata.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/section.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/register.cpp.inc>
            >();
    registerAttributes();
    registerTypes();
}

//
// Generated
//

#include <vpux/compiler/dialect/bytecode/dialect.cpp.inc>
