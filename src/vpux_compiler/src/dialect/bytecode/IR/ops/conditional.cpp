//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/conditional.hpp"
#include "npu_bytecode_utils/instructions.hpp"
#include "vpux/compiler/dialect/bytecode/utils/bytecode_writer.hpp"
#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

#include <cstdint>

using namespace vpux;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/bytecode/ops/conditional.cpp.inc>

void bytecode::SelectOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();
    const auto dstReg = getRegisterNumber(getDst());
    const auto condReg = getRegisterNumber(getCond());
    const auto trueReg = getRegisterNumber(getTrueVal());
    const auto falseReg = getRegisterNumber(getFalseVal());
    writer.appendInstruction(opcode, addrMode, SmallVector<int16_t>{dstReg, condReg, trueReg, falseReg});
}
