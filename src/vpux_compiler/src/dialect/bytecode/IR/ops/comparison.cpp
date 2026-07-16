//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/comparison.hpp"
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
#include <vpux/compiler/dialect/bytecode/ops/comparison.cpp.inc>

void bytecode::CmpI64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    const auto flag = static_cast<uint16_t>(getFlag());
    writer.appendInstruction(opcode, addrMode,
                             SmallVector<int16_t>{dstReg, lhsReg, rhsReg, static_cast<int16_t>(flag)});
}

void bytecode::CmpF64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    const auto flag = static_cast<uint16_t>(getFlag());
    writer.appendInstruction(opcode, addrMode,
                             SmallVector<int16_t>{dstReg, lhsReg, rhsReg, static_cast<int16_t>(flag)});
}
