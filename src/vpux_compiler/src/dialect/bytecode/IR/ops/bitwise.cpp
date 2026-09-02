//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/bitwise.hpp"
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
#include <vpux/compiler/dialect/bytecode/ops/bitwise.cpp.inc>

void bytecode::And64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::Not64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto srcReg = getRegisterNumber(getSrc());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, srcReg});
}

void bytecode::Or64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::Xor64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::Sll64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::Srl64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::Sra64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}
