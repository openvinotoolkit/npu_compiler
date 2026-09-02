//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/arithmetic.hpp"
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
#include <vpux/compiler/dialect/bytecode/ops/arithmetic.cpp.inc>

void bytecode::AddI64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::MulI64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::MinI64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::MaxI64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::SubI64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::DivI64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::DivU64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::MinU64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::AddU64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::MaxU64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::MulU64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::RemU64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::SubU64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::RemI64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::AbsI64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto srcReg = getRegisterNumber(getSrc());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, srcReg});
}

void bytecode::AddF64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::SubF64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::MulF64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::DivF64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::RemF64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::MaxF64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::MinF64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::AbsF64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto srcReg = getRegisterNumber(getSrc());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, srcReg});
}

void bytecode::NegF64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto srcReg = getRegisterNumber(getSrc());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, srcReg});
}

void bytecode::CeilF64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto srcReg = getRegisterNumber(getSrc());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, srcReg});
}

void bytecode::FloorF64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto srcReg = getRegisterNumber(getSrc());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, srcReg});
}

void bytecode::RoundF64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto srcReg = getRegisterNumber(getSrc());
    const auto flag = static_cast<uint16_t>(getFlag());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, srcReg, static_cast<int16_t>(flag)});
}
