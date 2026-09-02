//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/bytecode/utils/bytecode_writer.hpp"
#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <llvm/ADT/STLExtras.h>

#include <cstdint>

using namespace vpux;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/bytecode/ops/control_flow.cpp.inc>

namespace {

// falseDest is an implicit fallthrough — the VM advances the PC without a jump when the
// condition is not taken. This is only correct if falseDest is the physically next block.
mlir::LogicalResult verifyFalseDestIsNextBlock(mlir::Operation* op, mlir::Block* falseDest) {
    auto* nextBlock = op->getBlock()->getNextNode();
    if (nextBlock != falseDest) {
        return op->emitOpError("falseDest must be the physically next block in the region "
                               "(required for not-taken fallthrough during serialization)");
    }
    return mlir::success();
}

// Returns true if any op before `op` in its block implements SerializableOpInterface.
// Used to detect self-loop trueDest that would produce a zero PC-relative offset.
bool hasSerializablePredecessorInBlock(mlir::Operation* op) {
    auto& block = *op->getBlock();
    return llvm::any_of(llvm::make_range(block.begin(), op->getIterator()), [](mlir::Operation& blockOp) {
        return mlir::isa<bytecode::SerializableOpInterface>(&blockOp);
    });
}

// A trueDest self-loop where the jump op is the first serializable op in its block produces
// blockOffset[trueDest] == opOffset[jumpOp], i.e. offset = 0. The VM rejects zero-offset jumps.
mlir::LogicalResult verifyTrueDestNotZeroOffset(mlir::Operation* op, mlir::Block* trueDest) {
    if (trueDest == op->getBlock() && !hasSerializablePredecessorInBlock(op)) {
        return op->emitOpError("trueDest self-loop with no serializable predecessor produces a zero "
                               "PC-relative offset (VM rejects zero-offset jumps)");
    }
    return mlir::success();
}

}  // namespace

void bytecode::RetOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    writer.appendInstruction(opcode, /*operands=*/SmallVector<int16_t>{});
}

void bytecode::RetVOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto regs = getRegs();
    SmallVector<int16_t> regOperands = {static_cast<int16_t>(regs.size())};
    regOperands.reserve(regOperands.size() + regs.size());
    for (auto reg : regs) {
        regOperands.push_back(getRegisterNumber(reg));
    }
    writer.appendInstruction(opcode, regOperands);
}

size_t bytecode::RetVOp::getBinarySize() {
    // opcode + 1 operand for the number of registers + 1 operand per register
    return sizeof(uint16_t) + sizeof(int16_t) + getRegs().size() * sizeof(int16_t);
}

void bytecode::CallOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto resultCount = checked_cast<int16_t>(getResultsDst().size());
    const auto argCount = checked_cast<int16_t>(getArgs().size());

    SmallVector<int16_t> operands;
    operands.push_back(getRegisterNumber(getFuncIdx()));

    // N: number of destination (result) registers
    operands.push_back(resultCount);
    for (auto reg : getResultsDst()) {
        operands.push_back(getRegisterNumber(reg));
    }

    // M: number of argument registers
    operands.push_back(argCount);
    for (auto reg : getArgs()) {
        operands.push_back(getRegisterNumber(reg));
    }

    writer.appendInstruction(opcode, operands);
}

size_t bytecode::CallOp::getBinarySize() {
    // opcode + rs + (N + dest regs) + (M + arg regs)
    return sizeof(uint16_t) + sizeof(int16_t) + sizeof(int16_t) + getResultsDst().size() * sizeof(int16_t) +
           sizeof(int16_t) + getArgs().size() * sizeof(int16_t);
}

void bytecode::AssertOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto conditionReg = getRegisterNumber(getCondition());

    const auto msgIdxOpt =
            bytecode::getIndex<bytecode::StringSectionOp, bytecode::StringOp>(getOperation(), getMsgSym());
    VPUX_THROW_UNLESS(msgIdxOpt.has_value(), "Failed to resolve message symbol '{0}' in string section", getMsgSym());
    const auto msgIdx = checked_cast<int16_t>(msgIdxOpt.value());
    writer.appendInstruction(opcode, SmallVector<int16_t>{conditionReg, msgIdx});
}

mlir::LogicalResult bytecode::JmpOp::verify() {
    return verifyTrueDestNotZeroOffset(getOperation(), getDest());
}

void bytecode::JmpOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto offset = writer.getRelativeOffset(getOperation(), getDest());
    VPUX_THROW_WHEN(offset == 0, "JmpOp cannot have zero offset to target block");
    SmallVector<uint8_t> operandBytes;
    operandBytes.insert(operandBytes.end(), reinterpret_cast<const uint8_t*>(&offset),
                        reinterpret_cast<const uint8_t*>(&offset) + sizeof(offset));
    writer.appendInstruction(static_cast<uint16_t>(getOpcode()), operandBytes);
}

mlir::LogicalResult bytecode::JEOp::verify() {
    if (mlir::failed(verifyFalseDestIsNextBlock(getOperation(), getFalseDest()))) {
        return mlir::failure();
    }
    return verifyTrueDestNotZeroOffset(getOperation(), getTrueDest());
}

void bytecode::JEOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto offset = writer.getRelativeOffset(getOperation(), getTrueDest());
    VPUX_THROW_WHEN(offset == 0, "JEOp cannot have zero offset to target block");
    const auto lhs = getRegisterNumber(getLhs());
    const auto rhs = getRegisterNumber(getRhs());
    SmallVector<uint8_t> operandBytes;
    operandBytes.insert(operandBytes.end(), reinterpret_cast<const uint8_t*>(&offset),
                        reinterpret_cast<const uint8_t*>(&offset) + sizeof(offset));
    operandBytes.insert(operandBytes.end(), reinterpret_cast<const uint8_t*>(&lhs),
                        reinterpret_cast<const uint8_t*>(&lhs) + sizeof(lhs));
    operandBytes.insert(operandBytes.end(), reinterpret_cast<const uint8_t*>(&rhs),
                        reinterpret_cast<const uint8_t*>(&rhs) + sizeof(rhs));
    writer.appendInstruction(static_cast<uint16_t>(getOpcode()), operandBytes);
}

mlir::LogicalResult bytecode::JNEOp::verify() {
    if (mlir::failed(verifyFalseDestIsNextBlock(getOperation(), getFalseDest()))) {
        return mlir::failure();
    }
    return verifyTrueDestNotZeroOffset(getOperation(), getTrueDest());
}

void bytecode::JNEOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto offset = writer.getRelativeOffset(getOperation(), getTrueDest());
    VPUX_THROW_WHEN(offset == 0, "JNEOp cannot have zero offset to target block");
    const auto lhs = getRegisterNumber(getLhs());
    const auto rhs = getRegisterNumber(getRhs());
    SmallVector<uint8_t> operandBytes;
    operandBytes.insert(operandBytes.end(), reinterpret_cast<const uint8_t*>(&offset),
                        reinterpret_cast<const uint8_t*>(&offset) + sizeof(offset));
    operandBytes.insert(operandBytes.end(), reinterpret_cast<const uint8_t*>(&lhs),
                        reinterpret_cast<const uint8_t*>(&lhs) + sizeof(lhs));
    operandBytes.insert(operandBytes.end(), reinterpret_cast<const uint8_t*>(&rhs),
                        reinterpret_cast<const uint8_t*>(&rhs) + sizeof(rhs));
    writer.appendInstruction(static_cast<uint16_t>(getOpcode()), operandBytes);
}
