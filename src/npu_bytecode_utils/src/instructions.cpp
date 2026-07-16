//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/instructions.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>

namespace intel_npu::vm {

namespace {

std::optional<size_t> getInstructionSizeForOperandCount(size_t operandCount) {
    if (operandCount > (std::numeric_limits<size_t>::max() - OPCODE_SIZE) / OPERAND_SIZE) {
        return std::nullopt;
    }
    return OPCODE_SIZE + operandCount * OPERAND_SIZE;
}

std::optional<size_t> decodeRankBasedVariadicInstructionSize(const uint8_t* instructionBegin, size_t baseOperands,
                                                             size_t operandsPerDim, size_t rankOperandIndex) {
    const auto rank = getOperand(instructionBegin, rankOperandIndex);
    if (rank < 0) {
        return std::nullopt;
    }
    const auto checkedRank = static_cast<size_t>(rank);
    if (checkedRank > (std::numeric_limits<size_t>::max() - baseOperands) / operandsPerDim) {
        return std::nullopt;
    }
    return getInstructionSizeForOperandCount(baseOperands + checkedRank * operandsPerDim);
}

}  // namespace

std::optional<size_t> getStaticInstructionSize(OpCode opcode) {
    static const std::unordered_map<OpCode, size_t> staticInstructionSizes = {
            {OpCode::ADD_I64, OPCODE_SIZE + 3 * OPERAND_SIZE},  // opcode + 3 operands
            {OpCode::RET, OPCODE_SIZE},                         // opcode
            {OpCode::SET, OPCODE_SIZE + 2 * OPERAND_SIZE},      // opcode + 2 operands
            {OpCode::SET_IMM,
             OPCODE_SIZE + OPERAND_SIZE + sizeof(int64_t)},            // opcode + 1 16-bit operand + 1 64-bit operand
            {OpCode::ASSERT, OPCODE_SIZE + 2 * OPERAND_SIZE},          // opcode + 2 operands
            {OpCode::MUL_I64, OPCODE_SIZE + 3 * OPERAND_SIZE},         // opcode + 3 operands
            {OpCode::MIN_I64, OPCODE_SIZE + 3 * OPERAND_SIZE},         // opcode + 3 operands
            {OpCode::MAX_I64, OPCODE_SIZE + 3 * OPERAND_SIZE},         // opcode + 3 operands
            {OpCode::CMP_I64, OPCODE_SIZE + 4 * OPERAND_SIZE},         // opcode + 3 operands + 1 imm flag
            {OpCode::CMP_F64, OPCODE_SIZE + 4 * OPERAND_SIZE},         // opcode + 3 operands + 1 imm flag
            {OpCode::SUB_I64, OPCODE_SIZE + 3 * OPERAND_SIZE},         // opcode + 3 operands
            {OpCode::DIV_I64, OPCODE_SIZE + 3 * OPERAND_SIZE},         // opcode + 3 operands
            {OpCode::REM_I64, OPCODE_SIZE + 3 * OPERAND_SIZE},         // opcode + 3 operands
            {OpCode::ABS_I64, OPCODE_SIZE + 2 * OPERAND_SIZE},         // opcode + 2 operands
            {OpCode::AND_64, OPCODE_SIZE + 3 * OPERAND_SIZE},          // opcode + 3 operands
            {OpCode::NOT_64, OPCODE_SIZE + 2 * OPERAND_SIZE},          // opcode + 2 operands
            {OpCode::OR_64, OPCODE_SIZE + 3 * OPERAND_SIZE},           // opcode + 3 operands
            {OpCode::XOR_64, OPCODE_SIZE + 3 * OPERAND_SIZE},          // opcode + 3 operands
            {OpCode::SLL_64, OPCODE_SIZE + 3 * OPERAND_SIZE},          // opcode + 3 operands
            {OpCode::SRL_64, OPCODE_SIZE + 3 * OPERAND_SIZE},          // opcode + 3 operands
            {OpCode::SRA_64, OPCODE_SIZE + 3 * OPERAND_SIZE},          // opcode + 3 operands
            {OpCode::SELECT, OPCODE_SIZE + 4 * OPERAND_SIZE},          // opcode + 4 operands
            {OpCode::BUFFER_GET_DIM, OPCODE_SIZE + 3 * OPERAND_SIZE},  // opcode + 3 operands
            {OpCode::CMD_LIST_CREATE, OPCODE_SIZE + OPERAND_SIZE},     // opcode + 1 operand
            {OpCode::CMD_LIST_CLOSE, OPCODE_SIZE + OPERAND_SIZE},      // opcode + 1 operand
            {OpCode::CMD_LIST_EXEC, OPCODE_SIZE + 2 * OPERAND_SIZE},   // opcode + 2 operand
            {OpCode::ADD_F64, OPCODE_SIZE + 3 * OPERAND_SIZE},         // opcode + 3 operands
            {OpCode::SUB_F64, OPCODE_SIZE + 3 * OPERAND_SIZE},         // opcode + 3 operands
            {OpCode::MUL_F64, OPCODE_SIZE + 3 * OPERAND_SIZE},         // opcode + 3 operands
            {OpCode::DIV_F64, OPCODE_SIZE + 3 * OPERAND_SIZE},         // opcode + 3 operands
            {OpCode::REM_F64, OPCODE_SIZE + 3 * OPERAND_SIZE},         // opcode + 3 operands
            {OpCode::MAX_F64, OPCODE_SIZE + 3 * OPERAND_SIZE},         // opcode + 3 operands
            {OpCode::MIN_F64, OPCODE_SIZE + 3 * OPERAND_SIZE},         // opcode + 3 operands
            {OpCode::ABS_F64, OPCODE_SIZE + 2 * OPERAND_SIZE},         // opcode + 2 operands
            {OpCode::NEG_F64, OPCODE_SIZE + 2 * OPERAND_SIZE},         // opcode + 2 operands
            {OpCode::CEIL_F64, OPCODE_SIZE + 2 * OPERAND_SIZE},        // opcode + 2 operands
            {OpCode::FLOOR_F64, OPCODE_SIZE + 2 * OPERAND_SIZE},       // opcode + 2 operands
            {OpCode::ROUND_F64, OPCODE_SIZE + 3 * OPERAND_SIZE},       // opcode + 2 operands + 1 imm flag
            {OpCode::JMP, OPCODE_SIZE + sizeof(int64_t)},              // opcode + imm64(8)
            {OpCode::JE, OPCODE_SIZE + sizeof(int64_t) + 2 * OPERAND_SIZE},   // opcode + imm64(8) + lhs(2) + rhs(2)
            {OpCode::JNE, OPCODE_SIZE + sizeof(int64_t) + 2 * OPERAND_SIZE},  // opcode + imm64(8) + lhs(2) + rhs(2)
            {OpCode::DIV_U64, OPCODE_SIZE + 3 * OPERAND_SIZE},                // opcode + 3 operands
            {OpCode::MIN_U64, OPCODE_SIZE + 3 * OPERAND_SIZE},                // opcode + 3 operands
            {OpCode::ADD_U64, OPCODE_SIZE + 3 * OPERAND_SIZE},                // opcode + 3 operands
            {OpCode::MAX_U64, OPCODE_SIZE + 3 * OPERAND_SIZE},                // opcode + 3 operands
            {OpCode::MUL_U64, OPCODE_SIZE + 3 * OPERAND_SIZE},                // opcode + 3 operands
            {OpCode::REM_U64, OPCODE_SIZE + 3 * OPERAND_SIZE},                // opcode + 3 operands
            {OpCode::SUB_U64, OPCODE_SIZE + 3 * OPERAND_SIZE},                // opcode + 3 operands
            {OpCode::CONVERT_I8_TO_F32, OPCODE_SIZE + 2 * OPERAND_SIZE},      // opcode + dst + src
            {OpCode::CONVERT_I8_TO_F64, OPCODE_SIZE + 2 * OPERAND_SIZE},      // opcode + dst + src
            {OpCode::CONVERT_I16_TO_I8, OPCODE_SIZE + 2 * OPERAND_SIZE},      // opcode + dst + src
            {OpCode::CONVERT_I16_TO_F32, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_I16_TO_F64, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_I32_TO_I8, OPCODE_SIZE + 2 * OPERAND_SIZE},      // opcode + dst + src
            {OpCode::CONVERT_I32_TO_I16, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_I32_TO_F32, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_I32_TO_F64, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_I64_TO_I8, OPCODE_SIZE + 2 * OPERAND_SIZE},      // opcode + dst + src
            {OpCode::CONVERT_I64_TO_I16, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_I64_TO_I32, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_I64_TO_F32, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_I64_TO_F64, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_F32_TO_I8, OPCODE_SIZE + 2 * OPERAND_SIZE},      // opcode + dst + src
            {OpCode::CONVERT_F32_TO_I16, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_F32_TO_I32, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_F32_TO_I64, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_F32_TO_F64, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_F64_TO_I8, OPCODE_SIZE + 2 * OPERAND_SIZE},      // opcode + dst + src
            {OpCode::CONVERT_F64_TO_I16, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_F64_TO_I32, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_F64_TO_I64, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
            {OpCode::CONVERT_F64_TO_F32, OPCODE_SIZE + 2 * OPERAND_SIZE},     // opcode + dst + src
    };

    const auto it = staticInstructionSizes.find(opcode);
    if (it != staticInstructionSizes.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<size_t> getVariadicInstructionSize(OpCode opcode, const uint8_t* instructionBegin,
                                                 size_t availableBytes) {
    switch (opcode) {
    case OpCode::RETV: {
        const auto numResults = getOperand(instructionBegin, 0);
        if (numResults < 0) {
            return std::nullopt;
        }
        return getInstructionSizeForOperandCount(1 + static_cast<size_t>(numResults));
    }
    case OpCode::BUFFER_CREATE:
        return decodeRankBasedVariadicInstructionSize(instructionBegin, /*baseOperands=*/3, /*operandsPerDim=*/2,
                                                      /*rankOperandIndex=*/2);
    case OpCode::BUFFER_SUBVIEW:
        return decodeRankBasedVariadicInstructionSize(instructionBegin, /*baseOperands=*/3, /*operandsPerDim=*/3,
                                                      /*rankOperandIndex=*/2);
    case OpCode::BUFFER_VIEW:
        return decodeRankBasedVariadicInstructionSize(instructionBegin, /*baseOperands=*/5, /*operandsPerDim=*/2,
                                                      /*rankOperandIndex=*/4);
    case OpCode::BUFFER_STORE:
        return decodeRankBasedVariadicInstructionSize(instructionBegin, /*baseOperands=*/3, /*operandsPerDim=*/1,
                                                      /*rankOperandIndex=*/2);
    case OpCode::KERNEL_CREATE: {
        // Binary layout: [opcode:2] [dst:2] [kidx:2] [kNameIdx:2] [N:2] [N×input:2] [M:2] [M×output:2]
        // getInstructionSizeDecodeByteSize guarantees opcode+dst+kidx+kNameIdxN (10 bytes) are readable.
        const auto n = getOperand(instructionBegin, 3);
        if (n < 0) {
            return std::nullopt;
        }
        const auto nChecked = static_cast<size_t>(n);
        // M is at operand index (4 + N); verify it fits within the available buffer.
        const auto mByteOffset = OPCODE_SIZE + (4 + nChecked) * OPERAND_SIZE;
        if (availableBytes < mByteOffset + OPERAND_SIZE) {
            return std::nullopt;
        }
        const auto m = getOperand(instructionBegin, 4 + nChecked);
        if (m < 0) {
            return std::nullopt;
        }
        const auto mChecked = static_cast<size_t>(m);
        // Overflow-safe total: (6 + N + M) * sizeof(int16_t)
        constexpr size_t FIXED_OPERANDS = 6;  // opcode word + dst + kidx + kNameIndex + N + M
        if (nChecked > std::numeric_limits<size_t>::max() - mChecked ||
            nChecked + mChecked > std::numeric_limits<size_t>::max() - FIXED_OPERANDS) {
            return std::nullopt;
        }
        return (FIXED_OPERANDS + nChecked + mChecked) * sizeof(int16_t);
    }
    case OpCode::CMD_LIST_ADD_KERNEL: {
        // Layout: rs1, rs2, N, rN..., M, rM...
        const auto numSignalEvents = getOperand(instructionBegin, 2);
        if (numSignalEvents < 0) {
            return std::nullopt;
        }

        const auto checkedNumSignalEvents = static_cast<size_t>(numSignalEvents);
        if (checkedNumSignalEvents > std::numeric_limits<size_t>::max() - 4) {
            return std::nullopt;
        }

        const auto waitEventsCountOperandIndex = static_cast<size_t>(3) + checkedNumSignalEvents;
        // Verify we can read the numWaitEvents operand
        const auto mByteOffset = OPCODE_SIZE + waitEventsCountOperandIndex * OPERAND_SIZE;
        if (availableBytes < mByteOffset + OPERAND_SIZE) {
            return std::nullopt;
        }
        const auto numWaitEvents = getOperand(instructionBegin, waitEventsCountOperandIndex);
        if (numWaitEvents < 0) {
            return std::nullopt;
        }

        const auto checkedNumWaitEvents = static_cast<size_t>(numWaitEvents);
        if (checkedNumWaitEvents > std::numeric_limits<size_t>::max() - (4 + checkedNumSignalEvents)) {
            return std::nullopt;
        }

        const auto totalOperandCount = static_cast<size_t>(4) + checkedNumSignalEvents + checkedNumWaitEvents;
        return getInstructionSizeForOperandCount(totalOperandCount);
    }
    case OpCode::CALL: {
        return intel_npu::vm::getCallInstructionSize(instructionBegin, availableBytes);
    }
    default:
        return std::nullopt;
    }
}

std::optional<size_t> getInstructionSize(OpCode opcode, const uint8_t* instructionBegin, size_t availableBytes) {
    if (const auto staticInstructionSize = getStaticInstructionSize(opcode); staticInstructionSize.has_value()) {
        return staticInstructionSize;
    }
    return getVariadicInstructionSize(opcode, instructionBegin, availableBytes);
}

std::optional<size_t> getInstructionSizeDecodeByteSize(OpCode opcode) {
    if (const auto staticInstructionSize = getStaticInstructionSize(opcode); staticInstructionSize.has_value()) {
        return staticInstructionSize;
    }

    switch (opcode) {
    case OpCode::RETV:
        return OPCODE_SIZE + OPERAND_SIZE;
    case OpCode::BUFFER_CREATE:
        return OPCODE_SIZE + 3 * OPERAND_SIZE;
    case OpCode::BUFFER_SUBVIEW:
    case OpCode::BUFFER_STORE:
        return OPCODE_SIZE + 3 * OPERAND_SIZE;
    case OpCode::BUFFER_VIEW:
        return OPCODE_SIZE + 5 * OPERAND_SIZE;
    case OpCode::KERNEL_CREATE:
        // Minimum bytes to read N (input count): opcode + dst + kidx + kNameIdx + N
        return OPCODE_SIZE + 4 * OPERAND_SIZE;
    case OpCode::CMD_LIST_ADD_KERNEL:
        return OPCODE_SIZE + 3 * OPERAND_SIZE;  // rs1, rs2, N
    case OpCode::CALL:
        // CALL decode needs rs and N before full size can be derived
        return OPCODE_SIZE + 2 * OPERAND_SIZE;
    default:
        return std::nullopt;
    }
}

}  // namespace intel_npu::vm
