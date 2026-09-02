//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/conversion.hpp"
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
#include <vpux/compiler/dialect/bytecode/ops/conversion.cpp.inc>

// All conversion ops share the same 2-operand binary layout: opcode + dst + src.
// Each serialize implementation follows this uniform pattern.

#define DEFINE_CONVERT_SERIALIZE(ClassName)                                       \
    void bytecode::ClassName::serialize(vpux::bytecode::BytecodeWriter& writer) { \
        const auto opcode = static_cast<uint16_t>(getOpcode());                   \
        const auto dstReg = getRegisterNumber(getDst());                          \
        const auto srcReg = getRegisterNumber(getSrc());                          \
        writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, srcReg});   \
    }

DEFINE_CONVERT_SERIALIZE(ConvertI8ToF32Op)
DEFINE_CONVERT_SERIALIZE(ConvertI8ToF64Op)
DEFINE_CONVERT_SERIALIZE(ConvertI16ToI8Op)
DEFINE_CONVERT_SERIALIZE(ConvertI16ToF32Op)
DEFINE_CONVERT_SERIALIZE(ConvertI16ToF64Op)
DEFINE_CONVERT_SERIALIZE(ConvertI32ToI8Op)
DEFINE_CONVERT_SERIALIZE(ConvertI32ToI16Op)
DEFINE_CONVERT_SERIALIZE(ConvertI32ToF32Op)
DEFINE_CONVERT_SERIALIZE(ConvertI32ToF64Op)
DEFINE_CONVERT_SERIALIZE(ConvertI64ToI8Op)
DEFINE_CONVERT_SERIALIZE(ConvertI64ToI16Op)
DEFINE_CONVERT_SERIALIZE(ConvertI64ToI32Op)
DEFINE_CONVERT_SERIALIZE(ConvertI64ToF32Op)
DEFINE_CONVERT_SERIALIZE(ConvertI64ToF64Op)
DEFINE_CONVERT_SERIALIZE(ConvertF32ToI8Op)
DEFINE_CONVERT_SERIALIZE(ConvertF32ToI16Op)
DEFINE_CONVERT_SERIALIZE(ConvertF32ToI32Op)
DEFINE_CONVERT_SERIALIZE(ConvertF32ToI64Op)
DEFINE_CONVERT_SERIALIZE(ConvertF32ToF64Op)
DEFINE_CONVERT_SERIALIZE(ConvertF64ToI8Op)
DEFINE_CONVERT_SERIALIZE(ConvertF64ToI16Op)
DEFINE_CONVERT_SERIALIZE(ConvertF64ToI32Op)
DEFINE_CONVERT_SERIALIZE(ConvertF64ToI64Op)
DEFINE_CONVERT_SERIALIZE(ConvertF64ToF32Op)

#undef DEFINE_CONVERT_SERIALIZE
