//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/register.hpp"
#include "npu_bytecode_utils/instructions.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/external.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/IR/types.hpp"
#include "vpux/compiler/dialect/bytecode/utils/bytecode_writer.hpp"
#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>

#include <cstdint>

using namespace vpux;

namespace {

template <typename FuncOpT>
std::optional<uint64_t> getFunctionIndex(mlir::SymbolRefAttr calleeAttr, mlir::ModuleOp parentModule) {
    return bytecode::getIndex<bytecode::FuncSectionOp, FuncOpT>(calleeAttr, parentModule);
}

std::optional<uint64_t> getFunctionIndex(mlir::SymbolRefAttr calleeAttr, mlir::ModuleOp parentModule) {
    if (auto funcIdx = getFunctionIndex<bytecode::FuncOp>(calleeAttr, parentModule)) {
        return funcIdx;
    }
    return getFunctionIndex<bytecode::ExtFuncOp>(calleeAttr, parentModule);
}

}  // namespace

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/bytecode/ops/register.cpp.inc>

void bytecode::VirtualGeneralRegisterOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState) {
    build(odsBuilder, odsState, bytecode::RegisterType::get(odsBuilder.getContext()));
}

void bytecode::VirtualParameterRegisterOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                                 uint16_t paramIndex) {
    build(odsBuilder, odsState, bytecode::RegisterType::get(odsBuilder.getContext()), paramIndex);
}

void bytecode::SetOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto srcReg = getRegisterNumber(getSrc());
    writer.appendInstruction(opcode, SmallVector<int16_t>{dstReg, srcReg});
}

void bytecode::SetImmOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    const auto immValue = getImmValueAttr().getInt();
    SmallVector<uint8_t> operandBytes;
    operandBytes.insert(operandBytes.end(), reinterpret_cast<const uint8_t*>(&dstReg),
                        reinterpret_cast<const uint8_t*>(&dstReg) + sizeof(dstReg));
    operandBytes.insert(operandBytes.end(), reinterpret_cast<const uint8_t*>(&immValue),
                        reinterpret_cast<const uint8_t*>(&immValue) + sizeof(immValue));
    writer.appendInstruction(opcode, operandBytes);
}

void bytecode::SetImmIdxOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    auto parentModule = getOperation()->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(parentModule != nullptr, "Expected set_imm_idx op to be inside a module");

    const auto calleeIdxOpt = getFunctionIndex(getCalleeAttr(), parentModule);
    VPUX_THROW_UNLESS(calleeIdxOpt.has_value(), "Failed to resolve symbol '{0}' in bytecode.func_section",
                      getCalleeAttr());
    const auto immValue = static_cast<int64_t>(calleeIdxOpt.value());

    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());
    SmallVector<uint8_t> operandBytes;
    operandBytes.insert(operandBytes.end(), reinterpret_cast<const uint8_t*>(&dstReg),
                        reinterpret_cast<const uint8_t*>(&dstReg) + sizeof(dstReg));
    operandBytes.insert(operandBytes.end(), reinterpret_cast<const uint8_t*>(&immValue),
                        reinterpret_cast<const uint8_t*>(&immValue) + sizeof(immValue));
    writer.appendInstruction(opcode, operandBytes);
}

mlir::LogicalResult bytecode::SetImmIdxOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTables) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    if (!module) {
        return emitOpError("expected to be inside a module");
    }

    auto calleeAttr = getCalleeAttr();
    auto* resolved = symbolTables.lookupSymbolIn(module, calleeAttr);
    if (!resolved) {
        return emitOpError() << "callee '" << calleeAttr << "' not found in bytecode.func_section";
    }
    if (!mlir::isa<bytecode::FuncOp, bytecode::ExtFuncOp>(resolved)) {
        return emitOpError() << "callee '" << calleeAttr
                             << "' does not resolve to a bytecode function in bytecode.func_section";
    }
    return mlir::success();
}
