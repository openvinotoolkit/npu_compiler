//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/buffer.hpp"
#include "npu_bytecode_utils/instructions.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/utils/bytecode_writer.hpp"
#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>

#include <cstdint>
#include <limits>

using namespace vpux;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/bytecode/ops/buffer.cpp.inc>

namespace {

constexpr size_t BUFFER_CREATE_BASE_OPERAND_COUNT = 3;   // dst, elem_type, rank
constexpr size_t BUFFER_SUBVIEW_BASE_OPERAND_COUNT = 3;  // dst, src, rank
constexpr size_t BUFFER_VIEW_BASE_OPERAND_COUNT = 5;     // dst, src, byte_offset, elem_type, rank
constexpr size_t BUFFER_STORE_BASE_OPERAND_COUNT = 3;    // buffer, value, rank
constexpr size_t BUFFER_LOAD_BASE_OPERAND_COUNT = 3;     // dst, buffer, rank
void appendRegisterOperands(SmallVector<int16_t>& dst, mlir::ValueRange operands) {
    for (auto operand : operands) {
        dst.push_back(bytecode::getRegisterNumber(operand));
    }
}

mlir::LogicalResult verifyEncodedRank(mlir::Operation* op, size_t rank) {
    if (rank > static_cast<size_t>(intel_npu::vm::BufferType::MAX_RANK)) {
        return op->emitOpError("expected rank to fit in the instruction encoding (max ")
               << intel_npu::vm::BufferType::MAX_RANK << "), got " << rank;
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult bytecode::BufferCreateOp::verify() {
    if (getShape().size() != getStrides().size()) {
        return emitOpError("expected shape and strides to have the same number of registers, got ")
               << getShape().size() << " and " << getStrides().size();
    }
    return verifyEncodedRank(getOperation(), getShape().size());
}

size_t bytecode::BufferCreateOp::getBinarySize() {
    const auto numOperands = BUFFER_CREATE_BASE_OPERAND_COUNT + getShape().size() + getStrides().size();
    return intel_npu::vm::OPCODE_SIZE + numOperands * intel_npu::vm::OPERAND_SIZE;
}

void bytecode::BufferCreateOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    // Resolve the element-type symbol to its positional index in the type section.
    auto parentModule = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto typeSectionOps = parentModule.getOps<bytecode::TypeSectionOp>();
    VPUX_THROW_UNLESS(std::distance(typeSectionOps.begin(), typeSectionOps.end()) == 1,
                      "Expected exactly one TypeSectionOp in the module");
    const auto typeIndexMap = bytecode::buildTypeIndexMap(*typeSectionOps.begin());
    const auto elemTypeName = getElemTypeAttr().getLeafReference().getValue();
    auto it = typeIndexMap.find(elemTypeName);
    VPUX_THROW_UNLESS(it != typeIndexMap.end(), "Element type '@{0}' not found in type section", elemTypeName);
    const auto elemTypeIdx = it->second;
    VPUX_THROW_UNLESS(elemTypeIdx <= static_cast<uint64_t>(std::numeric_limits<int16_t>::max()),
                      "Element type index {0} exceeds int16_t range", elemTypeIdx);

    SmallVector<int16_t> operands;
    operands.push_back(getRegisterNumber(getDst()));
    operands.push_back(static_cast<int16_t>(elemTypeIdx));
    operands.push_back(checked_cast<int16_t>(getShape().size()));
    appendRegisterOperands(operands, getShape());
    appendRegisterOperands(operands, getStrides());
    writer.appendInstruction(static_cast<uint16_t>(getOpcode()), operands);
}

mlir::LogicalResult bytecode::BufferCreateOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTables) {
    auto elemTypeAttr = getElemTypeAttr();
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    if (!module) {
        return emitOpError("expected to be inside a module");
    }
    auto* resolved = symbolTables.lookupSymbolIn(module, elemTypeAttr);
    if (!resolved) {
        return emitOpError() << "references undefined symbol " << elemTypeAttr;
    }
    if (!mlir::isa<bytecode::TypeOp>(resolved)) {
        return emitOpError() << "symbol " << elemTypeAttr << " does not reference a bytecode.type op";
    }
    return mlir::success();
}

void bytecode::BufferGetDimOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    writer.appendInstruction(static_cast<uint16_t>(getOpcode()),
                             SmallVector<int16_t>{getRegisterNumber(getDst()), getRegisterNumber(getLhs()),
                                                  getRegisterNumber(getRhs())});
}

mlir::LogicalResult bytecode::BufferSubviewOp::verify() {
    if (getOffsets().size() != getSizes().size() || getOffsets().size() != getStrides().size()) {
        return emitOpError("expected offsets, sizes and strides to have the same number of registers, got ")
               << getOffsets().size() << ", " << getSizes().size() << " and " << getStrides().size();
    }

    const auto rank = getOffsets().size();
    return verifyEncodedRank(getOperation(), rank);
}

size_t bytecode::BufferSubviewOp::getBinarySize() {
    const auto numOperands =
            BUFFER_SUBVIEW_BASE_OPERAND_COUNT + getOffsets().size() + getSizes().size() + getStrides().size();
    return intel_npu::vm::OPCODE_SIZE + numOperands * intel_npu::vm::OPERAND_SIZE;
}

void bytecode::BufferSubviewOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    SmallVector<int16_t> operands;
    operands.push_back(getRegisterNumber(getDst()));
    operands.push_back(getRegisterNumber(getSrc()));
    operands.push_back(checked_cast<int16_t>(getOffsets().size()));
    appendRegisterOperands(operands, getOffsets());
    appendRegisterOperands(operands, getSizes());
    appendRegisterOperands(operands, getStrides());

    writer.appendInstruction(static_cast<uint16_t>(getOpcode()), operands);
}

mlir::LogicalResult bytecode::BufferViewOp::verify() {
    if (getShape().size() != getStrides().size()) {
        return emitOpError("expected shape and strides to have the same number of registers, got ")
               << getShape().size() << " and " << getStrides().size();
    }
    return verifyEncodedRank(getOperation(), getShape().size());
}

size_t bytecode::BufferViewOp::getBinarySize() {
    const auto numOperands = BUFFER_VIEW_BASE_OPERAND_COUNT + getShape().size() + getStrides().size();
    return intel_npu::vm::OPCODE_SIZE + numOperands * intel_npu::vm::OPERAND_SIZE;
}

void bytecode::BufferViewOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    auto parentModule = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto typeSectionOps = parentModule.getOps<bytecode::TypeSectionOp>();
    VPUX_THROW_UNLESS(std::distance(typeSectionOps.begin(), typeSectionOps.end()) == 1,
                      "Expected exactly one TypeSectionOp in the module");
    const auto typeIndexMap = bytecode::buildTypeIndexMap(*typeSectionOps.begin());
    const auto elemTypeName = getElemTypeAttr().getLeafReference().getValue();
    auto it = typeIndexMap.find(elemTypeName);
    VPUX_THROW_UNLESS(it != typeIndexMap.end(), "Element type '@{0}' not found in type section", elemTypeName);
    const auto elemTypeIdx = it->second;
    VPUX_THROW_UNLESS(elemTypeIdx <= static_cast<uint64_t>(std::numeric_limits<int16_t>::max()),
                      "Element type index {0} exceeds int16_t range", elemTypeIdx);

    SmallVector<int16_t> operands;
    operands.push_back(getRegisterNumber(getDst()));
    operands.push_back(getRegisterNumber(getSrc()));
    operands.push_back(getRegisterNumber(getByteOffset()));
    operands.push_back(static_cast<int16_t>(elemTypeIdx));
    operands.push_back(checked_cast<int16_t>(getShape().size()));
    appendRegisterOperands(operands, getShape());
    appendRegisterOperands(operands, getStrides());
    writer.appendInstruction(static_cast<uint16_t>(getOpcode()), operands);
}

mlir::LogicalResult bytecode::BufferViewOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTables) {
    auto elemTypeAttr = getElemTypeAttr();
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    if (!module) {
        return emitOpError("expected to be inside a module");
    }
    auto* resolved = symbolTables.lookupSymbolIn(module, elemTypeAttr);
    if (!resolved) {
        return emitOpError() << "references undefined symbol " << elemTypeAttr;
    }
    if (!mlir::isa<bytecode::TypeOp>(resolved)) {
        return emitOpError() << "symbol " << elemTypeAttr << " does not reference a bytecode.type op";
    }
    return mlir::success();
}

mlir::LogicalResult bytecode::BufferStoreOp::verify() {
    return verifyEncodedRank(getOperation(), getIndices().size());
}

size_t bytecode::BufferStoreOp::getBinarySize() {
    const auto numOperands = BUFFER_STORE_BASE_OPERAND_COUNT + getIndices().size();
    return intel_npu::vm::OPCODE_SIZE + numOperands * intel_npu::vm::OPERAND_SIZE;
}

void bytecode::BufferStoreOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    SmallVector<int16_t> operands;
    operands.push_back(getRegisterNumber(getBuffer()));
    operands.push_back(getRegisterNumber(getValue()));
    operands.push_back(checked_cast<int16_t>(getIndices().size()));
    appendRegisterOperands(operands, getIndices());

    writer.appendInstruction(static_cast<uint16_t>(getOpcode()), operands);
}

mlir::LogicalResult bytecode::BufferLoadOp::verify() {
    return verifyEncodedRank(getOperation(), getIndices().size());
}

size_t bytecode::BufferLoadOp::getBinarySize() {
    const auto numOperands = BUFFER_LOAD_BASE_OPERAND_COUNT + getIndices().size();
    return intel_npu::vm::OPCODE_SIZE + numOperands * intel_npu::vm::OPERAND_SIZE;
}

void bytecode::BufferLoadOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    SmallVector<int16_t> operands;
    operands.push_back(getRegisterNumber(getDst()));
    operands.push_back(getRegisterNumber(getBuffer()));
    operands.push_back(checked_cast<int16_t>(getIndices().size()));
    appendRegisterOperands(operands, getIndices());

    writer.appendInstruction(static_cast<uint16_t>(getOpcode()), operands);
}
