//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/kernel_submission.hpp"
#include "npu_bytecode_utils/instructions.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/utils/bytecode_writer.hpp"
#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>

#include <limits>

using namespace vpux;

namespace {

void appendRegisterOperands(SmallVector<int16_t>& dst, mlir::ValueRange operands) {
    for (auto operand : operands) {
        dst.push_back(bytecode::getRegisterNumber(operand));
    }
}

}  // namespace

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/bytecode/ops/kernel_submission.cpp.inc>

void bytecode::CmdListCreateOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto regNum = getRegisterNumber(getReg());
    writer.appendInstruction(opcode, SmallVector<int16_t>{regNum});
}

void bytecode::CmdListCloseOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto regNum = getRegisterNumber(getReg());
    writer.appendInstruction(opcode, SmallVector<int16_t>{regNum});
}

void bytecode::CmdListExecOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto regNum = getRegisterNumber(getSrc());
    const auto flag = static_cast<int16_t>(getFlag());
    writer.appendInstruction(opcode, SmallVector<int16_t>{regNum, flag});
}

size_t bytecode::KernelCreateOp::getBinarySize() {
    return (6 + getInputs().size() + getOutputs().size()) * sizeof(int16_t);
}

void bytecode::KernelCreateOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto dstReg = getRegisterNumber(getDst());

    // Resolve the kernel to its positional index in the kernel section using the leaf name.
    auto parentModule = getOperation()->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(parentModule != nullptr, "Expected kernel.create op to be inside a module");
    const auto kernelName = getKernelAttr().getLeafReference();
    const auto kernelSectionRef = mlir::SymbolRefAttr::get(getContext(), bytecode::KERNEL_SECTION_NAME,
                                                           {mlir::FlatSymbolRefAttr::get(kernelName)});
    const auto kernelIdxOpt =
            bytecode::getIndex<bytecode::KernelSectionOp, bytecode::KernelOp>(kernelSectionRef, parentModule);
    VPUX_THROW_UNLESS(kernelIdxOpt.has_value(), "Failed to resolve kernel '{0}' in bytecode.kernel_section",
                      kernelName);
    const auto kernelIdx = kernelIdxOpt.value();
    VPUX_THROW_UNLESS(kernelIdx <= static_cast<uint64_t>(std::numeric_limits<int16_t>::max()),
                      "Kernel index {0} exceeds int16_t range", kernelIdx);

    const auto stringSectionRef = mlir::SymbolRefAttr::get(getContext(), bytecode::STRING_SECTION_NAME,
                                                           {mlir::FlatSymbolRefAttr::get(kernelName)});
    const auto kernelNameIdxOpt =
            bytecode::getIndex<bytecode::StringSectionOp, bytecode::StringOp>(stringSectionRef, parentModule);
    VPUX_THROW_UNLESS(kernelNameIdxOpt.has_value(), "Kernel name '{0}' not found in bytecode.string_section",
                      kernelName);
    const auto kernelNameIdx = kernelNameIdxOpt.value();
    VPUX_THROW_UNLESS(kernelNameIdx <= static_cast<uint64_t>(std::numeric_limits<int16_t>::max()),
                      "Kernel name index {0} exceeds int16_t range", kernelNameIdx);

    const auto numInputs = getInputs().size();
    const auto numOutputs = getOutputs().size();
    VPUX_THROW_UNLESS(numInputs <= static_cast<size_t>(std::numeric_limits<int16_t>::max()),
                      "kernel.create input count {0} exceeds int16_t range", numInputs);
    VPUX_THROW_UNLESS(numOutputs <= static_cast<size_t>(std::numeric_limits<int16_t>::max()),
                      "kernel.create output count {0} exceeds int16_t range", numOutputs);

    SmallVector<int16_t> operands = {dstReg, static_cast<int16_t>(kernelIdx), static_cast<int16_t>(kernelNameIdx),
                                     static_cast<int16_t>(numInputs)};
    for (auto input : getInputs()) {
        operands.push_back(getRegisterNumber(input));
    }
    operands.push_back(static_cast<int16_t>(numOutputs));
    for (auto output : getOutputs()) {
        operands.push_back(getRegisterNumber(output));
    }
    writer.appendInstruction(opcode, operands);
}

mlir::LogicalResult bytecode::KernelCreateOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTables) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    if (!module) {
        return emitOpError("expected to be inside a module");
    }
    auto kernelAttr = getKernelAttr();
    auto* resolved = symbolTables.lookupSymbolIn(module, kernelAttr);
    if (!resolved) {
        return emitOpError() << "kernel '" << kernelAttr.getLeafReference() << "' not found in bytecode.kernel_section";
    }
    if (!mlir::isa<bytecode::KernelOp>(resolved)) {
        return emitOpError() << "kernel '" << kernelAttr.getLeafReference() << "' not found in bytecode.kernel_section";
    }
    return mlir::success();
}

size_t bytecode::CmdListAddKernelOp::getBinarySize() {
    return intel_npu::vm::OPCODE_SIZE +
           (2 + 1 + getSignalEvents().size() + 1 + getWaitEvents().size()) * intel_npu::vm::OPERAND_SIZE;
}

void bytecode::CmdListAddKernelOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    SmallVector<int16_t> operands;
    const auto signalEvents = getSignalEvents();
    const auto waitEvents = getWaitEvents();

    operands.reserve(4 + signalEvents.size() + waitEvents.size());
    operands.push_back(getRegisterNumber(getCmdList()));
    operands.push_back(getRegisterNumber(getKernel()));
    operands.push_back(checked_cast<int16_t>(signalEvents.size()));
    appendRegisterOperands(operands, signalEvents);
    operands.push_back(checked_cast<int16_t>(waitEvents.size()));
    appendRegisterOperands(operands, waitEvents);

    writer.appendInstruction(static_cast<uint16_t>(getOpcode()), operands);
}
