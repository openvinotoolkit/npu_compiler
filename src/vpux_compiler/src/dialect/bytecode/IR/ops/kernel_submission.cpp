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
    const auto addrMode = getAddressingMode();
    const auto regNum = getRegisterNumber(getReg());
    writer.appendInstruction(opcode, addrMode, SmallVector<int16_t>{regNum});
}

void bytecode::CmdListCloseOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();
    const auto regNum = getRegisterNumber(getReg());
    writer.appendInstruction(opcode, addrMode, SmallVector<int16_t>{regNum});
}

void bytecode::CmdListExecOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();

    const auto regNum = getRegisterNumber(getSrc());
    const auto flag = static_cast<int16_t>(getFlag());
    writer.appendInstruction(opcode, addrMode, SmallVector<int16_t>{regNum, flag});
}

size_t bytecode::KernelCreateOp::getBinarySize() {
    return (6 + getInputs().size() + getOutputs().size()) * sizeof(int16_t);
}

void bytecode::KernelCreateOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();
    const auto dstReg = getRegisterNumber(getDst());

    // Resolve the kernel symbol to its positional index in the kernel section.
    const auto kernelName = getKernelAttr().getLeafReference();
    auto parentModule = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto kernelSectionOps = parentModule.getOps<bytecode::KernelSectionOp>();
    auto stringSectionOps = parentModule.getOps<bytecode::StringSectionOp>();
    VPUX_THROW_UNLESS(std::distance(kernelSectionOps.begin(), kernelSectionOps.end()) == 1,
                      "Expected exactly one bytecode.kernel_section in the module");
    VPUX_THROW_UNLESS(std::distance(stringSectionOps.begin(), stringSectionOps.end()) == 1,
                      "Expected exactly one bytecode.string_section in the module");
    auto kernelSection = *kernelSectionOps.begin();
    auto stringSection = *stringSectionOps.begin();

    int64_t kernelIdx = -1;
    int64_t idx = 0;
    for (auto kernelOp : kernelSection.getContent().getOps<bytecode::KernelOp>()) {
        if (kernelOp.getSymName() == kernelName) {
            kernelIdx = idx;
            break;
        }
        ++idx;
    }
    int64_t kernelNameIdx = -1;
    idx = 0;
    for (auto stringOp : stringSection.getContent().getOps<bytecode::StringOp>()) {
        if (stringOp.getSymName() == kernelName) {
            kernelNameIdx = idx;
            break;
        }
        ++idx;
    }
    VPUX_THROW_UNLESS(kernelIdx >= 0, "Kernel '{0}' not found in bytecode.kernel_section", kernelName);
    VPUX_THROW_UNLESS(kernelIdx <= std::numeric_limits<int16_t>::max(), "Kernel index {0} exceeds int16_t range",
                      kernelIdx);
    VPUX_THROW_UNLESS(kernelNameIdx >= 0, "Kernel name '{0}' not found in bytecode.string_section", kernelName);
    VPUX_THROW_UNLESS(kernelNameIdx <= std::numeric_limits<int16_t>::max(),
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
    writer.appendInstruction(opcode, addrMode, operands);
}

mlir::LogicalResult bytecode::KernelCreateOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTables) {
    auto kernelAttr = getKernelAttr();
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto kernelSections = module.getOps<bytecode::KernelSectionOp>();
    if (kernelSections.empty()) {
        return emitOpError() << "no kernel section found in module";
    }
    auto* resolved = symbolTables.lookupNearestSymbolFrom((*kernelSections.begin()).getOperation(), kernelAttr);
    if (!resolved) {
        return emitOpError() << "references undefined symbol " << kernelAttr;
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

    writer.appendInstruction(static_cast<uint16_t>(getOpcode()), getAddressingMode(), operands);
}
