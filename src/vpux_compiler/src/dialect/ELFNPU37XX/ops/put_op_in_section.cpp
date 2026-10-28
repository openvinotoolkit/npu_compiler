//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

void vpux::ELFNPU37XX::PutOpInSectionOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto inputArgDefOp = getInputArg().getDefiningOp();

    if (llvm::dyn_cast<vpux::VPURT::DeclareBufferOp>(inputArgDefOp) != nullptr) {
        return;
    }

    auto binaryIface = llvm::dyn_cast_or_null<vpux::ELFNPU37XX::BinaryOpInterface>(inputArgDefOp);

    VPUX_THROW_WHEN(binaryIface == nullptr,
                    "PutOpInSectionOp::serialize(): inputArg is expected to define elf::BinaryOpInterface");

    binaryIface.serialize(binDataSection);
}

size_t vpux::ELFNPU37XX::PutOpInSectionOp::getBinarySize() {
    auto inputArgDefOp = getInputArg().getDefiningOp();

    if (llvm::dyn_cast<vpux::VPURT::DeclareBufferOp>(inputArgDefOp) != nullptr) {
        // VPURT::DeclareBufferOp is a simple logical operation with no data to serialize.
        auto ndTypeInterface = mlir::cast<vpux::NDTypeInterface>(getInputArg().getType());
        return ndTypeInterface.getTotalAllocSize().count();
    }

    auto binaryIface = llvm::dyn_cast_or_null<vpux::ELFNPU37XX::BinaryOpInterface>(inputArgDefOp);

    VPUX_THROW_WHEN(binaryIface == nullptr,
                    "PutOpInSectionOp::getBinarySize(): inputArg is expected to define elf::BinaryOpInterface");

    return binaryIface.getBinarySize();
}

vpux::VPURT::BufferSection vpux::ELFNPU37XX::PutOpInSectionOp::getMemorySpace() {
    mlir::Value op = getInputArg();

    vpux::ELFNPU37XX::BinaryOpInterface binaryOp =
            mlir::dyn_cast<vpux::ELFNPU37XX::BinaryOpInterface>(op.getDefiningOp());
    VPUX_THROW_WHEN(binaryOp == nullptr,
                    "PutOpInSectionOp::getMemorySpace(): inputArg is expected to define elf::BinaryOpInterface");

    return binaryOp.getMemorySpace();
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::ELFNPU37XX::PutOpInSectionOp::getAccessingProcs() {
    auto inputArgDefOp = getInputArg().getDefiningOp();
    auto binaryIface = llvm::dyn_cast_or_null<vpux::ELFNPU37XX::BinaryOpInterface>(inputArgDefOp);

    // TODO: Add VPUX_THROW_UNLESS if the op does not implement the BinaryOpInterface once relevant VPURT ops are
    // updated

    if (binaryIface) {
        return binaryIface.getAccessingProcs();
    } else {
        return vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE;
    }
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::ELFNPU37XX::PutOpInSectionOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

size_t vpux::ELFNPU37XX::PutOpInSectionOp::getAlignmentRequirements() {
    return ELFNPU37XX::VPUX_NO_ALIGNMENT;
}
