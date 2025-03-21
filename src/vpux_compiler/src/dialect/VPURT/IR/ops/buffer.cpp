//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, int64_t byteOffset) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section), /*sectionIndex=*/nullptr,
          getIntAttr(builder, byteOffset), /*swizzlingKey=*/nullptr);
}

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, ArrayRef<int64_t> sectionIndex,
                                         int64_t byteOffset) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section),
          getIntArrayAttr(builder, sectionIndex), getIntAttr(builder, byteOffset), /*swizzlingKey=*/nullptr);
}

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, int64_t sectionIndex, int64_t byteOffset) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section),
          getIntArrayAttr(builder, ArrayRef({sectionIndex})), getIntAttr(builder, byteOffset),
          /*swizzlingKey=*/nullptr);
}

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, ArrayRef<int64_t> sectionIndex,
                                         int64_t byteOffset, int64_t swizzlingKey) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section),
          getIntArrayAttr(builder, sectionIndex), getIntAttr(builder, byteOffset), getIntAttr(builder, swizzlingKey));
}

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, int64_t sectionIndex, int64_t byteOffset,
                                         int64_t swizzlingKey) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section),
          getIntArrayAttr(builder, ArrayRef({sectionIndex})), getIntAttr(builder, byteOffset),
          getIntAttr(builder, swizzlingKey));
}

mlir::LogicalResult vpux::VPURT::DeclareBufferOp::verify() {
    const auto op = getOperation();
    const auto type = getType().cast<vpux::NDTypeInterface>();
    const auto opSection = getSection();

    if (!VPURT::isMemoryCompatible(opSection, type)) {
        return errorAt(op, "BufferSection '{0}' is not compatible with memory space '{1}'", opSection,
                       type.getMemSpace());
    }

    const auto maybeSectionIndex = getSectionIndex();
    if (maybeSectionIndex.has_value()) {
        if (maybeSectionIndex.value().empty()) {
            return errorAt(op, "Empty section index is not supported");
        }
    }

    if (getSection() == VPURT::BufferSection::CMX_NN) {
        const auto checkSectionIndex = [&op, &type](ArrayRef<int64_t> sectionIdx) {
            if (auto distributedType = type.dyn_cast<VPUIP::DistributedBufferType>()) {
                const auto distribution = distributedType.getDistribution();
                const auto numClusters = checked_cast<size_t>(distribution.getNumClusters().getInt());
                if (numClusters != sectionIdx.size()) {
                    return errorAt(op, "Number of clusters '{0}' and section indexes '{1}' mismatch for op = {2}",
                                   numClusters, sectionIdx.size(), op);
                }
            }

            if (sectionIdx.size() == 1) {
                const auto memSpace = type.getMemSpace();
                if (memSpace == nullptr) {
                    return errorAt(op, "Output type must have CMX_NN memory space, op = {0}", op);
                }

                const auto maybeIdx = memSpace.getIndex();
                if (!maybeIdx.has_value()) {
                    return errorAt(op,
                                   "Output type must have memory space index equal to '{0}', but it doesn't. declare "
                                   "buffer op = {1}",
                                   sectionIdx[0], op);
                }

                const auto memSpaceIdx = maybeIdx.value();
                if (memSpaceIdx != sectionIdx[0]) {
                    return errorAt(op, "Section index '{0}' and memory space index '{1}' mismatch for op = {2}",
                                   sectionIdx[0], memSpaceIdx, op);
                }
            }

            if (sectionIdx.size() > 1) {
                if (!type.isa<VPUIP::DistributedBufferType>()) {
                    return errorAt(
                            op, "Array of section indexes is supported only for distributed buffer type, op = {0}", op);
                }
            }

            return mlir::success();
        };

        if (!maybeSectionIndex.has_value()) {
            if (!type.isa<VPUIP::DistributedBufferType>()) {
                return errorAt(op, "Section index is missing");
            }
        } else {
            const auto sectionIdx = parseIntArrayAttr<int64_t>(maybeSectionIndex.value());
            if (checkSectionIndex(sectionIdx).failed()) {
                return mlir::failure();
            }
        }
    } else if (getSection() == VPURT::BufferSection::DDR) {
        if (type.getMemSpace() == nullptr) {
            return errorAt(op, "Output type must have DDR memory space");
        }

        if (type.getMemSpace().getIndex().has_value() || maybeSectionIndex.has_value()) {
            return errorAt(op, "Output type with DDR memory space cannot have section index");
        }
    } else if (getSection() == VPURT::BufferSection::FunctionInput ||
               getSection() == VPURT::BufferSection::FunctionOutput) {
        if (type.getMemSpace() == nullptr) {
            return errorAt(op, "Output type must have DDR memory space");
        }

        if (!maybeSectionIndex.has_value()) {
            return errorAt(op, "Output type of FunctionInput/FunctionOutput must have a section index");
        }
    }

    return mlir::success();
}

SmallVector<int64_t> vpux::VPURT::DeclareBufferOp::getNonEmptySectionIndex() {
    if (getSectionIndex().has_value()) {
        return parseIntArrayAttr<int64_t>(getSectionIndex().value());
    }
    return SmallVector<int64_t>({0});
}

void vpux::VPURT::DeclareBufferOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    VPUX_UNUSED(binDataSection);
}

size_t vpux::VPURT::DeclareBufferOp::getBinarySize() {
    const auto type = getBuffer().getType().cast<vpux::NDTypeInterface>();
    return type.getTotalAllocSize().count();
}

size_t vpux::VPURT::DeclareBufferOp::getAlignmentRequirements() {
    return ELFNPU37XX::VPUX_NO_ALIGNMENT;
}

vpux::VPURT::BufferSection vpux::VPURT::DeclareBufferOp::getMemorySpace() {
    return getSection();
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPURT::DeclareBufferOp::getAccessingProcs() {
    auto tempFlagsVal = vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE;

    for (auto user : getResult().getUsers()) {
        if (auto binaryIface = mlir::dyn_cast<vpux::ELFNPU37XX::BinaryOpInterface>(user)) {
            tempFlagsVal = tempFlagsVal | binaryIface.getUserProcs();
        }
    }

    return tempFlagsVal;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPURT::DeclareBufferOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}
