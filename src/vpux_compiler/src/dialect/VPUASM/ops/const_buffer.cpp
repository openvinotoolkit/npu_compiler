//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

//
// ConstBufferOp
//

void VPUASM::ConstBufferOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto cnt = getProperties().getContent().fold();
    auto ptr = binDataSection.getCurrentWriteAddr() + getMemoryOffset();
    const auto size = getBinarySize(config::ArchKind::UNKNOWN);
    MutableArrayRef<char> inBlobView(reinterpret_cast<char*>(ptr), reinterpret_cast<char*>(ptr) + size);
    cnt.copyTo(inBlobView);
}

size_t VPUASM::ConstBufferOp::getBinarySize(config::ArchKind) {
    auto content = getProperties().getContent();
    VPUX_THROW_WHEN(content == nullptr, "This content is already deleted!");
    return content.getType().getTotalAllocSize().count();
}

size_t VPUASM::ConstBufferOp::getAlignmentRequirements(config::ArchKind) {
    // TODO: E#59169 measure if weights alignment has any impact on performance.
    return ELF::VPUX_DEFAULT_ALIGNMENT;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::ConstBufferOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("buffer", getBufferType(), "constant"),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::ConstBufferOp::hasMemoryFootprint() {
    return true;
}

void vpux::VPUASM::ConstBufferOp::build(mlir::OpBuilder&, mlir::OperationState& state, mlir::StringAttr symName,
                                        ::vpux::VPUASM::BufferType bufferType, vpux::Const::ContentAttr content) {
    auto& props = state.getOrAddProperties<Properties>();
    props.sym_name = symName;
    props.buffer_type = mlir::TypeAttr::get(bufferType);
    props.content = std::move(content);
}

vpux::VPURT::BufferSection VPUASM::ConstBufferOp::getMemorySection() {
    return getBufferType().getLocation().getSection();
}
