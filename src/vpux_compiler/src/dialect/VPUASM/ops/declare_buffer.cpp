//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

//
// DeclareBufferOp
//

size_t VPUASM::DeclareBufferOp::getBinarySize(config::ArchKind) {
    const auto type = mlir::cast<vpux::NDTypeInterface>(getBufferType().getMemref());
    return ELF::getOpBinarySize(type);
}

size_t VPUASM::DeclareBufferOp::getAlignmentRequirements(config::ArchKind) {
    // DeclareBuffers are addressed by the mem-schedulers, so can't override anything
    return ELF::VPUX_NO_ALIGNMENT;
}

void VPUASM::DeclareBufferOp::setMemoryOffset(mlir::IntegerAttr) {
    // declareBufferOp's offset is implicit in it's memLocation
    return;
}

uint64_t VPUASM::DeclareBufferOp::getMemoryOffset() {
    auto location = getBufferType().getLocation();
    return location.getByteOffset();
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DeclareBufferOp::getSectionSignature() {
    const auto buffType = getBufferType();
    const auto location = buffType.getLocation();
    const auto section = location.getSection();

    ELF::SectionTypeAttr type;
    ELF::SectionFlagsAttr flags;

    bool isInputOrOutputBuffer = false;

    switch (section) {
    case VPURT::BufferSection::CMX_NN:
        type = ELF::SectionTypeAttr::VPU_SHT_CMX_WORKSPACE;
        flags = ELF::SectionFlagsAttr::SHF_NONE;
        break;
    case VPURT::BufferSection::FunctionInput:
    case VPURT::BufferSection::FunctionOutput:
        VPUX_THROW("BufferSection::FunctionInput/BufferSection::FunctionOutput must not apprear in this context");
    case VPURT::BufferSection::DDR:
        type = ELF::SectionTypeAttr::SHT_NOBITS;
        flags = ELF::SectionFlagsAttr::SHF_WRITE | ELF::SectionFlagsAttr::SHF_ALLOC;
        break;
    case VPURT::BufferSection::NetworkInput:
        type = ELF::SectionTypeAttr::SHT_NOBITS;
        flags = ELF::SectionFlagsAttr::VPU_SHF_USERINPUT | ELF::SectionFlagsAttr::SHF_WRITE |
                ELF::SectionFlagsAttr::SHF_ALLOC;
        isInputOrOutputBuffer = true;
        break;
    case VPURT::BufferSection::NetworkOutput:
        type = ELF::SectionTypeAttr::SHT_NOBITS;
        flags = ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT | ELF::SectionFlagsAttr::SHF_WRITE |
                ELF::SectionFlagsAttr::SHF_ALLOC;
        isInputOrOutputBuffer = true;
        break;
    case VPURT::BufferSection::ProfilingOutput:
        type = ELF::SectionTypeAttr::SHT_NOBITS;
        flags = ELF::SectionFlagsAttr::VPU_SHF_PROFOUTPUT | ELF::SectionFlagsAttr::SHF_WRITE |
                ELF::SectionFlagsAttr::SHF_ALLOC;
        break;
    case VPURT::BufferSection::Register:
        type = ELF::SectionTypeAttr::SHT_NOBITS;
        flags = ELF::SectionFlagsAttr::SHF_NONE;
        break;
    default:
        return std::nullopt;
    }

    auto name = vpux::ELF::generateSignature("buffer", buffType);

    if (isInputOrOutputBuffer || section == VPURT::BufferSection::ProfilingOutput) {
        name = ELF::generateSignature("io", buffType);
    } else if (section == VPURT::BufferSection::Register) {
        name = ELF::generateSignature("reg", buffType);
    }

    return ELF::SectionSignature(std::move(name), flags, type);
}

bool vpux::VPUASM::DeclareBufferOp::hasMemoryFootprint() {
    return false;
}

vpux::VPURT::BufferSection VPUASM::DeclareBufferOp::getMemorySection() {
    return getBufferType().getLocation().getSection();
}
