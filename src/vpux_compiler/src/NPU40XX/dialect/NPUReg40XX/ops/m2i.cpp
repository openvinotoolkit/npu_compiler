//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

//
// M2IOp
//

namespace {
void setAddendForInAddr1AndInAddr2(size_t addend, uint64_t inFormat, uint64_t PSOB_inPS, size_t& addendInAddr1,
                                   size_t& addendInAddr2) {
    auto inFmt = VPU::symbolizeM2iColorFmt(inFormat);
    if (!inFmt.has_value()) {
        VPUX_THROW("invalid inFormat {0}", inFormat);
    }

    switch (inFmt.value()) {
    case VPU::M2iColorFmt::PL_RGB24:
    case VPU::M2iColorFmt::PL_YUV444_8:
    case VPU::M2iColorFmt::PL_FP16_RGB:
    case VPU::M2iColorFmt::SP_NV12_8:
        addendInAddr1 = addend + PSOB_inPS;
        addendInAddr2 = addendInAddr1 + PSOB_inPS;
        break;

    case VPU::M2iColorFmt::PL_GRAY8:
    case VPU::M2iColorFmt::IL_RGB888:
        addendInAddr1 = addend;
        addendInAddr2 = addendInAddr1;
        break;

    case VPU::M2iColorFmt::PL_YUV420_8:
        addendInAddr1 = addend + PSOB_inPS;
        addendInAddr2 = addendInAddr1 + PSOB_inPS / 4;
        break;

    case VPU::M2iColorFmt::PL_YUV422_8:
        addendInAddr1 = addend + PSOB_inPS;
        addendInAddr2 = addendInAddr1 + PSOB_inPS / 2;
        break;

    default:
        VPUX_THROW("Not supported inFormat {0}", inFmt.value());
        break;
    }
}
}  // namespace

void NPUReg40XX::M2IOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto m2iDescriptor = getDescriptor().getRegMapped();

    VPUX_THROW_UNLESS(sizeof(nn_public::VpuMediaTask) == m2iDescriptor.size(),
                      "HW M2iDescriptor size {0} != regMapped representation size {1}.",
                      sizeof(nn_public::VpuMediaTask), m2iDescriptor.size());

    auto serializedM2iDesc = m2iDescriptor.getStorage();
    binDataSection.appendData(serializedM2iDesc.data(), serializedM2iDesc.size());
}

size_t NPUReg40XX::M2IOp::getBinarySize(VPU::ArchKind) {
    return sizeof(nn_public::VpuMediaTask);
}

size_t NPUReg40XX::M2IOp::getAlignmentRequirements(VPU::ArchKind) {
    return alignof(nn_public::VpuMediaTask);
}

namespace {
size_t getSymRefOffsetForReloc(NPUReg40XX::M2IOp op, mlir::SymbolRefAttr ref) {
    if (ref == op.getNextLinkAttr()) {
        return offsetof(nn_public::VpuMediaTask, standard.buff_desc_) +
               offsetof(VpuMediaBuffDescriptor, nextDesc_offset);
    } else if (ref == op.getInputAttr()) {
        return offsetof(nn_public::VpuMediaTask, standard.buff_desc_) + offsetof(VpuMediaBuffDescriptor, inAddr0);
    } else if (ref == op.getOutputBuffAttr()) {
        return offsetof(nn_public::VpuMediaTask, standard.roi_desc_) + offsetof(VpuMediaROIDescriptor, roiDef) +
               offsetof(Media_RoiDef_t, roiBase_offset);
    }

    VPUX_THROW("Provided SymbolRefAttr is not linked to the M2I Op or getSymRefOffsetForReloc does not support it");
}
}  // namespace

std::vector<ELF::RelocationInfo> NPUReg40XX::M2IOp::getRelocationInfo(ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    auto buffDescOffset = offsetof(nn_public::VpuMediaTask, standard.buff_desc_);

    auto inputSymRef = getInput();
    auto m2iDescriptor = getDescriptor().getRegMapped();
    auto PSOB_inPS = m2iDescriptor.read<Fields::inPS>();
    auto inFormat = m2iDescriptor.read<Fields::inFormat>();

    auto addend = ELF::getOffsetOfSymRef(symRefMap, inputSymRef);
    size_t addendInAddr1(0), addendInAddr2(0);
    setAddendForInAddr1AndInAddr2(addend, inFormat, PSOB_inPS, addendInAddr1, addendInAddr2);

    //
    // input relocs
    //

    relocs.emplace_back(inputSymRef, targetSection, getSymRefOffsetForReloc(*this, inputSymRef),
                        ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET, addend, "Input (inputSymRef) in M2I reloc");

    relocs.emplace_back(inputSymRef, targetSection, buffDescOffset + offsetof(VpuMediaBuffDescriptor, inAddr1),
                        ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET, addendInAddr1,
                        "Input (inAddr1) in M2I reloc");

    relocs.emplace_back(inputSymRef, targetSection, buffDescOffset + offsetof(VpuMediaBuffDescriptor, inAddr2),
                        ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET, addendInAddr2,
                        "Input (inAddr2) in M2I reloc");

    //
    // output reloc
    //

    auto outputSymRef = getOutputBuff();

    relocs.emplace_back(outputSymRef, targetSection, getSymRefOffsetForReloc(*this, outputSymRef),
                        ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET,
                        ELF::getOffsetOfSymRef(symRefMap, outputSymRef), "Output (outputSymRef) in M2I reloc");

    //
    // next link reloc
    //

    if (auto nextLinkSymRef = getNextLink().value_or(nullptr)) {
        relocs.emplace_back(nextLinkSymRef, targetSection, getSymRefOffsetForReloc(*this, nextLinkSymRef),
                            ELF::RelocationType::R_VPU_32_BIT_OR_B21_B26_UNSET,
                            ELF::getOffsetOfSymRef(symRefMap, nextLinkSymRef),
                            "Next link (nextLinkSymRef) in M2I reloc");
    }

    return relocs;
}
