//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/core/IR/strided_dmas_utils.hpp"
#include "vpux/utils/core/error.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;
using namespace NPUReg40XX;

//
// NNDMAOp
//

void NPUReg40XX::NNDMAOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto dmaDescriptor = getProperties().getDescriptor();

    VPUX_THROW_UNLESS(sizeof(nn_public::VpuDMATask) == dmaDescriptor.size(),
                      "HW DmaDescriptor size {0} != regMapped representation size {1}.", sizeof(nn_public::VpuDMATask),
                      dmaDescriptor.size());
    auto serializedDmaDesc = dmaDescriptor.getStorage();

    binDataSection.appendData(serializedDmaDesc.data(), serializedDmaDesc.size());
}

size_t NPUReg40XX::NNDMAOp::getBinarySize(config::ArchKind) {
    return sizeof(nn_public::VpuDMATask);
}

size_t NPUReg40XX::NNDMAOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(nn_public::VpuDMATask);
}

namespace {
size_t getSymRefOffsetForReloc(NPUReg40XX::NNDMAOp op, mlir::SymbolRefAttr ref) {
    if (ref == op.getNextLinkAttr()) {
        return offsetof(nn_public::VpuDMATask, transaction_) + offsetof(DmaDescriptor, link_addr_offsetof);
    } else if (ref == op.getInputAttr()) {
        return offsetof(nn_public::VpuDMATask, transaction_) + offsetof(DmaDescriptor, src_offsetof);
    } else if (ref == mlir::cast<mlir::SymbolRefAttr>(op.getOutputBuffsAttr()[0])) {
        return offsetof(nn_public::VpuDMATask, transaction_) + offsetof(DmaDescriptor, dst_offsetof);
    } else if (op.getActCompressionSizeEntryAttr() == ref) {
        const auto& descriptor = op.getProperties().getDescriptor();
        const auto dma_cfg_fields_rws_en = descriptor.read<Fields::dma_cfg_fields_rws_en>();
        const auto dma_cfg_fields_rwf_en = descriptor.read<Fields::dma_cfg_fields_rwf_en>();
        if (dma_cfg_fields_rws_en == 1) {
            return offsetof(nn_public::VpuDMATask, transaction_) + offsetof(DmaDescriptor, remote_width_store);
        } else if (dma_cfg_fields_rwf_en == 1) {
            return offsetof(nn_public::VpuDMATask, transaction_) + offsetof(DmaDescriptor, remote_width_fetch);
        }
    } else if (op.getIndicesAttr() == ref) {
        return offsetof(nn_public::VpuDMATask, transaction_) + offsetof(DmaDescriptor, list_addr) +
               offsetof(decltype(DmaDescriptor::list_addr), src);
    }

    VPUX_THROW("Provided SymbolRefAttr is not linked to the NNDMA Op or getSymRefOffsetForReloc does not support it");
}
}  // namespace

std::vector<ELF::RelocationInfo> NPUReg40XX::NNDMAOp::getRelocationInfo(ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    auto thisDma = *(this);
    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    // Input reloc
    // Temporary, until SymRef lookup & interpretation is fixed
    // Skip input reloc for strided inputs as it is covered by a different relocation
    if (!getOperation()->hasAttr(vpux::stridedInputAttrName)) {
        auto input = getInputAttr();
        auto inputRelocType = VPUASM::getBufferLocation(symRefMap, input) == VPURT::BufferSection::CMX_NN
                                      ? ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET
                                      : ELF::RelocationType::R_VPU_64;
        relocs.emplace_back(input, targetSection, getSymRefOffsetForReloc(thisDma, input), inputRelocType,
                            ELF::getOffsetOfSymRef(symRefMap, input), "Input in NNDMA reloc");
    }

    // Output reloc
    // Skip output reloc for strided outputs as it is covered by a different relocation
    if (!getOperation()->hasAttr(vpux::stridedOutputAttrName)) {
        auto firstOutputBuff = mlir::cast<mlir::SymbolRefAttr>(getOutputBuffs()[0]);
        auto outputRelocType = VPUASM::getBufferLocation(symRefMap, firstOutputBuff) == VPURT::BufferSection::CMX_NN
                                       ? ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET
                                       : ELF::RelocationType::R_VPU_64;

        // Don't add relocations for Register type buffers as we use absolute HW address
        if (VPUASM::getBufferLocation(symRefMap, firstOutputBuff) != VPURT::BufferSection::Register) {
            relocs.emplace_back(firstOutputBuff, targetSection, getSymRefOffsetForReloc(thisDma, firstOutputBuff),
                                outputRelocType, ELF::getOffsetOfSymRef(symRefMap, firstOutputBuff),
                                "Output (firstOutputBuff) in NNDMA reloc");
        }
    }

    // Link Address reloc
    if (auto nextLink = getNextLink().value_or(nullptr)) {
        // TODO: (E#114625) refactor the way DMA knows if it has direct reloc or CMX reloc
        auto relocType = getOperation()->hasAttr("directLink") ? ELF::RelocationType::R_VPU_64
                                                               : ELF::RelocationType::R_VPU_32_BIT_OR_B21_B26_UNSET;

        relocs.emplace_back(nextLink, targetSection, getSymRefOffsetForReloc(thisDma, nextLink), relocType,
                            ELF::getOffsetOfSymRef(symRefMap, nextLink), "Link address (nextLink) in NNDMA reloc");
    }

    // ActCompressionSizeEntry reloc
    if (auto actCompressionSizeEntry = getActCompressionSizeEntry().value_or(nullptr)) {
        relocs.emplace_back(
                actCompressionSizeEntry, targetSection, getSymRefOffsetForReloc(thisDma, actCompressionSizeEntry),
                ELF::RelocationType::R_VPU_32_BIT_OR_B21_B26_UNSET,
                ELF::getOffsetOfSymRef(symRefMap, actCompressionSizeEntry), "actCompressionSizeEntry in NNDMA reloc");
    }

    // Indices reloc
    if (auto indices = getIndices().value_or(nullptr)) {
        VPUX_THROW_UNLESS(VPUASM::getBufferLocation(symRefMap, indices) == VPURT::BufferSection::CMX_NN,
                          "Indices must be in CMX");
        relocs.emplace_back(indices, targetSection, getSymRefOffsetForReloc(thisDma, indices),
                            ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET,
                            ELF::getOffsetOfSymRef(symRefMap, indices), "indices in NNDMA reloc");
    }

    return relocs;
}

mlir::SymbolRefAttr NPUReg40XX::NNDMAOp::getInputSymbol() {
    return getInputAttr();
}

mlir::SymbolRefAttr NPUReg40XX::NNDMAOp::getOutputSymbol() {
    return mlir::cast<mlir::SymbolRefAttr>(getOutputBuffs()[0]);
}

void NPUReg40XX::NNDMAOp::build(mlir::OpBuilder&, mlir::OperationState& state, mlir::StringAttr symName,
                                vpux::NPUReg40XX::Descriptors::DMARegister&& descriptor, mlir::SymbolRefAttr input,
                                mlir::ArrayAttr outputBuffs, mlir::SymbolRefAttr nextLink,
                                mlir::SymbolRefAttr actCompressionSizeEntry, mlir::SymbolRefAttr indices,
                                VPUIP::GatherAddressingModeAttr addressingMode) {
    auto& props = state.getOrAddProperties<Properties>();

    props.sym_name = symName;
    props.descriptor = std::move(descriptor);
    props.input = input;
    props.output_buffs = outputBuffs;
    props.next_link = nextLink;
    props.act_compression_size_entry = actCompressionSizeEntry;
    props.indices = indices;
    props.addressing_mode = addressingMode;
}
