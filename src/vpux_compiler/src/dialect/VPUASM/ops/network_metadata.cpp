//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_headers/serial_metadata.hpp>
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"
#include "vpux_headers/serial_metadata.hpp"

using namespace vpux;

void vpux::VPUASM::NetworkMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection,
                                                elf::NetworkMetadata& metadata) {
    auto operation = getOperation();
    auto mainModule = operation->getParentOfType<mlir::ModuleOp>();

    auto nBarrs = VPUIP::getNumAvailableBarriers(operation);
    metadata.mResourceRequirements.nn_barriers_ = nBarrs;
    metadata.mResourceRequirements.nn_slice_count_ = VPUIP::getNumTilesUsed(mainModule);

    metadata.mResourceRequirements.ddr_scratch_length_ =
            checked_cast<uint32_t>(IE::getAvailableMemory(mainModule, vpux::VPU::MemoryKind::DDR).getByteSize());
    metadata.mResourceRequirements.nn_slice_length_ =
            checked_cast<uint32_t>(IE::getAvailableMemory(mainModule, vpux::VPU::MemoryKind::CMX_NN).getByteSize());

    auto serializedMetadata = elf::MetadataSerialization::serialize(metadata);
    binDataSection.appendData(&serializedMetadata[0], serializedMetadata.size());
}

void vpux::VPUASM::NetworkMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for ActKernelRangeOp
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
}

size_t vpux::VPUASM::NetworkMetadataOp::getBinarySize() {
    return sizeof(elf::NetworkMetadata);
}

size_t vpux::VPUASM::NetworkMetadataOp::getAlignmentRequirements() {
    return alignof(elf::NetworkMetadata);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::NetworkMetadataOp::getAccessingProcs(mlir::SymbolUserMap&) {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::NetworkMetadataOp::getUserProcs() {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::NetworkMetadataOp::getSectionSignature() {
    return std::nullopt;
}

bool vpux::VPUASM::NetworkMetadataOp::hasMemoryFootprint() {
    return true;
}
