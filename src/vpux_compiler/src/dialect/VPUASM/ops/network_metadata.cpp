//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_headers/serial_metadata.hpp>
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux_headers/serial_metadata.hpp"

#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"

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
    // The NetworkMetadataOp has binaryOpInterface, this binaryOpInterface serialize need to be
    // implemented otherwise the build will failed due to the extra definition for the serialize.
    // TODO: E#153150
    VPUX_THROW("NetworkMetadataOp BinaryOpInterface method serialize should not be called! E#153150");
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
}

size_t vpux::VPUASM::NetworkMetadataOp::getBinarySize(VPU::ArchKind) {
    // calculate size based on serialized form, instead of just sizeof(NetworkMetadata)
    // serialization uses metadata that also gets stored in the blob and must be accounted for
    // also for non-POD types (e.g. have vector as member) account for all data to be serialized
    // (data owned by vector, instead of just pointer)
    auto metadataPtr =
            vpux::ELFNPU37XX::constructMetadata(getOperation()->getParentOfType<mlir::ModuleOp>(), Logger::global());
    auto& metadata = *metadataPtr;
    return elf::MetadataSerialization::serialize(metadata).size();
}

size_t vpux::VPUASM::NetworkMetadataOp::getAlignmentRequirements(VPU::ArchKind) {
    return alignof(elf::NetworkMetadata);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::NetworkMetadataOp::getSectionSignature() {
    return std::nullopt;
}

bool vpux::VPUASM::NetworkMetadataOp::hasMemoryFootprint() {
    return true;
}
