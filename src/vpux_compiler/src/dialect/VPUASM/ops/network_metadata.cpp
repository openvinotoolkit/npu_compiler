//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux_headers/serial_metadata.hpp"

#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"

using namespace vpux;

void vpux::VPUASM::NetworkMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection,
                                                elf::NetworkMetadata& metadata) {
    auto operation = getOperation();
    auto mainModule = operation->getParentOfType<mlir::ModuleOp>();

    VPUASM::setResourceRequirement(mainModule, metadata);

    auto serializedMetadata = elf::MetadataSerialization::serialize(metadata);
    binDataSection.appendData(&serializedMetadata[0], serializedMetadata.size());
}

size_t vpux::VPUASM::NetworkMetadataOp::getBinarySize(config::ArchKind) {
    // calculate size based on serialized form, instead of just sizeof(NetworkMetadata)
    // serialization uses metadata that also gets stored in the blob and must be accounted for
    // also for non-POD types (e.g. have vector as member) account for all data to be serialized
    // (data owned by vector, instead of just pointer)
    auto metadataPtr =
            vpux::ELFNPU37XX::constructMetadata(getOperation()->getParentOfType<mlir::ModuleOp>(), Logger::global());
    auto& metadata = *metadataPtr;
    return elf::MetadataSerialization::serialize(metadata).size();
}

size_t vpux::VPUASM::NetworkMetadataOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(elf::NetworkMetadata);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::NetworkMetadataOp::getSectionSignature() {
    return std::nullopt;
}

bool vpux::VPUASM::NetworkMetadataOp::hasMemoryFootprint() {
    return true;
}
