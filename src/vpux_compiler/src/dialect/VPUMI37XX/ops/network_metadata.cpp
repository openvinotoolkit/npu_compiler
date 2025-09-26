//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_headers/serial_metadata.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux_headers/serial_metadata.hpp"

using namespace vpux;

//
//  NetworkMetadataOp
//

void vpux::VPUMI37XX::NetworkMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection,
                                                   elf::NetworkMetadata& metadata) {
    auto operation = getOperation();
    auto mainModule = operation->getParentOfType<mlir::ModuleOp>();

    vpux::ELFNPU37XX::setResourceRequirement(mainModule, metadata);

    auto serializedMetadata = elf::MetadataSerialization::serialize(metadata);
    binDataSection.appendData(&serializedMetadata[0], serializedMetadata.size());
}

void vpux::VPUMI37XX::NetworkMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // serialize as part of the BinaryOpInterface has to be implemented, the actual serialize implementation of
    // NetworkMetadataOp is use the serialize declared in the extra class.
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
}

size_t vpux::VPUMI37XX::NetworkMetadataOp::getBinarySize() {
    // calculate size based on serialized form, instead of just sizeof(NetworkMetadata)
    // serialization uses metadata that also gets stored in the blob and must be accounted for
    // also for non-POD types (e.g. have vector as member) account for all data to be serialized
    // (data owned by vector, instead of just pointer)
    auto metadataPtr =
            vpux::ELFNPU37XX::constructMetadata(getOperation()->getParentOfType<mlir::ModuleOp>(), Logger::global());
    auto& metadata = *metadataPtr.get();
    return elf::MetadataSerialization::serialize(metadata).size();
}

size_t vpux::VPUMI37XX::NetworkMetadataOp::getAlignmentRequirements() {
    return alignof(elf::NetworkMetadata);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::NetworkMetadataOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::NetworkMetadataOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::NetworkMetadataOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}
