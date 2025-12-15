//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include "schema/profiling_generated.h"

using namespace vpux;

//
//  ProfilingMetadataOp
//

void vpux::VPUMI37XX::ProfilingMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto denseMetaAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(getMetadata());
    VPUX_THROW_UNLESS(denseMetaAttr != nullptr, "ProfilingMetadata's data is NULL");

    auto buf = denseMetaAttr.getRawData();
    binDataSection.appendData(reinterpret_cast<const uint8_t*>(buf.data()), buf.size());
}

size_t vpux::VPUMI37XX::ProfilingMetadataOp::getBinarySize() {
    // calculate size based on serialized form, instead of just sizeof(NetworkMetadata)
    // serialization uses metadata that also gets stored in the blob and must be accounted for
    // also for non-POD types (e.g. have vector as member) account for all data to be serialized
    // (data owned by vector, instead of just pointer)
    auto denseMetaAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(getMetadata());
    VPUX_THROW_UNLESS(denseMetaAttr != nullptr, "ProfilingMetadata's data is NULL");
    return denseMetaAttr.getRawData().size();
}

size_t vpux::VPUMI37XX::ProfilingMetadataOp::getAlignmentRequirements() {
    return alignof(ProfilingFB::ProfilingMeta);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::ProfilingMetadataOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::ProfilingMetadataOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::ProfilingMetadataOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}
