//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

void vpux::VPUASM::ProfilingMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto denseMetaAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(getMetadata());
    auto buf = denseMetaAttr.getRawData();
    binDataSection.appendData(reinterpret_cast<const uint8_t*>(buf.data()), buf.size());
}

size_t vpux::VPUASM::ProfilingMetadataOp::getBinarySize(config::ArchKind) {
    auto values = getMetadata().getValues<uint8_t>();
    return values.size();
}

size_t vpux::VPUASM::ProfilingMetadataOp::getAlignmentRequirements(config::ArchKind) {
    return ELF::VPUX_NO_ALIGNMENT;
}

// The operation is placed inside the section (CreateProfilingSectionOp) by a separate pass,
// see: AddProfilingSectionPass
std::optional<ELF::SectionSignature> vpux::VPUASM::ProfilingMetadataOp::getSectionSignature() {
    return std::nullopt;
}

bool vpux::VPUASM::ProfilingMetadataOp::hasMemoryFootprint() {
    return true;
}
