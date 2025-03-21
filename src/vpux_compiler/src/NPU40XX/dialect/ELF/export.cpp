//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/export.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <vpux_elf/writer.hpp>

namespace vpux::ELF {

namespace {

// current API forces to create & return elf::Writer object
// + pass section, symbol and symbol reference maps as calculation
// of blob size done together with populating ELF headers.
//
// refactor APIs, so that blob storage size calculation is
// decoupled from elf::Writer API and function returns just
// integral value
//
// consider getting rid of section and symbol maps here as well
// to calculate blob size
// ticket: <TBD>
elf::Writer calculateBlobSize(MainOp elfMain, Logger log, SectionMapType& sectionMap, SymbolMapType& symbolMap,
                              SymbolReferenceMap& symRefMap) {
    elf::Writer elfWriter;

    log.trace("Serialization setup '{0}' ops", CreateMetadataSectionOp::getOperationName());
    for (auto createMetadataSectionOp : elfMain.getOps<CreateMetadataSectionOp>()) {
        createMetadataSectionOp.preserialize(elfWriter, sectionMap, symRefMap);
    }

    auto createProfSectionOps = to_small_vector(elfMain.getOps<CreateProfilingSectionOp>());
    if (!createProfSectionOps.empty()) {
        VPUX_THROW_UNLESS(createProfSectionOps.size() == 1, "Expected exactly one CreateProfilingSectionOp. Got {0}",
                          createProfSectionOps.size());
        log.trace("Serialization setup '{0}' ops", CreateProfilingSectionOp::getOperationName());
        auto createProfSectionOp = createProfSectionOps[0];
        createProfSectionOp.preserialize(elfWriter, sectionMap, symRefMap);
    }

    log.trace("Serialization setup '{0}' ops", DataSectionOp::getOperationName());
    for (auto dataSectionOp : elfMain.getOps<DataSectionOp>()) {
        dataSectionOp.preserialize(elfWriter, sectionMap, symRefMap);
    }

    log.trace("Serialization setup '{0}' ops", LogicalSectionOp::getOperationName());
    for (auto logicalSectionOp : elfMain.getOps<LogicalSectionOp>()) {
        logicalSectionOp.preserialize(elfWriter, sectionMap, symRefMap);
    }

    // symbol tables and relocation sections don't implement preserialize step and store
    // their data into internal elf::Writer storage before copying into final blob storage
    // as they have internal state to be updated (relocation and symbol entries)
    // memory overhead is small
    // note: it needs to be called here (before elf::Writer::prepareWriter), to populate
    // sections data fields that are used during preparation
    // E#136375
    log.trace("Serializing '{0}' ops", CreateSymbolTableSectionOp::getOperationName());
    for (auto symTabOp : elfMain.getOps<CreateSymbolTableSectionOp>()) {
        symTabOp.serialize(elfWriter, sectionMap, symbolMap, symRefMap);
    }

    log.trace("Serializing '{0}' ops", CreateRelocationSectionOp::getOperationName());
    for (auto relocSectionOp : elfMain.getOps<CreateRelocationSectionOp>()) {
        relocSectionOp.serialize(elfWriter, sectionMap, symbolMap, symRefMap);
    }

    return elfWriter;
}

void serializeTo(uint8_t* storage, MainOp elfMain, Logger log, elf::Writer& elfWriter, SectionMapType& sectionMap,
                 SymbolMapType& symbolMap, SymbolReferenceMap& symRefMap) {
    elfWriter.generateELF(storage);
    elfWriter.setSectionsStartAddr(storage);

    log.trace("Serializing '{0}' ops", CreateMetadataSectionOp::getOperationName());
    for (auto createMetadataSectionOp : elfMain.getOps<CreateMetadataSectionOp>()) {
        auto metadataPtr = vpux::ELFNPU37XX::constructMetadata(elfMain->getParentOfType<mlir::ModuleOp>(), log.nest());
        auto& metadata = *metadataPtr;
        createMetadataSectionOp.serialize(elfWriter, sectionMap, symbolMap, metadata);
    }

    auto createProfSectionOps = to_small_vector(elfMain.getOps<CreateProfilingSectionOp>());
    if (!createProfSectionOps.empty()) {
        log.trace("Serializing '{0}' ops", CreateProfilingSectionOp::getOperationName());
        auto createProfSectionOp = createProfSectionOps[0];
        createProfSectionOp.serialize(elfWriter, sectionMap, symbolMap);
    }

    log.trace("Serializing '{0}' ops", DataSectionOp::getOperationName());
    for (auto dataSectionOp : elfMain.getOps<DataSectionOp>()) {
        dataSectionOp.serialize(elfWriter, sectionMap, symbolMap, symRefMap);
    }

    log.trace("Serializing '{0}' ops", LogicalSectionOp::getOperationName());
    for (auto logicalSectionOp : elfMain.getOps<LogicalSectionOp>()) {
        logicalSectionOp.serialize(elfWriter, sectionMap, symbolMap, symRefMap);
    }
}

}  // namespace

std::vector<uint8_t> exportToELF(mlir::ModuleOp module, Logger log) {
    log.setName("ELF BackEnd");

    // Associate the respective mlir::Operation* of
    //   DataSectionOp/LogicalSectionOp/CreateSymbolSectionOp/CreateRelocationSectionOp
    //   with the respective created elf::writer::Section* for it.
    SectionMapType sectionMap;
    // Associate the respective mlir::Operation* of a SymbolOp with the newly created
    //   elf::writer::Symbol* for it.
    SymbolMapType symbolMap;

    auto elfMain = getElfMainOp(module);

    SymbolReferenceMap symRefMap(elfMain, true);

    auto elfWriter = calculateBlobSize(elfMain, log, sectionMap, symbolMap, symRefMap);
    elfWriter.prepareWriter();

    std::vector<uint8_t> blob(elfWriter.getTotalSize());
    serializeTo(blob.data(), elfMain, log, elfWriter, sectionMap, symbolMap, symRefMap);

    return blob;
}

BlobView exportToELF(mlir::ModuleOp module, BlobAllocator& allocator, Logger log) {
    log.setName("ELF Backend - Export");

    // Associate the respective mlir::Operation* of
    //   DataSectionOp/LogicalSectionOp/CreateSymbolSectionOp/CreateRelocationSectionOp
    //   with the respective created elf::writer::Section* for it.
    SectionMapType sectionMap;
    // Associate the respective mlir::Operation* of a SymbolOp with the newly created
    //   elf::writer::Symbol* for it.
    SymbolMapType symbolMap;

    auto elfMain = getElfMainOp(module);

    SymbolReferenceMap symRefMap(elfMain, true);

    auto elfWriter = calculateBlobSize(elfMain, log, sectionMap, symbolMap, symRefMap);
    elfWriter.prepareWriter();

    const auto size = elfWriter.getTotalSize();
    auto blob = allocator.allocate(vpux::Byte{static_cast<int64_t>(size)});
    // For a consistent blob hash make sure that the memory is initialized before serializing
    // This fill_n is required as the writer will not cover the padding between the sections.
    // The writer will only override the memory for sections inside of the prealocated buffer.
    std::fill_n(blob, size, 0);
    serializeTo(blob, elfMain, log, elfWriter, sectionMap, symbolMap, symRefMap);

    return {blob, static_cast<uint64_t>(size)};
}

}  // namespace vpux::ELF
