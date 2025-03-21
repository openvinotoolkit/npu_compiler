//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/export.hpp"
#include <algorithm>
#include <cstring>
#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"

namespace vpux::ELFNPU37XX {

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
elf::Writer calculateBlobSize(mlir::func::FuncOp main, Logger log, SectionMapType& sectionMap,
                              SymbolMapType& symbolMap) {
    elf::Writer elfWriter;

    log.trace("Serialization setup '{0}' ops", CreateMetadataSectionOp::getOperationName());
    for (auto createMetadataSectionOp : main.getOps<CreateMetadataSectionOp>()) {
        createMetadataSectionOp.preserialize(elfWriter, sectionMap);
    }

    auto createProfSectionOps = to_small_vector(main.getOps<CreateProfilingSectionOp>());
    if (!createProfSectionOps.empty()) {
        VPUX_THROW_UNLESS(createProfSectionOps.size() == 1, "Expected exactly one CreateProfilingSectionOp. Got {0}",
                          createProfSectionOps.size());
        log.trace("Serialization setup '{0}' ops", CreateProfilingSectionOp::getOperationName());
        auto createProfSectionOp = createProfSectionOps[0];
        createProfSectionOp.preserialize(elfWriter, sectionMap);
    }

    log.trace("Serialization setup '{0}' ops", CreateSectionOp::getOperationName());
    for (auto createSectionOp : main.getOps<CreateSectionOp>()) {
        createSectionOp.preserialize(elfWriter, sectionMap);
    }

    log.trace("Serialization setup '{0}' ops", CreateLogicalSectionOp::getOperationName());
    for (auto logicalSectionOp : main.getOps<CreateLogicalSectionOp>()) {
        logicalSectionOp.preserialize(elfWriter, sectionMap);
    }

    // symbol tables and relocation sections don't implement preserialize step and store
    // their data into internal elf::Writer storage before copying into final blob storage
    // as they have internal state to be updated (relocation and symbol entries)
    // memory overhead is small
    // note: it needs to be called here (before elf::Writer::prepareWriter), to populate
    // sections data fields that are used during preparation
    // E#136375
    log.trace("Serializing '{0}' ops", CreateSymbolTableSectionOp::getOperationName());
    for (auto symTabOp : main.getOps<CreateSymbolTableSectionOp>()) {
        symTabOp.serialize(elfWriter, sectionMap, symbolMap);
    }

    log.trace("Serializing '{0}' ops", CreateRelocationSectionOp::getOperationName());
    for (auto relocSectionOp : main.getOps<CreateRelocationSectionOp>()) {
        relocSectionOp.serialize(elfWriter, sectionMap, symbolMap);
    }

    return elfWriter;
}

void serializeTo(uint8_t* storage, mlir::func::FuncOp main, Logger log, elf::Writer& elfWriter,
                 SectionMapType& sectionMap, SymbolMapType& symbolMap) {
    elfWriter.generateELF(storage);
    elfWriter.setSectionsStartAddr(storage);

    log.trace("Serializing '{0}' ops", CreateMetadataSectionOp::getOperationName());
    for (auto createMetadataSectionOp : main.getOps<CreateMetadataSectionOp>()) {
        auto metadataPtr = constructMetadata(main->getParentOfType<mlir::ModuleOp>(), log.nest());
        auto& metadata = *metadataPtr;
        createMetadataSectionOp.serialize(elfWriter, sectionMap, symbolMap, metadata);
    }

    auto createProfSectionOps = to_small_vector(main.getOps<CreateProfilingSectionOp>());
    if (!createProfSectionOps.empty()) {
        log.trace("Serializing '{0}' ops", CreateProfilingSectionOp::getOperationName());
        auto createProfSectionOp = createProfSectionOps[0];
        createProfSectionOp.serialize(elfWriter, sectionMap, symbolMap);
    }

    log.trace("Serializing '{0}' ops", CreateSectionOp::getOperationName());
    for (auto createSectionOp : main.getOps<CreateSectionOp>()) {
        createSectionOp.serialize(elfWriter, sectionMap, symbolMap);
    }

    log.trace("Serializing '{0}' ops", CreateLogicalSectionOp::getOperationName());
    for (auto logicalSectionOp : main.getOps<CreateLogicalSectionOp>()) {
        logicalSectionOp.serialize(elfWriter, sectionMap, symbolMap);
    }
}

}  // namespace

std::vector<uint8_t> exportToELF(mlir::ModuleOp module, Logger log) {
    log.setName("ELF Backend - Export");

    log.trace("Extract '{0}' from Module (ELF File)", IE::CNNNetworkOp::getOperationName());

    // Associate the respective mlir::Operation* of
    //   CreateSectionOp/CreateLogicalSectionOp/CreateSymbolSectionOp/CreateRelocationSectionOp
    //   with the respective created elf::writer::Section* for it.
    SectionMapType sectionMap;
    // Associate the respective mlir::Operation* of a SymbolOp with the newly created
    //   elf::writer::Symbol* for it.
    SymbolMapType symbolMap;

    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp main;
    IE::CNNNetworkOp::getFromModule(module, netOp, main);

    auto elfWriter = calculateBlobSize(main, log, sectionMap, symbolMap);
    elfWriter.prepareWriter();

    std::vector<uint8_t> blob(elfWriter.getTotalSize());
    serializeTo(blob.data(), main, log, elfWriter, sectionMap, symbolMap);

    return blob;
}

BlobView exportToELF(mlir::ModuleOp module, BlobAllocator& allocator, Logger log) {
    log.setName("ELFNPU37XX BackEnd");

    log.trace("Extract '{0}' from Module (ELF File)", IE::CNNNetworkOp::getOperationName());

    // Associate the respective mlir::Operation* of
    //   CreateSectionOp/CreateLogicalSectionOp/CreateSymbolSectionOp/CreateRelocationSectionOp
    //   with the respective created elf::writer::Section* for it.
    SectionMapType sectionMap;
    // Associate the respective mlir::Operation* of a SymbolOp with the newly created
    //   elf::writer::Symbol* for it.
    SymbolMapType symbolMap;

    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp main;
    IE::CNNNetworkOp::getFromModule(module, netOp, main);

    auto elfWriter = calculateBlobSize(main, log, sectionMap, symbolMap);
    elfWriter.prepareWriter();

    const auto size = elfWriter.getTotalSize();
    auto blob = allocator.allocate(Byte{static_cast<int64_t>(size)});
    // For a consistent blob hash make sure that the memory is initialized before serializing.
    // This fill_n is required as the writer will not cover the padding between the sections.
    // The writer will only override the memory for sections inside of the prealocated buffer.
    std::fill_n(blob, size, 0);
    serializeTo(blob, main, log, elfWriter, sectionMap, symbolMap);

    return {blob, static_cast<uint64_t>(size)};
}

}  // namespace vpux::ELFNPU37XX
