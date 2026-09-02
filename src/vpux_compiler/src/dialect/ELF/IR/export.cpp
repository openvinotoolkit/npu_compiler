//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/export.hpp"
#include "vpux/compiler/backend/export_utils.hpp"

#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"

#include <vpux_elf/writer.hpp>
#include "vpux/utils/core/error.hpp"

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
                              SymbolReferenceMap& symRefMap, DmaSymbolMapType& dmaSymbolMap) {
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
        symTabOp.serialize(elfWriter, sectionMap, symbolMap, dmaSymbolMap, symRefMap);
    }

    for (auto dmaSymTabOp : elfMain.getOps<DmaSymbolSectionOp>()) {
        dmaSymTabOp.serialize(elfWriter, sectionMap, symbolMap, dmaSymbolMap, symRefMap);
    }

    log.trace("Serializing '{0}' ops", CreateRelocationSectionOp::getOperationName());
    for (auto relocSectionOp : elfMain.getOps<CreateRelocationSectionOp>()) {
        relocSectionOp.serialize(elfWriter, sectionMap, symbolMap, dmaSymbolMap, symRefMap);
    }

    return elfWriter;
}

void serializeTo(uint8_t* storage, MainOp elfMain, Logger log, elf::Writer& elfWriter, SectionMapType& sectionMap,
                 SymbolMapType& symbolMap, SymbolReferenceMap& symRefMap, DmaSymbolMapType& dmaSymbolMap) {
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
        dataSectionOp.serialize(elfWriter, sectionMap, symbolMap, dmaSymbolMap, symRefMap);
    }

    log.trace("Serializing '{0}' ops", LogicalSectionOp::getOperationName());
    for (auto logicalSectionOp : elfMain.getOps<LogicalSectionOp>()) {
        logicalSectionOp.serialize(elfWriter, sectionMap, symbolMap, dmaSymbolMap, symRefMap);
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

    DmaSymbolMapType dmaSymbolMap;

    auto elfMain = getElfMainOp(module);
    SymbolReferenceMap symRefMap(elfMain, true);

    return backend::exportToELFCommon(
            [&]() {
                return calculateBlobSize(elfMain, log, sectionMap, symbolMap, symRefMap, dmaSymbolMap);
            },
            [&](uint8_t* storage, elf::Writer& elfWriter) {
                serializeTo(storage, elfMain, log, elfWriter, sectionMap, symbolMap, symRefMap, dmaSymbolMap);
            });
}

BlobView exportToELF(mlir::ModuleOp module, BlobAllocator& allocator, std::string& compatibilityString, Logger log) {
    log.setName("ELF Backend - Export");

    // Associate the respective mlir::Operation* of
    //   DataSectionOp/LogicalSectionOp/CreateSymbolSectionOp/CreateRelocationSectionOp
    //   with the respective created elf::writer::Section* for it.
    SectionMapType sectionMap;
    // Associate the respective mlir::Operation* of a SymbolOp with the newly created
    //   elf::writer::Symbol* for it.
    SymbolMapType symbolMap;

    DmaSymbolMapType dmaSymbolMap;

    auto elfMain = getElfMainOp(module);
    SymbolReferenceMap symRefMap(elfMain, true);

    auto walkResult = elfMain.walk([&](CompatibilityStringOp op) {
        compatibilityString = op.getCompatibilityString().str();
        return mlir::WalkResult::interrupt();
    });
    VPUX_THROW_WHEN(!walkResult.wasInterrupted(), "Compatibility string op not found in IR");
    log.info("Blob compatibility string: '{0}'", compatibilityString);

    return backend::exportToELFCommon(
            [&]() {
                return calculateBlobSize(elfMain, log, sectionMap, symbolMap, symRefMap, dmaSymbolMap);
            },
            [&](uint8_t* storage, elf::Writer& elfWriter) {
                serializeTo(storage, elfMain, log, elfWriter, sectionMap, symbolMap, symRefMap, dmaSymbolMap);
            },
            allocator);
}

}  // namespace vpux::ELF
