//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/export.hpp"
#include <mlir/Support/WalkResult.h>
#include <algorithm>
#include <cstring>
#include "vpux/compiler/backend/export_utils.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/ops_interfaces.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"

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

    log.trace("Extract '{0}' from Module (ELF File)", net::NetworkInfoOp::getOperationName());

    // Associate the respective mlir::Operation* of
    //   CreateSectionOp/CreateLogicalSectionOp/CreateSymbolSectionOp/CreateRelocationSectionOp
    //   with the respective created elf::writer::Section* for it.
    SectionMapType sectionMap;
    // Associate the respective mlir::Operation* of a SymbolOp with the newly created
    //   elf::writer::Symbol* for it.
    SymbolMapType symbolMap;

    auto main = net::getMainFunc(module);
    return backend::exportToELFCommon(
            [&]() {
                return calculateBlobSize(main, log, sectionMap, symbolMap);
            },
            [&](uint8_t* storage, elf::Writer& elfWriter) {
                serializeTo(storage, main, log, elfWriter, sectionMap, symbolMap);
            });
}

BlobView exportToELF(mlir::ModuleOp module, BlobAllocator& allocator, std::string& compatibilityString, Logger log) {
    log.setName("ELFNPU37XX BackEnd");

    log.trace("Extract '{0}' from Module (ELF File)", net::NetworkInfoOp::getOperationName());

    // Associate the respective mlir::Operation* of
    //   CreateSectionOp/CreateLogicalSectionOp/CreateSymbolSectionOp/CreateRelocationSectionOp
    //   with the respective created elf::writer::Section* for it.
    SectionMapType sectionMap;
    // Associate the respective mlir::Operation* of a SymbolOp with the newly created
    //   elf::writer::Symbol* for it.
    SymbolMapType symbolMap;

    auto main = net::getMainFunc(module);

    auto walkResult = main.walk([&](CompatibilityStringOp op) {
        compatibilityString = op.getCompatibilityString().str();
        return mlir::WalkResult::interrupt();
    });
    VPUX_THROW_WHEN(!walkResult.wasInterrupted(), "Compatibility string op not found in IR");
    log.info("Blob compatibility string: '{0}'", compatibilityString);

    return backend::exportToELFCommon(
            [&]() {
                return calculateBlobSize(main, log, sectionMap, symbolMap);
            },
            [&](uint8_t* storage, elf::Writer& elfWriter) {
                serializeTo(storage, main, log, elfWriter, sectionMap, symbolMap);
            },
            allocator);
}

}  // namespace vpux::ELFNPU37XX
