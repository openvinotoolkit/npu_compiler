//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/dialect.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Builders.h>

using namespace vpux;

//
// Dialect hooks
//

void config::ConfigDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/config/attributes.cpp.inc>
            >();
}

//
// Generated
//

#include <vpux/compiler/dialect/config/enums.cpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/config/attributes.cpp.inc>

//
// CompilationMode
//

namespace {
//
// Run-time resources
//

constexpr llvm::StringLiteral archAttrName = "config.arch";
constexpr Byte DDR_HEAP_SIZE = 64000_MB;

constexpr llvm::StringLiteral derateFactorAttrName = "config.derateFactor";
constexpr llvm::StringLiteral bandwidthAttrName = "config.bandwidth"; /*!< This attribute corresponds to a single JSON
                      field nested at header>resources>memory_bandwidth>number in the deserialized version of the blob.
                      */

constexpr StringLiteral compilationModeAttrName = "config.compilationMode";

}  // namespace

void vpux::config::setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode) {
    module->setAttr(compilationModeAttrName, config::CompilationModeAttr::get(module.getContext(), compilationMode));
}

bool vpux::config::hasCompilationMode(mlir::ModuleOp module) {
    return module->hasAttr(compilationModeAttrName);
}

config::CompilationMode vpux::config::getCompilationMode(mlir::Operation* op) {
    auto module = getModuleOp(op);

    if (auto attr = module->getAttr(compilationModeAttrName)) {
        VPUX_THROW_UNLESS(mlir::isa<vpux::config::CompilationModeAttr>(attr),
                          "Module attribute '{0}' has unsupported value '{1}'", compilationModeAttrName, attr);

        return mlir::cast<vpux::config::CompilationModeAttr>(attr).getValue();
    }

    // Use DefaultHW as a default mode
    return config::CompilationMode::DefaultHW;
}

StringLiteral vpux::config::getMemoryDerateAttrName() {
    return derateFactorAttrName;
}

StringLiteral vpux::config::getMemoryBandwidthAttrName() {
    return bandwidthAttrName;
}

//
// ArchKind
//

namespace {

struct Resources {
    int numOfDPUGroups = 1;
    std::optional<int> numOfDMAPorts = std::nullopt;
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    Resources(int numOfDPUGroups, std::optional<int> numOfDMAPorts, std::optional<vpux::Byte> availableCMXMemory)
            : numOfDPUGroups(numOfDPUGroups), numOfDMAPorts(numOfDMAPorts), availableCMXMemory(availableCMXMemory) {
    }
};

struct SetResourcesFuncs {
    using AddGlobalResourcesFuncType = FuncRef<config::ResourcesOp()>;
    using AddTileExecutorFuncType = FuncRef<config::ResourcesOp(size_t)>;
    using AddSubExecutorFuncType = FuncRef<config::ExecutorResourceOp(config::ResourcesOp, VPU::ExecutorKind, size_t)>;
    using AddInnerMemoryFuncType = FuncRef<config::MemoryResourceOp(config::ResourcesOp, mlir::SymbolRefAttr, Byte)>;
    using AddInnerMemoryWithAttrsFuncType =
            FuncRef<void(config::ResourcesOp, mlir::SymbolRefAttr, Byte, double, size_t)>;

    AddGlobalResourcesFuncType addGlobalResources;
    AddTileExecutorFuncType addTileExecutor;
    AddSubExecutorFuncType addSubExecutor;
    AddInnerMemoryFuncType addInnerMemory;
    AddInnerMemoryWithAttrsFuncType addInnerMemoryWithAttrs;

    SetResourcesFuncs(AddGlobalResourcesFuncType addGlobalResources, AddTileExecutorFuncType addTileExecutor,
                      AddSubExecutorFuncType addSubExecutor, AddInnerMemoryFuncType addInnerMemory,
                      AddInnerMemoryWithAttrsFuncType addInnerMemoryWithAttrs)
            : addGlobalResources(addGlobalResources),
              addTileExecutor(addTileExecutor),
              addSubExecutor(addSubExecutor),
              addInnerMemory(addInnerMemory),
              addInnerMemoryWithAttrs(addInnerMemoryWithAttrs) {
    }
};

void setArch(mlir::ModuleOp module, config::ArchKind kind, const Resources& res, const SetResourcesFuncs& funcs,
             bool allowCustom) {
    VPUX_THROW_WHEN(!allowCustom && module->hasAttr(archAttrName),
                    "Architecture is already defined, probably you run '--init-compiler' twice");

    if (!module->hasAttr(archAttrName)) {
        module->setAttr(archAttrName, config::ArchKindAttr::get(module.getContext(), kind));
    }

    auto numOfDPUGroups = res.numOfDPUGroups;
    auto numOfDMAPorts = res.numOfDMAPorts;
    auto availableCMXMemory = res.availableCMXMemory;

    const auto getNumOfDMAPortsVal = [&](int maxDmaPorts) {
        int numOfDMAPortsVal = numOfDMAPorts.has_value() ? numOfDMAPorts.value() : maxDmaPorts;
        return numOfDMAPortsVal;
    };

    config::ResourcesOp nceCluster;

    const auto ddrSymbolAttr = mlir::SymbolRefAttr::get(module.getContext(), stringifyEnum(VPU::MemoryKind::DDR));
    const auto cmxSymbolAttr = mlir::SymbolRefAttr::get(module.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto cmxFragAwareSymbolAttr = mlir::SymbolRefAttr::get(module.getContext(), VPU::CMX_NN_FragmentationAware);

    switch (kind) {
    case config::ArchKind::NPU37XX: {
        const auto workspaceCMXSize =
                availableCMXMemory.has_value() ? availableCMXMemory.value() : VPUX37XX_CMX_WORKSPACE_SIZE;
        const auto workspaceFragmentationAwareSize =
                availableCMXMemory.has_value()
                        ? Byte(static_cast<double>(availableCMXMemory.value().count()) * FRAGMENTATION_AVOID_RATIO)
                        : VPUX37XX_CMX_WORKSPACE_FRAGMENTATION_AWARE_SIZE;

        auto globalResource = funcs.addGlobalResources();
        funcs.addSubExecutor(globalResource, VPU::ExecutorKind::DMA_NN, getNumOfDMAPortsVal(VPUX37XX_MAX_DMA_PORTS));
        funcs.addInnerMemoryWithAttrs(globalResource, ddrSymbolAttr, DDR_HEAP_SIZE, 0.6, 8);

        nceCluster = funcs.addTileExecutor(numOfDPUGroups);
        funcs.addSubExecutor(nceCluster, VPU::ExecutorKind::DPU, 1);
        funcs.addSubExecutor(nceCluster, VPU::ExecutorKind::SHAVE_NN, 1);
        funcs.addSubExecutor(nceCluster, VPU::ExecutorKind::SHAVE_ACT, VPUX37XX_MAX_SHAVES_PER_TILE);
        funcs.addInnerMemoryWithAttrs(nceCluster, cmxSymbolAttr, workspaceCMXSize, 1.0, 32);
        funcs.addInnerMemory(nceCluster, cmxFragAwareSymbolAttr, workspaceFragmentationAwareSize);

        break;
    }
    case config::ArchKind::NPU40XX: {
        const auto workspaceCMXSize =
                availableCMXMemory.has_value() ? availableCMXMemory.value() : VPUX40XX_CMX_WORKSPACE_SIZE;
        const auto workspaceFragmentationAwareSize =
                availableCMXMemory.has_value()
                        ? Byte(static_cast<double>(availableCMXMemory.value().count()) * FRAGMENTATION_AVOID_RATIO)
                        : VPUX40XX_CMX_WORKSPACE_FRAGMENTATION_AWARE_SIZE;

        auto globalResource = funcs.addGlobalResources();
        funcs.addSubExecutor(globalResource, VPU::ExecutorKind::DMA_NN,
                             getNumOfDMAPortsVal(std::min(numOfDPUGroups, VPUX40XX_MAX_DMA_PORTS)));
        funcs.addSubExecutor(globalResource, VPU::ExecutorKind::M2I, 1);
        funcs.addInnerMemoryWithAttrs(globalResource, ddrSymbolAttr, DDR_HEAP_SIZE, 0.6, 64);

        nceCluster = funcs.addTileExecutor(numOfDPUGroups);
        funcs.addSubExecutor(nceCluster, VPU::ExecutorKind::DPU, 1);
        funcs.addSubExecutor(nceCluster, VPU::ExecutorKind::SHAVE_ACT, VPUX40XX_MAX_SHAVES_PER_TILE);
        funcs.addInnerMemoryWithAttrs(nceCluster, cmxSymbolAttr, workspaceCMXSize, 1.0, 64);
        funcs.addInnerMemory(nceCluster, cmxFragAwareSymbolAttr, workspaceFragmentationAwareSize);

        break;
    }
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }

    VPUX_THROW_WHEN(!allowCustom && nceCluster.hasProcessorFrequency(),
                    "Processor frequencyis already defined, probably you run '--init-compiler' twice");
}
}  // namespace

void vpux::config::setArch(mlir::ModuleOp module, config::ArchKind kind, int numOfDPUGroups,
                           std::optional<int> numOfDMAPorts, std::optional<vpux::Byte> availableCMXMemory,
                           bool allowCustomValues) {
    const auto addGlobalResource = [&]() {
        VPUX_THROW_WHEN(!allowCustomValues && config::hasGlobalResource(module),
                        "Available global resources was already added");
        if (config::hasGlobalResource(module)) {
            return config::getGlobalResource(module);
        }

        return config::addGlobalResource(module);
    };

    const auto addTileExecutor = [&](size_t count) {
        VPUX_THROW_WHEN(!allowCustomValues && config::hasTileExecutor(module),
                        "Available tile executor was already added");
        if (config::hasTileExecutor(module)) {
            return config::getTileExecutor(module);
        }

        return config::addTileExecutor(module, count);
    };

    const auto addSubExecutor = [&](config::ResourcesOp tileResOp, VPU::ExecutorKind kind, size_t count) {
        VPUX_THROW_WHEN(!allowCustomValues && tileResOp.hasSubExecutor(kind),
                        "Available executor kind '{0}' was already added", kind);
        if (tileResOp.hasSubExecutor(kind)) {
            return tileResOp.getSubExecutor(kind);
        }

        return tileResOp.addSubExecutor(kind, count);
    };

    const auto addInnerAvailableMemory = [&](config::ResourcesOp tileResOp, mlir::SymbolRefAttr memSpace, Byte size) {
        VPUX_THROW_WHEN(!allowCustomValues && tileResOp.hasAvailableMemory(memSpace),
                        "Available memory kind '{0}' was already added", memSpace);
        if (tileResOp.hasAvailableMemory(memSpace)) {
            return tileResOp.getAvailableMemory(memSpace);
        }

        return tileResOp.addAvailableMemory(memSpace, size);
    };

    const auto addInnerAvailableMemoryWithAttrs = [&](config::ResourcesOp tileResOp, mlir::SymbolRefAttr memSpace,
                                                      Byte size, double derateFactor, size_t bandwidth) {
        auto mem = addInnerAvailableMemory(tileResOp, memSpace, size);
        if (!mem->hasAttr(derateFactorAttrName)) {
            mem->setAttr(derateFactorAttrName, getFPAttr(module.getContext(), derateFactor));
        }

        if (!mem->hasAttr(bandwidthAttrName)) {
            mem->setAttr(bandwidthAttrName, getIntAttr(module.getContext(), bandwidth));
        }
    };

    ::Resources res(numOfDPUGroups, numOfDMAPorts, availableCMXMemory);
    ::SetResourcesFuncs funcs(addGlobalResource, addTileExecutor, addSubExecutor, addInnerAvailableMemory,
                              addInnerAvailableMemoryWithAttrs);

    return ::setArch(module, kind, res, funcs, allowCustomValues);
}

config::ArchKind vpux::config::getArch(mlir::Operation* op) {
    auto module = getModuleOp(op);

    if (auto attr = module->getAttr(archAttrName)) {
        VPUX_THROW_UNLESS(mlir::isa<vpux::config::ArchKindAttr>(attr),
                          "Module attribute '{0}' has unsupported value '{1}'", archAttrName, attr);
        return mlir::cast<vpux::config::ArchKindAttr>(attr).getValue();
    }

    return config::ArchKind::UNKNOWN;
}

bool vpux::config::isArchVPUX3XXX(config::ArchKind arch) {
    return (arch == config::ArchKind::NPU37XX);
}

//
// RevisionID
//

namespace {

constexpr StringLiteral revisionIDAttrName = "config.revisionID";

}  // namespace

void vpux::config::setRevisionID(mlir::ModuleOp module, RevisionID revisionID) {
    module->setAttr(revisionIDAttrName, config::RevisionIDAttr::get(module.getContext(), revisionID));
}

bool vpux::config::hasRevisionID(mlir::ModuleOp module) {
    return module->hasAttr(revisionIDAttrName);
}

config::RevisionID vpux::config::getRevisionID(mlir::Operation* op) {
    auto module = getModuleOp(op);

    if (module->hasAttr(revisionIDAttrName)) {
        if (auto attr = module->getAttr(revisionIDAttrName)) {
            VPUX_THROW_UNLESS(mlir::isa<vpux::config::RevisionIDAttr>(attr),
                              "Module attribute '{0}' has unsupported value '{1}'", revisionIDAttrName, attr);

            return mlir::cast<vpux::config::RevisionIDAttr>(attr).getValue();
        }
    }

    return config::RevisionID::REVISION_NONE;
}
