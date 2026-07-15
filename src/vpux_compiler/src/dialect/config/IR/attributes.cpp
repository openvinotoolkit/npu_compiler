//
// Copyright (C) 2025-2026 Intel Corporation
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

constexpr StringLiteral platformAttrName = config::PlatformAttr::name;
constexpr StringLiteral abiVersionName = "config.elf_version";
constexpr StringLiteral revisionIDAttrName = "config.revisionID";
constexpr Byte DDR_HEAP_SIZE = 64000_MB;

constexpr StringLiteral derateFactorAttrName = "config.derateFactor";
constexpr StringLiteral bandwidthAttrName = "config.bandwidth"; /*!< This attribute corresponds to a single JSON
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
        VPUX_THROW_UNLESS(mlir::isa<config::CompilationModeAttr>(attr),
                          "Module attribute '{0}' has unsupported value '{1}'", compilationModeAttrName, attr);

        return mlir::cast<config::CompilationModeAttr>(attr).getValue();
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

bool vpux::config::isHostCompileMode(CompilationMode mode) {
    return mode == CompilationMode::HostCompile || mode == CompilationMode::HostCompile_JIT ||
           mode == CompilationMode::HostCompile_Interpreter;
}

bool vpux::config::isHostCompileMode(mlir::Operation* op) {
    return isHostCompileMode(getCompilationMode(op));
}

bool vpux::config::isHostCompileInterpreterMode(CompilationMode mode) {
    return mode == CompilationMode::HostCompile_Interpreter;
}

bool vpux::config::isHostCompileInterpreterMode(mlir::Operation* op) {
    return isHostCompileInterpreterMode(getCompilationMode(op));
}

//
// ArchKind
//

namespace {

struct Resources {
    int numOfDPUGroups = 1;
    std::optional<int> numOfDMAPorts = std::nullopt;
    std::optional<Byte> availableCMXMemory = std::nullopt;

    Resources(int numOfDPUGroups, std::optional<int> numOfDMAPorts, std::optional<Byte> availableCMXMemory)
            : numOfDPUGroups(numOfDPUGroups), numOfDMAPorts(numOfDMAPorts), availableCMXMemory(availableCMXMemory) {
    }
};

struct SetResourcesFuncs {
    using AddGlobalResourcesFuncType = FuncRef<config::ResourcesOp()>;
    using AddTileExecutorFuncType = FuncRef<config::ResourcesOp(size_t)>;
    using AddSubExecutorFuncType =
            FuncRef<config::ExecutorResourceOp(config::ResourcesOp, config::ExecutorKind, size_t)>;
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

void setResources(mlir::ModuleOp module, const Resources& res, const SetResourcesFuncs& funcs, bool allowCustom) {
    const auto kind = config::getArch(module);

    auto numOfDPUGroups = res.numOfDPUGroups;
    auto numOfDMAPorts = res.numOfDMAPorts;
    auto availableCMXMemory = res.availableCMXMemory;

    const auto getNumOfDMAPortsVal = [&](int maxDmaPorts) {
        int numOfDMAPortsVal = numOfDMAPorts.has_value() ? numOfDMAPorts.value() : maxDmaPorts;
        return numOfDMAPortsVal;
    };

    const auto platform = config::getPlatform(module);
    const auto workspaceCMXSize = availableCMXMemory.value_or(VPU::getPlatformCapabilities(platform).cmxWorkspaceSize);

    const auto ddrSymbolAttr = mlir::SymbolRefAttr::get(module.getContext(), stringifyEnum(VPU::MemoryKind::DDR));
    const auto cmxSymbolAttr = mlir::SymbolRefAttr::get(module.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));

    // Common setup: every arch has exactly one global resource op, one tile executor, and one DPU sub-executor.
    config::ResourcesOp nceCluster;
    auto globalResource = funcs.addGlobalResources();
    nceCluster = funcs.addTileExecutor(numOfDPUGroups);
    funcs.addSubExecutor(nceCluster, config::ExecutorKind::DPU, 1);

    switch (kind) {
    case config::ArchKind::NPU37XX: {
        funcs.addSubExecutor(globalResource, config::ExecutorKind::DMA_NN, getNumOfDMAPortsVal(MAX_DMA_PORTS_2));
        funcs.addInnerMemoryWithAttrs(globalResource, ddrSymbolAttr, DDR_HEAP_SIZE, 0.6, 8);
        funcs.addSubExecutor(nceCluster, config::ExecutorKind::SHAVE_NN, 1);
        funcs.addSubExecutor(nceCluster, config::ExecutorKind::SHAVE_ACT, MAX_SHAVES_PER_TILE_2);
        funcs.addInnerMemoryWithAttrs(nceCluster, cmxSymbolAttr, workspaceCMXSize, 1.0, 32);
        break;
    }
    case config::ArchKind::NPU40XX: {
        funcs.addSubExecutor(globalResource, config::ExecutorKind::DMA_NN,
                             getNumOfDMAPortsVal(std::min(numOfDPUGroups, MAX_DMA_PORTS_2)));
        funcs.addSubExecutor(globalResource, config::ExecutorKind::M2I, 1);
        funcs.addInnerMemoryWithAttrs(globalResource, ddrSymbolAttr, DDR_HEAP_SIZE, 0.6, 64);
        funcs.addSubExecutor(nceCluster, config::ExecutorKind::SHAVE_ACT, MAX_SHAVES_PER_TILE_2);
        funcs.addInnerMemoryWithAttrs(nceCluster, cmxSymbolAttr, workspaceCMXSize, 1.0, 64);
        break;
    }
    case config::ArchKind::NPU50XX: {
        funcs.addInnerMemoryWithAttrs(globalResource, ddrSymbolAttr, DDR_HEAP_SIZE, 0.6, 64);
        funcs.addSubExecutor(globalResource, config::ExecutorKind::DMA_NN,
                             getNumOfDMAPortsVal(static_cast<int>(VPU::getMaxDMAPorts(platform))));
        funcs.addSubExecutor(globalResource, config::ExecutorKind::M2I, 1);
        funcs.addSubExecutor(nceCluster, config::ExecutorKind::SHAVE_ACT, MAX_SHAVES_PER_TILE_2);
        funcs.addInnerMemoryWithAttrs(nceCluster, cmxSymbolAttr, workspaceCMXSize, 1.0, 64);
        break;
    }
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }

    VPUX_THROW_WHEN(!allowCustom && nceCluster.hasProcessorFrequency(),
                    "Processor frequencies already defined, probably you run '--init-compiler' twice");
}
}  // namespace

void vpux::config::setParamInModule(mlir::ModuleOp module, config::Platform platform, int numOfDPUGroups,
                                    std::optional<int> numOfDMAPorts, std::optional<Byte> availableCMXMemory,
                                    bool allowCustomValues) {
    const bool hasPlatform = module->hasAttr(platformAttrName);
    VPUX_THROW_WHEN(hasPlatform && !allowCustomValues,
                    "Target platform is already set, probably you run '--init-compiler' twice");
    module->setAttr(platformAttrName, config::PlatformAttr::get(module.getContext(), platform));

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

    const auto addSubExecutor = [&](config::ResourcesOp tileResOp, config::ExecutorKind kind, size_t count) {
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

    Resources res(numOfDPUGroups, numOfDMAPorts, availableCMXMemory);
    SetResourcesFuncs funcs(addGlobalResource, addTileExecutor, addSubExecutor, addInnerAvailableMemory,
                            addInnerAvailableMemoryWithAttrs);

    return setResources(module, res, funcs, allowCustomValues);
}

config::ArchKind vpux::config::getArch(config::Platform platform) {
    switch (platform) {
    case config::Platform::NPU3720:
        return config::ArchKind::NPU37XX;
    case config::Platform::NPU4000:
        return config::ArchKind::NPU40XX;
    case config::Platform::NPU5010:
    case config::Platform::NPU5020:
        return config::ArchKind::NPU50XX;
    default:
        return config::ArchKind::UNKNOWN;
    }
}

config::Platform vpux::config::getPlatform(mlir::Operation* op) {
    auto attr = getModuleOp(op)->getAttr(platformAttrName);
    VPUX_THROW_WHEN(attr == nullptr, "Module does not have a platform attribute");
    return mlir::cast<config::PlatformAttr>(attr).getValue();
}

config::ArchKind vpux::config::getArch(mlir::Operation* op) {
    auto module = getModuleOp(op);
    if (auto attr = module->getAttr(platformAttrName)) {
        auto platform = mlir::cast<config::PlatformAttr>(attr).getValue();
        return getArch(platform);
    }
    return config::ArchKind::UNKNOWN;
}

bool vpux::config::isArchVPUX3XXX(config::ArchKind arch) {
    return (arch == config::ArchKind::NPU37XX);
}

//
// RevisionID
//

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
            VPUX_THROW_UNLESS(mlir::isa<config::RevisionIDAttr>(attr),
                              "Module attribute '{0}' has unsupported value '{1}'", revisionIDAttrName, attr);

            return mlir::cast<config::RevisionIDAttr>(attr).getValue();
        }
    }

    return config::RevisionID::REVISION_NONE;
}

namespace {
constexpr StringLiteral debatchCompileMethod = "config.debatch";
}  // namespace

void config::setCompileMethodDebatch(mlir::ModuleOp module) {
    module->setAttr(debatchCompileMethod, mlir::UnitAttr::get(module.getContext()));
}

bool config::hasCompileMethodDebatch(mlir::ModuleOp module) {
    return module != nullptr ? module->hasAttr(debatchCompileMethod) : false;
}

void config::removeCompileMethodDebatch(mlir::ModuleOp module) {
    if (module != nullptr) {
        module->removeAttr(debatchCompileMethod);
    }
}
//
// PureHostCompileFunc
//

namespace {
constexpr StringLiteral pureHostCompileFunc = "config.pureHostCompileFunc";
}  // namespace

void config::setPureHostCompileFuncAttribute(mlir::func::FuncOp func) {
    func->setAttr(pureHostCompileFunc, mlir::UnitAttr::get(func.getContext()));
}

bool config::isPureHostCompileFunc(mlir::func::FuncOp func) {
    return func != nullptr ? func->hasAttr(pureHostCompileFunc) : false;
}

//
// FunctionToPack
//

namespace {
constexpr StringLiteral functionToPackAttrName = "config.functionToPack";
}  // namespace

void config::setFunctionToPackAttribute(mlir::func::FuncOp func, llvm::StringRef targetModuleName) {
    VPUX_THROW_WHEN(targetModuleName.empty(), "Target module name must not be empty for FunctionToPack attribute");
    func->setAttr(functionToPackAttrName, mlir::StringAttr::get(func.getContext(), targetModuleName));
}

std::string config::getFunctionToPackTargetModule(mlir::func::FuncOp func) {
    if (func == nullptr || !func->hasAttr(functionToPackAttrName)) {
        return {};
    }

    auto attr = func->getAttr(functionToPackAttrName);
    if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr)) {
        return strAttr.getValue().str();
    }

    return {};
}

void config::removeFunctionToPackAttribute(mlir::func::FuncOp func) {
    if (func != nullptr) {
        func->removeAttr(functionToPackAttrName);
    }
}

//
// FunctionToPackEntryPoint
//

namespace {
constexpr StringLiteral functionToPackEntryPointAttrName = "config.functionToPackEntryPoint";
}  // namespace

void config::setFunctionToPackEntryPointAttribute(mlir::func::FuncOp func) {
    if (func != nullptr) {
        func->setAttr(functionToPackEntryPointAttrName, mlir::UnitAttr::get(func.getContext()));
    }
}

bool config::hasFunctionToPackEntryPointAttribute(mlir::func::FuncOp func) {
    return func != nullptr ? func->hasAttr(functionToPackEntryPointAttrName) : false;
}

void config::removeFunctionToPackEntryPointAttribute(mlir::func::FuncOp func) {
    if (func != nullptr) {
        func->removeAttr(functionToPackEntryPointAttrName);
    }
}

//
// PackedModule
//

namespace {
constexpr StringLiteral packedModuleAttrName = "config.packedModule";
}  // namespace

void config::setPackedModuleAttribute(mlir::ModuleOp module) {
    module->setAttr(packedModuleAttrName, mlir::UnitAttr::get(module.getContext()));
}

bool config::hasPackedModuleAttribute(mlir::ModuleOp module) {
    return module != nullptr ? module->hasAttr(packedModuleAttrName) : false;
}

void config::removePackedModuleAttribute(mlir::ModuleOp module) {
    if (module != nullptr) {
        module->removeAttr(packedModuleAttrName);
    }
}

//
// ABI Version
//

void vpux::config::setElfAbiVersion(mlir::ModuleOp moduleOp, const config::Version& version) {
    moduleOp->setAttr(abiVersionName, config::VersionAttr::get(moduleOp.getContext(), version));
}

std::optional<config::Version> vpux::config::getElfAbiVersion(mlir::Operation* op) {
    auto moduleOp = getModuleOp(op);
    if (auto attr = moduleOp->getAttr(abiVersionName)) {
        return mlir::cast<config::VersionAttr>(attr).getVersion();
    }
    return std::nullopt;
}
