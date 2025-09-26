//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>

using namespace vpux;

//
// Declarations
//

namespace {

bool isNceTileMemory(mlir::SymbolRefAttr memSpace) {
    auto memSpaceStr = memSpace.getRootReference().getValue();
    return memSpaceStr == stringifyEnum(VPU::MemoryKind::CMX_NN) || memSpaceStr == VPU::CMX_NN_FragmentationAware;
}

config::ResourcesOp getResources(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    auto resources =
            isNceTileMemory(memSpace) ? config::getTileExecutor(mainModule) : config::getGlobalResource(mainModule);
    VPUX_THROW_UNLESS(resources != nullptr, "Cannot find config.ResourcesOp in order to query '{0}' memspace.",
                      memSpace);
    return resources;
}

bool isNceTileExecutor(mlir::SymbolRefAttr executor) {
    auto nceExecutorList =
            SmallVector<StringRef>({stringifyEnum(VPU::ExecutorKind::DPU), stringifyEnum(VPU::ExecutorKind::SHAVE_ACT),
                                    stringifyEnum(VPU::ExecutorKind::SHAVE_NN)});
    auto executorStr = executor.getLeafReference().getValue();
    return std::find(nceExecutorList.begin(), nceExecutorList.end(), executorStr) != nceExecutorList.end();
}

mlir::Region* getRegionContainer(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    auto resources = getResources(mainModule, memSpace);
    VPUX_THROW_UNLESS(resources != nullptr, "Cannot find config.ResourcesOp in order to query '{0}' memspace.",
                      memSpace);
    return &resources.getRegion();
}

size_t getReservedMemoryStartOffset(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    size_t startMemoryOffset = 0;

    // DDR reserved memory starts at 0. Adjustment is required only for CMX
    auto cmxSpaceAttr = mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));
    if (memSpace == cmxSpaceAttr) {
        startMemoryOffset = config::getAvailableMemory(mainModule, cmxSpaceAttr).size().count();
        for (auto resource : config::getReservedMemoryResources(mainModule, memSpace)) {
            VPUX_THROW_WHEN(!resource.getOffset().has_value(), "reserved memory without offset");
            size_t offset = resource.getOffset().value();
            if (offset < startMemoryOffset) {
                startMemoryOffset = offset;
            }
        }
    }

    return startMemoryOffset;
}

size_t getReservedMemoryEndOffset(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    // CMX reserved memory always ends at the CMX end.
    auto cmxSpaceAttr = mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));
    if (memSpace == cmxSpaceAttr) {
        return config::getAvailableMemory(mainModule, cmxSpaceAttr).size().count() - 1;
    }

    size_t endMemoryOffset = 0;
    for (auto resource : config::getReservedMemoryResources(mainModule, memSpace)) {
        VPUX_THROW_WHEN(!resource.getOffset().has_value(), "reserved memory without offset value");
        size_t offset = resource.getOffset().value() + resource.getByteSize();
        if (offset > endMemoryOffset) {
            endMemoryOffset = offset;
        }
    }

    return endMemoryOffset;
}

config::MemoryResourceOp addReservedMemoryResource(mlir::ModuleOp mainModule, mlir::StringLiteral reservedMemorySection,
                                                   mlir::SymbolRefAttr memSpace, int64_t size, size_t alignment = 1) {
    auto region = getRegionContainer(mainModule, memSpace);
    auto resources = getResources(mainModule, memSpace);
    auto resMemTable = resources.lookupSymbol<mlir::ModuleOp>(config::resMemModuleName);
    if (resMemTable == nullptr) {
        auto mainBuilder = mlir::OpBuilder::atBlockBegin(&region->front());
        resMemTable = mainBuilder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(mainBuilder.getContext()),
                                                         config::resMemModuleName);
    }

    auto resMemBuilder = mlir::OpBuilder::atBlockBegin(resMemTable.getBody());
    auto cmxSpaceAttr = mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));
    // For DDR - reserve memory at the beginning of DDR space. The offset should be aligned according to the specified
    // alignment.
    // For CMX - reserve at the end of CMX space. This is done to satisfy the requirement of SW kernel data
    // prefetching. When prefetching SW kernel can exceed the input buffer size potentially reading the memory outside
    // the CMX range. To prevent this compiler reserves 1KiB of CMX at the end so that at worst reserved, but
    // accessible, memory is read by SW kernel.
    size_t offset = 0;
    if (memSpace == cmxSpaceAttr) {
        offset = getReservedMemoryStartOffset(mainModule, memSpace);
        VPUX_THROW_WHEN(static_cast<int64_t>(offset) < size, "Out of CMX memory for reservation");
        offset -= size;
        offset = alignValDown(offset, alignment);
    } else {
        offset = getReservedMemoryEndOffset(mainModule, memSpace);
        offset = alignValUp(offset, alignment);
    }

    auto resMemModule = resMemTable.lookupSymbol<mlir::ModuleOp>(reservedMemorySection);
    if (resMemModule == nullptr) {
        resMemModule = resMemBuilder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(resMemBuilder.getContext()),
                                                            reservedMemorySection);
    }

    auto* ctx = resources.getContext();
    auto byteSizeAttr = getIntAttr(ctx, size);

    auto res = resMemModule.lookupSymbol<config::MemoryResourceOp>(memSpace);
    if (res != nullptr) {
        res.setByteSizeAttr(byteSizeAttr);
        return res;
    }

    auto innerBuilder = mlir::OpBuilder::atBlockBegin(resMemModule.getBody());
    return innerBuilder.create<config::MemoryResourceOp>(mlir::UnknownLoc::get(resMemModule.getContext()),
                                                         memSpace.getLeafReference(), byteSizeAttr,
                                                         getIntAttr(mainModule.getContext(), offset));
};

config::MemoryResourceOp getReservedMemoryResource(mlir::ModuleOp mainModule, mlir::StringLiteral reservedMemorySection,
                                                   mlir::SymbolRefAttr memSpace) {
    auto resources = getResources(mainModule, memSpace);
    auto resMemTable = resources.lookupSymbol<mlir::ModuleOp>(config::resMemModuleName);
    if (resMemTable == nullptr) {
        return nullptr;
    }

    auto resMemModule = resMemTable.lookupSymbol<mlir::ModuleOp>(reservedMemorySection);
    if (resMemModule == nullptr) {
        return nullptr;
    }

    return resMemModule.lookupSymbol<config::MemoryResourceOp>(memSpace);
}

}  // namespace

bool vpux::config::isNceTile(mlir::SymbolRefAttr executor) {
    return executor.getLeafReference().getValue() == stringifyEnum(VPU::ExecutorKind::NCE);
}

config::MemoryResourceOp vpux::config::getAvailableMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    return getResources(mainModule, memSpace).getAvailableMemory(memSpace);
}

//
// Reserved memory resources
//

SmallVector<config::MemoryResourceOp> vpux::config::getReservedMemoryResources(mlir::ModuleOp mainModule,
                                                                               mlir::SymbolRefAttr memSpace) {
    auto resources = getResources(mainModule, memSpace);
    SmallVector<config::MemoryResourceOp> resMemVec;

    auto resMemModule = resources.lookupSymbol<mlir::ModuleOp>(resMemModuleName);
    if (resMemModule == nullptr) {
        return {};
    }

    for (auto&& resMemModuleOp : resMemModule.getOps<mlir::ModuleOp>()) {
        resMemVec.push_back(resMemModuleOp.lookupSymbol<config::MemoryResourceOp>(memSpace));
    }

    return resMemVec;
}

size_t vpux::config::getReservedMemorySize(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    return getReservedMemoryEndOffset(mainModule, memSpace) - getReservedMemoryStartOffset(mainModule, memSpace) + 1;
}

// Get information about reserved resources in given memory type
// This function should be called before performing memory allocation
SmallVector<std::pair<uint64_t, uint64_t>> vpux::config::getReservedMemOffsetAndSizeVec(
        mlir::ModuleOp module, mlir::SymbolRefAttr memSpaceAttr) {
    SmallVector<std::pair<uint64_t, uint64_t>> reservedMemVec;
    // Check for reserved memory which memory scheduler should take into account
    // so that they not overlap with other buffers. Those reserved resource might be related
    // to handling of additional special features (e.g. DMA HW profiling)
    for (auto& resMem : config::getReservedMemoryResources(module, memSpaceAttr)) {
        VPUX_THROW_UNLESS(resMem.getOffset().has_value(), "reserved memory resource without offset value");
        reservedMemVec.push_back(std::make_pair(resMem.getOffset().value(), resMem.getByteSize()));
    }

    return reservedMemVec;
}

//
// DMA profiling reserved memory
//

config::MemoryResourceOp vpux::config::setDmaProfilingReservedMemory(mlir::ModuleOp mainModule,
                                                                     mlir::SymbolRefAttr memSpace, int64_t size) {
    return addReservedMemoryResource(mainModule, dmaProfilingResMemModuleName, memSpace, size,
                                     dmaProfilingResMemAlignment.count());
}

config::MemoryResourceOp vpux::config::getDmaProfilingReservedMemory(mlir::ModuleOp mainModule,
                                                                     mlir::SymbolRefAttr memSpace) {
    return getReservedMemoryResource(mainModule, dmaProfilingResMemModuleName, memSpace);
}

//
// Compressed DMA reserved memory
//

config::MemoryResourceOp vpux::config::setCompressDmaReservedMemory(mlir::ModuleOp mainModule,
                                                                    mlir::SymbolRefAttr memSpace, int64_t size) {
    return addReservedMemoryResource(mainModule, compressDmaResMemModuleName, memSpace, size);
}

config::MemoryResourceOp vpux::config::getCompressDmaReservedMemory(mlir::ModuleOp mainModule,
                                                                    mlir::SymbolRefAttr memSpace) {
    return getReservedMemoryResource(mainModule, compressDmaResMemModuleName, memSpace);
}

//
// SW Kernel prefetching reserved memory
//

config::MemoryResourceOp vpux::config::setSWKernelPrefetchingReservedMemory(mlir::ModuleOp mainModule,
                                                                            mlir::SymbolRefAttr memSpace,
                                                                            int64_t size) {
    return addReservedMemoryResource(mainModule, swKernelPrefetchingResMemModuleName, memSpace, size);
}

config::MemoryResourceOp vpux::config::getSWKernelPrefetchingReservedMemory(mlir::ModuleOp mainModule,
                                                                            mlir::SymbolRefAttr memSpace) {
    return getReservedMemoryResource(mainModule, swKernelPrefetchingResMemModuleName, memSpace);
}

//
// Dummy SW Kernel prefetch reserved memory
//

config::MemoryResourceOp vpux::config::setDummySwKernelsForInstructionPrefetchReservedMemory(
        mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace, int64_t size) {
    return addReservedMemoryResource(mainModule, dummySwKernelsForInstructionPrefetchResMemModuleName, memSpace, size);
}

config::MemoryResourceOp vpux::config::getDummySwKernelsForInstructionPrefetchReservedMemory(
        mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    return getReservedMemoryResource(mainModule, dummySwKernelsForInstructionPrefetchResMemModuleName, memSpace);
}

//
// ExecutorResourceOp
//

config::ExecutorResourceOp vpux::config::getAvailableExecutor(mlir::ModuleOp mainModule,
                                                              mlir::SymbolRefAttr executorAttr) {
    VPUX_THROW_UNLESS(!config::isNceTile(executorAttr), "Unexpected '{0}' during executor query.", executorAttr);
    auto resources = isNceTileExecutor(executorAttr) ? config::getTileExecutor(mainModule)
                                                     : config::getGlobalResource(mainModule);
    VPUX_THROW_UNLESS(resources != nullptr, "Cannot find config.ResourcesOp in order to query '{0}' executor.",
                      executorAttr);
    return resources.getSubExecutor(executorAttr);
}

mlir::LogicalResult vpux::config::ExecutorResourceOp::verify() {
    if (getCount() <= 0) {
        return errorAt(*this, "Number of executor units should be a positive integer, while it is {0}", getCount());
    }
    return mlir::success();
}

//
// ResourcesOp
//

config::ExecutorResourceOp vpux::config::ResourcesOp::addSubExecutor(mlir::SymbolRefAttr executorAttr, size_t count) {
    auto& region = getRegion();
    VPUX_THROW_UNLESS(count > 0, "Trying to set zero count of executor kind '{0}'", executorAttr);
    VPUX_THROW_UNLESS(!config::isNceTile(executorAttr), "Unexpected '{0}' during executor query.", executorAttr);
    auto* ctx = region.getContext();
    const auto countAttr = getIntAttr(ctx, count);
    auto builder = mlir::OpBuilder::atBlockBegin(&region.front());
    return builder.create<config::ExecutorResourceOp>(mlir::UnknownLoc::get(ctx), executorAttr.getLeafReference(),
                                                      countAttr, nullptr);
}

bool vpux::config::ResourcesOp::hasSubExecutor(mlir::SymbolRefAttr executorAttr) {
    VPUX_THROW_UNLESS(!config::isNceTile(executorAttr), "Unexpected '{0}' during executor query.", executorAttr);
    return lookupSymbol<config::ExecutorResourceOp>(executorAttr.getLeafReference()) != nullptr;
}

config::ExecutorResourceOp vpux::config::ResourcesOp::getSubExecutor(mlir::SymbolRefAttr executorAttr) {
    VPUX_THROW_WHEN(!hasSubExecutor(executorAttr), "Cannot find executor kind '{0}' in order to query its information",
                    executorAttr);
    return lookupSymbol<config::ExecutorResourceOp>(executorAttr.getLeafReference());
}

config::MemoryResourceOp vpux::config::ResourcesOp::addAvailableMemory(mlir::SymbolRefAttr memSpace, Byte size) {
    auto& region = getRegion();
    VPUX_THROW_UNLESS(size.count() > 0, "Trying to set zero size of memory kind '{0}'", memSpace);
    const auto byteSizeAttr = getIntAttr(region.getContext(), size.count());
    auto builder = mlir::OpBuilder::atBlockBegin(&region.front());
    return builder.create<config::MemoryResourceOp>(mlir::UnknownLoc::get(region.getContext()),
                                                    memSpace.getLeafReference(), byteSizeAttr, /*offset*/ nullptr);
}

bool vpux::config::ResourcesOp::hasAvailableMemory(mlir::SymbolRefAttr memSpace) {
    return lookupSymbol<config::MemoryResourceOp>(memSpace.getLeafReference()) != nullptr;
}

config::MemoryResourceOp vpux::config::ResourcesOp::getAvailableMemory(mlir::SymbolRefAttr memSpace) {
    VPUX_THROW_WHEN(!hasAvailableMemory(memSpace), "Cannot find memory kind '{0}' in order to query its information",
                    memSpace);
    return lookupSymbol<config::MemoryResourceOp>(memSpace);
}

mlir::LogicalResult vpux::config::ResourcesOp::verify() {
    if (getCount() <= 0) {
        return errorAt(*this, "Number of executor units should be a positive integer, while it is {0}", getCount());
    }
    return mlir::success();
}

//
// EngineResources
//

int64_t vpux::config::getTotalNumOfEngines(mlir::ModuleOp moduleOp, VPU::ExecutorKind execKind) {
    auto tileOp = getTileExecutor(moduleOp);
    VPUX_THROW_UNLESS(tileOp != nullptr, "Expected tileOp executor in order to query {0} executor.", execKind);
    auto executorPerTile = tileOp.getSubExecutor(execKind);
    VPUX_THROW_UNLESS(executorPerTile != nullptr, "Failed to get {0} information", execKind);
    return tileOp.getCount() * executorPerTile.getCount();
}

int64_t vpux::config::getTotalNumOfEngines(mlir::Operation* op, VPU::ExecutorKind execKind) {
    return getTotalNumOfEngines(op->getParentOfType<mlir::ModuleOp>(), execKind);
}

config::ResourcesOp config::addTileExecutor(mlir::ModuleOp mainModule, size_t count) {
    VPUX_THROW_UNLESS(count > 0, "Trying to set zero count of tile resource kind.");

    auto builder = mlir::OpBuilder::atBlockBegin(mainModule.getBody());
    return builder.create<config::ResourcesOp>(appendLoc(mainModule.getLoc(), "_tile_resources"),
                                               stringifyEnum(VPU::ExecutorKind::NCE), count);
}

bool config::hasTileExecutor(mlir::ModuleOp mainModule) {
    auto res = mainModule.lookupSymbol<config::ResourcesOp>(stringifyEnum(VPU::ExecutorKind::NCE));
    return res != nullptr;
}

config::ResourcesOp config::getTileExecutor(mlir::ModuleOp mainModule) {
    return mainModule.lookupSymbol<config::ResourcesOp>(stringifyEnum(VPU::ExecutorKind::NCE));
}

config::ResourcesOp config::getTileExecutor(mlir::func::FuncOp funcOp) {
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    return config::getTileExecutor(moduleOp);
}

namespace {

static constexpr auto GLOBAL_RESOURCE_NAME = "global";

}

config::ResourcesOp config::addGlobalResource(mlir::ModuleOp moduleOp) {
    auto builder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());
    return builder.create<config::ResourcesOp>(appendLoc(moduleOp.getLoc(), "_global_resources"), GLOBAL_RESOURCE_NAME);
}

bool config::hasGlobalResource(mlir::ModuleOp moduleOp) {
    auto res = moduleOp.lookupSymbol<config::ResourcesOp>(GLOBAL_RESOURCE_NAME);
    return res != nullptr;
}

config::ResourcesOp config::getGlobalResource(mlir::ModuleOp moduleOp) {
    return moduleOp.lookupSymbol<config::ResourcesOp>(GLOBAL_RESOURCE_NAME);
}

config::ResourcesOp config::getGlobalResource(mlir::func::FuncOp funcOp) {
    return config::getGlobalResource(funcOp->getParentOfType<mlir::ModuleOp>());
}
