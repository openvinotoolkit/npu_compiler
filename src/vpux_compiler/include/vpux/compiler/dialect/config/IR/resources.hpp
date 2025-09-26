//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::VPU {
enum class ExecutorKind : uint64_t;
}

namespace vpux {
namespace config {

//
// Hierarchy aware utils
//

bool isNceTile(mlir::SymbolRefAttr executor);

//
// MemoryResourceOp
//

template <typename Enum, typename OutT = config::MemoryResourceOp>
using memory_resource_if = enable_t<OutT, std::is_enum<Enum>, vpux::details::HasStringifyEnum<Enum>>;

config::MemoryResourceOp getAvailableMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getAvailableMemory(mlir::ModuleOp mainModule, Enum kind) {
    return getAvailableMemory(mainModule, mlir::SymbolRefAttr::get(mainModule->getContext(), stringifyEnum(kind)));
}

//
// Reserved memory resource
//
static constexpr StringLiteral resMemModuleName = "ReservedMemory";

SmallVector<config::MemoryResourceOp> getReservedMemoryResources(mlir::ModuleOp mainModule,
                                                                 mlir::SymbolRefAttr memSpace);

SmallVector<std::pair<uint64_t, uint64_t>> getReservedMemOffsetAndSizeVec(mlir::ModuleOp module,
                                                                          mlir::SymbolRefAttr memSpaceAttr);

size_t getReservedMemorySize(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

//
// DMA profiling reserved memory
//
static constexpr StringLiteral dmaProfilingResMemModuleName = "DmaProfilingReservedMemory";

static constexpr auto dmaProfilingResMemAlignment = Byte(64);

config::MemoryResourceOp setDmaProfilingReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace,
                                                       int64_t size);

config::MemoryResourceOp getDmaProfilingReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getDmaProfilingReservedMemory(mlir::ModuleOp mainModule, Enum kind) {
    return getDmaProfilingReservedMemory(mainModule,
                                         mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(kind)));
}

//
// Compressed DMA reserved memory
//
static constexpr StringLiteral compressDmaResMemModuleName = "CompressDmaReservedMemory";

config::MemoryResourceOp setCompressDmaReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace,
                                                      int64_t size);

config::MemoryResourceOp getCompressDmaReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getCompressDmaReservedMemory(mlir::ModuleOp mainModule, Enum kind) {
    return getCompressDmaReservedMemory(mainModule,
                                        mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(kind)));
}

//
// SW Kernel prefetching reserved memory
//
static constexpr StringLiteral swKernelPrefetchingResMemModuleName = "SWKernelPrefetchingReservedMemory";

config::MemoryResourceOp setSWKernelPrefetchingReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace,
                                                              int64_t size);

config::MemoryResourceOp getSWKernelPrefetchingReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getSWKernelPrefetchingReservedMemory(mlir::ModuleOp mainModule, Enum kind) {
    return getSWKernelPrefetchingReservedMemory(mainModule,
                                                mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(kind)));
}

//
// Dummy SW kernels for instruction prefetch reserved memory
//
static constexpr StringLiteral dummySwKernelsForInstructionPrefetchResMemModuleName =
        "DummySWKernelsForInstructionPrefetchReservedMemory";

config::MemoryResourceOp setDummySwKernelsForInstructionPrefetchReservedMemory(mlir::ModuleOp mainModule,
                                                                               mlir::SymbolRefAttr memSpace,
                                                                               int64_t size);

config::MemoryResourceOp getDummySwKernelsForInstructionPrefetchReservedMemory(mlir::ModuleOp mainModule,
                                                                               mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getDummySwKernelsForInstructionPrefetchReservedMemory(mlir::ModuleOp mainModule, Enum kind) {
    return getDummySwKernelsForInstructionPrefetchReservedMemory(
            mainModule, mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(kind)));
}

//
// ExecutorResourceOp
//

template <typename Enum, typename OutT = config::ExecutorResourceOp>
using exec_resource_if = enable_t<OutT, std::is_enum<Enum>, vpux::details::HasStringifyEnum<Enum>>;

config::ExecutorResourceOp getAvailableExecutor(mlir::ModuleOp mainModule, mlir::SymbolRefAttr executorAttr);

template <typename Enum>
exec_resource_if<Enum> getAvailableExecutor(mlir::ModuleOp mainModule, Enum kind) {
    return getAvailableExecutor(mainModule, mlir::SymbolRefAttr::get(mainModule->getContext(), stringifyEnum(kind)));
}

//
// EngineResources
//

int64_t getTotalNumOfEngines(mlir::ModuleOp moduleOp, VPU::ExecutorKind execKind);
int64_t getTotalNumOfEngines(mlir::Operation* op, VPU::ExecutorKind execKind);

config::ResourcesOp addTileExecutor(mlir::ModuleOp mainModule, size_t count);

bool hasTileExecutor(mlir::ModuleOp mainModule);

config::ResourcesOp getTileExecutor(mlir::ModuleOp mainModule);

config::ResourcesOp getTileExecutor(mlir::func::FuncOp funcOp);

config::ResourcesOp addGlobalResource(mlir::ModuleOp moduleOp);
bool hasGlobalResource(mlir::ModuleOp moduleOp);
config::ResourcesOp getGlobalResource(mlir::ModuleOp moduleOp);
config::ResourcesOp getGlobalResource(mlir::func::FuncOp funcOp);

}  // namespace config
}  // namespace vpux
