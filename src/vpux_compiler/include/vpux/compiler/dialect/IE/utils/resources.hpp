//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"

namespace vpux {
namespace IE {

//
// MemoryResourceOp
//

static constexpr StringLiteral usedMemModuleName = "UsedMemory";

MemoryResourceOp addAvailableMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace, Byte size);

template <typename Enum, typename OutT = MemoryResourceOp>
using memory_resource_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

template <typename Enum>
memory_resource_if<Enum> addAvailableMemory(mlir::ModuleOp mainModule, Enum kind, Byte size) {
    return addAvailableMemory(mainModule, mlir::StringAttr::get(mainModule.getContext(), stringifyEnum(kind)), size);
}

MemoryResourceOp getAvailableMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getAvailableMemory(mlir::ModuleOp mainModule, Enum kind) {
    return mainModule.template lookupSymbol<MemoryResourceOp>(stringifyEnum(kind));
}

MemoryResourceOp setUsedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace, Byte size);

template <typename Enum>
memory_resource_if<Enum> setUsedMemory(mlir::ModuleOp mainModule, Enum kind, Byte size) {
    return setUsedMemory(mainModule, mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(kind)), size);
}

MemoryResourceOp getUsedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getUsedMemory(mlir::ModuleOp mainModule, Enum kind) {
    return getUsedMemory(mainModule, mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(kind)));
}

SmallVector<MemoryResourceOp> getUsedMemory(mlir::ModuleOp mainModule);

//
// ExecutorResourceOp
//

namespace details {

ExecutorResourceOp addExecutor(mlir::SymbolTable mainModule, mlir::Region& region, mlir::StringAttr executorAttr,
                               uint32_t count);

}

template <typename Enum, typename OutT = ExecutorResourceOp>
using exec_resource_if = enable_t<OutT, std::is_enum<Enum>, vpux::details::HasStringifyEnum<Enum>>;

template <typename Enum>
exec_resource_if<Enum> addAvailableExecutor(mlir::ModuleOp mainModule, Enum kind, uint32_t count) {
    const auto executorAttr = mlir::StringAttr::get(mainModule->getContext(), stringifyEnum(kind));
    return details::addExecutor(mainModule.getOperation(), mainModule.body(), executorAttr, count);
}

ExecutorResourceOp getAvailableExecutor(mlir::ModuleOp mainModule, mlir::SymbolRefAttr executorAttr);

template <typename Enum>
exec_resource_if<Enum> getAvailableExecutor(mlir::ModuleOp mainModule, Enum kind) {
    return getAvailableExecutor(mainModule, mlir::SymbolRefAttr::get(mainModule->getContext(), stringifyEnum(kind)));
}

}  // namespace IE
}  // namespace vpux
