//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"

#include <mlir/IR/MLIRContext.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

using namespace vpux;

vpux::SmallString ShaveBinaryResources::getSwKernelArchString(config::ArchKind archKind) {
    switch (archKind) {
    case config::ArchKind::NPU37XX:
        return vpux::SmallString("3720xx");
    case config::ArchKind::NPU40XX:
        return vpux::SmallString("4000xx");
    case config::ArchKind::NPU50XX:
        return vpux::SmallString("5000xx");
    default:
        VPUX_THROW("unsupported archKind {0}", archKind);
        return vpux::SmallString("");
    }
}

llvm::ArrayRef<uint8_t> ShaveBinaryResources::getElf(llvm::StringRef kernelPath) const {
    auto symbolName = printToString("{0}_elf", kernelPath);
    const auto it = _shaveBinaryResourcesMap.find(symbolName);

    VPUX_THROW_UNLESS(it != _shaveBinaryResourcesMap.end(), "Can't find 'elf' for kernel symbol '{0}'", symbolName);

    const auto [symbolData, symbolSize] = it->second;
    return llvm::ArrayRef<uint8_t>(symbolData, symbolSize);
}

void ShaveBinaryResources::addCompiledElf(llvm::StringRef funcName, llvm::ArrayRef<uint8_t> binary,
                                          config::ArchKind archKind, bool overwrite) {
    auto arch = getSwKernelArchString(archKind);
    auto symbolName = printToString("{0}_{1}_elf", funcName, arch);
    auto data = _shaveBinaryResourcesMap.find(symbolName);

    if (data != _shaveBinaryResourcesMap.end()) {
        if (!overwrite) {
            return;
        }
        _shaveBinaryResourcesMap.erase(symbolName);
    }

    // Store data in unique_ptr for memory leak prevention
    auto permArray = std::make_unique<uint8_t[]>(binary.size());
    // Transfer ownership to vector s.t. the unique_ptr is alive for the lifetime of ShaveBinaryResources
    _elfPermStorage.push_back(std::move(permArray));
    auto& ref = _elfPermStorage.back();

    memcpy(ref.get(), binary.data(), binary.size());
    _shaveBinaryResourcesMap.insert(std::make_pair(symbolName, std::make_pair(ref.get(), binary.size())));
}

void ShaveBinaryResources::addCompiledElf(llvm::StringRef funcName, std::unique_ptr<uint8_t[]> binary, size_t size,
                                          config::ArchKind archKind, bool overwrite) {
    auto arch = getSwKernelArchString(archKind);
    auto symbolName = printToString("{0}_{1}_elf", funcName, arch);
    auto data = _shaveBinaryResourcesMap.find(symbolName);

    if (data != _shaveBinaryResourcesMap.end()) {
        if (!overwrite) {
            return;
        }
        _shaveBinaryResourcesMap.erase(symbolName);
    }
    _elfPermStorage.push_back(std::move(binary));
    auto& ref = _elfPermStorage.back();

    _shaveBinaryResourcesMap.insert(std::make_pair(symbolName, std::make_pair(ref.get(), size)));
}

void ShaveBinaryResources::loadElfData(mlir::ModuleOp module) {
#if defined(_WIN32) || defined(_WIN64)
    return;
#endif
    ShaveBinaryResources& sbr = ShaveBinaryResourcesCache::getCache(module->getContext());

    std::string line;

    std::ifstream ifileList("FileList.in", std::ifstream::in);
    if (!ifileList.is_open()) {
        return;
    }

    while (std::getline(ifileList, line)) {
        std::vector<uint8_t> binary;

        std::ifstream ifileElf(line, std::ifstream::in);
        VPUX_THROW_UNLESS(ifileElf.is_open(), "ELF file not found.");

        // Get length of file:
        ifileElf.seekg(0, std::ios::end);
        int length = ifileElf.tellg();
        ifileElf.seekg(0, std::ios::beg);

        auto buffer = std::vector<char>(length);
        ifileElf.read(buffer.data(), length);
        ifileElf.close();

        binary.insert(binary.end(), buffer.begin(), buffer.end());

        std::string funcName;
        std::getline(ifileList, funcName);

        config::ArchKind archKind = config::getArch(module.getOperation());

        sbr.addCompiledElf(funcName, binary, archKind, true);
    }

    ifileList.close();
}

ShaveBinaryResources& ShaveBinaryResourcesCache::getCache(mlir::MLIRContext* ctx) {
    auto* dialect = ctx->getOrLoadDialect<vpux::Core::CoreDialect>();
    assert(dialect != nullptr && "CoreDialect must be present in the context");

    auto* iface = dialect->getRegisteredInterface<ShaveBinaryResourcesCache>();
    assert(iface != nullptr && "The requested cache must be registered in the context");
    return iface->_cache;
}
