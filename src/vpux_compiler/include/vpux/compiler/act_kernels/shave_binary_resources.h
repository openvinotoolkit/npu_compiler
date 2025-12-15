//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>

#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/small_string.hpp"

#include <unordered_map>
#include <utility>

extern std::unordered_map<std::string, const std::pair<const uint8_t*, size_t>> shaveBinaryResourcesMap;
extern std::unordered_map<std::string, const std::pair<const uint8_t*, size_t>> shaveBitcodeResourcesMap;
extern std::unordered_map<std::string, const std::pair<const uint8_t*, size_t>> shaveAsmArchiveResourcesMap;

namespace vpux {

class ShaveBinaryResourcesCache;

class ShaveBinaryResources {
private:
    ShaveBinaryResources(): _shaveBinaryResourcesMap(shaveBinaryResourcesMap) {
    }

public:
    ShaveBinaryResources(ShaveBinaryResources const&) = delete;
    void operator=(ShaveBinaryResources const&) = delete;
    ~ShaveBinaryResources() = default;

    template <typename... Args>
    std::string concatenateArgs(Args&&... args) const {
        std::string result;
        ((result += ("_" + std::forward<Args>(args).str())), ...);
        return result;
    }

    template <typename... Args>
    llvm::ArrayRef<uint8_t> getElf(llvm::StringRef entry, llvm::StringRef cpu, Args&&... args) const {
        auto result = printToString("{0}_{1}", entry, cpu);
        auto argsConcat = concatenateArgs(std::forward<Args>(args)...);
        auto symbolName = printToString("{0}{1}_elf", result, argsConcat);
        const auto it = _shaveBinaryResourcesMap.find(symbolName);

        VPUX_THROW_UNLESS(it != _shaveBinaryResourcesMap.end(), "Can't find 'elf' for kernel symbol '{0}'", symbolName);

        const auto [symbolData, symbolSize] = it->second;
        return llvm::ArrayRef<uint8_t>(symbolData, symbolSize);
    }

    llvm::ArrayRef<uint8_t> getElf(llvm::StringRef kernelPath) const;

    void addCompiledElf(llvm::StringRef funcName, llvm::ArrayRef<uint8_t> binary, config::ArchKind archKind,
                        bool overwrite = false);

    void addCompiledElf(llvm::StringRef funcName, std::unique_ptr<uint8_t[]> binary, size_t size,
                        config::ArchKind archKind, bool overwrite = false);

    static void loadElfData(mlir::ModuleOp module);

    static vpux::SmallString getSwKernelArchString(config::ArchKind archKind);

private:
    // Storage for dynamically-compiled SHAVE kernels
    std::vector<std::unique_ptr<uint8_t[]>> _elfPermStorage;
    // Mapping to the address and size of the pre-compiled and dynamically-compiled SHAVE kernels
    std::unordered_map<std::string, const std::pair<const uint8_t*, size_t>> _shaveBinaryResourcesMap;

    friend ShaveBinaryResourcesCache;
};

class ShaveBinaryResourcesCache final : public mlir::DialectInterface::Base<ShaveBinaryResourcesCache> {
private:
    ShaveBinaryResources _cache;

public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShaveBinaryResourcesCache)

    ShaveBinaryResourcesCache(mlir::Dialect* dialect): Base(dialect) {
    }

    static ShaveBinaryResources& getCache(mlir::MLIRContext* ctx);
};

}  // namespace vpux
