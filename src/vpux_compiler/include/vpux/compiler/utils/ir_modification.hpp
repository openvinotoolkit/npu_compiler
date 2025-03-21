//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

#include <algorithm>

namespace vpux::utils {

/** @brief Caches identical function block arguments specified by cache entries.

    This cache handles identical block arguments, that are defined generically
    via the cache entries, and thus allows implicit argument deduplication.
    Deduplication is useful to ensure optimal I/O bandwidth utilization.
*/
template <typename CacheEntry>
class ArgumentCache {
    using Cache = mlir::DenseMap<CacheEntry, mlir::BlockArgument>;

    mlir::Block* _block;
    mlir::Location _loc;
    Cache _cache;

public:
    //! @brief Creates a new cache object. Assumes block is not nullptr.
    ArgumentCache(mlir::Block* block, mlir::Location loc): _block(block), _loc(loc) {
        assert(block != nullptr && "Specified block shouldn't be nullptr");
    }
    //! @brief Creates a new cache object. Assumes function has a body.
    ArgumentCache(mlir::func::FuncOp funcOp): ArgumentCache(&funcOp.getFunctionBody().front(), funcOp.getLoc()) {
    }

    //! @brief Returns a block argument that corresponds to the specified entry.
    mlir::BlockArgument addArgument(const CacheEntry& entry, mlir::Type type) {
        auto& value = _cache[entry];
        if (value == nullptr) {
            const auto newLoc = appendLoc(_loc, "arg_{0}", _block->getNumArguments());
            value = _block->addArgument(type, newLoc);
        }

        return value;
    }

    //! @brief Finds a block argument associated with the specified entry.
    mlir::BlockArgument findArgument(const CacheEntry& entry) const {
        const auto it = _cache.find(entry);
        VPUX_THROW_WHEN(it == _cache.end(), "Failed to find cached function argument");
        return it->second;
    }

    //! @brief Returns {entry, block arg} sorted by block argument index.
    SmallVector<typename Cache::const_iterator> getSortedArgs() const {
        // sort the args by block argument's position to ensure stability of the
        // returned results (users depend on this)
        using ArgElement = typename Cache::const_iterator;
        SmallVector<ArgElement> args;
        args.reserve(_cache.size());
        for (auto begin = _cache.begin(); begin != _cache.end(); ++begin) {
            args.push_back(begin);
        }
        std::sort(args.begin(), args.end(), [](ArgElement x, ArgElement y) {
            return x->second.getArgNumber() < y->second.getArgNumber();
        });
        return args;
    }
};

}  // namespace vpux::utils
