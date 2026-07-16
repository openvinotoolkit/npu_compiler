//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/thread_safe_hash_map.hpp"
#include "vpux/utils/core/dense_map.hpp"

#ifdef TBB_AVAILABLE
#include <tbb/concurrent_hash_map.h>
#endif

#include <llvm/ADT/Hashing.h>
#include <mutex>
#include <optional>

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/thread_safe_accessors.hpp"

namespace vpux {
namespace details {

// ============================================================================
// MutexHashMapImpl Implementation
// ============================================================================

template <typename KeyT, typename ValueT>
class MutexHashMapImpl {
public:
    std::optional<ValueT> find(const KeyT& key) const {
        auto handle = _map.lock();
        const auto& map = *handle;
        auto it = map.find(key);
        if (it != map.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    void insert(const KeyT& key, const ValueT& value) {
        _map->insert({key, value});
    }

    void clear() {
        _map->clear();
    }

private:
    SimpleThreadSafeAccessor<DenseMap<KeyT, ValueT>> _map;
};

// ============================================================================
// TBBHashMapImpl Implementation
// ============================================================================

#ifdef TBB_AVAILABLE
template <typename KeyT, typename ValueT>
class TBBHashMapImpl {
public:
    std::optional<ValueT> find(const KeyT& key) const {
        typename MapType::const_accessor accessor;
        if (_map.find(accessor, key)) {
            return accessor->second;
        }
        return std::nullopt;
    }

    void insert(const KeyT& key, const ValueT& value) {
        typename MapType::accessor accessor;
        _map.insert(accessor, key);
        accessor->second = value;
    }

    void clear() {
        _map.clear();
    }

private:
    tbb::concurrent_hash_map<KeyT, ValueT> _map;
};
#endif

}  // namespace details

// ============================================================================
// ThreadSafeHashMap Implementation
// ============================================================================

template <typename KeyT, typename ValueT>
class ThreadSafeHashMap<KeyT, ValueT>::Impl {
#ifdef TBB_AVAILABLE
    details::TBBHashMapImpl<KeyT, ValueT> _map;
#else
    details::MutexHashMapImpl<KeyT, ValueT> _map;
#endif

public:
    std::optional<ValueT> find(const KeyT& key) const {
        return _map.find(key);
    }

    void insert(const KeyT& key, const ValueT& value) {
        _map.insert(key, value);
    }

    void clear() {
        _map.clear();
    }
};

template <typename KeyT, typename ValueT>
ThreadSafeHashMap<KeyT, ValueT>::ThreadSafeHashMap(): _impl(std::make_unique<Impl>()) {
}

template <typename KeyT, typename ValueT>
ThreadSafeHashMap<KeyT, ValueT>::~ThreadSafeHashMap() = default;

template <typename KeyT, typename ValueT>
ThreadSafeHashMap<KeyT, ValueT>::ThreadSafeHashMap(ThreadSafeHashMap&&) noexcept = default;

template <typename KeyT, typename ValueT>
ThreadSafeHashMap<KeyT, ValueT>& ThreadSafeHashMap<KeyT, ValueT>::operator=(ThreadSafeHashMap&&) noexcept = default;

template <typename KeyT, typename ValueT>
std::optional<ValueT> ThreadSafeHashMap<KeyT, ValueT>::find(const KeyT& key) const {
    return _impl->find(key);
}

template <typename KeyT, typename ValueT>
void ThreadSafeHashMap<KeyT, ValueT>::insert(const KeyT& key, const ValueT& value) {
    _impl->insert(key, value);
}

template <typename KeyT, typename ValueT>
void ThreadSafeHashMap<KeyT, ValueT>::clear() {
    _impl->clear();
}

// ============================================================================
// Explicit Instantiations
// ============================================================================
// Required for PIMPL pattern: the destructor and other special member functions
// must be instantiated explicitly because Impl is an incomplete type in the header.

using NTilesOnDim = Shape;
using PerClusterShapeCacheItem = std::optional<SmallVector<Shape>>;
using DimArr = SmallVector<Dim>;

template class ThreadSafeHashMap<llvm::hash_code, std::optional<NTilesOnDim>>;
template class ThreadSafeHashMap<llvm::hash_code, std::optional<::llvm::hash_code>>;
template class ThreadSafeHashMap<llvm::hash_code, SmallVector<uint32_t>>;
template class ThreadSafeHashMap<llvm::hash_code, uint32_t>;
template class ThreadSafeHashMap<llvm::hash_code, PerClusterShapeCacheItem>;
template class ThreadSafeHashMap<llvm::hash_code, SmallVector<DimArr>>;
template class ThreadSafeHashMap<llvm::hash_code, DimArr>;
template class ThreadSafeHashMap<llvm::hash_code, SmallVector<vpux::NDTypeInterface>>;
template class ThreadSafeHashMap<llvm::hash_code, size_t>;
template class ThreadSafeHashMap<llvm::hash_code, std::optional<VPU::TemporalTilingInfo>>;
}  // namespace vpux
