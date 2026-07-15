//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <memory>
#include <mutex>

namespace vpux {

/** @brief Associates an object with a mutex to ensure thread-safe access.

    This is a simple generic primitive that couples a given object with a mutex
    to ensure always-thread-safe access to the object's state and APIs.

    This is inspired by similar implementations in other libraries such as
    [CopperSpice's "guarded"](https://github.com/copperspice/cs_libguarded)
    primitives, and [folly's
    Synchronized<T>](https://github.com/facebook/folly/blob/main/folly/docs/Synchronized.md).
 */
template <typename T, typename Mutex = std::mutex>
class SimpleThreadSafeAccessor {
    T _object;
    mutable Mutex _mutex;

    // RAII unlock
    class SafetyHandleDeleter {
        std::unique_lock<Mutex> _lock;

    public:
        SafetyHandleDeleter(std::unique_lock<Mutex>&& lock): _lock(std::move(lock)) {
        }
        void operator()(const T* /* object ptr*/) const {
            assert(_lock.owns_lock() && "The lock must own the mutex");
            // Note: std::unique_lock<> unlocks on destruction. Since this
            // object "owns" the lock, unlock would happen on this object's
            // destruction, which is when the std::unique_ptr is destroyed. Just
            // need to make sure the lock owns the mutex, so that there are no
            // exceptions during destruction.
        }
    };
    // Note: std::unique_ptr is a convenient RAII primitive.
    using SafetyHandle = std::unique_ptr<T, SafetyHandleDeleter>;
    using ConstSafetyHandle = std::unique_ptr<const T, SafetyHandleDeleter>;

public:
    /** @brief A "catch-all" ctor to forward arguments to the underlying object.
     */
    template <typename... Args>
    SimpleThreadSafeAccessor(Args&&... args): _object(std::forward<Args>(args)...) {
    }

    /** @brief Returns a handle that guarantees that the subsequent access to
               the object is thread-safe as long as the handle is alive.

        Returns a "safety" handle that maintains the lock on the mutex. The
        synchronized (mutex-protected) access to the object is bound to this
        handle's lifetime.

        Usage:
        ```cpp
        SimpleThreadSafeAccessor<std::vector<int>> threadSafeWrapper =
            getThreadSafeWrapper();

        {
            auto handle = threadSafeWrapper.lock(); // locks the mutex
            handle->push_back(42); // this is thread-safe
            // --- mutex is still locked here --- //
            // --- any access below is thread-safe -- //
            auto& object = *handle;
            auto& fortyTwo = object.emplace_back(42);
            fortyTwo = 44;
        } // handle is destroyed at the end of scope, mutex is unlocked

        // new access has to be acquired here:
        auto handle2 = threadSafeWrapper.lock(); // locks the mutex again
        // ...
        ```
     */
    [[nodiscard]] auto lock() {
        return SafetyHandle(&_object, SafetyHandleDeleter(std::unique_lock<Mutex>(_mutex)));
    }

    /** @brief Const overload of lock().
     */
    [[nodiscard]] auto lock() const {
        return ConstSafetyHandle(&_object, SafetyHandleDeleter(std::unique_lock<Mutex>(_mutex)));
    }

    /** @brief Provides "single-shot" access to the underlying object's methods
        via operator->() chaining.

        This is a convenient, very "simple", but also very *dangerous* API
        (bugs-wise and performance-wise) that allows thread-safe access to the
        underlying object for the purposes of a *single* API call.

        @warning This API is suitable only for a single, isolated call to a
        particular method of the underlying object. Any more complex expression
        involving the object's API is almost never thread-safe.

        Usage:
        ```cpp
        SimpleThreadSafeAccessor<std::vector<int>> threadSafeWrapper =
            getThreadSafeWrapper();

        // this IS thread-safe:
        threadSafeWrapper->push_back(42);

        // this is NOT thread-safe:
        auto& x = threadSafeWrapper->emplace_back(42); // emplace_back() is thread-safe; reference assignment is not
        // --- mutex is unlocked here --- //
        x = 43; // this is NOT thread-safe
        ```
     */
    [[nodiscard]] auto operator->() {
        return SafetyHandle(&_object, SafetyHandleDeleter(std::unique_lock<Mutex>(_mutex)));
    }

    /** @brief Const overload of operator->().
     */
    [[nodiscard]] auto operator->() const {
        return ConstSafetyHandle(&_object, SafetyHandleDeleter(std::unique_lock<Mutex>(_mutex)));
    }
};

}  // namespace vpux
