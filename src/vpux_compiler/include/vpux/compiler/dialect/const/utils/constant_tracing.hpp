//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Attributes.h>

#include <cstddef>
#include <functional>

namespace vpux::Const {

struct TraceId {
    const void* _ptr = nullptr;

    TraceId() = default;
    ~TraceId() = default;

    TraceId(const void* ptr): _ptr(ptr) {
    }

    // Intentionally allowing construction of ContentSetup without a specific Attribute.
    // Passing nullptr, which means no real Attribute, would make the constant be
    // ignored by the tracing system.
    TraceId(mlir::Attribute attr): _ptr(attr.getAsOpaquePointer()) {
    }

    TraceId(const TraceId& other) = default;
    TraceId& operator=(const TraceId& other) = default;
    bool operator==(TraceId other) const;
    bool operator!=(TraceId other) const;
    bool operator<(TraceId other) const;
    explicit operator bool() const;
    bool operator!() const;
};

}  // namespace vpux::Const

namespace std {

template <>
struct hash<vpux::Const::TraceId> final {
    size_t operator()(vpux::Const::TraceId id) const noexcept {
        return hash<const void*>{}(id._ptr);
    }
};

}  // namespace std
