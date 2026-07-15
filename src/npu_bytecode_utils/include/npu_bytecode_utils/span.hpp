//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cstddef>

namespace intel_npu::vm {

// Represents a contiguous sequence of objects, with the first element of the sequence at position zero
template <typename T>
class Span {
    T* _data{};
    size_t _size{};

public:
    Span() noexcept = default;

    Span(T* data, size_t size) noexcept: _data{data}, _size{size} {
        assert((data != nullptr || size == 0) && "Span cannot have a null data pointer if size is greater than zero");
    }

    T& operator[](size_t i) noexcept {
        assert(i < _size && "Span index out of bounds");
        return *(begin() + i);
    }

    T const& operator[](size_t i) const noexcept {
        assert(i < _size && "Span index out of bounds");
        return *(begin() + i);
    }

    bool empty() const noexcept {
        return size() == 0;
    }

    size_t size() const noexcept {
        return _size;
    }

    T* begin() noexcept {
        return _data;
    }

    const T* begin() const noexcept {
        return _data;
    }

    T* end() noexcept {
        return _data + _size;
    }

    const T* end() const noexcept {
        return _data + _size;
    }

    Span<T> subspan(size_t offset) const noexcept {
        if (offset > _size) {
            return {nullptr, 0};
        }
        return {_data + offset, _size - offset};
    }

    Span<T> subspan(size_t offset, size_t size) const noexcept {
        if (offset > _size || size > _size - offset) {
            return {nullptr, 0};
        }
        return {_data + offset, size};
    }
};

}  // namespace intel_npu::vm
