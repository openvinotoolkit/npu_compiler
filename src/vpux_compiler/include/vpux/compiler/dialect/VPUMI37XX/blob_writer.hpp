//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mvcnn.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <limits>
#include <unordered_map>

namespace vpux {
namespace VPUMI37XX {

template <typename T>
auto createVector(ArrayRef<T> arr) {
    return MVCNN::Vector<T>(arr.begin(), arr.end());
}

template <class Range>
auto createVector(const Range& range) {
    using ValueType = std::decay_t<decltype(*std::begin(range))>;
    const auto vec = to_small_vector(range);
    return MVCNN::Vector<ValueType>(vec.begin(), vec.end());
}

template <class UnderlyingType>
auto arrayCast(ArrayRef<int64_t> source) {
    SmallVector<UnderlyingType> casted(source.size());
    std::transform(source.begin(), source.end(), casted.begin(), [](auto value) {
        return checked_cast<UnderlyingType>(value);
    });
    return createVector(casted);
}

MVCNN::TensorReference createTensorRef(vpux::NDTypeInterface type, ArrayRef<int64_t> sectionIndex, int64_t byteOffset,
                                       ArrayRef<int64_t> mult, ArrayRef<int64_t> shift, ArrayRef<uint8_t> zeroPoints,
                                       std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                       std::optional<int64_t> storageElementOffset = std::nullopt,
                                       std::optional<int64_t> storageElementSize = std::nullopt,
                                       std::optional<int64_t> swizzlingKey = std::nullopt);
MVCNN::TensorReference createTensorRef(vpux::NDTypeInterface type, ArrayRef<int64_t> sectionIndex, int64_t byteOffset,
                                       std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                       std::optional<int64_t> storageElementOffset = std::nullopt,
                                       std::optional<int64_t> storageElementSize = std::nullopt,
                                       std::optional<int64_t> swizzlingKey = std::nullopt);
MVCNN::TensorReference createTensorRef(vpux::NDTypeInterface type, int64_t sectionIndex, int64_t byteOffset,
                                       std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                       std::optional<int64_t> storageElementOffset = std::nullopt,
                                       std::optional<int64_t> storageElementSize = std::nullopt,
                                       std::optional<int64_t> swizzlingKey = std::nullopt);
MVCNN::TensorReference createTensorRef(mlir::Value val, ArrayRef<int64_t> sectionIndex, int64_t byteOffset,
                                       std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                       std::optional<int64_t> storageElementOffset = std::nullopt,
                                       std::optional<int64_t> storageElementSize = std::nullopt,
                                       std::optional<int64_t> swizzlingKey = std::nullopt);
MVCNN::TensorReference createTensorRef(mlir::Value val, int64_t sectionIndex, int64_t byteOffset,
                                       std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                       std::optional<int64_t> storageElementOffset = std::nullopt,
                                       std::optional<int64_t> storageElementSize = std::nullopt,
                                       std::optional<int64_t> swizzlingKey = std::nullopt);

MVCNN::Vector<uint32_t> createDims(ShapeRef shape);
MVCNN::Vector<uint32_t> createDims(vpux::NDTypeInterface type);
template <typename T>
MVCNN::Vector<T> createStrides(StridesRef strides, Bit elemSize);
template <typename T>
MVCNN::Vector<T> createStrides(vpux::NDTypeInterface type);
MVCNN::IndirectDataReference createIndirectDataReference(int64_t dataIndex,
                                                         std::optional<int64_t> sparsityIndex = std::nullopt,
                                                         std::optional<int64_t> storageElementIndex = std::nullopt,
                                                         std::optional<int64_t> storageElementSize = std::nullopt);

}  // namespace VPUMI37XX
}  // namespace vpux
