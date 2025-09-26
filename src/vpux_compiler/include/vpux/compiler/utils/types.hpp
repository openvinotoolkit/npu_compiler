//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>

namespace vpux::IE {
enum class TypeComparisonMode : uint64_t;
}  // namespace vpux::IE

namespace vpux {

template <typename Enum, typename OutT = mlir::RankedTensorType>
using ranked_tensor_type_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

template <typename Enum, typename OutT = mlir::MemRefType>
using memref_type_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

//
// get<scalar>Type
//

mlir::IntegerType getInt1Type(mlir::MLIRContext* ctx);
mlir::IntegerType getInt2Type(mlir::MLIRContext* ctx);
mlir::IntegerType getInt3Type(mlir::MLIRContext* ctx);
mlir::IntegerType getInt4Type(mlir::MLIRContext* ctx);
mlir::IntegerType getInt5Type(mlir::MLIRContext* ctx);
mlir::IntegerType getInt6Type(mlir::MLIRContext* ctx);
mlir::IntegerType getInt8Type(mlir::MLIRContext* ctx);
mlir::IntegerType getInt16Type(mlir::MLIRContext* ctx);
mlir::IntegerType getInt32Type(mlir::MLIRContext* ctx);
mlir::IntegerType getInt64Type(mlir::MLIRContext* ctx);

mlir::IntegerType getSInt2Type(mlir::MLIRContext* ctx);
mlir::IntegerType getSInt4Type(mlir::MLIRContext* ctx);
mlir::IntegerType getSInt8Type(mlir::MLIRContext* ctx);
mlir::IntegerType getSInt16Type(mlir::MLIRContext* ctx);
mlir::IntegerType getSInt32Type(mlir::MLIRContext* ctx);
mlir::IntegerType getSInt64Type(mlir::MLIRContext* ctx);

mlir::IntegerType getUInt1Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt2Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt3Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt4Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt5Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt6Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt8Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt16Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt32Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt64Type(mlir::MLIRContext* ctx);
mlir::IntegerType getBool8Type(mlir::MLIRContext* ctx);

mlir::Float16Type getFp16Type(mlir::MLIRContext* ctx);

//
// TypeSize
//

Bit getElemTypeSize(mlir::Type type);

// Calculates tensor size based on stride information
// Example:
// #map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 48 + d2 * 3 + d3)>
// memref<2x3x8x4xf32, #NHWC, #map0>
// result: 768 * 2 * 4 = 6144, where
// 768 - stride, to get the next element by N dim
// 2 - size of N dim, 4 - element size
Byte getTotalSize(mlir::Value val);

// Calculates tensor size ignoring stride information
// Example:
// #map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 48 + d2 * 3 + d3)>
// memref<2x3x8x4xui8, #NHWC, #map0>
// result: 2 * 3 * 8 * 4 = 192
Byte getCompactSize(mlir::Value val);

// compute axis permutation
std::optional<int32_t> getQuantizedAxis(int32_t axis, ShapeRef prevShape, ShapeRef newShape);

/// @brief Returns the expected buffer size that is described by the given type.
/// This also supports sub-byte types.
Byte getExpectedBufferSize(mlir::Type type);

//
// MemRefType utilities
//

mlir::MemRefType getMemRefType(ShapeRef shape, mlir::Type elemType, DimsOrder order, IndexedSymbolAttr memSpace,
                               StridesRef strides = {}, VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr = nullptr,
                               VPUIP::SparsityCompressionAttr sparsityCompressionAttr = nullptr,
                               mlir::IntegerAttr allocSizeAttr = nullptr,
                               VPUIP::CompressionStateAttr compressionState = nullptr);
template <typename Enum>
memref_type_if<Enum> getMemRefType(ShapeRef shape, mlir::Type elemType, DimsOrder order, Enum kind,
                                   StridesRef strides = StridesRef(),
                                   VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr = nullptr,
                                   VPUIP::SparsityCompressionAttr sparsityCompressionAttr = nullptr,
                                   mlir::IntegerAttr allocSizeAttr = nullptr) {
    return getMemRefType(shape, elemType, order, IndexedSymbolAttr::get(elemType.getContext(), stringifyEnum(kind)),
                         strides, swizzlingSchemeAttr, sparsityCompressionAttr, allocSizeAttr);
}

IndexedSymbolAttr getMemorySpace(mlir::MemRefType type);

mlir::SmallVector<float> getFloatStrides(StridesRef strides);

//
// RankedTensorType utilities
//

mlir::RankedTensorType getTensorType(ShapeRef shape, mlir::Type elemType, DimsOrder order, IndexedSymbolAttr memSpace);
mlir::RankedTensorType getTensorType(ShapeRef shape, mlir::Type elemType, DimsOrder order, IndexedSymbolAttr memSpace,
                                     BoundsRef bounds, DynamicDimsMaskRef dynamicDimsMask);

///
/// \brief Convert RankedTensor type to Memref type
/// \param [in] tensorType - ranked tensor type
/// \return mlir::MemRefType memref type
///

mlir::MemRefType convertToMemRef(mlir::RankedTensorType tensorType);

//
// NDTypeInterface utilities
//

bool isCompatibleForInplaceOp(NDTypeInterface inInterface, NDTypeInterface outInterface, Logger log);

//
// Type comparison
//

bool areTypesCompatible(mlir::TypeRange lhs, mlir::TypeRange rhs, IE::TypeComparisonMode elemComparisonMode,
                        bool checkDimsOrder, bool checkMemSpace);

// Checks if the only difference between two UniformQuantizedPerAxisType is the quantized dimension
bool isQuantizedDimensionPermutation(mlir::quant::UniformQuantizedPerAxisType inputElemType,
                                     mlir::quant::UniformQuantizedPerAxisType newElemType);

bool isSubByteType(mlir::Type elemType);

bool isBufferType(mlir::Type type);

}  // namespace vpux
