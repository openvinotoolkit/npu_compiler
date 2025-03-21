//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>
#include <numeric>

using namespace vpux;

//
// get<scalar>Type
//

mlir::IntegerType vpux::getInt1Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 1);
}

mlir::IntegerType vpux::getInt2Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 2);
}

mlir::IntegerType vpux::getInt3Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 3);
}

mlir::IntegerType vpux::getInt4Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 4);
}

mlir::IntegerType vpux::getInt5Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 5);
}

mlir::IntegerType vpux::getInt6Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 6);
}

mlir::IntegerType vpux::getInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8);
}

mlir::IntegerType vpux::getInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 16);
}

mlir::IntegerType vpux::getInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 32);
}

mlir::IntegerType vpux::getInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 64);
}

mlir::IntegerType vpux::getSInt4Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 4, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getUInt1Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 1, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt2Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 2, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt3Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 3, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt4Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 4, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt5Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 5, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt6Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 6, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getBool8Type(mlir::MLIRContext* ctx) {
    // Signless 8-bit integer use for BOOL, to distinguish it from U8
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signless);
}

//
// TypeSize
//

Bit vpux::getElemTypeSize(mlir::Type type) {
    if (const auto quantileFloatType = mlir::dyn_cast<vpux::type::QuantileFloatType>(type)) {
        return Bit(quantileFloatType.getWidth());
    }

    if (const auto ndType = type.dyn_cast<vpux::NDTypeInterface>()) {
        return getElemTypeSize(ndType.getElementType());
    }

    if (type.isIntOrFloat()) {
        return Bit(type.getIntOrFloatBitWidth());
    }

    if (const auto qType = type.dyn_cast<mlir::quant::QuantizedType>()) {
        return Bit(qType.getStorageTypeIntegralWidth());
    }

    VPUX_THROW("Can't get type size for '{0}'", type);
}

Byte vpux::getTotalSize(mlir::Value val) {
    const auto type = val.getType().dyn_cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non vpux::NDTypeInterface '{1}'", val, val.getType());
    return type.getTotalAllocSize();
}

Byte vpux::getCompactSize(mlir::Value val) {
    const auto type = val.getType().dyn_cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non vpux::NDTypeInterface '{1}'", val, val.getType());
    return type.getCompactAllocSize();
}

std::optional<int32_t> vpux::getQuantizedAxis(int32_t axis, ShapeRef prevShape, ShapeRef newShape) {
    if (checked_cast<size_t>(axis) >= prevShape.size() || prevShape[Dim(axis)] == 1) {
        return std::nullopt;
    }

    const auto sizeAtAxis = prevShape[Dim(axis)];
    const auto fullSizeIncludingAxis =
            std::accumulate(prevShape.begin(), prevShape.begin() + axis + 1, int64_t(1), std::multiplies<int64_t>());

    int64_t accumSize = 1;
    for (auto it = newShape.begin(); it != newShape.end(); ++it) {
        accumSize *= *it;
        if (accumSize == fullSizeIncludingAxis && *it == sizeAtAxis) {
            return static_cast<int32_t>(std::distance(newShape.begin(), it));
        }
        if (accumSize > fullSizeIncludingAxis) {
            break;
        }
    }

    return std::nullopt;
}

Byte vpux::getExpectedBufferSize(mlir::Type type) {
    const Bit elemSize = getElemTypeSize(type);
    const auto ndType = mlir::cast<NDTypeInterface>(type);
    const auto elemCount = ndType.getNumElements();

    const auto totalBits = elemCount * elemSize.count();
    VPUX_THROW_UNLESS(totalBits >= 0, "Cannot calculate buffer size required for '{0}'", type);

    // practically, total bits must be always byte-aligned (think of aligned
    // malloc analogy here):
    //
    // offset = 0b111, ~offset = 0b11111...1000
    // 1. already aligned case:
    //  * totalBits = N * 8 = N * 0b1000
    //  * totalBits + offset = N * 0b1000 + 0b0111
    //  * (totalBits + offset) & ~offset = (N * 0b1000 + 0b0111) & 0b111...1000
    //    = N * 0b1000 + 0b0000 = N * 0b1000 -> same value
    //
    // 2. unaligned case:
    //  * totalBits = N * 8 + 3 = N * 0b1000 + 0b11
    //  * totalBits + offset = N * 0b1000 + 0b11 + 0b0111 = N * 0b1000 + 0b1010
    //                                      (3 + 7 = 10)
    //  * (totalBits + offset) & ~offset = (N * 0b1000 + 0b1010) & 0b111...1000
    //    = N * 0b1000 + 0b1000 = (N + 1) * 0b1000 = (N + 1) * 8 -> original
    //    value promoted to "next divisible by 8"
    constexpr size_t offset = CHAR_BIT - 1;
    const auto alignedBits = (static_cast<size_t>(totalBits) + offset) & ~offset;
    return Byte(alignedBits / CHAR_BIT);
}

//
// MemRefType utilities
//

mlir::MemRefType vpux::getMemRefType(ShapeRef shape, mlir::Type elemType, DimsOrder order, IndexedSymbolAttr memSpace,
                                     StridesRef strides, VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr,
                                     VPUIP::SparsityCompressionAttr sparsityCompressionAttr,
                                     mlir::IntegerAttr allocSizeAttr, VPUIP::CompressionStateAttr compressionState) {
    VPUX_THROW_UNLESS(order.numDims() == shape.size(), "Shape '{0}' doesn't match order '{1}'", shape, order);
    VPUX_THROW_UNLESS(strides.empty() || shape.size() == strides.size(), "Strides '{0}' doesn't match shape '{1}'",
                      strides, shape);

    auto* ctx = elemType.getContext();
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));

    mlir::ArrayAttr stridesAttr = nullptr;
    if (strides != StridesRef()) {
        const Bit elemSize = getElemTypeSize(elemType);
        const auto memShape = order.toMemoryOrder(shape);
        const auto memStrides = order.toMemoryOrder(strides);
        const auto compactReqs = StrideReqs::compact(shape.size());
        if (!compactReqs.checkStrides(memStrides, elemSize, memShape)) {
            // Have strides only if they are not compact
            const auto elemStrides = to_small_vector(strides | transformed([&](Bit stride) {
                                                         return stride.count() / elemSize.count();
                                                     }));

            stridesAttr = getIntArrayAttr(ctx, elemStrides);
        }
    }

    mlir::MemRefType::Builder builder(shape.raw(), elemType);
    builder.setMemorySpace(memSpace);
    if (stridesAttr == nullptr && swizzlingSchemeAttr == nullptr && sparsityCompressionAttr == nullptr &&
        allocSizeAttr == nullptr && compressionState == nullptr) {
        builder.setLayout(orderAttr);
    } else {
        const auto layoutAttr =
                vpux::MemRefAttr::get(orderAttr, stridesAttr, allocSizeAttr,
                                      {swizzlingSchemeAttr, sparsityCompressionAttr, compressionState}, ctx);
        builder.setLayout(layoutAttr.cast<mlir::MemRefLayoutAttrInterface>());
    }
    return builder;
}

mlir::SmallVector<float> vpux::getFloatStrides(StridesRef strides) {
    Strides temp(strides.begin(), strides.end());

    const auto cvtBitStrideToByteFP = [](Bit val) {
        if (val.count() % CHAR_BIT == 0) {
            return checked_cast<float>(Byte(val).count());
        }

        return checked_cast<float>(val.count()) / CHAR_BIT;
    };

    return to_small_vector(temp | transformed(cvtBitStrideToByteFP));
}

//
// RankedTensorType utilities
//

mlir::RankedTensorType vpux::getTensorType(ShapeRef shape, mlir::Type elemType, DimsOrder order,
                                           IndexedSymbolAttr memSpace, mlir::ArrayAttr bounds) {
    VPUX_THROW_UNLESS(order.numDims() == shape.size(), "DimsOrder '{0}' doesn't match to shape '{1}'", order, shape);

    const auto tensorDesc = vpux::getTensorAttr(elemType.getContext(), order, memSpace, bounds);
    const auto newType = mlir::RankedTensorType::get(shape.raw(), elemType, tensorDesc);

    const auto loc = mlir::UnknownLoc::get(elemType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::MemRefType vpux::convertToMemRef(mlir::RankedTensorType tensorType) {
    const auto type = tensorType.cast<vpux::NDTypeInterface>();
    const auto shape = type.getShape();
    const auto elemType = type.getElementType();
    const auto order = type.getDimsOrder();
    const auto memSpace = type.getMemSpace();
    return getMemRefType(shape, elemType, order, memSpace);
}

//
// NDTypeInterface utilities
//

bool vpux::isCompatibleForInplaceOp(vpux::NDTypeInterface inInterface, vpux::NDTypeInterface outInterface,
                                    vpux::Logger log) {
    if (inInterface.getShape() != outInterface.getShape()) {
        log.trace("Different in and out shape {0} != {1}", inInterface.getShape(), outInterface.getShape());
        return false;
    }

    if (inInterface.getStrides() != outInterface.getStrides()) {
        log.trace("Different in and out strides {0} != {1}", inInterface.getStrides(), outInterface.getStrides());
        return false;
    }

    if (inInterface.getDimsOrder() != outInterface.getDimsOrder()) {
        log.trace("Different in and out order {0} != {1}", inInterface.getDimsOrder(), outInterface.getDimsOrder());
        return false;
    }

    if (inInterface.getTotalAllocSize() < outInterface.getTotalAllocSize()) {
        /* #65422 Case with different tensor sizes.
        If op is eltwise and it is a part of dequantize chain then input is U8 and output is float. */
        log.trace("Input tensor size is smaller than output tensor size, the case is not supported {0} < {1}",
                  inInterface.getTotalAllocSize(), outInterface.getTotalAllocSize());
        return false;
    }

    return true;
}

//
// Type comparison
//

bool vpux::areTypesCompatible(mlir::TypeRange lhs, mlir::TypeRange rhs, IE::TypeComparisonMode elemComparisonModes,
                              bool checkDimsOrder, bool checkMemSpace) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (const auto p : zip(lhs, rhs)) {
        auto lhsOrigType = std::get<0>(p);
        auto rhsOrigType = std::get<1>(p);

        if (lhsOrigType.getTypeID() != rhsOrigType.getTypeID()) {
            if (IE::bitEnumContainsAny(elemComparisonModes, (IE::TypeComparisonMode::ALLOW_GROUPED_OUTPUT |
                                                             IE::TypeComparisonMode::ALLOW_DISTRIBUTED_OUTPUT))) {
                const auto oneIsGrouped = (mlir::isa<vpux::GroupedTypeInterface>(lhsOrigType) &&
                                           !mlir::isa<vpux::GroupedTypeInterface>(rhsOrigType)) ||
                                          (!mlir::isa<vpux::GroupedTypeInterface>(lhsOrigType) &&
                                           mlir::isa<vpux::GroupedTypeInterface>(rhsOrigType));
                const auto oneIsDistributed = (mlir::isa<vpux::VPU::DistributedTensorType>(lhsOrigType) &&
                                               !mlir::isa<vpux::VPU::DistributedTensorType>(rhsOrigType)) ||
                                              (!mlir::isa<vpux::VPU::DistributedTensorType>(lhsOrigType) &&
                                               mlir::isa<vpux::VPU::DistributedTensorType>(rhsOrigType));

                if (!oneIsGrouped && !oneIsDistributed) {
                    return false;
                }
            } else {
                return false;
            }
        }

        auto lhsType = mlir::dyn_cast<NDTypeInterface>(lhsOrigType);
        auto rhsType = mlir::dyn_cast<NDTypeInterface>(rhsOrigType);

        if (lhsType == nullptr || rhsType == nullptr) {
            return false;
        }

        if (lhsType.getShape() != rhsType.getShape()) {
            return false;
        }

        if (lhsType.getElementType() != rhsType.getElementType()) {
            if (IE::bitEnumContainsAny(elemComparisonModes, IE::TypeComparisonMode::STRICT_EQUAL)) {
                return false;
            }

            const auto lhsQuantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(lhsType.getElementType());
            const auto rhsQuantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(rhsType.getElementType());

            if (lhsQuantizedType && rhsQuantizedType) {
                if ((lhsQuantizedType.getExpressedType() != rhsQuantizedType.getExpressedType()) ||
                    (lhsQuantizedType.getStorageType() != rhsQuantizedType.getStorageType())) {
                    if (!IE::bitEnumContainsAny(elemComparisonModes, IE::TypeComparisonMode::ALLOW_DIFFERENT_QUANT)) {
                        return false;
                    }
                }
            } else if (!IE::bitEnumContainsAny(elemComparisonModes, IE::TypeComparisonMode::ALLOW_MIXED_PRECISION)) {
                return false;
            }
        }

        if (checkDimsOrder) {
            const auto order1 = lhsType.getDimsOrder();
            const auto order2 = rhsType.getDimsOrder();

            if (order1 != order2) {
                return false;
            }
        }

        if (checkMemSpace) {
            const auto memSpace1 = lhsType.getMemSpace();
            const auto memSpace2 = rhsType.getMemSpace();

            if (memSpace1 != memSpace2) {
                // Allow different memory spaces only if both types are in DDR, since a null value also represents DDR
                if (!(lhsType.getMemoryKind() == VPU::MemoryKind::DDR &&
                      rhsType.getMemoryKind() == VPU::MemoryKind::DDR)) {
                    return false;
                }
            }
        }
    }

    return true;
}

//
// Quantized dimension permutation
//

bool vpux::isQuantizedDimensionPermutation(mlir::quant::UniformQuantizedPerAxisType inputElemType,
                                           mlir::quant::UniformQuantizedPerAxisType newElemType) {
    if (inputElemType.getExpressedType() != newElemType.getExpressedType() ||
        inputElemType.getStorageType() != newElemType.getStorageType() ||
        inputElemType.isSigned() != newElemType.isSigned() || inputElemType.getFlags() != newElemType.getFlags() ||
        inputElemType.getStorageTypeMin() != newElemType.getStorageTypeMin() ||
        inputElemType.getStorageTypeMax() != newElemType.getStorageTypeMax() ||
        inputElemType.getQuantizedDimension() == newElemType.getQuantizedDimension()) {
        return false;
    }

    auto inputElemTypeScales = inputElemType.getScales();
    auto newElemTypeScales = newElemType.getScales();
    if (inputElemTypeScales.size() != newElemTypeScales.size() ||
        !std::equal(inputElemTypeScales.begin(), inputElemTypeScales.end(), newElemTypeScales.begin())) {
        return false;
    }

    auto inputElemTypeZPs = inputElemType.getZeroPoints();
    auto newElemTypeZPs = newElemType.getZeroPoints();
    if (inputElemTypeZPs.size() != newElemTypeZPs.size() ||
        !std::equal(inputElemTypeZPs.begin(), inputElemTypeZPs.end(), newElemTypeZPs.begin())) {
        return false;
    }

    return true;
}

bool vpux::isSubByteType(mlir::Type elemType) {
    return getElemTypeSize(elemType).count() < CHAR_BIT;
}

bool vpux::isBufferType(mlir::Type type) {
    // Note: BaseMemRefType covers MemRefType, UnrankedMemRefType,
    // VPUIP::DistributedBufferType, VPUIP::SparseBufferType and VPUIP::BoundedBufferType
    return mlir::isa<mlir::BaseMemRefType>(type);
}
