//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI37XX/blob_writer.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPUIP/device.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/custom_float.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

#include <algorithm>

using namespace vpux;

namespace {

VPUIP::DType createDType(mlir::Type type) {
    if (type.isF64()) {
        return VPUIP::DType_FP64;
    } else if (type.isF32()) {
        return VPUIP::DType_FP32;
    } else if (type.isF16()) {
        return VPUIP::DType_FP16;
    } else if (type.isBF16()) {
        return VPUIP::DType_BFP16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int64_t))) {
        return VPUIP::DType_I64;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int32_t))) {
        return VPUIP::DType_I32;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int16_t))) {
        return VPUIP::DType_I16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return VPUIP::DType_I8;
    } else if (type.isSignedInteger(4)) {
        return VPUIP::DType_I4;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint64_t))) {
        return VPUIP::DType_U64;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint32_t))) {
        return VPUIP::DType_U32;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint16_t))) {
        return VPUIP::DType_U16;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint8_t))) {
        return VPUIP::DType_U8;
    } else if (type.isInteger(4)) {
        return VPUIP::DType_U4;
    } else if (type.isInteger(2)) {
        return VPUIP::DType_I2;
    } else if (type.isInteger(1)) {
        return VPUIP::DType_BIN;
    } else if (mlir::isa<mlir::quant::QuantizedType>(type)) {
        auto quant = mlir::cast<mlir::quant::QuantizedType>(type);
        auto quantStorageType = quant.getStorageType();
        if (auto intType = mlir::cast<mlir::IntegerType>(quantStorageType)) {
            auto signedness = quant.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
            return createDType(mlir::IntegerType::get(intType.getContext(), intType.getWidth(), signedness));
        }
        return createDType(quantStorageType);
    } else {
        VPUX_THROW("Unsupported element type {0}", type);
    }
}

}  // namespace

namespace vpux::VPUMI37XX {

MVCNN::TensorReference createTensorRef(vpux::NDTypeInterface type, ArrayRef<int64_t> sectionIndex, int64_t byteOffset,
                                       ArrayRef<int64_t> mult, ArrayRef<int64_t> shift, ArrayRef<uint8_t> zeroPoints,
                                       std::optional<int64_t> sparsityMapOffset,
                                       std::optional<int64_t> storageElementOffset,
                                       std::optional<int64_t> storageElementSize, std::optional<int64_t> swizzlingKey) {
    const auto serializedDataType = static_cast<MVCNN::DType>(createDType(type.getElementType()));
    const auto serializedDims = createDims(type);

    const auto serializedDataReference =
            createIndirectDataReference(byteOffset, sparsityMapOffset, storageElementOffset, storageElementSize);

    MVCNN::Vector<uint8_t> serializedQuantZero = createVector(zeroPoints);

    const auto serializedLocaleIndex = arrayCast<uint32_t>(sectionIndex);
    // Unchecked casting of quant_mult since it stores both u16 and i16 values
    auto castedQuantMult = SmallVector<uint16_t>(mult.size());
    std::transform(mult.begin(), mult.end(), castedQuantMult.begin(), [](auto value) {
        return value & 0xFFFF;
    });
    const auto serializedQuantMult = createVector(castedQuantMult);
    const auto serializedQuantShift = arrayCast<uint8_t>(shift);

    auto strides = createStrides<uint64_t>(type);

    MVCNN::TensorReference tensor;
    tensor.add_dimensions(serializedDims);
    tensor.add_bit_strides(strides);
    tensor.add_data(serializedDataReference);
    tensor.add_locale_index(serializedLocaleIndex);
    tensor.add_data_dtype(serializedDataType);
    tensor.add_quant_zero(serializedQuantZero);
    tensor.add_quant_mult(serializedQuantMult);
    tensor.add_quant_shift(serializedQuantShift);
    if (swizzlingKey.has_value()) {
        tensor.add_swizzling_key(checked_cast<uint8_t>(swizzlingKey.value()));
    }
    return tensor;
}

MVCNN::TensorReference createTensorRef(vpux::NDTypeInterface type, ArrayRef<int64_t> sectionIndex, int64_t byteOffset,
                                       std::optional<int64_t> sparsityMapOffset,
                                       std::optional<int64_t> storageElementOffset,
                                       std::optional<int64_t> storageElementSize, std::optional<int64_t> swizzlingKey) {
    SmallVector<uint8_t> zeroPoints;
    SmallVector<int64_t> mults;
    SmallVector<int64_t> shifts;

    if (const auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(type.getElementType())) {
        zeroPoints.push_back(checked_cast<uint8_t>(qType.getZeroPoint()));
        const auto scaleApproximation = QuantizationApproximation(qType.getScale());
        mults.push_back(scaleApproximation.mult());
        shifts.push_back(scaleApproximation.shift());
    } else if (const auto qPerAxisType =
                       mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(type.getElementType())) {
        auto qtype_quant_zp = qPerAxisType.getZeroPoints();
        auto qtype_quant_scale = qPerAxisType.getScales();

        zeroPoints.resize(qtype_quant_zp.size());
        std::transform(qtype_quant_zp.begin(), qtype_quant_zp.end(), zeroPoints.begin(), [](int64_t val) {
            return checked_cast<uint8_t>(val);
        });

        mults.resize(qtype_quant_scale.size());
        shifts.resize(qtype_quant_scale.size());
        for (std::size_t i = 0; i < qtype_quant_scale.size(); ++i) {
            const auto scaleApproximation = QuantizationApproximation(qtype_quant_scale[i]);
            mults[i] = scaleApproximation.mult();
            shifts[i] = scaleApproximation.shift();
        }
    } else {
        zeroPoints.push_back(0);
        mults.push_back(1);
        shifts.push_back(0);
    }

    return createTensorRef(type, sectionIndex, byteOffset, mults, shifts, zeroPoints, sparsityMapOffset,
                           storageElementOffset, storageElementSize, swizzlingKey);
}

MVCNN::TensorReference createTensorRef(vpux::NDTypeInterface type, int64_t sectionIndex, int64_t byteOffset,
                                       std::optional<int64_t> sparsityMapOffset,
                                       std::optional<int64_t> storageElementOffset,
                                       std::optional<int64_t> storageElementSize, std::optional<int64_t> swizzlingKey) {
    return createTensorRef(type, ArrayRef({sectionIndex}), byteOffset, sparsityMapOffset, storageElementOffset,
                           storageElementSize, swizzlingKey);
}

MVCNN::TensorReference createTensorRef(mlir::Value val, ArrayRef<int64_t> sectionIndex, int64_t byteOffset,
                                       std::optional<int64_t> sparsityMapOffset,
                                       std::optional<int64_t> storageElementOffset,
                                       std::optional<int64_t> storageElementSize, std::optional<int64_t> swizzlingKey) {
    return createTensorRef(mlir::cast<vpux::NDTypeInterface>(val.getType()), sectionIndex, byteOffset,
                           sparsityMapOffset, storageElementOffset, storageElementSize, swizzlingKey);
}

MVCNN::TensorReference createTensorRef(mlir::Value val, int64_t sectionIndex, int64_t byteOffset,
                                       std::optional<int64_t> sparsityMapOffset,
                                       std::optional<int64_t> storageElementOffset,
                                       std::optional<int64_t> storageElementSize, std::optional<int64_t> swizzlingKey) {
    return createTensorRef(val, ArrayRef({sectionIndex}), byteOffset, sparsityMapOffset, storageElementOffset,
                           storageElementSize, swizzlingKey);
}

MVCNN::Vector<uint32_t> createDims(ShapeRef shape) {
    return createVector(shape | transformed([](int64_t val) {
                            return checked_cast<uint32_t>(val);
                        }));
}

MVCNN::Vector<uint32_t> createDims(vpux::NDTypeInterface type) {
    return createDims(type.getShape());
}

template <typename T>
MVCNN::Vector<T> createStrides(StridesRef strides, Bit elemSize) {
    Strides temp;
    temp.push_back(elemSize);
    temp.append(strides.begin(), strides.end());

    const auto cvtBitStride = [](Bit val) {
        if constexpr (!std::is_floating_point<T>::value) {
            return checked_cast<T>(val.count());
        }

        if (val.count() % CHAR_BIT == 0) {
            return checked_cast<T>(Byte(val).count());
        }

        return checked_cast<T>(val.count()) / CHAR_BIT;
    };

    return createVector(temp | transformed(cvtBitStride));
}

template <typename T>
MVCNN::Vector<T> createStrides(vpux::NDTypeInterface type) {
    const auto strides = type.getStrides();
    return createStrides<T>(strides, type.getElemTypeSize());
}

MVCNN::IndirectDataReference createIndirectDataReference(int64_t dataIndex, std::optional<int64_t> sparsityIndex,
                                                         std::optional<int64_t> storageElementIndex,
                                                         std::optional<int64_t> storageElementSize) {
    MVCNN::IndirectDataReference indDataRef;
    indDataRef.add_data_index(checked_cast<uint64_t>(dataIndex));
    if (sparsityIndex.has_value()) {
        indDataRef.add_sparsity_index(checked_cast<uint64_t>(sparsityIndex.value()));
    }
    if (storageElementIndex.has_value()) {
        indDataRef.add_storage_element_index(checked_cast<uint64_t>(storageElementIndex.value()));
    }
    if (storageElementSize.has_value()) {
        indDataRef.add_storage_element_size(checked_cast<uint32_t>(storageElementSize.value()));
    }
    return indDataRef;
}

}  // namespace vpux::VPUMI37XX
