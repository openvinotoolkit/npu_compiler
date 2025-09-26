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

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <algorithm>

using namespace vpux;

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

VPUIP::MemoryLocation createMemoryLocation(VPURT::BufferSection section) {
    switch (section) {
    case VPURT::BufferSection::NetworkInput:
        return VPUIP::MemoryLocation_ProgrammableInput;
    case VPURT::BufferSection::NetworkOutput:
        return VPUIP::MemoryLocation_ProgrammableOutput;
    case VPURT::BufferSection::ProfilingOutput:
        return VPUIP::MemoryLocation_ProfilingOutput;
    case VPURT::BufferSection::Constant:
        return VPUIP::MemoryLocation_GraphFile;
    case VPURT::BufferSection::SW_KernelText:
        return VPUIP::MemoryLocation_GFEmbeddedKernel;
    case VPURT::BufferSection::FunctionInput:
    case VPURT::BufferSection::FunctionOutput:
    case VPURT::BufferSection::DDR:
        return VPUIP::MemoryLocation_VPU_DDR_Heap;
    case VPURT::BufferSection::CSRAM:
        return VPUIP::MemoryLocation_VPU_CSRAM;
    case VPURT::BufferSection::CMX_NN:
        return VPUIP::MemoryLocation_VPU_CMX_NN;
    case VPURT::BufferSection::Register:
        return VPUIP::MemoryLocation_AbsoluteAddr;
    case VPURT::BufferSection::MAC_Accumulators:
        return VPUIP::MemoryLocation_MAC_Accumulators;
    default:
        VPUX_THROW("Unsupported BufferSection {0}", section);
    }
}

VPUMI37XX::BlobWriter::TensorReference vpux::VPUMI37XX::BlobWriter::createTensorRef(
        StringRef name, vpux::NDTypeInterface type, VPURT::BufferSection section, ArrayRef<int64_t> sectionIndex,
        int64_t byteOffset, ArrayRef<int64_t> mult, ArrayRef<int64_t> shift, int64_t postShift,
        ArrayRef<uint8_t> zeroPoints, std::optional<int64_t> sparsityMapOffset,
        std::optional<int64_t> storageElementOffset, std::optional<int64_t> storageElementSize,
        std::optional<int64_t> swizzlingKey, std::optional<uint64_t> descriptor) {
    const auto serializedName = createString(name);

    const auto serializedDataType = static_cast<MVCNN::DType>(createDType(type.getElementType()));
    const auto serializedDims = createDims(type);
    const auto dimsOrder = type.getDimsOrder();

    const auto serializedDataReference =
            createIndirectDataReference(byteOffset, sparsityMapOffset, storageElementOffset, storageElementSize);

    const auto serializedLocale = static_cast<MVCNN::MemoryLocation>(createMemoryLocation(section));

    Vector<uint8_t> serializedQuantZero = createVector(zeroPoints);

    const auto serializedLocaleIndex = arrayCast<uint32_t>(sectionIndex);
    // Unchecked casting of quant_mult since it stores both u16 and i16 values
    auto castedQuantMult = SmallVector<uint16_t>(mult.size());
    std::transform(mult.begin(), mult.end(), castedQuantMult.begin(), [](auto value) {
        return value & 0xFFFF;
    });
    const auto serializedQuantMult = createVector(castedQuantMult);
    const auto serializedQuantShift = arrayCast<uint8_t>(shift);

    const auto basePtrs = createVector(std::vector<uint16_t>{});

    auto strides = createStrides<uint64_t>(type);
    MVCNN::TensorReferenceBuilder builder(_impl);

    builder.add_name(serializedName);
    builder.add_dimensions(serializedDims);
    builder.add_bit_strides(strides);
    builder.add_data(serializedDataReference);
    builder.add_locale(serializedLocale);
    builder.add_locale_index(serializedLocaleIndex);
    builder.add_data_dtype(serializedDataType);
    builder.add_quant_zero(serializedQuantZero);
    builder.add_quant_mult(serializedQuantMult);
    builder.add_quant_shift(serializedQuantShift);
    builder.add_quant_post_shift_right(checked_cast<int8_t>(postShift));
    builder.add_order(dimsOrder.code());
    builder.add_base_ptrs(basePtrs);
    if (swizzlingKey.has_value()) {
        builder.add_swizzling_key(checked_cast<uint8_t>(swizzlingKey.value()));
    }
    if (descriptor.has_value()) {
        builder.add_descriptor(checked_cast<uint64_t>(descriptor.value()));
    }
    return builder.Finish();
}

VPUMI37XX::BlobWriter::TensorReference vpux::VPUMI37XX::BlobWriter::createTensorRef(
        StringRef name, vpux::NDTypeInterface type, VPURT::BufferSection section, ArrayRef<int64_t> sectionIndex,
        int64_t byteOffset, std::optional<int64_t> sparsityMapOffset, std::optional<int64_t> storageElementOffset,
        std::optional<int64_t> storageElementSize, std::optional<int64_t> swizzlingKey,
        std::optional<uint64_t> descriptor) {
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

    return createTensorRef(name, type, section, sectionIndex, byteOffset, mults, shifts, 0, zeroPoints,
                           sparsityMapOffset, storageElementOffset, storageElementSize, swizzlingKey, descriptor);
}

VPUMI37XX::BlobWriter::TensorReference vpux::VPUMI37XX::BlobWriter::createTensorRef(
        StringRef name, vpux::NDTypeInterface type, VPURT::BufferSection section, int64_t sectionIndex,
        int64_t byteOffset, std::optional<int64_t> sparsityMapOffset, std::optional<int64_t> storageElementOffset,
        std::optional<int64_t> storageElementSize, std::optional<int64_t> swizzlingKey,
        std::optional<uint64_t> descriptor) {
    return createTensorRef(name, type, section, ArrayRef({sectionIndex}), byteOffset, sparsityMapOffset,
                           storageElementOffset, storageElementSize, swizzlingKey, descriptor);
}

VPUMI37XX::BlobWriter::TensorReference vpux::VPUMI37XX::BlobWriter::createTensorRef(
        mlir::Value val, StringRef name, VPURT::BufferSection section, ArrayRef<int64_t> sectionIndex,
        int64_t byteOffset, std::optional<int64_t> sparsityMapOffset, std::optional<int64_t> storageElementOffset,
        std::optional<int64_t> storageElementSize, std::optional<int64_t> swizzlingKey,
        std::optional<uint64_t> descriptor) {
    VPUX_THROW_UNLESS(_tensors.count(val) == 0, "Value '{0}' was already serialized", val.getLoc());
    const auto ref =
            createTensorRef(name, mlir::cast<vpux::NDTypeInterface>(val.getType()), section, sectionIndex, byteOffset,
                            sparsityMapOffset, storageElementOffset, storageElementSize, swizzlingKey, descriptor);
    _tensors.insert({val, ref});
    return ref;
}

VPUMI37XX::BlobWriter::TensorReference vpux::VPUMI37XX::BlobWriter::createTensorRef(
        mlir::Value val, StringRef name, VPURT::BufferSection section, int64_t sectionIndex, int64_t byteOffset,
        std::optional<int64_t> sparsityMapOffset, std::optional<int64_t> storageElementOffset,
        std::optional<int64_t> storageElementSize, std::optional<int64_t> swizzlingKey,
        std::optional<uint64_t> descriptor) {
    return createTensorRef(val, name, section, ArrayRef({sectionIndex}), byteOffset, sparsityMapOffset,
                           storageElementOffset, storageElementSize, swizzlingKey, descriptor);
}

VPUMI37XX::BlobWriter::Vector<uint32_t> vpux::VPUMI37XX::BlobWriter::createDims(ShapeRef shape) {
    return createVector(shape | transformed([](int64_t val) {
                            return checked_cast<uint32_t>(val);
                        }));
}

VPUMI37XX::BlobWriter::Vector<uint32_t> vpux::VPUMI37XX::BlobWriter::createDims(vpux::NDTypeInterface type) {
    return createDims(type.getShape());
}

template <typename T>
VPUMI37XX::BlobWriter::Vector<T> vpux::VPUMI37XX::BlobWriter::createStrides(StridesRef strides, Bit elemSize) {
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
VPUMI37XX::BlobWriter::Vector<T> vpux::VPUMI37XX::BlobWriter::createStrides(vpux::NDTypeInterface type) {
    const auto strides = type.getStrides();
    return createStrides<T>(strides, type.getElemTypeSize());
}

VPUMI37XX::BlobWriter::IndirectDataReference vpux::VPUMI37XX::BlobWriter::createIndirectDataReference(
        int64_t dataIndex, std::optional<int64_t> sparsityIndex, std::optional<int64_t> storageElementIndex,
        std::optional<int64_t> storageElementSize) {
    MVCNN::IndirectDataReferenceBuilder builder(_impl);
    builder.add_data_index(checked_cast<uint64_t>(dataIndex));
    if (sparsityIndex.has_value()) {
        builder.add_sparsity_index(checked_cast<uint64_t>(sparsityIndex.value()));
    }
    if (storageElementIndex.has_value()) {
        builder.add_storage_element_index(checked_cast<uint64_t>(storageElementIndex.value()));
    }
    if (storageElementSize.has_value()) {
        builder.add_storage_element_size(checked_cast<uint32_t>(storageElementSize.value()));
    }
    return builder.Finish();
}
