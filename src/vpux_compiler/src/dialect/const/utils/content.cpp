//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/utils/content.hpp"
#include <mlir/Support/LLVM.h>
#include <cstdint>

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/const/utils/sub_byte.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <type_traits>
#include <vpux/utils/core/mem_size.hpp>

using namespace vpux;

//
// Content::fromRawBuffer
//

Const::Content vpux::Const::Content::fromRawBuffer(vpux::NDTypeInterface type, ArrayRef<char> data,
                                                   mlir::Type storageElemType, bool isSplat) {
    Const::Content content;
    content._type = type;
    content._data = Const::ConstData::fromRawBuffer(static_cast<const void*>(data.data()), data.size());
    content._storageElemType = storageElemType;
    content._isSplat = isSplat;
    return content;
}

//
// Content::allocTempBuffer
//

Const::Content vpux::Const::Content::allocTempBuffer(vpux::NDTypeInterface type, mlir::Type storageElemType,
                                                     bool isSplat) {
    const int64_t tempBufSize = isSplat ? 1 : type.getNumElements();
    const Bit tempElemBitSize = vpux::getElemTypeSize(storageElemType);
    const auto tempBufRawBitSize = alignMemSize(tempElemBitSize * tempBufSize, Byte(1));
    auto data = Const::ConstData::allocate<char>(Byte(tempBufRawBitSize).count());
    return Const::Content(type, std::move(data), storageElemType, isSplat);
}

Const::Content vpux::Const::Content::allocTempBuffer(vpux::NDTypeInterface type, mlir::Type storageElemType,
                                                     bool isSplat, size_t tempBufRawSize) {
    // Overloading for sub-byte cases.
    auto data = Const::ConstData::allocate<char>(tempBufRawSize);
    return Const::Content(type, std::move(data), storageElemType, isSplat);
}

//
// Content::moveBuffer
//

Const::Content vpux::Const::Content::moveBuffer(vpux::NDTypeInterface type, Const::Content&& other) {
    Const::Content content(std::move(other));
    content._type = type;
    return content;
}

// The Content object might not own the referred data. This function ensures the returned Content object owns the
// referred data by copying it into a new buffer when needed
Const::Content vpux::Const::Content::copyUnownedBuffer(Const::Content&& origin) {
    if (origin._data.isMutable()) {  // is internal already
        return std::move(origin);
    }

    auto oldData = origin._data.data();
    if (!oldData.empty()) {  // could be the case after --fuse-constants?
        const auto dataSize = oldData.size();
        origin._data = Const::ConstData::allocate<char>(dataSize);
        auto mutableData = origin._data.mutableData<char>();
        std::copy_n(oldData.begin(), dataSize, mutableData.begin());
    }

    return std::move(origin);
}

//
// Content::copyTo
//

namespace {

template <typename DstType, typename SrcType>
void fillBuf(ArrayRef<SrcType> src, MutableArrayRef<char> dst) {
    constexpr auto VALUE_BYTE_SIZE = sizeof(DstType);

    const auto convertedSrcSize = src.size() * VALUE_BYTE_SIZE;
    VPUX_THROW_UNLESS(src.size() == 1 || dst.size() == convertedSrcSize,
                      "Target buffer size '{0}' does not match source buffer size '{1}'", dst.size(), convertedSrcSize);

    const auto doCast = [](SrcType value) -> DstType {
        // Note: unconditionally use CvtHelper as this is what the previous
        // implementation did (`getValues<DstType>()`) to ensure the
        // compatibility of results.
        return Const::details::CvtHelper<DstType>::cvt(value);
    };

    auto* dstPtr = reinterpret_cast<DstType*>(dst.data());
    if (bool isSplat = src.size() == 1; isSplat) {
        std::fill_n(dstPtr, dst.size() / VALUE_BYTE_SIZE, doCast(src.front()));
        return;
    }
    std::transform(src.begin(), src.end(), dstPtr, doCast);
}

}  // namespace

void vpux::Const::Content::copySubByteContent(MutableArrayRef<char> targetData, mlir::Type elemType) const {
    if (_isSplat) {
        const auto bits = vpux::getElemTypeSize(elemType).count();
        uint8_t byteValue = 0;
        const auto subByteValue = [this, bits]() {
            const auto mask = checked_cast<uint8_t>((1 << bits) - 1);
            if (_storageElemType.isSignedInteger() || _storageElemType.isFloat()) {
                return static_cast<uint8_t>(getSplatValue<int8_t>() & mask);
            } else {
                return static_cast<uint8_t>(getSplatValue<uint8_t>() & mask);
            }
        }();

        const auto numShifts = CHAR_BIT / bits;
        for (int64_t shift = 0; shift < numShifts; shift++) {
            byteValue = (byteValue << bits) | subByteValue;
        }
        std::fill(targetData.begin(), targetData.end(), byteValue);
        return;
    }

    // The source data might be stored with each sub-byte element into an individual byte.
    // This can happen when using the `CastElemType<i1>` transformation which does not alter the underlying data.
    // If the target buffer for the copy is smaller than the source buffer, the data will be packed if that is possible.
    // e.g. source data contains bytes with values 1 or 0 representing (i.e. boolean element type) while the target
    // buffer contains 1/8th of elements - the source values will be packed into bytes in the target buffer
    if (targetData.size() < _data.size()) {
        VPUX_THROW_UNLESS(_data.size() % targetData.size() == 0,
                          "Cannot pack sub-byte elements into buffer: source data size '{0}', target data size '{1}'",
                          _data.size(), targetData.size());
        auto sourceValues = getValues<int8_t>();
        const auto elemPerByte = sourceValues.size() / targetData.size();
        VPUX_THROW_UNLESS(elemPerByte <= CHAR_BIT && vpux::isPowerOfTwo(elemPerByte),
                          "Invalid number of elements per byte '{0}'", elemPerByte);

        const auto bits = CHAR_BIT / elemPerByte;
        const char mask = checked_cast<uint8_t>(checked_cast<uint16_t>(std::pow(2, bits)) - 1);
        for (size_t idx = 0; idx < sourceValues.size(); idx += elemPerByte) {
            uint8_t byte = 0;
            uint8_t shift = 0;
            for (uint64_t elemIdx = 0; elemIdx < elemPerByte; ++elemIdx) {
                byte |= (sourceValues[idx + elemIdx] & mask) << shift;
                shift += bits;
            }
            targetData[idx / elemPerByte] = byte;
        }
        return;
    }
    auto srcData = _data.data();
    std::memcpy(targetData.data(), srcData.data(), srcData.size());
}

mlir::Type vpux::Const::Content::getNormalizedQuantStorageType(mlir::quant::QuantizedType qType) {
    return vpux::normalizeQuantStorageType(qType);
}

void vpux::Const::Content::copyTo(MutableArrayRef<char> targetData) const {
    const auto elemType = getType().getElementType();
    const Bit elemSize = vpux::getElemTypeSize(elemType);
    const auto isSubByte = elemSize.count() < CHAR_BIT;
    if (isSubByte) {
        copySubByteContent(targetData, elemType);
        return;
    }

    read(elemType, [&](auto srcData, auto dummy) {
        fillBuf<decltype(dummy)>(srcData, targetData);
    });
}

//
// Content::fillWithZero
//

void vpux::Const::Content::fillWithZero() {
    auto fillQuantileTypeWithZero = [&](mlir::quant::QuantizedType type) -> mlir::LogicalResult {
        const auto quantileStorageType = mlir::dyn_cast_if_present<vpux::type::QuantileType>(type.getStorageType());
        if (quantileStorageType == nullptr) {
            return mlir::failure();
        }

        const auto bitWidth = quantileStorageType.getStorageWidth();
        const auto quantiles = quantileStorageType.getQuantiles();

        const uint64_t zeroIdx = std::distance(quantiles.begin(), std::find(quantiles.begin(), quantiles.end(), 0.0));
        VPUX_THROW_UNLESS(zeroIdx != quantiles.size(),
                          "Missing zero (0) value from palletization LUT, which must be present for padding.");

        const auto isSubByte = bitWidth < CHAR_BIT;
        uint8_t packedVal = 0;
        if (isSubByte) {
            constexpr int64_t bitsInByte = 8;
            const auto numVals = bitsInByte / bitWidth;
            const uint64_t mask = (1 << bitWidth) - 1;
            VPUX_THROW_WHEN(zeroIdx > mask, "Zero index {0} cannot be represented with {1} bits", zeroIdx, bitWidth);

            const auto maskedIdx = zeroIdx & mask;
            for (auto i : irange(numVals)) {
                packedVal |= maskedIdx << (i * bitWidth);
            }
        }

        // Quantile type has the same zero value even in the case of per axis quantization (only scales differ per
        // channel)
        const auto fillBuffer = [&](auto buffer) {
            using BufferType = std::decay_t<decltype(buffer)>;
            using ElemType = typename BufferType::value_type;

            // Using static_cast for packedVal, as it is a value comprised of multiple subbyte put together
            // in a byte and does not represent a real number on its own.
            const auto val = isSubByte ? static_cast<ElemType>(packedVal) : checked_cast<ElemType>(zeroIdx);
            std::fill_n(buffer.data(), buffer.size(), val);
        };

        mutate(fillBuffer);
        return mlir::success();
    };

    if (auto perAxisQuantizedType =
                mlir::dyn_cast_if_present<mlir::quant::UniformQuantizedPerAxisType>(getType().getElementType())) {
        VPUX_THROW_UNLESS(perAxisQuantizedType.getQuantizedDimension() == 0,
                          "Only per-channel quantization is supported");

        // Treat quantile storage type case
        const auto res = fillQuantileTypeWithZero(perAxisQuantizedType);
        if (res.failed()) {
            // If storage type is not quantile, fill with zero points per channel
            const auto outShape = getType().getShape();
            const auto order = getType().getDimsOrder();
            const auto outMemShape = order.toMemoryOrder(outShape);

            VPUX_THROW_UNLESS(outShape.size() == 4, "Unsupported shape size {0}", outShape.size());

            const auto OC = outShape[Dims4D::Filter::OC];
            const auto IC = outShape[Dims4D::Filter::IC];
            const auto H = outShape[Dims4D::Filter::KY];
            const auto W = outShape[Dims4D::Filter::KX];

            const auto zeroPoints = perAxisQuantizedType.getZeroPoints();
            loop_4d(LoopExecPolicy::Parallel, getType().getContext(), OC, IC, H, W,
                    [&](int64_t i, int64_t ic, int64_t h, int64_t w) {
                        const auto zp = zeroPoints[i];

                        const auto fillChannel = [&](auto buffer) {
                            using BufferType = std::decay_t<decltype(buffer)>;
                            using ElemType = typename BufferType::value_type;

                            const auto inMemIndND = order.toMemoryOrder(Shape{i, ic, h, w});
                            const auto inMemInd1D = getMemIndex1D(inMemIndND, outMemShape);

                            buffer[inMemInd1D] = checked_cast<ElemType>(zp);
                        };

                        mutate(fillChannel);
                    });
        }
    } else if (auto uniformQuantizedType =
                       mlir::dyn_cast_if_present<mlir::quant::UniformQuantizedType>(getType().getElementType())) {
        // Treat quantile storage type case
        const auto res = fillQuantileTypeWithZero(uniformQuantizedType);
        if (res.failed()) {
            // If storage type is not quantile, fill with zero points, same per tensor
            const auto zp = uniformQuantizedType.getZeroPoint();

            const auto fillBuffer = [&](auto buffer) {
                using BufferType = std::decay_t<decltype(buffer)>;
                using ElemType = typename BufferType::value_type;

                std::fill_n(buffer.data(), buffer.size(), checked_cast<ElemType>(zp));
            };

            mutate(fillBuffer);
        }
    } else {
        auto outBuf = getRawTempBuf();
        std::fill_n(outBuf.data(), outBuf.size(), char(0));
    }
}

//
// Content::setStorageElemType
//

void vpux::Const::Content::setStorageElemType(mlir::Type newStorageElemType) {
    _storageElemType = newStorageElemType;
}
