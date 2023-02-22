//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/IE/float16.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

#include <llvm/Support/TypeName.h>

#include <memory>

namespace vpux {
namespace Const {

namespace details {

//
// ConvertCb
//

template <typename OutT>
using ConvertCb = OutT (*)(const char*);

template <typename OutT>
struct CvtHelper final {
    template <typename InT>
    static OutT cvt(InT val) {
        return checked_cast<OutT>(val);
    }
};

template <>
struct CvtHelper<float16> final {
    template <typename InT>
    static float16 cvt(InT val) {
        return float16(checked_cast<float>(val));
    }
};

template <>
struct CvtHelper<bfloat16> final {
    template <typename InT>
    static bfloat16 cvt(InT val) {
        return bfloat16(checked_cast<float>(val));
    }
};

template <typename InT, typename OutT>
ConvertCb<OutT> makeConvertCb() {
    return [](const char* rawPtr) {
        return CvtHelper<OutT>::cvt(*reinterpret_cast<const InT*>(rawPtr));
    };
}

//
// ContentRangeBase
//

template <typename OutT>
class ContentRangeBase final {
public:
    ContentRangeBase(ArrayRef<char> data, bool isSplat, Byte elemSize, ConvertCb<OutT> cvtOp)
            : _data(data), _isSplat(isSplat), _elemSize(elemSize), _cvtOp(std::move(cvtOp)) {
        if (_isSplat) {
            VPUX_THROW_UNLESS(_data.size() == checked_cast<size_t>(_elemSize.count()),
                              "Splat data store size '{0}' doesn't match element type size '{1}'", _data.size(),
                              _elemSize);
        }
    }

public:
    OutT getItem(ptrdiff_t ind) const {
        if (_isSplat) {
            return _cvtOp(_data.data());
        }

        const auto rawIndex = checked_cast<size_t>(ind * _elemSize.count());
        VPUX_THROW_UNLESS(rawIndex < _data.size(), "Out-of-bound access in ContentRangeBase");

        return _cvtOp(_data.data() + rawIndex);
    }

public:
    bool operator==(const ContentRangeBase& other) const {
        return _elemSize == other._elemSize && _isSplat == other._isSplat && _data == other._data;
    }
    bool operator!=(const ContentRangeBase& other) const {
        return !(*this == other);
    }

private:
    ArrayRef<char> _data;
    bool _isSplat = false;
    Byte _elemSize;
    ConvertCb<OutT> _cvtOp;
};

//
// ContentRange
//

template <typename OutT>
class ContentRange final :
        public llvm::indexed_accessor_range<ContentRange<OutT>, ContentRangeBase<OutT>, OutT, OutT, OutT> {
    using BaseType = llvm::indexed_accessor_range<ContentRange<OutT>, ContentRangeBase<OutT>, OutT, OutT, OutT>;

public:
    ContentRange(ArrayRef<char> data, bool isSplat, Byte elemSize, ptrdiff_t count, ConvertCb<OutT> cvtOp)
            : BaseType(ContentRangeBase<OutT>(data, isSplat, elemSize, std::move(cvtOp)), 0, count) {
    }

public:
    static OutT dereference(const ContentRangeBase<OutT>& base, ptrdiff_t ind) {
        return base.getItem(ind);
    }
};

}  // namespace details

//
// Content
//

class Content final {
public:
    // `data` storage might have different element type than base `type`.
    // The `getValues` / `getSplatValue` methods accept template type parameter and convert element type on the fly.
    static Content fromRawBuffer(vpux::NDTypeInterface type, ArrayRef<char> data, mlir::Type storageElemType,
                                 bool isSplat);
    static Content allocTempBuffer(vpux::NDTypeInterface type, mlir::Type storageElemType, bool isSplat);
    static Content allocTempBuffer(vpux::NDTypeInterface type, mlir::Type storageElemType, bool isSplat,
                                   size_t tempBufRawSize);
    static Content moveBuffer(vpux::NDTypeInterface type, Content&& other);

public:
    vpux::NDTypeInterface getType() const {
        return _type;
    }

public:
    template <typename OutT>
    details::ContentRange<OutT> getValues() const& {
        auto cvtOp = dispatchByElemType<details::ConvertCb<OutT>>(getStorageElemType(), [](auto dummy) {
            using InT = std::decay_t<decltype(dummy)>;
            return details::makeConvertCb<InT, OutT>();
        });

        const auto storageElemTypeSize = vpux::getElemTypeSize(_storageElemType);
        return details::ContentRange<OutT>(_data, _isSplat, storageElemTypeSize, getType().getNumElements(),
                                           std::move(cvtOp));
    }

    template <typename OutT>
    void getValues() && = delete;

public:
    bool isSplat() const {
        return _isSplat;
    }

    template <typename OutT>
    auto getSplatValue() const {
        VPUX_THROW_UNLESS(isSplat(), "Expected the attribute to be a splat value");
        return *getValues<OutT>().begin();
    }

public:
    void copyTo(MutableArrayRef<char> buf) const;

    void fillWithZero();

    template <typename Caller>
    void mutate(Caller&& caller) & {
        dispatchByElemType<void>(getStorageElemType(), [this, caller](auto dummy) {
            using ElemT = std::decay_t<decltype(dummy)>;
            caller(this->getTempBuf<ElemT>());
        });
    }

public:
    template <typename OutT>
    MutableArrayRef<OutT> getTempBuf() & {
        VPUX_THROW_UNLESS(_tempBuf != nullptr, "Temp buffer was not allocated");

        const Byte storageElemSize = vpux::getElemTypeSize(_storageElemType);
        VPUX_THROW_UNLESS(storageElemSize.count() == sizeof(OutT),
                          "Temp buffer type '{0}' mismatch with storage element size '{1}'", llvm::getTypeName<OutT>(),
                          storageElemSize);

        return MutableArrayRef<OutT>(reinterpret_cast<OutT*>(_tempBuf.get()), _data.size() / sizeof(OutT));
    }

    template <typename OutT>
    MutableArrayRef<OutT> getTempBuf() && = delete;

public:
    mlir::Type getStorageElemType() const {
        return _storageElemType;
    }

    void setStorageElemType(mlir::Type newStorageElemType);

    ArrayRef<char> getRawStorageBuf() const& {
        return _data;
    }

    ArrayRef<char> getRawStorageBuf() && = delete;

    MutableArrayRef<char> getRawTempBuf() & {
        VPUX_THROW_UNLESS(_tempBuf != nullptr, "Temp buffer was not allocated");
        return makeMutableArrayRef(_tempBuf.get(), _data.size());
    }

    MutableArrayRef<char> getRawTempBuf() && = delete;

private:
    Content() = default;

    template <typename RetT, class Caller>
    static RetT dispatchByElemType(mlir::Type elemType, Caller&& caller) {
        if (elemType.isUnsignedInteger(8)) {
            return caller(uint8_t(0));
        } else if (elemType.isUnsignedInteger(16)) {
            return caller(uint16_t(0));
        } else if (elemType.isUnsignedInteger(32)) {
            return caller(uint32_t(0));
        } else if (elemType.isUnsignedInteger(64)) {
            return caller(uint64_t(0));
        } else if (elemType.isSignedInteger(8)) {
            return caller(int8_t(0));
        } else if (elemType.isSignedInteger(16)) {
            return caller(int16_t(0));
        } else if (elemType.isSignedInteger(32)) {
            return caller(int32_t(0));
        } else if (elemType.isSignedInteger(64)) {
            return caller(int64_t(0));
        } else if (elemType.isF32()) {
            return caller(float(0));
        } else if (elemType.isF64()) {
            return caller(double(0));
        } else if (elemType.isF16()) {
            return caller(float16(0.0f));
        } else if (elemType.isBF16()) {
            return caller(bfloat16(0.0f));
        } else if (const auto qType = elemType.dyn_cast<mlir::quant::QuantizedType>()) {
            const auto quantStorageType = vpux::normalizeQuantStorageType(qType);
            if (quantStorageType.isSignedInteger(8)) {
                return caller(int8_t(0));
            } else if (quantStorageType.isUnsignedInteger(8)) {
                return caller(uint8_t(0));
            } else {
                VPUX_THROW("Unsupported quantized storage type '{0}'", quantStorageType);
            }
        } else {
            VPUX_THROW("Unsupported element type '{0}'", elemType);
        }
    }

private:
    vpux::NDTypeInterface _type;
    ArrayRef<char> _data;
    mlir::Type _storageElemType;
    bool _isSplat = false;
    std::unique_ptr<char[]> _tempBuf;
};

}  // namespace Const
}  // namespace vpux
