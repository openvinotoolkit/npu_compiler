//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/const/utils/const_data.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/custom_float.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/type/float16.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

#include <llvm/Support/TypeName.h>

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
struct CvtHelper<vpux::type::float16> final {
    template <typename InT>
    static vpux::type::float16 cvt(InT val) {
        return vpux::type::float16(val);
    }
};

template <>
struct CvtHelper<vpux::type::bfloat16> final {
    template <typename InT>
    static vpux::type::bfloat16 cvt(InT val) {
        return vpux::type::bfloat16(checked_cast<float>(val));
    }
};

template <>
struct CvtHelper<vpux::type::float8_e4m3> final {
    template <typename InT>
    static vpux::type::float8_e4m3 cvt(InT val) {
        return vpux::type::float8_e4m3(checked_cast<float>(val));
    }
};

template <>
struct CvtHelper<vpux::type::float8_e5m2> final {
    template <typename InT>
    static vpux::type::float8_e5m2 cvt(InT val) {
        return vpux::type::float8_e5m2(checked_cast<float>(val));
    }
};

template <>
struct CvtHelper<bool> final {
    template <typename InT>
    static bool cvt(InT val) {
        return val != static_cast<InT>(0);
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
        return _elemSize == other._elemSize && _isSplat == other._isSplat && _data.data() == other._data.data() &&
               _data.size() == other._data.size();
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
    Content() = default;
    Content(vpux::NDTypeInterface type, ConstData&& data, mlir::Type storageElemType, bool isSplat)
            : _type(type), _data(std::move(data)), _storageElemType(storageElemType), _isSplat(isSplat) {
    }

    // `data` storage might have different element type than base `type`.
    // The `getValues` / `getSplatValue` methods accept template type parameter and convert element type on the fly.
    static Content fromRawBuffer(vpux::NDTypeInterface type, ArrayRef<char> data, mlir::Type storageElemType,
                                 bool isSplat);
    static Content allocTempBuffer(vpux::NDTypeInterface type, mlir::Type storageElemType, bool isSplat);
    static Content allocTempBuffer(vpux::NDTypeInterface type, mlir::Type storageElemType, bool isSplat,
                                   size_t tempBufRawSize);
    static Content moveBuffer(vpux::NDTypeInterface type, Content&& other);
    static Content copyUnownedBuffer(Content&& origin);

public:
    vpux::NDTypeInterface getType() const {
        return _type;
    }

public:
    template <typename OutT>
    details::ContentRange<OutT> getValues() const& {
        auto cvtOp = dispatchByElemType({getStorageElemType()}, [](auto dummy) {
            using InT = std::decay_t<decltype(dummy)>;
            return details::makeConvertCb<InT, OutT>();
        });

        const Bit logicalElemBitSize = vpux::getElemTypeSize(_storageElemType);
        auto isSubbyte = logicalElemBitSize.count() < CHAR_BIT;
        const Bit storageElemTypeSize = isSubbyte ? Bit(8) : logicalElemBitSize;

        return details::ContentRange<OutT>(_data.data(), _isSplat, storageElemTypeSize, getType().getNumElements(),
                                           std::move(cvtOp));
    }

    template <typename OutT>
    void getValues() && = delete;

    template <typename OutT>
    std::vector<OutT> vec() const {
        auto allocSize = getType().getTotalAllocSize().count();
        auto outTsize = sizeof(OutT);
        VPUX_THROW_UNLESS(allocSize % outTsize == 0, "size of Content is expected to be multiple of {0} but found {1}",
                          outTsize, allocSize);

        std::vector<OutT> outValues(allocSize / outTsize);
        MutableArrayRef<char> buf(reinterpret_cast<char*>(outValues.data()), allocSize);
        copyTo(buf);
        return outValues;
    }

public:
    bool isSplat() const {
        return _isSplat;
    }

    template <typename OutT>
    auto getSplatValue() const {
        VPUX_THROW_UNLESS(isSplat(), "Expected the attribute to be a splat value");
        return read([](auto values) {
            if constexpr (std::is_same<OutT, bool>::value) {
                // E#160869: checked_cast<bool> works poorly due to MSVC warning
                // C4804. fixing checked_cast<> overload is also not simple, so
                // for now this could act as a workaround.
                return static_cast<bool>(values[0]);
            } else {
                return checked_cast<OutT>(values[0]);
            }
        });
    }

public:
    void copyTo(MutableArrayRef<char> buf) const;

    void fillWithZero();

    template <typename Caller>
    void mutate(Caller&& caller) & {
        dispatchByElemType({getStorageElemType()}, [this, caller](auto dummy) {
            using ElemT = std::decay_t<decltype(dummy)>;
            caller(this->getTempBuf<ElemT>());
        });
    }

    /** @brief Provides read access to the underlying data.

        Provides storage buffer of the specific type to the callable. The buffer
        type is determined based on the internal details and is guaranteed to
        match real underlying data type.

        @note The buffer type may not be equal to Content::getType().
     */
    template <typename Caller>
    auto read(Caller&& caller) const& {
        return dispatchByElemType({getStorageElemType()}, [this, caller](auto dummy) {
            using ElemT = std::decay_t<decltype(dummy)>;
            return caller(this->getStorageBuf<ElemT>());
        });
    }

    template <typename Caller>
    auto read(Caller&&) && = delete;

    /** @brief Provides read access to the underlying data.

        Provides storage buffer of the specific type to the callable as well as
        a default-constructed value associated with some other specified type.
        The buffer type is determined based on the internal details and is
        guaranteed to match real underlying data type.

        This is an overload of another Content::read() that allows one to
        combine direct data access with an additional type-dispatch. For
        instance, one can use this in contexts where storage buffer values must
        be converted to some other type and thus one needs:
        * data buffer of a runtime-defined type - ArrayRef<T>
        * other runtime-defined type - U

        resulting in: `auto operator()(ArrayRef<T> rawData, U dummy)`.

        A pseudo-example of such a situation is the following:
        ```cpp
        // Convert constant to a new element type
        auto newElemType = getNewElemType(...);
        auto outputs = allocateOutputs(...);

        constant.read(newElemType, [&outputs](auto rawValues, auto dummy) {
            // rawValues is an ArrayRef<T>, where T is deduced from the internal
            // type of the constant.

            using NewType = decltype(dummy);
            // NewType is a static type deduced from MLIR type `newElemType`:
            // * if newElemType is 'f16' in MLIR, NewType is vpux::type::float16
            // * if newElemType is 'ui8' in MLIR, NewType is uint8_t
            // ...

            for (size_t i = 0; i < rawValues.size(); ++i) {
                // access raw data by index and then cast it to `newElemType`
                // store result into the output
                outputs[i] = checked_cast<NewType>(rawValues[i]);
            }
        });
        ```

        @note The buffer type may not be equal to Content::getType().
     */
    template <typename Caller>
    auto read(mlir::Type otherType, Caller&& caller) const& {
        return dispatchByElemType({getStorageElemType(), otherType}, [this, caller](auto dummy, auto otherTypeDummy) {
            using ElemT = std::decay_t<decltype(dummy)>;
            return caller(this->getStorageBuf<ElemT>(), otherTypeDummy);
        });
    }

    template <typename Caller>
    auto read(mlir::Type, Caller&&) && = delete;

public:
    template <typename OutT>
    MutableArrayRef<OutT> getTempBuf() & {
        VPUX_THROW_WHEN(!_data.isMutable(), "This data is read-only");

        VPUX_THROW_UNLESS(_data.size() % sizeof(OutT) == 0,
                          "Size of tempBuf needs to be multiple of '{0}' but is '{1}'", sizeof(OutT), _data.size());

        return _data.mutableData<OutT>();
    }

    template <typename OutT>
    MutableArrayRef<OutT> getTempBuf() && = delete;

public:
    mlir::Type getStorageElemType() const {
        return _storageElemType;
    }

    void setStorageElemType(mlir::Type newStorageElemType);

    ArrayRef<char> getRawStorageBuf() const& {
        return _data.data();
    }

    ArrayRef<char> getRawStorageBuf() && = delete;

    template <typename OutT>
    ArrayRef<OutT> getStorageBuf() const& {
        VPUX_THROW_UNLESS(_data.size() % sizeof(OutT) == 0, "Size of buffer needs to be multiple of '{0}' but is '{1}'",
                          sizeof(OutT), _data.size());

        return _data.data<OutT>();
    }

    MutableArrayRef<char> getRawTempBuf() & {
        return getTempBuf<char>();
    }

    MutableArrayRef<char> getRawTempBuf() && = delete;

private:
    /// Implements multi-type dispatch.
    template <size_t I, size_t N, class Caller, typename... Types>
    static auto dispatchByElemTypeImpl(const mlir::Type (&types)[N], Caller&& caller) {
        constexpr bool allTypesDispatched = (I == N);
        if constexpr (allTypesDispatched) {
            // Note: base case - call the callable for the sequence of types
            return std::forward<Caller>(caller)(Types()...);
        } else {
            // Note: recursive step - takes I-th runtime type and converts it to
            // static type, then calls dispatch for I + 1
            auto elemType = types[I];
            if (elemType.isUnsignedInteger(8) || elemType.isSignlessInteger(8)) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., uint8_t>(types, std::forward<Caller>(caller));
            } else if (elemType.isUnsignedInteger(4) || elemType.isSignlessInteger(4)) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., uint8_t>(types, std::forward<Caller>(caller));
            } else if (elemType.isUnsignedInteger(16) || elemType.isSignlessInteger(16)) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., uint16_t>(types,
                                                                                    std::forward<Caller>(caller));
            } else if (elemType.isUnsignedInteger(32) || elemType.isSignlessInteger(32)) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., uint32_t>(types,
                                                                                    std::forward<Caller>(caller));
            } else if (elemType.isUnsignedInteger(64) || elemType.isSignlessInteger(64)) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., uint64_t>(types,
                                                                                    std::forward<Caller>(caller));
            } else if (elemType.isSignedInteger(8)) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., int8_t>(types, std::forward<Caller>(caller));
            } else if (elemType.isSignedInteger(4)) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., int8_t>(types, std::forward<Caller>(caller));
            } else if (elemType.isSignedInteger(16)) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., int16_t>(types, std::forward<Caller>(caller));
            } else if (elemType.isSignedInteger(32)) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., int32_t>(types, std::forward<Caller>(caller));
            } else if (elemType.isSignedInteger(64)) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., int64_t>(types, std::forward<Caller>(caller));
            } else if (elemType.isF32()) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., float>(types, std::forward<Caller>(caller));
            } else if (elemType.isF64()) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., double>(types, std::forward<Caller>(caller));
            } else if (elemType.isF16()) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., vpux::type::float16>(
                        types, std::forward<Caller>(caller));
            } else if (elemType.isBF16()) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., vpux::type::bfloat16>(
                        types, std::forward<Caller>(caller));
            } else if (mlir::isa<mlir::Float8E4M3FNType>(elemType)) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., vpux::type::float8_e4m3>(
                        types, std::forward<Caller>(caller));
            } else if (mlir::isa<mlir::Float8E5M2Type>(elemType)) {
                return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., vpux::type::float8_e5m2>(
                        types, std::forward<Caller>(caller));
            } else if (const auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
                const auto quantStorageType = getNormalizedQuantStorageType(qType);
                if (quantStorageType.isSignedInteger(8)) {
                    return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., int8_t>(types,
                                                                                      std::forward<Caller>(caller));
                } else if (quantStorageType.isUnsignedInteger(8)) {
                    return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., uint8_t>(types,
                                                                                       std::forward<Caller>(caller));
                } else if (quantStorageType.isSignedInteger(4)) {
                    return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., int8_t>(types,
                                                                                      std::forward<Caller>(caller));
                } else if (quantStorageType.isUnsignedInteger(4)) {
                    return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., uint8_t>(types,
                                                                                       std::forward<Caller>(caller));
                } else if (mlir::isa<mlir::Float8E4M3FNType>(quantStorageType)) {
                    return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., vpux::type::float8_e4m3>(
                            types, std::forward<Caller>(caller));
                } else if (mlir::isa<mlir::Float8E5M2Type>(quantStorageType)) {
                    return dispatchByElemTypeImpl<I + 1, N, Caller, Types..., vpux::type::float8_e5m2>(
                            types, std::forward<Caller>(caller));
                } else {
                    VPUX_THROW("Unsupported quantized storage type '{0}'", quantStorageType);
                }
            } else {
                VPUX_THROW("Unsupported element type '{0}'", elemType);
            }
        }
    }

    template <size_t N, typename Caller>
    static auto dispatchByElemType(const mlir::Type (&types)[N], Caller&& caller) {
        // Note: impl is recursive and walks through N types, starting from 0
        return dispatchByElemTypeImpl<0>(types, std::forward<Caller>(caller));
    }

private:
    void copySubByteContent(MutableArrayRef<char> targetData, mlir::Type elemType) const;

    // helper function to hide quantization.hpp header
    static mlir::Type getNormalizedQuantStorageType(mlir::quant::QuantizedType qType);

private:
    vpux::NDTypeInterface _type;
    ConstData _data;
    mlir::Type _storageElemType;
    bool _isSplat = false;
};

}  // namespace Const
}  // namespace vpux
