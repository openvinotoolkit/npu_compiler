//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/const/utils/content.hpp"

#include "vpux/utils/IE/loop.hpp"

using namespace vpux;

//
// Content::fromRawBuffer
//

Const::Content vpux::Const::Content::fromRawBuffer(vpux::NDTypeInterface type, ArrayRef<char> data,
                                                   mlir::Type storageElemType, bool isSplat) {
    Const::Content content;
    content._type = type;
    content._data = data;
    content._storageElemType = storageElemType;
    content._isSplat = isSplat;
    return content;
}

//
// Content::allocTempBuffer
//

Const::Content vpux::Const::Content::allocTempBuffer(vpux::NDTypeInterface type, mlir::Type storageElemType,
                                                     bool isSplat) {
    Const::Content content;
    content._type = type;
    content._storageElemType = storageElemType;
    content._isSplat = isSplat;

    const size_t tempBufSize = isSplat ? 1 : checked_cast<size_t>(type.getNumElements());
    const Byte tempElemSize = vpux::getElemTypeSize(storageElemType);
    const size_t tempBufRawSize = tempBufSize * checked_cast<size_t>(tempElemSize.count());

    content._tempBuf.reset(new char[tempBufRawSize]);
    content._data = makeArrayRef(content._tempBuf.get(), tempBufRawSize);

    return content;
}

Const::Content vpux::Const::Content::allocTempBuffer(vpux::NDTypeInterface type, mlir::Type storageElemType,
                                                     bool isSplat, size_t tempBufRawSize) {
    // Overloading for sub-byte cases.
    Const::Content content;
    content._type = type;
    content._storageElemType = storageElemType;
    content._isSplat = isSplat;

    content._tempBuf.reset(new char[tempBufRawSize]);
    content._data = makeArrayRef(content._tempBuf.get(), tempBufRawSize);

    return content;
}

//
// Content::moveBuffer
//

Const::Content vpux::Const::Content::moveBuffer(vpux::NDTypeInterface type, Const::Content&& other) {
    Const::Content content;
    content._type = type;
    content._storageElemType = other._storageElemType;
    content._isSplat = other._isSplat;

    content._data = other._data;
    other._data = None;

    if (other._tempBuf != nullptr) {
        content._tempBuf = std::move(other._tempBuf);
    }

    return content;
}

//
// Content::copyTo
//

namespace {

template <class Range>
void fillBuf(const Range& range, MutableArrayRef<char> buf) {
    using value_type = typename Range::iterator::value_type;
    static const auto VALUE_BYTE_SIZE = sizeof(value_type);

    VPUX_THROW_UNLESS(buf.size() == range.size() * VALUE_BYTE_SIZE,
                      "Buffer with byte size '{0}' is not enough to hold actual elements with '{1}' byte size",
                      buf.size(), range.size() * VALUE_BYTE_SIZE);

    loop_1d(LoopExecPolicy::Parallel, range.size(), [&](size_t i) {
        auto* bufPtr = reinterpret_cast<value_type*>(buf.data() + i * VALUE_BYTE_SIZE);
        *bufPtr = range[i];
    });
}

}  // namespace

void vpux::Const::Content::copyTo(MutableArrayRef<char> buf) const {
    const Bit elemSize = vpux::getElemTypeSize(getElementType());
    const bool isTrivialStorage = (getElementType() == _storageElemType);
    const bool isSubByte = elemSize.count() < CHAR_BIT;
    // Dispatch is required when:
    // 1. The buffer is splat and expressed type doesn't match stored type (non-trivial).
    // 2. The buffer is splat and elements are not packed (fillBuf doesn't work after bitPack).
    // Otherwise, plain copy will do the trick.
    if (!_isSplat && (isTrivialStorage || isSubByte)) {
        VPUX_THROW_UNLESS(buf.size() == _data.size(),
                          "Byte sizes of the input buffer '{0}' and stored elements '{1}' are different.", buf.size(),
                          _data.size());
        std::memcpy(buf.data(), _data.data(), buf.size());
    } else {
        dispatchByElemType<void>(getElementType(), [this, buf](auto dummy) {
            using ElemT = std::decay_t<decltype(dummy)>;
            fillBuf(this->getValues<ElemT>(), buf);
        });
    }
}
