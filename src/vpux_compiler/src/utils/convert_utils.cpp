//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/convert_utils.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

// Unpack subbyte constants.
// Consider this configuration for a bit width of 4:
// 0x89, 0x88
// Will end like:
// 0x9, 0x8, 0x8, 0x8
vpux::Const::Content vpux::subByteConversion(vpux::Const::Content& input, vpux::NDTypeInterface outputType,
                                             bool outputIsSplat, size_t bitWidth) {
    const auto elementType = outputType.getElementType();
    auto output = Const::Content::allocTempBuffer(outputType, elementType, outputIsSplat);

    const auto elemPerByte = CHAR_BIT / bitWidth;
    VPUX_THROW_UNLESS(vpux::isPowerOfTwo(elemPerByte), "Invalid number of elements per byte '{0}'", elemPerByte);

    // When bitWidth is 1, we treat it as unsigned numbers
    if (elementType.isUnsignedInteger() || elementType.isSignlessIntOrIndex() || bitWidth == 1) {
        const auto sourceData = input.getRawStorageBuf();
        auto targetData = output.getRawTempBuf();
        const size_t mask = checked_cast<uint8_t>(checked_cast<uint16_t>(std::pow(2, bitWidth)) - 1);

        if (input.isSplat()) {
            // Unpack first element
            auto firstElem = checked_cast<uint8_t>(sourceData.front() & mask);
            std::fill_n(targetData.data(), targetData.size(), firstElem);
        } else {
            const auto numBytes = sourceData.size();
            for (size_t byteIdx = 0; byteIdx < numBytes; byteIdx++) {
                for (size_t elemIdxPerByte = 0; elemIdxPerByte < elemPerByte; elemIdxPerByte++) {
                    size_t shift = elemIdxPerByte * bitWidth;
                    targetData[byteIdx * elemPerByte + elemIdxPerByte] = (sourceData[byteIdx] >> shift) & mask;
                }
            }
        }
    } else if (elementType.isSignedInteger()) {
        // For signed integer, we need maintain the sign bit when unpacking
        const auto sourceData = input.getStorageBuf<int8_t>();
        auto targetData = output.getTempBuf<int8_t>();
        const size_t rShift = CHAR_BIT - bitWidth;

        if (input.isSplat()) {
            // Unpack the last element in the first byte
            auto secondElem = sourceData.front() >> rShift;
            std::fill_n(targetData.data(), targetData.size(), secondElem);
        } else {
            const auto numBytes = sourceData.size();
            for (size_t byteIdx = 0; byteIdx < numBytes; byteIdx++) {
                for (size_t elemIdxPerByte = 0; elemIdxPerByte < elemPerByte; elemIdxPerByte++) {
                    size_t lShift = rShift - (elemIdxPerByte * bitWidth);
                    targetData[byteIdx * elemPerByte + elemIdxPerByte] = (sourceData[byteIdx] << lShift);
                    targetData[byteIdx * elemPerByte + elemIdxPerByte] =
                            targetData[byteIdx * elemPerByte + elemIdxPerByte] >> rShift;
                }
            }
        }
    } else {
        VPUX_THROW("Unsupported subByte conversion type '{0}'", elementType);
    }

    return output;
}
