//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <vpux/utils/core/logger.hpp>
using namespace vpux;

//
// BitPackAttr::verify
//

mlir::LogicalResult vpux::Const::BitPackAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::IntegerAttr width) {
    if (width == nullptr) {
        return printTo(emitError(), "Got NULL 'width' in 'BitPackAttr'");
    }

    if (width.getInt() > 6) {
        return printTo(emitError(), "BitPackAttr supports only 1-6 sub-byte bit widths.");
    }

    return mlir::success();
}

//
// BitPackAttr::print
//

void vpux::Const::BitPackAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getWidth());
    printer << ">";
}

//
// BitPackAttr::parse
//

mlir::Attribute vpux::Const::BitPackAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::IntegerAttr width;
    if (mlir::failed(parser.parseAttribute(width))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::BitPackAttr::get(width);
}

//
// BitPackAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::BitPackAttr::inferOutputType(vpux::NDTypeInterface input) const {
    // Check that we're not trying to pack any floating point values.
    VPUX_THROW_WHEN(input.getElementType().isa<mlir::FloatType>(), "Bit pack does not support float inputs.");
    const auto bitWidth = checked_cast<unsigned>(getWidth().getInt());
    mlir::Type outElementType;

    if (auto quantInType = mlir::dyn_cast_or_null<mlir::quant::QuantileQuantizedType>(input.getElementType())) {
        const auto minVal = quantInType.getStorageTypeMin();
        const auto maxVal = quantInType.getStorageTypeMax();
        const auto signedness = quantInType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
        const auto elementIntegerType = mlir::IntegerType::get(getContext(), bitWidth, signedness);
        outElementType = mlir::quant::QuantileQuantizedType::get(
                quantInType.getFlags(), elementIntegerType, quantInType.getQuantileType(),
                quantInType.getExpressedType(), quantInType.getQuantiles(), quantInType.getScale(),
                quantInType.getZeroPoint(), minVal, maxVal);
    } else if (auto quantInType =
                       mlir::dyn_cast_or_null<mlir::quant::QuantileQuantizedPerAxisType>(input.getElementType())) {
        const auto minVal = quantInType.getStorageTypeMin();
        const auto maxVal = quantInType.getStorageTypeMax();
        const auto signedness = quantInType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
        const auto elementIntegerType = mlir::IntegerType::get(getContext(), bitWidth, signedness);
        outElementType = mlir::quant::QuantileQuantizedPerAxisType::get(
                quantInType.getFlags(), elementIntegerType, quantInType.getQuantileType(),
                quantInType.getExpressedType(), quantInType.getQuantiles(), quantInType.getScales(),
                quantInType.getZeroPoints(), quantInType.getQuantizedDimension(), minVal, maxVal);
    } else if (auto quantInType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(input.getElementType())) {
        const auto minVal = quantInType.getStorageTypeMin();
        const auto maxVal = quantInType.getStorageTypeMax();
        const auto signedness = quantInType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
        const auto elementIntegerType = mlir::IntegerType::get(getContext(), bitWidth, signedness);
        outElementType = mlir::quant::UniformQuantizedType::get(quantInType.getFlags(), elementIntegerType,
                                                                quantInType.getExpressedType(), quantInType.getScale(),
                                                                quantInType.getZeroPoint(), minVal, maxVal);
    } else if (auto quantInType =
                       mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(input.getElementType())) {
        const auto minVal = quantInType.getStorageTypeMin();
        const auto maxVal = quantInType.getStorageTypeMax();
        const auto signedness = quantInType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
        const auto elementIntegerType = mlir::IntegerType::get(getContext(), bitWidth, signedness);
        outElementType = mlir::quant::UniformQuantizedPerAxisType::get(
                quantInType.getFlags(), elementIntegerType, quantInType.getExpressedType(), quantInType.getScales(),
                quantInType.getZeroPoints(), quantInType.getQuantizedDimension(), minVal, maxVal);
    } else if (auto intInType = mlir::dyn_cast_or_null<mlir::IntegerType>(input.getElementType())) {
        outElementType = mlir::IntegerType::get(getContext(), bitWidth, intInType.getSignedness());
    } else {
        VPUX_THROW("Got unsupported input element type '{0}' in bitpack", input.getElementType());
    }
    return input.changeElemType(outElementType);
}

bool vpux::Const::BitPackAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) {
    VPUX_THROW_WHEN(inputIsSplat, "Bit pack does not support splat inputs.");  // as per ::transform()
    return false;
}

//
// Pack sub-byte values in LSB format. (Values are continuous)
// e.g. for 3-bit width: 0b00000x2x1x0, 0b00000y2y1y0, 0b00000z2z1z0
//   packed values ->  0bz1z0y2y1y0x2x1x0, 0b0000000z2
//

uint8_t* packSubByteLSB(const vpux::Const::details::ContentRange<unsigned char> input, uint8_t* packedBuffer,
                        uint8_t bitWidth) {
    auto packedIdx = 0;
    auto bitPosition = 0;
    uint8_t currentByte = 0;
    auto size = input.size();
    for (size_t idx = 0; idx < size; idx++) {
        auto value = static_cast<uint8_t>(input[idx]);
        auto leftOverBits = 8 - bitPosition;
        if (bitWidth <= leftOverBits) {
            currentByte |= (value << bitPosition);
            bitPosition += bitWidth;
        } else {
            currentByte |= (value << bitPosition);
            packedBuffer[packedIdx] = currentByte;
            packedIdx++;

            currentByte = 0;
            bitPosition = 0;
            currentByte |= (value >> leftOverBits);
            bitPosition += bitWidth - leftOverBits;
        }
    }
    if (bitPosition > 0) {
        packedBuffer[packedIdx] = currentByte;
        packedIdx++;
    }
    return packedBuffer;
}

//
// BitPackAttr::transform
//

Const::Content vpux::Const::BitPackAttr::transform(vpux::Const::Content& input) const {
    VPUX_THROW_WHEN(input.isSplat(), "Bit pack does not support splat inputs.");
    const auto widthParam = getWidth().getInt();
    VPUX_THROW_UNLESS(widthParam < 7, "Bit pack does not support bit widths greater than 6.");
    const auto inBuf = input.getValues<uint8_t>();
    const auto outputType = inferOutputType(input.getType());
    const Byte outputByteSize = outputType.getTotalAllocSize();
    const size_t tempBufferSize = outputByteSize.count();
    auto output = Const::Content::allocTempBuffer(outputType, getUInt8Type(getContext()),
                                                  inferOutputSplat(input.isSplat(), input.getType()), tempBufferSize);

    auto outBuf = output.getRawTempBuf();
    auto outBlobPtr = reinterpret_cast<uint8_t*>(outBuf.data());
    if (widthParam == 1) {
        for (size_t idx = 0; idx < inBuf.size(); idx += 8) {
            const auto bit1 = static_cast<uint8_t>(inBuf[idx + 0] & 0x01);
            const auto bit2 = static_cast<uint8_t>(inBuf[idx + 1] & 0x01);
            const auto bit3 = static_cast<uint8_t>(inBuf[idx + 2] & 0x01);
            const auto bit4 = static_cast<uint8_t>(inBuf[idx + 3] & 0x01);
            const auto bit5 = static_cast<uint8_t>(inBuf[idx + 4] & 0x01);
            const auto bit6 = static_cast<uint8_t>(inBuf[idx + 5] & 0x01);
            const auto bit7 = static_cast<uint8_t>(inBuf[idx + 6] & 0x01);
            const auto bit8 = static_cast<uint8_t>(inBuf[idx + 7] & 0x01);
            const auto byte = static_cast<uint8_t>((bit8 << 7) + (bit7 << 6) + (bit6 << 5) + (bit5 << 4) + (bit4 << 3) +
                                                   (bit3 << 2) + (bit2 << 1) + bit1);
            outBlobPtr[idx / 8] = byte;
        }
    } else if (widthParam == 2) {
        for (size_t idx = 0; idx < inBuf.size(); idx += 4) {
            const auto bits12 = static_cast<uint8_t>(inBuf[idx + 0] & 0x03);
            const auto bits34 = static_cast<uint8_t>(inBuf[idx + 1] & 0x03);
            const auto bits56 = static_cast<uint8_t>(inBuf[idx + 2] & 0x03);
            const auto bits78 = static_cast<uint8_t>(inBuf[idx + 3] & 0x03);
            const auto byte = static_cast<uint8_t>((bits78 << 6) + (bits56 << 4) + (bits34 << 2) + bits12);
            outBlobPtr[idx / 4] = byte;
        }
    } else if (widthParam == 4) {
        for (size_t idx = 0; idx < inBuf.size(); idx += 2) {
            const auto lsn = static_cast<uint8_t>(inBuf[idx + 0] & 0x0f);
            const auto msn = static_cast<uint8_t>(inBuf[idx + 1] & 0x0f);
            const auto byte = static_cast<uint8_t>((msn << 4) + lsn);
            outBlobPtr[idx / 2] = byte;
        }
    } else {
        outBlobPtr = packSubByteLSB(inBuf, outBlobPtr, widthParam);
    }
    return output;
}

//
// BitPackAttr::getPositionRequirement
//

Const::details::PositionRequirement vpux::Const::BitPackAttr::getPositionRequirement() const {
    return Const::details::PositionRequirement::LAST;
}

//
// BitPackAttr::getStableHashValue
//

llvm::hash_code vpux::Const::BitPackAttr::getStableHashValue() const {
    const auto width = getWidth().getValue();
    return llvm::hash_combine(getMnemonic(), width);
}
