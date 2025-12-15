//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/swizzle_transform.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

#include <mlir/IR/DialectImplementation.h>
#include <numeric>

using namespace vpux;
using namespace vpux::BufferTransform;

//
// vpux::BufferTransform::BufferSwizzleTransform
//
BufferSwizzleTransform::BufferSwizzleTransform(uint32_t swizzleKey, config::ArchKind archKind)
        : _addressTransform(swizzleKey, archKind) {
}

//
// vpux::BufferTransform::BufferSwizzleTransform::getSwizzlePatternStride
//

uint32_t BufferSwizzleTransform::getSwizzlePatternStride() {
    const auto log2RamCutDataWidth = _addressTransform.getLog2RamCutDataWidth();
    return (1u << (log2RamCutDataWidth + 5));
}

//
// vpux::BufferTransform::AddressTransform::setStaggerBits
//

void AddressTransform::setStaggerBits(uint32_t bits) {
    _staggerAddressBits = bits % (MAX_SWIZZLE_KEY + 1u);
    _staggerAddressMask = (1 << _staggerAddressBits) - 1;
    _shift = LOG2_RAM_CUT_BYTES - _staggerAddressBits;

    switch (_archKind) {
    case config::ArchKind::NPU50XX:  // NPU50XX - NN CMX ram cut data width = 32B
        _shift++;
        _log2RamCutDataWidth++;
        _ramCutAddressMask = (1u << (LOG2_RAM_CUT_BYTES + 1)) - 1u;
        break;
    case config::ArchKind::NPU40XX:  // NPU40XX - NN CMX ram cut data width = 32B
        _shift++;
        _log2RamCutDataWidth++;
        _ramCutAddressMask = (1u << (LOG2_RAM_CUT_BYTES + 1)) - 1u;
        break;
    case config::ArchKind::NPU37XX:
        break;
    default:
        VPUX_THROW("Unsuported ArchKind {0}", _archKind);
        break;
    }
}

//
// vpux::BufferTransform::AddressTransform::getRamCut
//

uint32_t AddressTransform::getRamCut(uint32_t addr) {
    const uint32_t cutAddr{(addr >> _log2RamCutDataWidth) & 0x1f};
    return cutAddr;
}

//
// vpux::BufferTransform::AddressTransform::getPhysicalAddress
//

uint32_t AddressTransform::getPhysicalAddress(uint32_t dpuAddr) {
    uint32_t addrStagger{dpuAddr >> _log2RamCutDataWidth};
    addrStagger &= CUT_ADDRESS_MASK_10b;
    addrStagger >>= MAX_SWIZZLE_KEY;
    addrStagger &= _staggerAddressMask;
    addrStagger <<= _shift;

    uint32_t phyAddr{dpuAddr + addrStagger};
    phyAddr &= _ramCutAddressMask;
    phyAddr = phyAddr + (dpuAddr & ~_ramCutAddressMask);
    return phyAddr;
}

//
// SwizzleConstantAttr::print
//

void vpux::Const::SwizzleConstantAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getSwizzleKey());
    printer << ", ";
    printer.printAttribute(getArch());
    printer << ">";
}

//
// SwizzleConstantAttr::parse
//

mlir::Attribute vpux::Const::SwizzleConstantAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::IntegerAttr swizzleKey;
    if (mlir::failed(parser.parseAttribute(swizzleKey))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::IntegerAttr arch;
    if (mlir::failed(parser.parseAttribute(arch))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }
    return Const::SwizzleConstantAttr::get(swizzleKey, arch);
}

//
// SwizzleConstantAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::SwizzleConstantAttr::inferOutputType(vpux::NDTypeInterface inputType) const {
    const uint32_t arch = static_cast<int32_t>(*getArch().getValue().getRawData());
    config::ArchKind archKind = static_cast<config::ArchKind>(arch);
    const auto newSize =
            alignSizeForSwizzling(inputType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind));
    // Create a flat type with aligned size based on HW requirements
    SmallVector<int64_t> newShapeVec(inputType.getShape().size(), 1);
    if (inputType.getElemTypeSize().count() == 1) {
        // For sub-byte type (i1) use same type on output
        // to align with swizzle transform
        newShapeVec[0] = newSize * CHAR_BIT;
        auto newShape = Shape(newShapeVec);
        return inputType.changeShape(newShape);
    } else if (inputType.getElementType().isF16()) {
        // For FP16 maintain same type
        newShapeVec[0] = newSize / static_cast<int64_t>(sizeof(vpux::type::float16));
        auto newShape = Shape(newShapeVec);
        return inputType.changeShape(newShape);
    } else {
        // For any other type use U8
        newShapeVec[0] = newSize;
        auto newShape = Shape(newShapeVec);
        return inputType.changeShapeElemType(newShape, getUInt8Type(inputType.getContext()));
    }
}

bool vpux::Const::SwizzleConstantAttr::inferOutputSplat(bool, vpux::NDTypeInterface) const {
    return false;
}

//
// SwizzleConstantAttr::transform
//

Const::Content vpux::Const::SwizzleConstantAttr::transform(vpux::Const::Content& input) const {
    const uint32_t swizzleKey = checked_cast<int32_t>(*getSwizzleKey().getValue().getRawData());
    const uint32_t dataWidth = checked_cast<uint32_t>(input.getType().getElemTypeSize().count());
    const uint32_t arch = static_cast<int32_t>(getArch().getValue().getSExtValue());
    config::ArchKind archKind = static_cast<config::ArchKind>(arch);
    auto outputType = inferOutputType(input.getType());

    BufferSwizzleTransform bufferSwizzleTransform{swizzleKey, archKind};
    auto inputTotalSize = checked_cast<size_t>(input.getType().getTotalAllocSize().count());
    auto newSize = checked_cast<size_t>(
            alignSizeForSwizzling(input.getType().getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));
    auto output = Const::Content::allocTempBuffer(outputType, outputType.getElementType(),
                                                  inferOutputSplat(input.isSplat(), input.getType()), newSize);

    auto swizzledBuffer = output.getRawTempBuf();

    // the copyTo will not pack the input data when the copy buffer larger than _data.size().
    // Therefore it needs to pass the TotalAllocSize of the input first, and then the copyTo function will pack the
    // sub-bytes data. After that, resize the buffer to alignSize.
    std::vector<char> inputValues(newSize);
    input.copyTo(MutableArrayRef(inputValues.data(), inputTotalSize));

    // Pad if final aligned size is larger than input size
    // If input constant was splat then pad with the same value to allow
    // having splat constant also after swizzling transformation
    if (newSize > inputTotalSize) {
        char padVal = 0;
        if (input.isSplat()) {
            padVal = inputValues[0];
        }

        std::fill(inputValues.begin() + inputTotalSize, inputValues.end(), padVal);
    }

    VPUX_THROW_WHEN(inputValues.size() != swizzledBuffer.size(), "Mismatch of buffer sizes");

    // Set storage element type to be equal to the sub-byte element type in order to have trivial storage
    // This allows functionality such as copying the buffer to be done as a simple memcpy
    if (dataWidth == 1) {
        output.setStorageElemType(input.getStorageElemType());
    }

    // If input is splat no need to performa actual swizzling transformation
    if (input.isSplat()) {
        std::memcpy(swizzledBuffer.data(), inputValues.data(), swizzledBuffer.size());
        return output;
    }

    bufferSwizzleTransform.swizzle<char>(inputValues, swizzledBuffer);

    return output;
}

//
// SwizzleConstantAttr::getPositionRequirement
//

Const::details::PositionRequirement Const::SwizzleConstantAttr::getPositionRequirement() const {
    return Const::details::PositionRequirement::LAST;
}

//
// SwizzleConstantAttr::supportsSubByteStorageType
//

bool Const::SwizzleConstantAttr::supportsSubByteStorageType() const {
    return true;
}

//
// SwizzleConstantAttr::getStableHashValue
//

llvm::hash_code vpux::Const::SwizzleConstantAttr::getStableHashValue() const {
    const auto arch = getArch().getValue();
    const auto swizzleKey = getSwizzleKey().getValue();
    return llvm::hash_combine(getMnemonic(), arch, swizzleKey);
}
