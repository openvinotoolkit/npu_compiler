//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/DialectImplementation.h>
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <numeric>

using namespace vpux;

//
// BroadcastAttr::print
//

void vpux::Const::BroadcastAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getAxis());
    printer << ", ";
    printer.printAttribute(getValue());
    printer << ">";
}

//
// PadWithZeroAttr::parse
//

mlir::Attribute vpux::Const::BroadcastAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::IntegerAttr axis;
    if (mlir::failed(parser.parseAttribute(axis))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::IntegerAttr value;
    if (mlir::failed(parser.parseAttribute(value))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::BroadcastAttr::get(axis, value);
}

//
// BroadcastAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::BroadcastAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto value = getValue().getInt();
    const auto axis = Dim(getAxis().getInt());

    const auto inShape = input.getShape();

    VPUX_THROW_UNLESS(value >= inShape[axis],
                      "Value cannot be broadcasted due to new value's size is less than old one: {0} < {1}", value,
                      inShape[axis]);

    const Shape padBefore(inShape.size(), 0);

    Shape padAfter(inShape.size(), 0);
    padAfter[axis] = value - inShape[axis];

    return input.pad(padBefore, padAfter);
}

bool vpux::Const::BroadcastAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

//
// BroadcastAttr::transform
//

Const::Content vpux::Const::BroadcastAttr::transform(vpux::Const::Content& input) const {
    auto output = Const::Content::allocTempBuffer(inferOutputType(input.getType()), input.getStorageElemType(),
                                                  inferOutputSplat(input.isSplat(), input.getType()));

    const auto inBuf = input.getRawStorageBuf();
    auto outBuf = output.getRawTempBuf();
    if (input.isSplat()) {
        std::copy_n(inBuf.data(), inBuf.size(), outBuf.data());
    } else {
        const auto value = getValue().getInt();
        auto cstType = input.getType();
        auto physicalShape = cstType.getMemShape().raw();
        auto index = cstType.getDimsOrder().dimPos(Dim(getAxis().getInt()));
        const auto preDims = std::accumulate(physicalShape.begin(), physicalShape.begin() + index, (int64_t)1,
                                             std::multiplies<int64_t>());
        const auto expandDim = physicalShape[index];
        const auto afterDims = std::accumulate(physicalShape.begin() + index, physicalShape.end(), (int64_t)1,
                                               std::multiplies<int64_t>());
        VPUX_THROW_WHEN(value % expandDim, "Can't broadcast from {0} to {1}", expandDim, value);
        const auto factor = value / expandDim;
        const auto elemSize = vpux::getElemTypeSize(input.getStorageElemType()).to<Byte>().count();
        const auto singleCopySize = afterDims * elemSize;

        loop_2d(LoopExecPolicy::Parallel, input.getStorageElemType().getContext(), preDims, factor,
                [&](int64_t preIndex, int64_t factorIndex) {
                    auto inOffset = preIndex * afterDims * elemSize;
                    auto outBase = singleCopySize * preIndex * factor;
                    std::copy_n(inBuf.data() + inOffset, singleCopySize,
                                outBuf.data() + outBase + (singleCopySize * factorIndex));
                });
    }

    return output;
}

//
// BroadcastAttr::getStableHashValue
//

llvm::hash_code vpux::Const::BroadcastAttr::getStableHashValue() const {
    const auto axis = getAxis().getValue();
    const auto value = getValue().getValue();
    return llvm::hash_combine(getMnemonic(), axis, value);
}
