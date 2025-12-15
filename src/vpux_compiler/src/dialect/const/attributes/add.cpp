//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/loop.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// AddAttr::print
//

void vpux::Const::AddAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getBias());
    printer << ">";
}

//
// PadWithZeroAttr::parse
//

mlir::Attribute vpux::Const::AddAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::FloatAttr bias;
    if (mlir::failed(parser.parseAttribute(bias))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::AddAttr::get(bias);
}

//
// AddAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::AddAttr::inferOutputType(vpux::NDTypeInterface input) const {
    return input;
}

bool vpux::Const::AddAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

//
// AddAttr::transform
//

Const::Content vpux::Const::AddAttr::transform(vpux::Const::Content& input) const {
    auto output =
            Const::Content::allocTempBuffer(inferOutputType(input.getType()), mlir::Float32Type::get(getContext()),
                                            inferOutputSplat(input.isSplat(), input.getType()));

    auto shiftedVals = output.getTempBuf<float>();

    const auto bias = static_cast<float>(getBias().getValue().convertToDouble());
    input.read([&](auto values) {
        for (size_t i = 0; i < shiftedVals.size(); ++i) {
            shiftedVals[i] = checked_cast<float>(values[i]) + bias;
        }
    });

    return output;
}

//
// AddAttr::getStableHashValue
//

llvm::hash_code vpux::Const::AddAttr::getStableHashValue() const {
    const auto bias = getBias().getValue();
    return llvm::hash_combine(getMnemonic(), bias);
}
