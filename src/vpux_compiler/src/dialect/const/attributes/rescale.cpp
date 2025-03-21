//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// RescaleAttr::print
//

void vpux::Const::RescaleAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getScale());
    printer << ">";
}

//
// RescaleAttr::parse
//

mlir::Attribute vpux::Const::RescaleAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::FloatAttr scale;
    if (mlir::failed(parser.parseAttribute(scale))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::RescaleAttr::get(scale);
}

//
// RescaleAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::RescaleAttr::inferOutputType(vpux::NDTypeInterface input) const {
    return input;
}

bool vpux::Const::RescaleAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) {
    return inputIsSplat;
}

//
// RescaleAttr::transform
//

Const::Content vpux::Const::RescaleAttr::transform(vpux::Const::Content& input) const {
    auto output =
            Const::Content::allocTempBuffer(inferOutputType(input.getType()), mlir::Float32Type::get(getContext()),
                                            inferOutputSplat(input.isSplat(), input.getType()));

    const auto values = input.getValues<float>();
    auto scaledVals = output.getTempBuf<float>();

    const auto scale = static_cast<float>(getScale().getValue().convertToDouble());

    for (size_t i = 0; i < scaledVals.size(); ++i) {
        scaledVals[i] = values[i] * scale;
    }

    return output;
}

//
// RescaleAttr::getStableHashValue
//

llvm::hash_code vpux::Const::RescaleAttr::getStableHashValue() const {
    const auto scale = getScale().getValue();
    return llvm::hash_combine(getMnemonic(), scale);
}
