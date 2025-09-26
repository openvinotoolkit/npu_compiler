//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
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
    if (getScale().isSplat()) {
        auto foldedFloatContent = getScale().fold();
        auto splatValue = foldedFloatContent.getSplatValue<double>();
        auto floatAttr = mlir::FloatAttr::get(mlir::FloatType::getF64(getContext()), splatValue);
        printer.printAttribute(floatAttr);
    } else {
        printer << "Content";
        printer << "<";
        vpux::Const::printContentAttr(printer, getScale());
        printer << ">";
    }
    printer << ">";
}

//
// RescaleAttr::parse
//

mlir::Attribute vpux::Const::RescaleAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }
    Const::ContentAttr contentAttr;
    mlir::FloatAttr floatAttr;

    if (parser.parseOptionalAttribute(floatAttr).has_value()) {
    } else if (mlir::succeeded(parser.parseOptionalKeyword("Content"))) {
        if (mlir::failed(parser.parseLess())) {
            return nullptr;
        }
        if (mlir::failed(Const::parseContentAttr(parser, contentAttr))) {
            return nullptr;
        }
        if (mlir::failed(parser.parseGreater())) {
            return nullptr;
        }
    } else {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }
    return (floatAttr != nullptr) ? Const::RescaleAttr::get(floatAttr) : Const::RescaleAttr::get(contentAttr);
}

//
// RescaleAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::RescaleAttr::inferOutputType(vpux::NDTypeInterface input) const {
    return input;
}

bool vpux::Const::RescaleAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

//
// RescaleAttr::transform
//

Const::Content vpux::Const::RescaleAttr::transform(vpux::Const::Content& input) const {
    auto output =
            Const::Content::allocTempBuffer(inferOutputType(input.getType()), mlir::Float32Type::get(getContext()),
                                            inferOutputSplat(input.isSplat(), input.getType()));
    auto scaledValues = output.getTempBuf<float>();
    auto scaleContent = getScale().fold();
    VPUX_THROW_WHEN(input.isSplat() && !scaleContent.isSplat(), "scalar * tensor is not supported");
    if (scaleContent.isSplat()) {
        if (std::isnan(scaleContent.getSplatValue<double>())) {  // Check for NaN scale value - When scale is NaN, all
                                                                 // output values must be NaN to prevent undefined
                                                                 // behavior during type casting.
            for (size_t i = 0; i < scaledValues.size(); ++i) {
                scaledValues[i] = std::numeric_limits<float>::quiet_NaN();
            }
            return output;
        }
        float scale = scaleContent.getSplatValue<float>();
        input.read([&](auto inputValues) {
            for (size_t i = 0; i < scaledValues.size(); ++i) {
                scaledValues[i] = checked_cast<float>(inputValues[i]) * scale;
            }
        });
    } else {
        scaleContent.read([&](auto scaleValues) {
            input.read([&](auto inputValues) {
                VPUX_THROW_UNLESS(scaleValues.size() == inputValues.size(), "Vector size ({0}) doesn't match ({1})",
                                  scaleValues.size(), inputValues.size());
                for (size_t i = 0; i < scaledValues.size(); ++i) {
                    float scale = scaleValues[i];
                    if (std::isnan(scale)) {  // Check for NaN in individual scale elements
                        scaledValues[i] = std::numeric_limits<float>::quiet_NaN();
                    } else {
                        scaledValues[i] = checked_cast<float>(inputValues[i]) * scale;
                    }
                }
            });
        });
    }
    return output;
}

//
// RescaleAttr::getStableHashValue
//

llvm::hash_code vpux::Const::RescaleAttr::getStableHashValue() const {
    VPUX_THROW_UNLESS(getScale().isSplat(), "RescaleAttr scale must be splat");
    const auto scale = getScale().fold().getSplatValue<double>();
    return llvm::hash_combine(getMnemonic(), llvm::APFloat(scale));
}
