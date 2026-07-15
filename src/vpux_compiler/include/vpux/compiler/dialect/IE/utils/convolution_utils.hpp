//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux {
namespace IE {

/** Clone a Convolution operation with new inputs.
 *
 *  This function clones the given Convolution operation, replacing its inputs with the provided new inputs.
 *
 *  Please note that rest of attributes (like stride, padding, post-ops, etc.) are preserved from the original
 * operation.
 */
template <typename Builder>
IE::ConvolutionOp cloneConvolutionOp(Builder& builder, IE::ConvolutionOp convOp, mlir::Value input, mlir::Value filter,
                                     mlir::Value bias, mlir::Value scale, mlir::Value zeroPoints,
                                     std::optional<mlir::Location> loc = std::nullopt) {
    return builder.template create<IE::ConvolutionOp>(
            loc.value_or(convOp.getLoc()), input, filter, bias, scale, zeroPoints, convOp.getStridesAttr(),
            convOp.getPadsBegin(), convOp.getPadsEnd(), convOp.getDilations(), convOp.getPostOpAttr(),
            convOp.getClampAttr(), convOp.getStaticScaleAttr(), convOp.getOutputPaddingAttr(),
            convOp.getInputPaddingAttr());
}

/** Clone a Convolution operation with new inputs (without bias and scale).
 *
 *  This function clones the given Convolution operation, replacing its inputs with the provided new inputs.
 *
 *  Please note that rest of attributes (like stride, padding, post-ops, etc.) are preserved from the original
 * operation.
 */
template <typename Builder>
IE::ConvolutionOp cloneConvolutionOp(Builder& builder, IE::ConvolutionOp convOp, mlir::Value input, mlir::Value filter,
                                     std::optional<mlir::Location> loc = std::nullopt) {
    return builder.template create<IE::ConvolutionOp>(
            loc.value_or(convOp.getLoc()), input, filter, convOp.getBias(), convOp.getScale(), convOp.getZeroPoints(),
            convOp.getStridesAttr(), convOp.getPadsBegin(), convOp.getPadsEnd(), convOp.getDilations(),
            convOp.getPostOpAttr(), convOp.getClampAttr(), convOp.getStaticScaleAttr(), convOp.getOutputPaddingAttr(),
            convOp.getInputPaddingAttr());
}

/** Clone a Convolution operation with new inputs and output type.
 *
 *  This function clones the given Convolution operation, replacing its inputs with the provided new inputs
 *  and setting the output type.
 *
 *  Please note that rest of attributes (like stride, padding, post-ops, etc.) are preserved from the original
 * operation.
 */
template <typename Builder>
IE::ConvolutionOp cloneConvolutionOp(Builder& builder, IE::ConvolutionOp convOp, mlir::Type outputType,
                                     mlir::Value input, mlir::Value filter, mlir::Value bias, mlir::Value scale,
                                     mlir::Value zeroPoints, std::optional<mlir::Location> loc = std::nullopt) {
    return builder.template create<IE::ConvolutionOp>(
            loc.value_or(convOp.getLoc()), outputType, input, filter, bias, scale, zeroPoints, convOp.getStridesAttr(),
            convOp.getPadsBegin(), convOp.getPadsEnd(), convOp.getDilations(), convOp.getPostOpAttr(),
            convOp.getClampAttr(), convOp.getStaticScaleAttr(), convOp.getOutputPaddingAttr(),
            convOp.getInputPaddingAttr());
}

/** Clone a Convolution operation with new inputs(without bias and scale) and output type.
 *
 *  This function clones the given Convolution operation, replacing its inputs with the provided new inputs
 *  and setting the output type.
 *
 *  Please note that rest of attributes (like stride, padding, post-ops, etc.) are preserved from the original
 * operation.
 */
template <typename Builder>
IE::ConvolutionOp cloneConvolutionOp(Builder& builder, IE::ConvolutionOp convOp, mlir::Type outputType,
                                     mlir::Value input, mlir::Value filter,
                                     std::optional<mlir::Location> loc = std::nullopt) {
    return builder.template create<IE::ConvolutionOp>(
            loc.value_or(convOp.getLoc()), outputType, input, filter, convOp.getBias(), convOp.getScale(),
            convOp.getZeroPoints(), convOp.getStridesAttr(), convOp.getPadsBegin(), convOp.getPadsEnd(),
            convOp.getDilations(), convOp.getPostOpAttr(), convOp.getClampAttr(), convOp.getStaticScaleAttr(),
            convOp.getOutputPaddingAttr(), convOp.getInputPaddingAttr());
}

/** Clone a Convolution operation with new inputs(without bias and scale) and strides, pads dilations
 *
 *  This function clones the given Convolution operation, replacing its inputs with the provided new inputs
 *
 *  Please note that rest of attributes (like static scale, post-ops, etc.) are preserved from the original
 * operation.
 */
template <typename Builder>
IE::ConvolutionOp cloneConvolutionOp(Builder& builder, IE::ConvolutionOp convOp, mlir::Value input, mlir::Value filter,
                                     mlir::ArrayAttr strides, mlir::ArrayAttr padsBegin, mlir::ArrayAttr padsEnd,
                                     mlir::ArrayAttr dilations, std::optional<mlir::Location> loc = std::nullopt) {
    return builder.template create<IE::ConvolutionOp>(
            loc.value_or(convOp.getLoc()), input, filter, convOp.getBias(), convOp.getScale(), convOp.getZeroPoints(),
            strides, padsBegin, padsEnd, dilations, convOp.getPostOpAttr(), convOp.getClampAttr(),
            convOp.getStaticScaleAttr(), convOp.getOutputPaddingAttr(), convOp.getInputPaddingAttr());
}

/** Clone a Convolution operation with new inputs(without bias and scale) and strides, pads dilations
 *
 *  This function clones the given Convolution operation, replacing its inputs with the provided new inputs
 *  and setting the output type.
 *
 *  Please note that rest of attributes (like static scale, post-ops, etc.) are preserved from the original
 * operation.
 */
template <typename Builder>
IE::ConvolutionOp cloneConvolutionOp(Builder& builder, IE::ConvolutionOp convOp, mlir::Type outputType,
                                     mlir::Value input, mlir::Value filter, mlir::ArrayAttr strides,
                                     mlir::ArrayAttr padsBegin, mlir::ArrayAttr padsEnd, mlir::ArrayAttr dilations,
                                     std::optional<mlir::Location> loc = std::nullopt) {
    return builder.template create<IE::ConvolutionOp>(loc.value_or(convOp.getLoc()), outputType, input, filter,
                                                      convOp.getBias(), convOp.getScale(), convOp.getZeroPoints(),
                                                      strides, padsBegin, padsEnd, dilations, convOp.getPostOpAttr(),
                                                      convOp.getClampAttr(), convOp.getStaticScaleAttr(),
                                                      convOp.getOutputPaddingAttr(), convOp.getInputPaddingAttr());
}

/** Clone a Convolution operation with new inputs and strides, pads dilations
 *
 *  This function clones the given Convolution operation, replacing its inputs with the provided new inputs
 *
 *  Please note that rest of attributes (like static scale, post-ops, etc.) are preserved from the original
 * operation.
 */
template <typename Builder>
IE::ConvolutionOp cloneConvolutionOp(Builder& builder, IE::ConvolutionOp convOp, mlir::Value input, mlir::Value filter,
                                     mlir::Value bias, mlir::Value scale, mlir::Value zeroPoints,
                                     mlir::ArrayAttr strides, mlir::ArrayAttr padsBegin, mlir::ArrayAttr padsEnd,
                                     mlir::ArrayAttr dilations, std::optional<mlir::Location> loc = std::nullopt) {
    return builder.template create<IE::ConvolutionOp>(
            loc.value_or(convOp.getLoc()), input, filter, bias, scale, zeroPoints, strides, padsBegin, padsEnd,
            dilations, convOp.getPostOpAttr(), convOp.getClampAttr(), convOp.getStaticScaleAttr(),
            convOp.getOutputPaddingAttr(), convOp.getInputPaddingAttr());
}

/** Clone a Convolution operation with new inputs and strides, pads dilations
 *
 *  This function clones the given Convolution operation, replacing its inputs with the provided new inputs
 *  and setting the output type.
 *
 *  Please note that rest of attributes (like static scale, post-ops, etc.) are preserved from the original
 * operation.
 */
template <typename Builder>
IE::ConvolutionOp cloneConvolutionOp(Builder& builder, IE::ConvolutionOp convOp, mlir::Type outputType,
                                     mlir::Value input, mlir::Value filter, mlir::Value bias, mlir::Value scale,
                                     mlir::Value zeroPoints, mlir::ArrayAttr strides, mlir::ArrayAttr padsBegin,
                                     mlir::ArrayAttr padsEnd, mlir::ArrayAttr dilations,
                                     std::optional<mlir::Location> loc = std::nullopt) {
    return builder.template create<IE::ConvolutionOp>(
            loc.value_or(convOp.getLoc()), outputType, input, filter, bias, scale, zeroPoints, strides, padsBegin,
            padsEnd, dilations, convOp.getPostOpAttr(), convOp.getClampAttr(), convOp.getStaticScaleAttr(),
            convOp.getOutputPaddingAttr(), convOp.getInputPaddingAttr());
}

mlir::LogicalResult canConvertGroupConvToConv(IE::GroupConvolutionOp groupconv, bool isAttrCheckEnabled = true,
                                              bool checkHandleLargePads = false);
bool isEltwiseGroupConv(IE::GroupConvolutionOp convOp, bool isConstFilter = true);

//
// FuseConvAndBias
//

class FuseConvAndBias final : public mlir::OpRewritePattern<IE::ScaleShiftOp> {
public:
    using mlir::OpRewritePattern<IE::ScaleShiftOp>::OpRewritePattern;

    void initialize() {
        setDebugName("FuseConvAndBias");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp biasOp, mlir::PatternRewriter& rewriter) const final;
};

}  // namespace IE
}  // namespace vpux
