//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux {
namespace IE {

mlir::LogicalResult canConvertTransposedConvToConv(IE::TransposedConvolutionOp transposedConv);
mlir::LogicalResult canConvertGroupTransposedConvToGroupConv(IE::GroupTransposedConvolutionOp groupTransposedConv);

// For 2D GroupTransposedConvolution:
// input tensor layout is [N, C_IN * GROUPS, H, W]
// kernel tensor layout is [GROUPS, C_OUT, C_IN, kH, kW]
const int64_t GROUP_TRANSPOSED_CONV_GROUPS_DIM_INDEX = 0;
const int64_t GROUP_TRANSPOSED_CONV_C_OUT_DIM_INDEX = 1;
const int64_t GROUP_TRANSPOSED_CONV_C_IN_DIM_INDEX = 2;
const int64_t GROUP_TRANSPOSED_CONV_KY_DIM_INDEX = 3;
const int64_t GROUP_TRANSPOSED_CONV_KX_DIM_INDEX = 4;

template <class ConcreteOp>
mlir::FailureOr<mlir::Value> createUpsampling(mlir::PatternRewriter& rewriter, mlir::Location loc, ConcreteOp origOp,
                                              vpux::Shape& padsOutput, bool isGroupTransposedConv) {
    auto ctx = rewriter.getContext();

    const auto padsBeginVector = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()));
    const auto padsEndVector = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsEnd()));
    const auto stridesVector = Shape(parseIntArrayAttr<int64_t>(origOp.getStrides()));

    int64_t padL, padR, padT, padB;

    if (isGroupTransposedConv) {
        auto origFilterShape =
                to_small_vector(mlir::dyn_cast<mlir::ShapedType>(origOp.getFilter().getType()).getShape());
        if (origFilterShape.size() != 5) {
            return errorAt(origOp, "Only 2D GroupTransposedConvolution is supported, expected 5D filter but got {0}",
                           origFilterShape.size());
        }

        padL = origFilterShape[GROUP_TRANSPOSED_CONV_KX_DIM_INDEX] - 1 - padsBeginVector[Dims4D::PadsBegin::Left];
        padR = origFilterShape[GROUP_TRANSPOSED_CONV_KX_DIM_INDEX] - 1 - padsEndVector[Dims4D::PadsEnd::Right];
        padT = origFilterShape[GROUP_TRANSPOSED_CONV_KY_DIM_INDEX] - 1 - padsBeginVector[Dims4D::PadsBegin::Top];
        padB = origFilterShape[GROUP_TRANSPOSED_CONV_KY_DIM_INDEX] - 1 - padsEndVector[Dims4D::PadsEnd::Bottom];
    } else {
        auto filterShape = getShape(origOp.getFilter()).toValues();
        if (filterShape.size() != 4) {
            return errorAt(origOp, "Only 2D TransposedConvolution is supported, expected 4D filter but got {0}",
                           filterShape.size());
        }

        // Update the pad based on whether the outputShape is specified. For example:
        // Input: 1x16x128x128xf16    Weights: 32x16x2x2xf16    OutputShape: 2xsi32 = dense<128>
        //                   \                |               /
        //                   TransposedConv: 1x32x128x128xf16
        //                   (strides = [2, 2], spatial_output_padding = [0, 0], pads_begin = [64, 64], pads_end = [64,
        //                   64])
        // The padL/padR/padT/padB should be non-negative integer, here will be set to 0 instead of -63.
        // Then the Upsampling output shape will be 1x32x255x255xf16 with strides = [2, 2].
        // So the sliceOp wiil be added after NCEConv (1x32x254x254xf16) for crop to 1x32x128x128xf16.
        padL = filterShape[Dims4D::Filter::KX] - 1 - padsBeginVector[Dims4D::PadsBegin::Left];
        padR = filterShape[Dims4D::Filter::KX] - 1 - padsEndVector[Dims4D::PadsEnd::Right];
        padT = filterShape[Dims4D::Filter::KY] - 1 - padsBeginVector[Dims4D::PadsBegin::Top];
        padB = filterShape[Dims4D::Filter::KY] - 1 - padsEndVector[Dims4D::PadsEnd::Bottom];
        if (origOp.getOutputShape() != nullptr) {
            padL = std::max<int64_t>(padL, 0);
            padR = std::max<int64_t>(padR, 0);
            padT = std::max<int64_t>(padT, 0);
            padB = std::max<int64_t>(padB, 0);
        }
    }

    // Output padding refers to padding 0 for convolutional input data.
    if (padsOutput[Dims4D::PadsOutput::Y] > 0) {
        padB += padsOutput[Dims4D::PadsOutput::Y];
        padsOutput[Dims4D::PadsOutput::Y] = 0;
    }
    if (padsOutput[Dims4D::PadsOutput::X] > 0) {
        padR += padsOutput[Dims4D::PadsOutput::X];
        padsOutput[Dims4D::PadsOutput::X] = 0;
    }

    if ((padL < 0) || (padR < 0) || (padT < 0) || (padB < 0)) {
        return errorAt(origOp, "Upsampling layer does not support negative paddings");
    }

    auto padChannelAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    auto padHeightAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{padT, padB});
    auto padWidthAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{padL, padR});
    auto padAttr = IE::UpsamplingPadAttr::get(ctx, padChannelAttr, padHeightAttr, padWidthAttr);

    auto upsamplingFactor = getIntArrayAttr(
            ctx, SmallVector<int64_t>{stridesVector[Dims4D::Strides::X], stridesVector[Dims4D::Strides::Y], 1});

    return rewriter
            .create<IE::UpsamplingOp>(loc, origOp.getInput(), upsamplingFactor, padAttr, origOp.getOutputPaddingAttr(),
                                      origOp.getInputPaddingAttr())
            .getOutput();
}

mlir::Value createPadding(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input, Dim axis,
                          int64_t nums, IE::FakeQuantizeOp inputFQ);

}  // namespace IE
}  // namespace vpux
