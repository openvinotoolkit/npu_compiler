//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <llvm/ADT/SmallVector.h>

namespace vpux::VPU {

bool isNCEConvSupported(mlir::Operation* op, NDTypeInterface inputType, NDTypeInterface filterType,
                        NDTypeInterface outputType, ArrayRef<int64_t> dilations, int64_t KY, int64_t KX, int64_t SY,
                        int64_t SX, PadInfo pads, bool checkLayout, bool checkChannelAlignment, LogCb logCb,
                        bool supportsInputActCompression = false);

bool isSupportedConv(IE::ConvolutionOp op, LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                     bool supportsInputActCompression = false);

bool isSupportedSEPTransposedConv(IE::TransposedConvolutionOp op, LogCb logCb, bool checkLayout,
                                  bool checkChannelAlignment, bool supportsInputActCompression = false);

bool isSupportedSEPTransposedConv(IE::GroupTransposedConvolutionOp op, LogCb logCb, bool checkLayout,
                                  bool checkChannelAlignment, bool supportsInputActCompression = false);

bool isSupportedSEPTransposedConv(VPU::TransposedConvolutionOp op, LogCb logCb, bool checkLayout,
                                  bool checkChannelAlignment, bool supportsInputActCompression = false);

std::optional<bool> isSEPConvCompatibleWithClusterStrategy(VPU::NCEConvolutionOp nceConv,
                                                           VPU::MultiClusterStrategy strategy);

mlir::LogicalResult verifyConvUtil(mlir::Location loc, mlir::Operation* op, ShapeRef filterShape,
                                   ShapeRef kernelStrides, PaddingAttr padAttr, ShapeRef weightsTableShape,
                                   mlir::Value output);

PadInfo shrinkPadsForDilatedConvolution(const PadInfo& pads, const ArrayRef<int64_t> dilations);

template <typename ConvTypeOp>
static bool areConvInputOutputs4d(ConvTypeOp convOp, LogCb logCb) {
    const auto operands = convOp->getOperands();
    const auto results = convOp->getResults();
    for (const auto operand : operands) {
        const auto operandType = operand.getType().template cast<NDTypeInterface>();
        if (operandType.getShape().size() != 4) {
            logCb(formatv("Only 4D inputs are supported, got {0} dimensions", operandType.getShape().size()));
            return false;
        }
    }
    for (const auto result : results) {
        const auto resultType = result.getType().template cast<NDTypeInterface>();
        if (resultType.getShape().size() != 4) {
            logCb(formatv("Only 4D outputs are supported, got {0} dimensions", resultType.getShape().size()));
            return false;
        }
    }
    return true;
}

inline bool isSupportedSEPDilatedConvPadding(PadInfo pads, ArrayRef<int64_t> dilations) {
    const auto paddingZero = pads.bottom == 0 && pads.top == 0 && pads.left == 0 && pads.right == 0;

    const auto dilationY = dilations[Dims4D::Dilation::Y.ind()];
    const auto dilationX = dilations[Dims4D::Dilation::X.ind()];

    const auto symmetricDilation = dilationY == dilationX;

    const auto paddingYEqualtoDilationY = pads.top == dilationY && pads.bottom == dilationY;
    const auto paddingXEqualtoDilationX = pads.right == dilationX && pads.left == dilationX;

    const auto paddingEqualDilation = symmetricDilation && paddingYEqualtoDilationY && paddingXEqualtoDilationX;

    return paddingZero || paddingEqualDilation;
}

template <typename GroupConvOpType>
bool isSupportedSEPDilatedConv(GroupConvOpType groupConvOp, LogCb logCb, bool checkLayout, bool checkChannelAlignment) {
    if (!areConvInputOutputs4d(groupConvOp, logCb)) {
        return false;
    }
    auto dilations = parseIntArrayAttr<int64_t>(groupConvOp.getDilations());
    const auto dilationY = dilations[Dims4D::Dilation::Y.ind()];
    const auto dilationX = dilations[Dims4D::Dilation::X.ind()];
    if (dilationY == 1 && dilationX == 1) {
        return false;
    }
    auto pads = PadInfo(groupConvOp.getPadsBegin(), groupConvOp.getPadsEnd());
    if (!isSupportedSEPDilatedConvPadding(pads, dilations)) {
        return false;
    }
    const auto filterShape = getShape(groupConvOp.getFilter());
    const auto inputType = groupConvOp.getInput().getType().template cast<NDTypeInterface>();
    const auto filterType = groupConvOp.getFilter().getType().template cast<NDTypeInterface>();
    const auto outputType = groupConvOp.getOutput().getType().template cast<NDTypeInterface>();

    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(groupConvOp.getStrides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    pads = shrinkPadsForDilatedConvolution(pads, dilations);

    // Normal isNCEConvSupported can be used to check if Op can be run on NCE,
    // when padding and dilation is adjusted
    dilations[Dims4D::Dilation::X.ind()] = 1;
    dilations[Dims4D::Dilation::Y.ind()] = 1;

    return VPU::isNCEConvSupported(groupConvOp, inputType, filterType, outputType, dilations, KY, KX, SY, SX, pads,
                                   checkLayout, checkChannelAlignment, logCb,
                                   /*supportsInputActCompression*/ false);
}

template <typename GroupConvOpType>
bool isDilatedGroupConv(GroupConvOpType groupConvOp) {
    const auto dilations = parseIntArrayAttr<int64_t>(groupConvOp.getDilations());
    const auto isDilated = dilations[Dims4D::Dilation::X.ind()] > 1 || dilations[Dims4D::Dilation::Y.ind()] > 1;

    return isDilated;
}

}  // namespace vpux::VPU
