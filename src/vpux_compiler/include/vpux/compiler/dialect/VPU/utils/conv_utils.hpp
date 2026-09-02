//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/SmallVector.h>

namespace vpux::VPU {
enum class MultiClusterStrategy : uint64_t;
}  // namespace vpux::VPU

namespace vpux::VPU {

bool isNCEConvSupported(mlir::Operation* op, NDTypeInterface inputType, NDTypeInterface filterType,
                        NDTypeInterface outputType, ArrayRef<int64_t> dilations, int64_t KY, int64_t KX, int64_t SY,
                        int64_t SX, PadInfo pads, bool checkLayout, bool checkChannelAlignment, LogCb logCb,
                        bool supportsInputActCompression = false);

bool isSupportedConv(IE::ConvolutionOp op, LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                     bool supportsInputActCompression = false);

std::optional<bool> isSEPConvCompatibleWithClusterStrategy(VPU::NCEConvolutionOp nceConv,
                                                           VPU::MultiClusterStrategy strategy);

mlir::LogicalResult verifyConvUtil(mlir::Location loc, mlir::Operation* op, ShapeRef filterShape,
                                   ShapeRef kernelStrides, PaddingAttr padAttr,
                                   std::optional<ShapeRef> weightsTableShape, mlir::Value output);

PadInfo shrinkPadsForDilatedConvolution(const PadInfo& pads, const ArrayRef<int64_t> dilations);

bool hasReduceOutputs(VPU::NCEConvolutionOp op);

SmallVector<mlir::Value> splitNCEConvolutionOverIC(VPU::NCEConvolutionOp origOp, mlir::Value weightInput,
                                                   SmallVector<VPU::NCEConvolutionOp>& convOps,
                                                   SmallVector<VPU::NCEEltwiseOp>& addOps,
                                                   SmallVector<VPU::DequantizeOp>& dequantizeOps,
                                                   const OutputTiling& tiles, VPU::DequantizeOp weightDequantizeOp,
                                                   mlir::PatternRewriter& rewriter, Logger log);

template <typename ConvTypeOp>
static bool areConvInputOutputs4d(ConvTypeOp convOp, LogCb logCb) {
    const auto operands = convOp->getOperands();
    const auto results = convOp->getResults();
    for (const auto operand : operands) {
        const auto operandType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
        if (operandType.getShape().size() != 4) {
            logCb(formatv("Only 4D inputs are supported, got {0} dimensions", operandType.getShape().size()));
            return false;
        }
    }
    for (const auto result : results) {
        const auto resultType = mlir::cast<vpux::NDTypeInterface>(result.getType());
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
bool isDilatedGroupConv(GroupConvOpType groupConvOp) {
    const auto dilations = parseIntArrayAttr<int64_t>(groupConvOp.getDilations());
    const auto isDilated = dilations[Dims4D::Dilation::X.ind()] > 1 || dilations[Dims4D::Dilation::Y.ind()] > 1;

    return isDilated;
}

/// Resolve rawFilterShape by constant-folding dynamic dimensions.
/// For each position in staticRawFilterShape: static values are kept as-is;
/// dynamic sentinels (kDynamic) are replaced with the integer value of the
/// corresponding SSA operand from dynamicRawFilterShapeValues when it
/// constant-folds, or kept as kDynamic otherwise.
SmallVector<int64_t> resolveRawFilterShape(ArrayRef<int64_t> staticRawFilterShape,
                                           mlir::ValueRange dynamicRawFilterShapeValues);

}  // namespace vpux::VPU
