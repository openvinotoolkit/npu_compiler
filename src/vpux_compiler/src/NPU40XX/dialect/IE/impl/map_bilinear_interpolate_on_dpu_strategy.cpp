//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/IE/impl/map_bilinear_interpolate_on_dpu_strategy.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/map_bilinear_interpolate_on_DPU.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::IE::arch40xx {

void MapBilinearInterpolateOnDPUStrategy::prepareInterpolate(mlir::ConversionTarget& target, LogCb logCb) const {
    target.addDynamicallyLegalOp<IE::InterpolateOp>([this, logCb](IE::InterpolateOp op) {
        auto inputElemType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType()).getElementType();
        auto outputElemType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType()).getElementType();
        if (inputElemType != outputElemType && !mlir::isa<mlir::quant::QuantizedType>(inputElemType) &&
            !mlir::isa<mlir::quant::QuantizedType>(outputElemType)) {
            // There has been IE.Convert fused into this IE.InterpolateOp.
            return false;
        }
        const auto inputShape = getShape(op.getInput());
        const auto outputShape = getShape(op.getOutput());

        const auto attr = op.getAttr();
        const auto coordModeAttr = attr.getCoordMode();
        bool isAlignCorners = coordModeAttr.getValue() == IE::InterpolateCoordMode::ALIGN_CORNERS;

        const auto axesValue = parseIntArrayAttr<int64_t>(op.getAxesAttrAttr());
        const bool isIntegerRatioOnly = std::all_of(axesValue.begin(), axesValue.end(), [&](const auto& axis) {
            auto outputDim = outputShape[Dim(axis)];
            auto inputDim = inputShape[Dim(axis)];

            if (isAlignCorners && !isDoubleEqual(axis, 1.0f)) {
                outputDim = outputDim == 1 ? 1 : (outputDim - 1);
                inputDim = inputDim == 1 ? 1 : (inputDim - 1);
            }

            return (outputDim % inputDim == 0) || (inputDim % outputDim == 0);
        });
        // SW kernel performance is bigger than DPU decomposition performance for floating scale factors.
        if (!isIntegerRatioOnly) {
            return true;
        }
        return isLegalInterpolateOp(op, _interpolateAsSEOpInStrategy, logCb);
    });
}
}  // namespace vpux::IE::arch40xx
