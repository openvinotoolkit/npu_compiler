//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/IE/impl/map_bilinear_interpolate_on_dpu_strategy.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/map_bilinear_interpolate_on_DPU.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/Support/LLVM.h>

namespace vpux::IE::arch40xx {

bool MapBilinearInterpolateOnDPUStrategy::shouldConvertInterpolateOpForMapBilinear(IE::InterpolateOp op,
                                                                                   LogCb logCb) const {
    // Runtime scales produce dynamic spatial dims, skip transformation
    if (IE::isScalesAsParameter(op.getScales(), op.getScalesAttr())) {
        logCb(llvm::formatv("InterpolateOp '{0}' has runtime scales", op->getLoc()));
        return false;
    }

    auto inputElemType = mlir::cast<NDTypeInterface>(op.getInput().getType()).getElementType();
    auto outputElemType = mlir::cast<NDTypeInterface>(op.getOutput().getType()).getElementType();
    const auto hasFusedConvert = inputElemType != outputElemType &&
                                 !mlir::isa<mlir::quant::QuantizedType>(inputElemType) &&
                                 !mlir::isa<mlir::quant::QuantizedType>(outputElemType);
    const auto isLegalOp = isLegalInterpolateOp(op, _interpolateAsSEOpInStrategy, logCb);
    if (hasFusedConvert && !isLegalOp) {
        logCb(llvm::formatv(
                "InterpolateOp '{0}' has fused convert and is not already legal, so it will be mapped to DPU",
                op->getLoc()));
        return true;
    }

    const auto inputShape = getShape(op.getInput());
    const auto outputShape = getShape(op.getOutput());

    const auto attr = op.getAttr();
    const auto coordModeAttr = attr.getCoordMode();
    const bool isAlignCorners = coordModeAttr.getValue() == IE::InterpolateCoordMode::ALIGN_CORNERS;

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
        logCb(llvm::formatv("InterpolateOp '{0}' has non-integer ratio between input and output shapes", op->getLoc()));
        return false;
    }

    if (isLegalOp) {
        logCb(llvm::formatv("InterpolateOp '{0}' is already legal, so it will not be mapped to DPU", op->getLoc()));
        return false;
    }
    logCb(llvm::formatv("InterpolateOp '{0}' will be mapped to DPU", op->getLoc()));
    return true;
}
}  // namespace vpux::IE::arch40xx
