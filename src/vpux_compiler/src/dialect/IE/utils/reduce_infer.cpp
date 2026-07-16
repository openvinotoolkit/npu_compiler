//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/error.hpp"

mlir::LogicalResult vpux::IE::inferReduceReturnTypeComponents(
        mlir::Location loc, mlir::Value input, bool keepDims, SmallVector<int64_t>& axes,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes, mlir::ArrayAttr inputPadding,
        mlir::ArrayAttr outputPadding) {
    const auto inType = mlir::cast<mlir::ShapedType>(input.getType());
    const auto inRank = inType.getRank();
    auto inShape = SmallVector<int64_t>(inType.getShape());

    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inRank;
        }
    }

    bool isAllUnique = std::unique(axes.begin(), axes.end()) == axes.end();
    if (!isAllUnique) {
        return errorAt(loc, "Axes values should be unique");
    }

    if (mlir::failed(IE::unpadInputShape(inShape, inputPadding, loc))) {
        return errorAt(loc, "Input padding {0} incompatible with input shape {1}", inputPadding, inShape);
    }

    // Add to outShape the values with indices not found in axes_set.
    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inShape.size(); i++) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
            outShape.push_back(inShape[i]);
        } else if (keepDims) {
            outShape.push_back(1);
        }
    }

    if (outShape.size() == 0) {
        outShape.push_back(1);
    }

    if (mlir::failed(IE::padOutputShape(outShape, outputPadding, loc))) {
        return errorAt(loc, "Output padding {0} incompatible with output shape {1}", outputPadding, outShape);
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

// This function calculate outputDimOrder in case some axes are reduced.
// Example:
// NCHW 1234, remove N 1 -> 234 -> 123 (CHW)
//            remove H 2 -> 134 -> 123 (CHW)
//            remove W 3 -> 124 -> 123 (CHW)
//            remove W 4 -> 123 -> 123 (CHW)
//
// NHWC 1342, remove N 1 -> 342 -> 231 (HWC)
//            remove H 3 -> 142 -> 132 (CWH)
//            remove W 4 -> 132 -> 132 (CWH)
//            remove C 2 -> 134 -> 123 (CHW)
vpux::DimsOrder vpux::IE::calculateReducedOutputLayout(const vpux::DimsOrder& inputDimOrder,
                                                       const SmallVector<int64_t>& axes) {
    auto inputCodeOrder = inputDimOrder.code();
    vpux::DimsOrder::StorageType outputCodeOrder = 0;
    uint64_t multiply = 0;
    while (inputCodeOrder) {
        auto it = std::find(axes.begin(), axes.end(), inputCodeOrder % (1 << vpux::DimsOrder::BITS_PER_DIM));
        if (it == axes.end()) {
            int64_t numberToInsert = inputCodeOrder % (1 << vpux::DimsOrder::BITS_PER_DIM);
            auto numSmallerReducedAxes = 0;
            for (auto axis : axes) {
                if (axis < numberToInsert) {
                    numSmallerReducedAxes++;
                }
            }
            numberToInsert -= numSmallerReducedAxes;
            outputCodeOrder +=
                    static_cast<uint64_t>(numberToInsert) * (1ULL << (vpux::DimsOrder::BITS_PER_DIM * multiply));
            multiply++;
        }
        inputCodeOrder = inputCodeOrder / (1 << vpux::DimsOrder::BITS_PER_DIM);
    }

    // If axes contains all dimensions of input data, the output tensor has a single dimension
    outputCodeOrder = outputCodeOrder != 0 ? outputCodeOrder : 0x1;

    return vpux::DimsOrder::fromCode(outputCodeOrder);
}

bool vpux::IE::isChannelAxisReductionWithMatchingLayout(vpux::NDTypeInterface parentInputType,
                                                        vpux::NDTypeInterface parentOutputType, ArrayRef<int64_t> axes,
                                                        Logger log) {
    if (axes.size() != 1) {
        log.trace("Expected single reduction axis, got {0}", axes.size());
        return false;
    }

    const auto rank = checked_cast<int64_t>(parentOutputType.getShape().size());
    int64_t axis = axes.front();
    if (axis < 0) {
        axis += rank;
    }

    // DPU-supported ops with rank-5 output always use DimsGroups5D semantics (group conv, NCEMatMul).
    // No DPU-supported op produces rank-5 output in the 3D-spatial (Dims5D) axis system.
    int64_t channelAxis = (rank == 5) ? DimsGroups5D::Act::C.ind() : Dims4D::Act::C.ind();

    if (axis != channelAxis) {
        log.trace("Axis for reduction {0} is not channel axis {1}", axis, channelAxis);
        return false;
    }

    if (parentInputType.getDimsOrder() != parentOutputType.getDimsOrder()) {
        log.trace("Parent op input and output have different layouts, permutation active");
        return false;
    }

    // TODO: E#218334 remove this check once the support for tiling on channel axis is added to the reduce fusion path.
    // Fusing requires all channels to fit in a single DPU tile; tiling on the
    // channel axis is not supported for the reduce fusion path.
    const auto numChannels = parentOutputType.getShape()[Dim(channelAxis)];
    if (numChannels > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        log.trace("Number of channels {0} exceeds maximum {1} for channel-axis reduction fusion", numChannels,
                  VPU::NCEInvariant::VPU_DIMENSION_LIMIT);
        return false;
    }

    return true;
}

bool vpux::IE::isChannelAxisReductionWithDPUParent(mlir::Operation* op, ArrayRef<int64_t> axes, Logger log) {
    if (!mlir::isa<IE::ReduceMinOp, IE::ReduceMaxOp>(op) || axes.size() != 1) {
        return false;
    }

    auto parentOp = op->getOperand(0).getDefiningOp();
    if (parentOp == nullptr || mlir::failed(VPU::NCEInvariant::isSupported(parentOp, log))) {
        log.trace("Parent op is not DPU-supported");
        return false;
    }

    const auto parentInputType = mlir::cast<vpux::NDTypeInterface>(parentOp->getOperand(0).getType());
    const auto parentOutputType = mlir::cast<vpux::NDTypeInterface>(parentOp->getResult(0).getType());

    return isChannelAxisReductionWithMatchingLayout(parentInputType, parentOutputType, axes, log);
}
