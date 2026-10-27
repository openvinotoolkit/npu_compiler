//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/error.hpp"

mlir::LogicalResult vpux::IE::inferReduceReturnTypeComponents(
        mlir::Location loc, mlir::Value input, bool keepDims, SmallVector<int64_t>& axes,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes, mlir::ArrayAttr inputPadding,
        mlir::ArrayAttr outputPadding) {
    const auto inType = mlir::cast<mlir::ShapedType>(input.getType());
    auto inShape = SmallVector<int64_t>(inType.getShape());

    // expect axes to be normalized - easily done during import
    assert(std::all_of(axes.begin(), axes.end(),
                       [&](int64_t axis) {
                           return axis >= 0 && axis < static_cast<int64_t>(inShape.size());
                       }) &&
           "Axes must already be non-negative");
    assert(std::is_sorted(axes.begin(), axes.end()) && "Axes must already be sorted");

    bool isAllUnique = std::unique(axes.begin(), axes.end()) == axes.end();
    if (!isAllUnique) {
        return errorAt(loc, "Axes values should be unique");
    }

    if (mlir::failed(IE::unpadInputShape(inShape, inputPadding, loc))) {
        return errorAt(loc, "Input padding {0} incompatible with input shape {1}", inputPadding, inShape);
    }

    // Add to the result the values with indices not found in axes_set.
    const auto reduceDims = [&](ArrayRef<int64_t> shape) {
        SmallVector<int64_t> result;
        for (size_t i = 0; i < shape.size(); i++) {
            if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                result.push_back(shape[i]);
            } else if (keepDims) {
                result.push_back(1);
            }
        }

        if (result.size() == 0) {
            result.push_back(1);
        }

        return result;
    };

    SmallVector<int64_t> outShape = reduceDims(inShape);

    if (mlir::failed(IE::padOutputShape(outShape, outputPadding, loc))) {
        return errorAt(loc, "Output padding {0} incompatible with output shape {1}", outputPadding, outShape);
    }

    // Attach bounds only when the output shape is still dynamic. Test the output shape and not the
    // input type: reducing tensor<1x16x?x?xf16> with bounds [1, 16, 48, 400] over axes 2 and 3 gives
    // the static tensor<1x16x1x1xf16>, which needs no bounds. Reducing over axis 1 instead leaves
    // height and width dynamic, so the bounds are carried over.
    //
    // Nothing behind this check would catch a mistake. Core::BoundedTensorType::get rejects a static
    // shape, but this function never calls get: it builds the encoding with getTensorAttr below and
    // lets MLIR assemble a plain RankedTensorType. Core::BoundedTensorType::classof only asks whether
    // the bounds vector is non-empty, so isa<Core::BoundedTensorType> keeps answering true for a
    // static shape that carries bounds, and no verifier compares the bounds against the shape. Every
    // pass that branches on that isa then takes the dynamic path for a value whose shape is a
    // compile-time constant, and sizes its work from the bounds. No error is reported anywhere.
    //
    // A fully static Reduce gets no encoding at all, because getTensorAttr returns null for an
    // identity order with no bounds and no memory space.
    //
    // VPU::inferReduceReturnTypes in type_infer.cpp applies the same rule after lowering.
    const auto bounds = [&]() -> mlir::FailureOr<Bounds> {
        if (!vpux::details::isDynamicDimValues(outShape)) {
            return Bounds();
        }

        const auto boundedInType = mlir::dyn_cast<Core::BoundedTensorType>(inType);
        if (boundedInType == nullptr) {
            return Bounds();
        }

        auto inBounds = SmallVector<int64_t>(boundedInType.getBounds().raw());
        if (mlir::failed(IE::unpadInputShape(inBounds, inputPadding, loc))) {
            return errorAt(loc, "Input padding {0} incompatible with input bounds {1}", inputPadding, inBounds);
        }

        auto outBounds = reduceDims(inBounds);
        if (mlir::failed(IE::padOutputShape(outBounds, outputPadding, loc))) {
            return errorAt(loc, "Output padding {0} incompatible with output bounds {1}", outputPadding, outBounds);
        }

        return Bounds(ArrayRef<int64_t>(outBounds));
    }();

    if (mlir::failed(bounds)) {
        return mlir::failure();
    }

    // Keep the layout the input came in with, instead of the identity order this function used to
    // write out. With keep_dims the rank does not move, so the order carries over unchanged. Without
    // keep_dims the reduced axes leave the shape, and calculateReducedOutputLayout gives the order
    // for the dimensions that are left. VPU::inferReduceReturnTypes and VPU::verifyReduceLayoutInfo
    // apply the same two rules, so the order no longer moves when a Reduce lowers to VPU.
    //
    // A rank-0 input is the one shape where neither order fits the result, because reducing over no
    // axis still gives a rank-1 result through the empty-result fallback above.
    const auto inOrder = mlir::cast<vpux::NDTypeInterface>(input.getType()).getDimsOrder();
    const auto reducedOrder = keepDims ? inOrder : IE::calculateReducedOutputLayout(inOrder, axes);
    const auto outOrder =
            reducedOrder.numDims() == outShape.size() ? reducedOrder : DimsOrder::fromNumDims(outShape.size());

    const auto outDesc = vpux::getTensorAttr(inType.getContext(), outOrder, /*memSpace=*/nullptr, *bounds);

    inferredReturnShapes.emplace_back(outShape, inType.getElementType(), outDesc);

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

mlir::FailureOr<mlir::SmallVector<int64_t>> vpux::IE::getReduceAxes(mlir::Location loc, mlir::Value axesInput,
                                                                    size_t inputRank) {
    auto axesValue = IE::constInputToData(loc, axesInput);
    if (mlir::failed(axesValue)) {
        return mlir::failure();
    }

    auto& axes = *axesValue;
    for (auto& axis : axes) {
        if (axis < 0) {
            axis += static_cast<int64_t>(inputRank);
        }
    }
    std::sort(axes.begin(), axes.end());
    return axesValue;
}
