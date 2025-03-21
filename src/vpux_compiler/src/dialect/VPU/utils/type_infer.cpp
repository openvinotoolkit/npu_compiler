//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace vpux {
namespace VPU {

mlir::LogicalResult inferReduceReturnTypes(mlir::Location loc, mlir::Value input, bool keepDims,
                                           SmallVector<int64_t>& axes,
                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto inType = input.getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();

    bool isAllUnique = std::unique(axes.begin(), axes.end()) == axes.end();
    if (!isAllUnique) {
        return errorAt(loc, "Axes values should be unique");
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

    // If axes contains all dimensions of input data, a single reduction value is calculated for the entire input tensor
    if (outShape.empty()) {
        outShape = {1};
    }

    const auto newOutputType =
            TypeComponents()
                    .setDimsOrder(keepDims ? inType.getDimsOrder()
                                           : vpux::IE::calculateReducedOutputLayout(inType.getDimsOrder(), axes))
                    .setShape(Shape(outShape));

    const auto bounds =
            mlir::isa<BoundedTypeInterface>(inType) ? mlir::cast<BoundedTypeInterface>(inType).getBounds() : nullptr;
    vpux::DimsOrder outOrder = newOutputType.dimsOrder.value();

    auto outTensorAttr =
            vpux::getTensorAttr(outOrder.toAffineMap(input.getType().getContext()), inType.getMemSpace(), bounds);

    auto outputType = mlir::RankedTensorType::get(outShape, inType.getElementType(), outTensorAttr);

    inferredReturnTypes.push_back(outputType);

    return mlir::success();
}

void inferPermuteReturnTypes(mlir::Value input, mlir::AffineMap memPerm, mlir::AffineMap dstOrder,
                             SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromAffineMap(dstOrder);
    const auto inType = input.getType().cast<vpux::NDTypeInterface>();

    const auto inShape = getShape(input);
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    const auto outMemShape = applyPerm(inMemShape, memPerm);
    const auto outShape = outOrder.toLogicalOrder(outMemShape);
    const auto outBoundsAttr = permuteBounds(input.getContext(), inType.dyn_cast_or_null<vpux::BoundedTypeInterface>(),
                                             inOrder, outOrder, memPerm);

    auto getOutputType = [&]() {
        auto elemType = inType.getElementType();
        if (auto perAxisType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
            const auto origAxis = perAxisType.getQuantizedDimension();
            const auto inMemAxis = inOrder.dimPos(Dim(origAxis));
            const auto outMemAxis = DimsOrder::fromAffineMap(memPerm).dimPos(Dim(inMemAxis));
            const auto outAxis = outOrder.dimAt(outMemAxis);
            elemType = changeAxis(perAxisType, outAxis.ind());
        }

        if (auto distributedInput = inType.dyn_cast<VPU::DistributedTensorType>()) {
            auto outDistribution = applyPermutationOnDistributionInfoAttr(
                    distributedInput, memPerm, inType.getDimsOrder(), outOrder, inShape, outShape);

            VPUX_THROW_WHEN(
                    mlir::failed(outDistribution),
                    "Cannot infer output distribution for Permute Op, intype = {0}, memPerm = {1}, dstOrder = {2}",
                    inType, memPerm, dstOrder);

            const auto dstDimsOrderAttr = mlir::AffineMapAttr::get(dstOrder);
            return DistributedTensorType::get(inType.getContext(), outShape.raw(), elemType, dstDimsOrderAttr,
                                              inType.getMemSpace(), outDistribution.value())
                    .cast<NDTypeInterface>();
        }

        return inType.changeDimsOrder(outOrder).changeShapeElemType(outShape, elemType);
    };

    auto outType = getOutputType();

    if (auto boundedOutType = outType.dyn_cast<vpux::BoundedTypeInterface>()) {
        outType = boundedOutType.changeBounds(outBoundsAttr);
    }

    inferredReturnTypes.push_back(outType);
}

vpux::TensorAttr createTensorAttrFromType(vpux::NDTypeInterface inType) {
    auto ctx = inType.getContext();
    const auto bounds =
            mlir::isa<BoundedTypeInterface>(inType) ? mlir::cast<BoundedTypeInterface>(inType).getBounds() : nullptr;
    return vpux::getTensorAttr(inType.getDimsOrder().toAffineMap(ctx), inType.getMemSpace(), bounds);
}

}  // namespace VPU
}  // namespace vpux
