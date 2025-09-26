//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace vpux {
namespace VPU {

mlir::LogicalResult inferReduceReturnTypes(mlir::Location loc, mlir::Value input, bool keepDims,
                                           SmallVector<int64_t>& axes,
                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes,
                                           mlir::ArrayAttr inputPadding, mlir::ArrayAttr outputPadding) {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    auto inShape = SmallVector<int64_t>(inType.getShape().raw());

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

    // If axes contains all dimensions of input data, a single reduction value is calculated for the entire input tensor
    if (outShape.empty()) {
        outShape = {1};
    }

    if (mlir::failed(IE::padOutputShape(outShape, outputPadding, loc))) {
        return errorAt(loc, "Output padding {0} incompatible with output shape {1}", outputPadding, outShape);
    }

    const auto newOutputType =
            TypeComponents()
                    .setDimsOrder(keepDims ? inType.getDimsOrder()
                                           : vpux::IE::calculateReducedOutputLayout(inType.getDimsOrder(), axes))
                    .setShape(ShapeRef(outShape));
    vpux::DimsOrder outOrder = newOutputType.dimsOrder.value();
    const auto tensorAttr = vpux::getTensorAttr(inType.getContext(), outOrder.toAffineMap(input.getType().getContext()),
                                                inType.getMemSpace(), getBounds(input.getType()));
    auto outputType = mlir::RankedTensorType::get(outShape, inType.getElementType(), tensorAttr);

    inferredReturnTypes.push_back(outputType);

    return mlir::success();
}

void inferPermuteReturnTypes(mlir::Value input, mlir::AffineMap memPerm, mlir::AffineMap dstOrder,
                             SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromAffineMap(dstOrder);
    auto inType = mlir::cast<vpux::NDTypeInterface>(input.getType());

    const auto outShape = callOnShapeOf(inType, [&](const auto& shape) {
        const auto inMemShape = inOrder.toMemoryOrder(shape);
        const auto outMemShape = applyPerm(inMemShape, memPerm);
        const auto outShape = outOrder.toLogicalOrder(outMemShape);

        auto [outStaticShape, outBounds, outDimMask] = splitShapeAndRepresentation(outShape);
        if (!outBounds.empty()) {
            inType = mlir::cast<Core::BoundedTensorType>(inType).changeBounds(outBounds);
        } else if (!outDimMask.empty()) {
            inType = mlir::cast<Core::DynamicDimsMaskTensorType>(inType).changeDynamicDimsMask(outDimMask);
        }

        return std::move(outStaticShape);
    });

    auto getOutputType = [&]() {
        auto elemType = inType.getElementType();
        if (auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
            const auto origAxis = perAxisType.getQuantizedDimension();
            const auto inMemAxis = inOrder.dimPos(Dim(origAxis));
            const auto outMemAxis = DimsOrder::fromAffineMap(memPerm).dimPos(Dim(inMemAxis));
            const auto outAxis = outOrder.dimAt(outMemAxis);
            elemType = changeAxis(perAxisType, outAxis.ind());
        }

        if (auto distributedInput = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(inType)) {
            auto outDistribution = applyPermutationOnDistributionInfoAttr(
                    distributedInput, memPerm, inType.getDimsOrder(), outOrder, getShape(input), outShape);

            VPUX_THROW_WHEN(
                    mlir::failed(outDistribution),
                    "Cannot infer output distribution for Permute Op, intype = {0}, memPerm = {1}, dstOrder = {2}",
                    inType, memPerm, dstOrder);

            const auto dstDimsOrderAttr = mlir::AffineMapAttr::get(dstOrder);
            return mlir::cast<vpux::NDTypeInterface>(
                    DistributedTensorType::get(inType.getContext(), outShape.raw(), elemType, dstDimsOrderAttr,
                                               inType.getMemSpace(), outDistribution.value()));
        }

        return inType.changeDimsOrder(outOrder).changeShapeElemType(outShape, elemType);
    };

    auto outType = getOutputType();
    inferredReturnTypes.push_back(outType);
}

vpux::TensorAttr createTensorAttrFromType(vpux::NDTypeInterface inType) {
    auto ctx = inType.getContext();
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(inType)) {
        return vpux::getTensorAttr(ctx, inType.getDimsOrder().toAffineMap(ctx), inType.getMemSpace(),
                                   boundedType.getBounds());
    }
    return vpux::getTensorAttr(ctx, inType.getDimsOrder().toAffineMap(ctx), inType.getMemSpace());
}

}  // namespace VPU
}  // namespace vpux
