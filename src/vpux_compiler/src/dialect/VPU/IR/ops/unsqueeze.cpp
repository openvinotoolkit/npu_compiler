//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/layout_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include "vpux/compiler/dialect/IE/utils/unsqueeze.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::UnsqueezeOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::UnsqueezeOpAdaptor unsqueeze(operands, attrs, prop);
    if (mlir::failed(unsqueeze.verify(loc))) {
        return mlir::failure();
    }

    const auto axes = IE::getAxes(unsqueeze, loc);
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto input = unsqueeze.getInput();
    const auto inType = input.getType().cast<NDTypeInterface>();
    const auto inShape = inType.getShape();
    const auto inOrder = DimsOrder::fromValue(input);

    const auto outShape = IE::propagateShape(loc, inShape, axes.value());
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    const auto outBounds = IE::propagateBoundsAttr(ctx, loc, input, axes.value());
    if (mlir::failed(outBounds)) {
        return mlir::failure();
    }

    const auto typeComponents =
            TypeComponents()
                    .setShape(Shape(outShape.value()))
                    .setDimsOrder(vpux::VPU::inferUnsqueezeOutputLayout(inOrder.toPermutation(), axes.value(), inShape))
                    .setBounds(outBounds.value());

    auto outType = inType.changeTypeComponents(typeComponents);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
