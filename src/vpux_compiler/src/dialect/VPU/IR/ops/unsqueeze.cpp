//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/unsqueeze.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/utils/layout_utils.hpp"

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
    const auto inType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder =
            vpux::VPU::inferUnsqueezeOutputLayout(inOrder.toPermutation(), axes.value(), inType.getShape());

    return callOnShapeOf(inType, [&](const auto& inShape) {
        auto outShape = IE::unsqueezeShape(loc, inShape, *axes);
        if (mlir::failed(outShape)) {
            return mlir::failure();
        }

        const auto outType = inType.changeTypeComponents(
                TypeComponents().setShapeWithRepresentation(std::move(*outShape)).setDimsOrder(outOrder));
        inferredReturnTypes.push_back(outType);
        return mlir::success();
    });
}
