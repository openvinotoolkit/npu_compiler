//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::DetectionOutputOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::DetectionOutputOpAdaptor detectionOutput(operands, attrs);
    if (mlir::failed(detectionOutput.verify(loc))) {
        return mlir::failure();
    }

    const auto boxLogitsType = detectionOutput.in_box_logits().getType().cast<mlir::ShapedType>();

    auto origN{0}, origC{1};
    const auto num_images = boxLogitsType.getShape()[origN];
    const auto num_loc_classes =
            detectionOutput.attr().share_location().getValue() ? 1 : detectionOutput.attr().num_classes().getInt();

    if (num_loc_classes <= 0) {
        return errorAt(loc, "Number of classes should be a natural number");
    }

    if (boxLogitsType.getShape()[origC] % (num_loc_classes * 4) != 0) {
        return errorAt(loc, "C dimension should be divisible by num_loc_classes * 4");
    }

    const auto num_prior_boxes = boxLogitsType.getShape()[origC] / (num_loc_classes * 4);
    const auto keep_top_k = detectionOutput.attr().keep_top_k()[0].cast<mlir::IntegerAttr>().getInt();
    const auto top_k = detectionOutput.attr().top_k().getInt();
    const auto num_classes = detectionOutput.attr().num_classes().getInt();

    SmallVector<int64_t> output_shape{1, 1};
    if (keep_top_k > 0) {
        output_shape.push_back(num_images * keep_top_k);
    } else if (top_k > 0) {
        output_shape.push_back(num_images * top_k * num_classes);
    } else {
        output_shape.push_back(num_images * num_prior_boxes * num_classes);
    }
    output_shape.push_back(7);

    inferredReturnShapes.emplace_back(output_shape, boxLogitsType.getElementType());

    return mlir::success();
}
