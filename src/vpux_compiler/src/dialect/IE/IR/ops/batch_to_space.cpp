//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/const/utils/attributes_utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

#include <numeric>

using namespace vpux;

mlir::LogicalResult vpux::IE::BatchToSpace::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::BatchToSpaceAdaptor bsp(operands, attrs, prop);
    if (mlir::failed(bsp.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(bsp.getInput().getType());
    const auto inputShape = inputType.getShape().raw();

    auto blockShapeVal = parseIntArrayAttr<int64_t>(bsp.getBlockShapeValue());
    auto cropsBeginVal = parseIntArrayAttr<int64_t>(bsp.getCropsBeginValue());
    auto cropsEndVal = parseIntArrayAttr<int64_t>(bsp.getCropsEndValue());

    if (inputShape.size() < 2) {
        return errorAt(loc, "Input tensor rank should be 2 or greater.");
    }

    if (inputShape.size() != blockShapeVal.size() || inputShape.size() != cropsBeginVal.size() ||
        inputShape.size() != cropsEndVal.size()) {
        return errorAt(loc,
                       "blockShape, cropsBegin, cropsEnd shape[N] should be equal to the size of Input shape. Got "
                       "blockShape [{0}], cropsBegin [{1}], cropsEnd [{2}]",
                       blockShapeVal.size(), cropsBeginVal.size(), cropsEndVal.size());
    }

    auto outShape = SmallVector<int64_t>(inputShape.size());

    outShape[0] = inputShape[0] /
                  std::accumulate(blockShapeVal.begin(), blockShapeVal.end(), int64_t(1), std::multiplies<int64_t>());

    for (size_t i = 1; i < inputShape.size(); i++) {
        outShape[i] = inputShape[i] * blockShapeVal[i] - cropsBeginVal[i] - cropsEndVal[i];
    }

    VPUX_THROW_UNLESS(!mlir::isa<Core::BoundedTensorType>(inputType), "{0} doesn't support dynamic shapes",
                      IE::BatchToSpace::getOperationName());
    const auto outDesc = vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace());
    inferredReturnShapes.emplace_back(outShape, inputType.getElementType(), outDesc);

    return mlir::success();
}
