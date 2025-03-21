//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/BuiltinTypes.h>

#include <numeric>

using namespace vpux;

mlir::LogicalResult vpux::IE::NonZeroOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::NonZeroOpAdaptor nonZero(operands, attrs, prop);
    if (mlir::failed(nonZero.verify(loc))) {
        return mlir::failure();
    }

    auto inType = mlir::cast<NDTypeInterface>(nonZero.getInput().getType());
    const auto inRank = inType.getRank();
    const auto outShape = Shape{inRank, mlir::ShapedType::kDynamic};

    const auto numElements = details::calcTotalShapeSize(inType.getShape());
    const auto bounds = SmallVector<int64_t>{inRank, numElements};

    const auto typeComponents = TypeComponents()
                                        .setShape(outShape)
                                        .setDimsOrder(DimsOrder::fromNumDims(outShape.size()))
                                        .setElementType(nonZero.getDstElemType())
                                        .setBounds(getIntArrayAttr(ctx, bounds));

    auto outType = inType.changeTypeComponents(typeComponents);

    const auto encoding = mlir::cast<mlir::RankedTensorType>(outType).getEncoding();

    inferredReturnShapes.emplace_back(outShape.raw(), nonZero.getDstElemType(), encoding);

    return mlir::success();
}
