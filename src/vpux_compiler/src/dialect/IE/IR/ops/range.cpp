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
#include "vpux/compiler/utils/range_bound.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::RangeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::RangeOpAdaptor range(operands, attrs, prop);
    if (mlir::failed(range.verify(loc))) {
        return mlir::failure();
    }

    auto inType = mlir::cast<NDTypeInterface>(range.getStart().getType());

    const auto outShape = Shape{mlir::ShapedType::kDynamic};
    const auto bounds = SmallVector<int64_t>{RANGEBOUND};

    const auto typeComponents = TypeComponents()
                                        .setShape(outShape)
                                        .setDimsOrder(DimsOrder::fromNumDims(outShape.size()))
                                        .setElementType(range.getDstElemType())
                                        .setBounds(getIntArrayAttr(ctx, bounds));

    auto outType = inType.changeTypeComponents(typeComponents);

    const auto encoding = mlir::cast<mlir::RankedTensorType>(outType).getEncoding();

    inferredReturnShapes.emplace_back(outShape.raw(), range.getDstElemType(), encoding);

    return mlir::success();
}
