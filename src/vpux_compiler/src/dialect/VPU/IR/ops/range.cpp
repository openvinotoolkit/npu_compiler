//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/utils/dynamic_shape_propagation.hpp"
#include "vpux/compiler/utils/range_bound.hpp"

using namespace vpux;

mlir::LogicalResult VPU::RangeOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                   mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                   mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                   mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::RangeOpAdaptor range(operands, attrs, prop);
    if (mlir::failed(range.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(range.getStart().getType());

    const auto outShape = Shape{mlir::ShapedType::kDynamic};
    const auto outBounds = SmallVector<int64_t>{RANGEBOUND};

    auto typeComponents = TypeComponents()
                                  .setDimsOrder(DimsOrder::fromNumDims(outShape.size()))
                                  .setElementType(range.getDstElemType());

    assignDynamicTypeComponents(typeComponents, range.getBoundsRepresentation(), outShape.raw(), outBounds);

    auto outType = inType.changeTypeComponents(typeComponents);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::VPU::RangeOp::verify() {
    const auto shape = getShape(getStart());

    if (shape.size() > 1) {
        return errorAt(*this, "Range kernel supports only up to 1D shapes, got '{0}'", shape.size());
    }

    return mlir::success();
}
