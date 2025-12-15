//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/utils/dynamic_shape_propagation.hpp"

using namespace vpux;

mlir::LogicalResult VPU::NonZeroOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                     mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                     mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                     mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::NonZeroOpAdaptor nonZero(operands, attrs, prop);
    if (mlir::failed(nonZero.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(nonZero.getInput().getType());
    const auto inRank = inType.getRank();
    const auto outShape = Shape{inRank, mlir::ShapedType::kDynamic};

    const auto numElements = inType.getNumElements();
    const auto outBounds = SmallVector<int64_t>{inRank, numElements};

    auto typeComponents = TypeComponents()
                                  .setElementType(mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed))
                                  .setDimsOrder(DimsOrder::fromNumDims(outShape.size()));

    assignDynamicTypeComponents(typeComponents, nonZero.getBoundsRepresentation(), outShape.raw(), outBounds);

    auto outType = inType.changeTypeComponents(typeComponents);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::VPU::NonZeroOp::verify() {
    const auto shape = getShape(getInput());

    if (shape.size() > 4) {
        return errorAt(*this, "NonZero kernel supports only up to 4D shapes, got '{0}'", shape.size());
    }

    return mlir::success();
}
