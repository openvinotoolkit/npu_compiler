//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

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

    const auto inType = nonZero.getInput().getType().cast<NDTypeInterface>();
    const auto inRank = inType.getRank();
    const auto outShape = Shape{inRank, mlir::ShapedType::kDynamic};

    const auto inShape = inType.getShape();
    const auto numElements = std::accumulate(inShape.begin(), inShape.end(), ShapeRef::ValueType{1},
                                             std::multiplies<ShapeRef::ValueType>());
    const auto bounds = SmallVector<int64_t>{inRank, numElements};

    const auto typeComponents = TypeComponents()
                                        .setShape(outShape)
                                        .setDimsOrder(DimsOrder::fromNumDims(outShape.size()))
                                        .setElementType(mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed))
                                        .setBounds(getIntArrayAttr(ctx, bounds));

    auto outType = inType.changeTypeComponents(typeComponents);

    inferredReturnTypes.emplace_back(outType);

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
