//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::SubtractOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SubtractOpAdaptor subtract(operands, attrs, prop);
    if (mlir::failed(subtract.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = subtract.getInput1().getType().cast<mlir::ShapedType>();
    const auto in2Type = subtract.getInput2().getType().cast<mlir::ShapedType>();

    const auto outShapeRes =
            IE::broadcastEltwiseShape(in1Type.getShape(), in2Type.getShape(), subtract.getAutoBroadcast(), loc);
    if (mlir::succeeded(outShapeRes)) {
        auto outShapeResVec = outShapeRes.value();
        if (subtract.getOutputChannels().has_value()) {
            outShapeResVec[Dims4D::Act::C.ind()] = subtract.getOutputChannels().value();
        }

        inferredReturnShapes.emplace_back(outShapeResVec, in1Type.getElementType());
    }

    return outShapeRes;
}

mlir::OpFoldResult vpux::IE::SubtractOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    VPUX_THROW_UNLESS(operands.size() == 2, "Wrong number of operands : {0}", operands.size());

    const bool shapeChanges = getShape(getInput1()) != getShape(getOutput());
    if (shapeChanges) {
        return nullptr;
    }

    const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[1]);
    if (attr == nullptr || !attr.isSplat()) {
        return nullptr;
    }

    const auto content = static_cast<Const::ContentAttr>(attr).fold();
    return isDoubleEqual(content.getSplatValue<double>(), 0.0f) ? getInput1() : nullptr;
}
