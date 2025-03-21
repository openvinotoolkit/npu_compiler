//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ConvertOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ConvertOpAdaptor cvt(operands, attrs, prop);
    if (mlir::failed(cvt.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = cvt.getInput().getType().cast<mlir::RankedTensorType>();
    const auto dstElemType = cvt.getDstElemType();

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType, inType.getEncoding());
    return mlir::success();
}

bool vpux::IE::ConvertOp::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return false;
    }

    const auto input = inputs.front().cast<vpux::NDTypeInterface>();
    const auto output = outputs.front().cast<vpux::NDTypeInterface>();

    return input.getShape() == output.getShape();
}

namespace {

#include <vpux/compiler/dialect/IE/convert.hpp.inc>

}  // namespace

void vpux::IE::ConvertOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext*) {
    populateWithGenerated(patterns);
}

mlir::OpFoldResult vpux::IE::ConvertOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    VPUX_THROW_UNLESS(operands.size() == 1, "Wrong number of operands : {0}", operands.size());

    if (auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return static_cast<Const::ContentAttr>(attr).transform().castElemType(getDstElemType()).get();
    }

    return nullptr;
}

//
// verify
//

mlir::LogicalResult vpux::IE::ConvertOp::verify() {
    const auto inTy = getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outTy = getOutput().getType().cast<vpux::NDTypeInterface>();

    if (inTy.getShape().isDynamic()) {
        const auto boundedInTy = inTy.cast<vpux::BoundedTypeInterface>();
        if (boundedInTy.getBounds() == nullptr) {
            return errorAt(*this, "Missed bounds for input with dynamic dims");
        }
    }
    if (outTy.getShape().isDynamic()) {
        const auto boundedOutTy = outTy.cast<vpux::BoundedTypeInterface>();
        if (boundedOutTy.getBounds() == nullptr) {
            return errorAt(*this, "Missed bounds for output with dynamic dims");
        }
    }

    return mlir::success();
}
