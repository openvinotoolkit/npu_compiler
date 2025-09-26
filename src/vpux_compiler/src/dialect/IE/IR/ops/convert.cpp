//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
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

    const auto inType = mlir::cast<mlir::RankedTensorType>(cvt.getInput().getType());
    const auto dstElemType = cvt.getDstElemType();

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType, inType.getEncoding());
    return mlir::success();
}

bool vpux::IE::ConvertOp::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return false;
    }

    const auto input = mlir::cast<vpux::NDTypeInterface>(inputs.front());
    const auto output = mlir::cast<vpux::NDTypeInterface>(outputs.front());

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
    VPUX_THROW_UNLESS(operands.size() == 1, "Expected exactly one operand, but got {0}", operands.size());

    if (auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        return attr.transform().castElemType(getDstElemType()).get();
    }

    return nullptr;
}

//
// verify
//

mlir::LogicalResult vpux::IE::ConvertOp::verify() {
    const auto inTy = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto outTy = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());

    if (inTy.getShape().isDynamic()) {
        if (!mlir::isa<Core::BoundedTensorType>(inTy)) {
            return errorAt(*this, "Missed bounds for input with dynamic dims");
        }
    }
    if (outTy.getShape().isDynamic()) {
        if (!mlir::isa<Core::BoundedTensorType>(outTy)) {
            return errorAt(*this, "Missed bounds for output with dynamic dims");
        }
    }

    return mlir::success();
}

//
// ShaveCodeGenSupportedOpInterface
//

bool vpux::IE::ConvertOp::shouldJITCompile() {
    auto inType = getInput().getType().getElementType();
    auto outType = getOutput().getType().getElementType();

    if ((inType.getIntOrFloatBitWidth() == outType.getIntOrFloatBitWidth()) && inType.isIntOrIndex() &&
        outType.isIntOrIndex()) {
        return false;
    }
    if (inType.isBF16() || outType.isBF16()) {
        return false;
    }
    return true;
}
