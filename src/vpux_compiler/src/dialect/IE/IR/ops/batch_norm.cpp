//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/import/batch_norm.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::BatchNormInferenceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::BatchNormInferenceOpAdaptor norm(operands, attrs, prop);
    if (mlir::failed(norm.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(norm.getInput().getType());
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

namespace vpux::IE::BatchNormInference {
llvm::FailureOr<BatchNormAttrs> parseAttributes(mlir::MLIRContext* ctx, ArrayRef<mlir::Value> inputs) {
    if (inputs.size() != 5) {
        return mlir::failure();
    }
    auto gammaConst = inputs[1].getDefiningOp<Const::DeclareOp>();
    auto betaConst = inputs[2].getDefiningOp<Const::DeclareOp>();
    auto meanConst = inputs[3].getDefiningOp<Const::DeclareOp>();
    auto varConst = inputs[4].getDefiningOp<Const::DeclareOp>();

    // fail if not constants
    if ((gammaConst == nullptr) || (betaConst == nullptr) || (meanConst == nullptr) || (varConst == nullptr)) {
        return mlir::failure();
    }

    const auto gammaContent = gammaConst.getContent();
    const auto betaContent = betaConst.getContent();
    const auto meanContent = meanConst.getContent();
    const auto varContent = varConst.getContent();

    return std::make_tuple(
            getFPArrayAttr(ctx, gammaContent.getValues<double>()), getFPArrayAttr(ctx, betaContent.getValues<double>()),
            getFPArrayAttr(ctx, meanContent.getValues<double>()), getFPArrayAttr(ctx, varContent.getValues<double>()));
}
}  // namespace vpux::IE::BatchNormInference
