//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Value.h>
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::MultiplyOp::verifyShapeInfo() {
    if (mlir::failed(vpux::IE::verifyInputIs4D(getInput1()))) {
        return mlir::failure();
    }

    return vpux::IE::verifyInputIs4D(getInput2());
}

mlir::LogicalResult vpux::IE::MultiplyOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MultiplyOpAdaptor multiply(operands, attrs, prop);
    if (mlir::failed(multiply.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<vpux::NDTypeInterface>(multiply.getInput1().getType());
    const auto in2Type = mlir::cast<vpux::NDTypeInterface>(multiply.getInput2().getType());

    auto outShapeInfo = inferEltwiseOutputShapeInfo(ShapeInfo::fromNDType(in1Type), ShapeInfo::fromNDType(in2Type),
                                                    multiply.getAutoBroadcast(), loc);

    const auto outDesc = vpux::getTensorAttr(ctx, inferOrder(in1Type, in2Type), /*memSpace=*/nullptr,
                                             BoundsRef(outShapeInfo.bounds));
    inferredReturnShapes.emplace_back(outShapeInfo.shape, in1Type.getElementType(), outDesc);

    return mlir::success();
}

mlir::LogicalResult vpux::IE::MultiplyOp::verify() {
    if (getScale()) {
        return errorAt(*this, "scales operand is not yet supported; "
                              "implement scale tensor input support before enabling this path");
    }
    return mlir::success();
}

mlir::OpFoldResult vpux::IE::MultiplyOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    VPUX_THROW_UNLESS(operands.size() == 2 || operands.size() == 3, "Wrong number of operands : {0}", operands.size());

    const auto lhsInputShape = getShape(getInput1());
    const auto rhsInputShape = getShape(getInput2());
    const auto outputShape = getShape(getOutput());

    const bool shapeChanges = (lhsInputShape != outputShape);
    if (shapeChanges) {
        return nullptr;
    }

    auto rhsAttr = mlir::dyn_cast_or_null<vpux::Const::ContentAttr>(operands[1]);
    if (rhsAttr && rhsAttr.isSplat()) {
        const auto folded = rhsAttr.fold();
        const double splatVal = folded.getSplatValue<double>();
        if (isDoubleEqual(splatVal, 1.0)) {
            return getInput1();
        }
        return nullptr;
    }

    auto lhsAttr = mlir::dyn_cast_or_null<vpux::Const::ContentAttr>(operands[0]);
    // Do not fold constants with different shapes
    const bool equalShapes = (lhsInputShape == rhsInputShape);
    if (!equalShapes) {
        return nullptr;
    }

    if (lhsAttr && rhsAttr) {
        return lhsAttr.transform().rescale(rhsAttr).get();
    }
    return nullptr;
}

mlir::LogicalResult vpux::IE::MultiplyOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                            mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    auto loc = getLoc();

    auto outShape = reifyEltwiseTensors(builder, getInput1(), getInput2(), getAutoBroadcast(), loc);

    if (mlir::failed(outShape)) {
        return outShape;
    }

    reifiedReturnShapes.emplace_back(std::move(outShape.value()));
    return mlir::success();
}

bool vpux::IE::MultiplyOp::requiresStaticShape() {
    // MultiplyOp might require static shapes in some cases. Currently we handle only one case when MultiplyOp is
    // involved into BoundaryCorrection logic
    // return true in case one of the operand is a DynamicDataMaskOp or all operands have static shapes or operand will
    // require static shape, to avoid shape mismatch during static shape propagation across dynamic subgraphs.
    auto allParentsAreStatic = llvm::all_of(getOperands(), [](mlir::Value operand) {
        return mlir::dyn_cast<vpux::NDTypeInterface>(operand.getType()).getShape().isStatic();
    });
    auto hasDynamicAndWillRequireStaticShape = llvm::any_of(getOperands(), [](mlir::Value operand) {
        auto isDynamic = mlir::dyn_cast<vpux::NDTypeInterface>(operand.getType()).getShape().isDynamic();
        auto requiresStaticShape = false;
        if (auto staticShapeOpInterface = mlir::dyn_cast_or_null<StaticShapeOpInterface>(operand.getDefiningOp())) {
            requiresStaticShape = staticShapeOpInterface.requiresStaticShape();
        }
        return isDynamic && requiresStaticShape;
    });
    auto hasDynamicDataMaskOp = llvm::any_of(getOperands(), [](mlir::Value operand) {
        auto defOp = operand.getDefiningOp();
        return mlir::isa_and_nonnull<IE::DynamicDataMaskOp>(defOp);
    });
    return allParentsAreStatic || hasDynamicAndWillRequireStaticShape || hasDynamicDataMaskOp;
}
