//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::VPU::LoopSelectOp::verify() {
    if (!getDoConcatAttr()) {
        return errorAt(*this, "Attribute do_concat is required by LoopSelect op.");
    }

    if (getDoConcat()) {
        return errorAt(*this, "Concat cases is not supported for now.");
    }

    if (getDoConcat() && !getAxisAttr()) {
        return errorAt(*this, "Attribute axis is required by LoopSelect op concat cases.");
    }

    if (getDoConcat() && !getStrideAttr()) {
        return errorAt(*this, "Attribute stride is required by LoopSelect op concat cases.");
    }
    return mlir::success();
}

mlir::LogicalResult vpux::VPU::LoopSelectOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    VPU::LoopSelectOpAdaptor loopSelect(operands, attrs, prop);
    if (mlir::failed(loopSelect.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<mlir::RankedTensorType>(loopSelect.getInput().getType());
    const auto inputShape = inputType.getShape();
    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.push_back(inputShape[i]);
    }

    const auto execCondsType = mlir::cast<mlir::ShapedType>(loopSelect.getExecConds().getType());
    const auto numIterations = execCondsType.getShape()[0];
    outShape[0] = outShape[0] / numIterations;

    const auto outType =
            mlir::RankedTensorType::get(outShape, inputType.getElementType(), vpux::getTensorAttr(inputType));
    inferredReturnTypes.push_back(outType);
    return mlir::success();
}
