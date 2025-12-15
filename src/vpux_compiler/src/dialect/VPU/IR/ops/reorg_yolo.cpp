//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ReorgYoloOp::verify() {
    if (getStride() <= 0) {
        return errorAt(*this, "Stride should be a natural number, while it is {0}", getStride());
    }
    return mlir::success();
}

mlir::LogicalResult vpux::VPU::ReorgYoloOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ReorgYoloOpAdaptor reorgYolo(operands, attrs, prop);
    if (mlir::failed(reorgYolo.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(reorgYolo.getInput().getType());
    auto stride = reorgYolo.getStride();

    if (stride <= 0) {
        return errorAt(loc, "Stride should be a natural number");
    }
    if (inType.getShape().raw()[2] % stride != 0) {
        return errorAt(loc, "Input H should be divisible by stride.");
    }
    if (inType.getShape().raw()[3] % stride != 0) {
        return errorAt(loc, "Input W should be divisible by stride.");
    }
    if (inType.getShape().raw()[1] < stride * stride) {
        return errorAt(loc, "Input C >= (stride*stride) is required.");
    }

    SmallVector<int64_t> outputShape{inType.getShape().raw()[0], inType.getShape().raw()[1]};
    for (size_t i = 2; i < inType.getShape().size(); i++) {
        outputShape.push_back(inType.getShape().raw()[i] / stride);
        outputShape[1] *= stride;
    }

    const auto outType = inType.changeShape(ShapeRef(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
