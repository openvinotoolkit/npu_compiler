//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DeformableConvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DeformableConvolutionOpAdaptor defConv(operands, attrs, prop);
    if (mlir::failed(defConv.verify(loc))) {
        return mlir::failure();
    }

    const auto maskInput = defConv.getMask();
    if (maskInput == nullptr) {
        return errorAt(loc, "The case without input mask is not supported");
    }

    const auto group = defConv.getGroup();
    if (group < 0) {
        return errorAt(loc, "Attribute 'group' must have a positive value");
    }

    const auto deformableGroup = defConv.getDeformableGroup();
    if (deformableGroup < 0) {
        return errorAt(loc, "Attribute 'deformable group' must have a positive value");
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(defConv.getInput().getType());
    const auto inShape = getShape(defConv.getInput()).raw();
    const auto kernelShape = getShape(defConv.getKernel()).raw();
    const auto offsetShape = getShape(defConv.getOffset()).raw();

    SmallVector<int64_t> outputShape{inShape[0],       // number of batches
                                     kernelShape[0],   // number of kernel output channels
                                     offsetShape[2],   // spatial axes Y
                                     offsetShape[3]};  // spatial axes X

    const auto outType = inType.changeShape(ShapeRef(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
