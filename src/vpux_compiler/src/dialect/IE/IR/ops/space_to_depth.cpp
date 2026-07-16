//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::SpaceToDepthOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SpaceToDepthOpAdaptor spd(operands, attrs, prop);
    if (mlir::failed(spd.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(spd.getInput().getType());

    const auto elementType = inputType.getElementType();
    if (!(elementType.isF16() || elementType.isF32() || elementType.isUnsignedInteger(8) ||
          mlir::isa<mlir::quant::QuantizedType>(elementType))) {
        return errorAt(loc, "SpaceToDepth only supports FP16, FP32, U8 or Quantized input data type");
    }

    const auto inputShape = inputType.getShape().raw();
    const auto block_size = spd.getBlockSize();

    if (inputShape.size() < 3) {
        return errorAt(loc, "Input tensor rank must be greater than 2. Got {0}D tensor", inputShape.size());
    }

    if (block_size <= 0) {
        return errorAt(loc, "Invalid block size {0}, should be greater than zero", block_size);
    }

    using namespace Dims4D::Act;

    if (inputShape[H.ind()] % block_size || inputShape[W.ind()] % block_size) {
        return errorAt(loc, "Invalid block_size {0} , height {1} and width {2} must be divisible by block_size",
                       block_size, inputShape[H.ind()], inputShape[W.ind()]);
    }

    const auto outN = inputShape[N.ind()];
    const auto outC = inputShape[C.ind()] * block_size * block_size;
    const auto outH = inputShape[H.ind()] / block_size;
    const auto outW = inputShape[W.ind()] / block_size;

    SmallVector<int64_t> outShape{outN, outC, outH, outW};

    VPUX_THROW_UNLESS(!mlir::isa<Core::BoundedTensorType>(inputType), "{0} doesn't support dynamic shapes",
                      IE::SpaceToDepthOp::getOperationName());
    const auto outDesc = vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace());
    inferredReturnShapes.emplace_back(outShape, inputType.getElementType(), outDesc);

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::SpaceToDepthOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    VPUX_THROW_UNLESS(operands.size() == 1, "Wrong number of operands : {0}", operands.size());

    if (getBlockSize() == 1) {
        return getInput();
    }

    return nullptr;
}

mlir::LogicalResult IE::SpaceToDepthOp::verify() {
    if (getBlockSize() <= 0) {
        return errorAt(*this, "Block size should be a positive integer, while it is {0}", getBlockSize());
    }
    return mlir::success();
}
