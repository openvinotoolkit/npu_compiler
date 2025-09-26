//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ReverseSequenceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ReverseSequenceOpAdaptor rev(operands, attrs, prop);
    if (mlir::failed(rev.verify(loc))) {
        return mlir::failure();
    }

    const auto dataType = mlir::cast<vpux::NDTypeInterface>(rev.getData().getType());
    const auto dataShape = dataType.getShape().raw();

    if (dataShape.size() < 2) {
        return errorAt(loc, "First input tensor's size should not be less than 2D. Got {0}D tensor", dataShape.size());
    }

    const auto seqShape = getShape(rev.getSeqLength());

    if (seqShape.size() != 1) {
        return errorAt(loc, "Second input tensor should be 1D Tensor. Got {0}D tensor", seqShape.size());
    }

    const auto dataDims = checked_cast<int64_t>(dataShape.size());

    const auto batch_axis = rev.getBatchAxis();

    if (batch_axis >= dataDims || batch_axis < -dataDims) {
        return errorAt(loc, "ReverseSequence Parameter batch axis {0} out of the tensor rank range [{1}, {2}].",
                       batch_axis, -dataDims, dataDims - 1);
    }

    const auto seq_axis = rev.getSeqAxis();

    if (seq_axis >= dataDims || seq_axis < -dataDims) {
        return errorAt(loc, "ReverseSequence Parameter sequence axis {0} out of the tensor rank range [{1}, {2}].",
                       seq_axis, -dataDims, dataDims - 1);
    }

    if (seqShape[Dims4D::Act::N] != dataShape[batch_axis]) {
        return errorAt(loc, "Sequence lengths input size {0} is not equal to batch axis dimension of data input {1}",
                       seqShape[Dims4D::Act::N], dataShape[batch_axis]);
    }

    const auto elementType = dataType.getElementType();
    if (!(elementType.isF16() || elementType.isF32() || elementType.isInteger(8))) {
        return errorAt(loc, "Reverse Sequence only support FP16, FP32, INT8 (I8/U8/SI8) data type");
    }

    auto outType = dataType.changeElemType(elementType);
    outType = outType.changeShape(ShapeRef(dataShape));

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
