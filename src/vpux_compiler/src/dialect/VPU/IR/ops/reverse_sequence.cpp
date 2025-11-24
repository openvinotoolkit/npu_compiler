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
    const auto dataDims = checked_cast<int64_t>(dataShape.size());
    const auto batchAxis = rev.getBatchAxis();

    if (batchAxis >= dataDims || batchAxis < -dataDims) {
        return errorAt(loc, "ReverseSequence Parameter batch axis {0} out of the tensor rank range [{1}, {2}].",
                       batchAxis, -dataDims, dataDims - 1);
    }

    const auto seqAxis = rev.getSeqAxis();

    if (seqAxis >= dataDims || seqAxis < -dataDims) {
        return errorAt(loc, "ReverseSequence Parameter sequence axis {0} out of the tensor rank range [{1}, {2}].",
                       seqAxis, -dataDims, dataDims - 1);
    }

    const auto batchAxisNorm = (batchAxis < 0) ? (batchAxis + dataDims) : batchAxis;
    const auto seqCheckAxis = (seqShape.size() == 1) ? Dim(0) : Dim(batchAxisNorm);
    if (static_cast<size_t>(seqCheckAxis.ind()) >= seqShape.size()) {
        return errorAt(loc, "Sequence-lengths axis check {0} exceeds sequence-lengths rank {1}", seqCheckAxis.ind(),
                       seqShape.size());
    }
    if (seqShape[seqCheckAxis] != dataShape[batchAxisNorm]) {
        return errorAt(loc, "Sequence lengths input size {0} is not equal to batch axis dimension of data input {1}",
                       seqShape[seqCheckAxis], dataShape[batchAxisNorm]);
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
