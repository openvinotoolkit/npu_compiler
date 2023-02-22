//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ReverseSequenceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::ReverseSequenceOpAdaptor rev(operands, attrs);
    if (mlir::failed(rev.verify(loc))) {
        return mlir::failure();
    }

    const auto dataType = rev.data().getType().cast<vpux::NDTypeInterface>();
    const auto dataShape = dataType.getShape().raw();

    if (dataShape.size() < 2) {
        return errorAt(loc, "First input tensor's size should not be less than 2D. Got {0}D tensor", dataShape.size());
    }

    const auto seqShape = getShape(rev.seq_length());

    if (seqShape.size() != 1) {
        return errorAt(loc, "Second input tensor should be 1D Tensor. Got {0}D tensor", seqShape.size());
    }

    const auto dataDims = checked_cast<int64_t>(dataShape.size());

    const auto batch_axis = rev.batch_axis();

    if (batch_axis >= dataDims || batch_axis < -dataDims) {
        return errorAt(loc, "ReverseSequence Parameter batch axis {0} out of the tensor rank range [{1}, {2}].",
                       batch_axis, -dataDims, dataDims - 1);
    }

    const auto seq_axis = rev.seq_axis();

    if (seq_axis >= dataDims || seq_axis < -dataDims) {
        return errorAt(loc, "ReverseSequence Parameter sequence axis {0} out of the tensor rank range [{1}, {2}].",
                       seq_axis, -dataDims, dataDims - 1);
    }

    if (seqShape[Dims4D::Act::N] != dataShape[batch_axis]) {
        return errorAt(loc, "Sequence lengths input size {0} is not equal to batch axis dimension of data input {1}",
                       seqShape[Dims4D::Act::N], dataShape[batch_axis]);
    }

    const auto elementType = dataType.getElementType();
    if (!(elementType.isF16() || elementType.isF32() || elementType.isUnsignedInteger(8))) {
        return errorAt(loc, "Reverse Sequence only support FP16, FP32, U8 data type");
    }

    auto outType = dataType.changeElemType(elementType);
    outType = outType.changeShape(Shape(dataShape));

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ReverseSequenceOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::ReversesequenceParamsBuilder builder(writer);
    builder.add_seq_axis(checked_cast<int32_t>(seq_axis()));
    builder.add_batch_axis(checked_cast<int32_t>(batch_axis()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReversesequenceParams});
}
