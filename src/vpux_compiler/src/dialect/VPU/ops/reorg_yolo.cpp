//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ReorgYoloOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             mlir::Optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::ReorgYoloOpAdaptor reorgYolo(operands, attrs);
    if (mlir::failed(reorgYolo.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = reorgYolo.input().getType().cast<vpux::NDTypeInterface>();

    if (reorgYolo.stride() <= 0) {
        return errorAt(loc, "Stride should be a natural number");
    }
    if (inType.getShape().raw()[2] % reorgYolo.stride() != 0) {
        return errorAt(loc, "Input H should be divisible by stride.");
    }
    if (inType.getShape().raw()[3] % reorgYolo.stride() != 0) {
        return errorAt(loc, "Input W should be divisible by stride.");
    }
    if (inType.getShape().raw()[1] < reorgYolo.stride() * reorgYolo.stride()) {
        return errorAt(loc, "Input C >= (stride*stride) is required.");
    }

    SmallVector<int64_t> outputShape{inType.getShape().raw()[0], inType.getShape().raw()[1]};
    for (size_t i = 2; i < inType.getShape().size(); i++) {
        outputShape.push_back(inType.getShape().raw()[i] / reorgYolo.stride());
        outputShape[1] *= reorgYolo.stride();
    }

    const auto outType = inType.changeShape(Shape(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ReorgYoloOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("Not implemented in low level dialects.");
}
