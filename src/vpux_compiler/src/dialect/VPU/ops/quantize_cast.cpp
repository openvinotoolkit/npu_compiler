//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::QuantizeCastOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::QuantizeCastOpAdaptor quantizeCast(operands, attrs);
    if (mlir::failed(quantizeCast.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = quantizeCast.input().getType().cast<vpux::NDTypeInterface>();
    const auto dstElemType = quantizeCast.dstElemType();
    unsigned int outputWidth;
    if (auto quantizedOutput = dstElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        outputWidth = quantizedOutput.getStorageTypeIntegralWidth();
    } else if (auto quantizedOutput = dstElemType.dyn_cast<mlir::IntegerType>()) {
        outputWidth = quantizedOutput.getWidth();
    } else {
        return errorAt(loc, "Unsupported output type: {0}", dstElemType);
    }

    if (auto integerInput = inType.getElementType().dyn_cast<mlir::IntegerType>()) {
        const auto inputWidth = integerInput.getWidth();
        if (inputWidth != outputWidth) {
            return errorAt(loc, "Integer input width ({0}) differs from output width ({1})", inputWidth, outputWidth);
        }
    } else if (auto quantizedInput = inType.getElementType().dyn_cast<mlir::quant::QuantizedType>()) {
        const auto inputWidth = quantizedInput.getStorageTypeIntegralWidth();
        if (inputWidth != outputWidth) {
            return errorAt(loc, "Quantized input width ({0}) differs from output width ({1})", inputWidth, outputWidth);
        }
    } else {
        return errorAt(loc, "Unsupported combination of input and output element types: {0} -> {1}",
                       inType.getElementType(), dstElemType);
    }

    const auto outType = inType.changeElemType(dstElemType);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::QuantizeCastOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::CopyParamsBuilder builder(writer);
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_CopyParams});
}
