//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/m2i_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::M2IColorConvertOp::fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input,
                                              vpux::NDTypeInterface output, Byte reservedMem) {
    auto totalAvailableCMXSize =
            reservedMem.count() == 0 ? getTotalCMXSize(op).count() : getTotalCMXFragmentationAwareSize(op).count();
    // Note: for 1xPlane config, 1st input fully dictates the size
    SmallVector<Byte> buffers = {input.getTotalAllocSize(), output.getTotalAllocSize()};
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(op), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::M2IColorConvertOp::fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input,
                                              vpux::NDTypeInterface output) {
    // Note: for 1xPlane config, 1st input fully dictates the size
    return fitIntoCMX(op, input, output, Byte(0));
}

//
// isSupported
//

bool vpux::VPU::M2IColorConvertOp::isSupported(IE::YuvToRgbOp op, LogCb logCb, bool /*checkLayout*/,
                                               bool /*checkChannelAlignment*/) {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInput1().getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());

    if (!fitIntoCMX(op, inType, outType)) {
        logCb(llvm::formatv("Op doesn't fit into CMX memory"));
        return false;
    }

    if ((op.getInput2() != nullptr) || (op.getInput3() != nullptr)) {
        logCb(llvm::formatv("Convert to M2I : only single-plane supported for now, got {0}", op.getNumOperands()));
        return false;
    }

    // M2I only supports UI8 for NV12/I420. Other modes could be enabled
    if (!(inType.getElementType().isUnsignedInteger(8) &&
          ((op.getInFmt() == IE::ColorFmt::NV12) || (op.getInFmt() == IE::ColorFmt::I420)))) {
        logCb(llvm::formatv("Convert to M2I : unsupported {0} in format and type {1}", op.getInFmt(), inType));
        return false;
    }

    const auto lnStride = getM2iLineStride(inType, 2);  // NHW(2)C
    if (!VPU::isM2iLineStrideSupported(lnStride)) {
        logCb(llvm::formatv("Convert to M2I : line-stride NOT multiple of 16, got {0}", lnStride));
        return false;
    }

    return true;
}

//
// inferReturnTypes
//

mlir::LogicalResult vpux::VPU::M2IColorConvertOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    M2IColorConvertOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto inShape = inType.getShape().raw();

    // Y,UV (1 or 2 plane configs) are exected to have C = 1
    if (inShape[3] != 1) {
        return errorAt(loc, "Incorrect number of channels: expecting 1, got '{0}'", inShape[3]);
    }

    // OK for NV12/I420 -> RGB/BGR
    SmallVector<int64_t> outShape{inShape[0], inShape[1], inShape[2], 3};
    // input Height is big enough to include Chroma, so lower for RGB output
    outShape[1] = outShape[1] * 2 / 3;

    const auto outType = inType.changeShape(ShapeRef(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
