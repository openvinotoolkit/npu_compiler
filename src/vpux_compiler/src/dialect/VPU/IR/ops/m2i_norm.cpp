//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/m2i.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/m2i_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::M2INormOp::fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output,
                                      Byte reservedMem) {
    auto totalAvailableCMXSize =
            reservedMem.count() == 0 ? getTotalCMXSize(op).count() : getTotalCMXFragmentationAwareSize(op).count();
    SmallVector<Byte> buffers = {input.getTotalAllocSize(), output.getTotalAllocSize()};
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(op), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::M2INormOp::fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output) {
    return fitIntoCMX(op, input, output, Byte(0));
}

//
// isSupported
//

bool vpux::VPU::M2INormOp::isSupported(IE::BatchNormInferenceOp op, LogCb logCb, bool /*checkLayout*/,
                                       bool /*checkChannelAlignment*/) {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());

    if (!fitIntoCMX(op, inType, outType)) {
        logCb(llvm::formatv("Op doesn't fit into CMX memory"));
        return false;
    }

    return VPU::isM2IBatchNormSupported(op.getInput(), op.getOutput(), logCb);
}
//
// inferReturnTypes
//

mlir::LogicalResult vpux::VPU::M2INormOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    M2INormOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = op.getInput().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}
