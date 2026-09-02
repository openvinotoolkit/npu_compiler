//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// ViewLikeOpInterface
//

mlir::Value VPUIP::NonDistributedCastOp::getViewSource() {
    return getInput();
}

namespace {

bool isSupportedNonDistributedCastMode(VPUIP::DistributedBufferType distType) {
    // NonDistributedCast is a view between a distributed buffer and the same
    // logical plain buffer. It is only safe when all clusters describe the same
    // full memory view, e.g. DUPLICATED or SEGMENTED|MULTICASTED.
    const auto mode = distType.getDistribution().getMode().getValue();
    return VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
           mode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED);
}

bool isSupportedNonDistributedCastType(mlir::Type inputType, mlir::Type outputType, LogCb logCb = emptyLogCb) {
    auto inputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(inputType);
    auto outputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(outputType);
    if (inputDistType == nullptr) {
        logCb(formatv("NonDistributedCast input must be DistributedBufferType, got '{0}'", inputType));
        return false;
    }
    if (outputNDType == nullptr) {
        logCb(formatv("NonDistributedCast output must implement NDTypeInterface, got '{0}'", outputType));
        return false;
    }
    if (mlir::isa<VPUIP::DistributedBufferType>(outputType)) {
        logCb(formatv("NonDistributedCast output must be non-distributed, got '{0}'", outputType));
        return false;
    }

    if (!isSupportedNonDistributedCastMode(inputDistType)) {
        logCb(formatv("NonDistributedCast supports only DUPLICATED or SEGMENTED|MULTICASTED input distribution, got "
                      "'{0}' for input type '{1}'",
                      inputDistType.getDistribution(), inputDistType));
        return false;
    }
    if (inputDistType.getShape() != outputNDType.getShape()) {
        logCb(formatv("NonDistributedCast input shape '{0}' doesn't match output shape '{1}'", inputDistType.getShape(),
                      outputNDType.getShape()));
        return false;
    }
    if (inputDistType.getElementType() != outputNDType.getElementType()) {
        logCb(formatv("NonDistributedCast input element type '{0}' doesn't match output element type '{1}'",
                      inputDistType.getElementType(), outputNDType.getElementType()));
        return false;
    }
    if (inputDistType.getMemoryKind() != outputNDType.getMemoryKind()) {
        logCb(formatv("NonDistributedCast input memory kind '{0}' doesn't match output memory kind '{1}'",
                      inputDistType.getMemoryKind(), outputNDType.getMemoryKind()));
        return false;
    }
    if (inputDistType.getStrides() != outputNDType.getStrides()) {
        logCb(formatv("NonDistributedCast input strides '{0}' don't match output strides '{1}'",
                      inputDistType.getStrides(), outputNDType.getStrides()));
        return false;
    }
    if (inputDistType.getDimsOrder() != outputNDType.getDimsOrder()) {
        logCb(formatv("NonDistributedCast input dims order '{0}' doesn't match output dims order '{1}'",
                      inputDistType.getDimsOrder(), outputNDType.getDimsOrder()));
        return false;
    }

    return true;
}

}  // namespace

//
// BackInferViewTypeOpInterface
//

std::optional<mlir::Type> VPUIP::NonDistributedCastOp::inferOutputTypeFromInput(mlir::Type newInputType) {
    auto newInputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(newInputType);
    if (newInputNDType == nullptr || !mlir::isa<VPUIP::DistributedBufferType>(newInputNDType)) {
        return std::nullopt;
    }

    // Drop the distributed wrapper while preserving the rewritten input's
    // visible buffer facets. Example: DistributedBuffer<1x128x16x16xf16,
    // DUPLICATED, CMX> becomes memref<1x128x16x16xf16, CMX> with the original
    // plain output memory space.
    auto origOutputNDType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto newInputStrides = newInputNDType.getStrides();
    auto typeComps = TypeComponents()
                             .setShape(newInputNDType.getShape())
                             .setElementType(newInputNDType.getElementType())
                             .setDimsOrder(newInputNDType.getDimsOrder())
                             .setStrides(newInputStrides)
                             .setMemSpace(origOutputNDType.getMemSpace());
    auto outType = origOutputNDType.changeTypeComponents(typeComps);
    const auto outputType = mlir::cast<mlir::Type>(outType);
    if (!isSupportedNonDistributedCastType(newInputType, outputType)) {
        return std::nullopt;
    }
    return outputType;
}

std::optional<mlir::Type> VPUIP::NonDistributedCastOp::inferInputTypeFromOutput(mlir::Type desiredOutputType) {
    auto desiredOutputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(desiredOutputType);
    auto origInputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(getInput().getType());
    if (desiredOutputNDType == nullptr || origInputDistType == nullptr) {
        return std::nullopt;
    }
    const auto desiredOutputStrides = desiredOutputNDType.getStrides();
    auto typeComps = TypeComponents()
                             .setShape(desiredOutputNDType.getShape())
                             .setElementType(desiredOutputNDType.getElementType())
                             .setDimsOrder(desiredOutputNDType.getDimsOrder())
                             .setStrides(desiredOutputStrides)
                             .setMemSpace(desiredOutputNDType.getMemSpace());
    auto inType = origInputDistType.changeTypeComponentsForExplicitDistribution(typeComps,
                                                                                origInputDistType.getDistribution());
    auto inputType = mlir::cast<mlir::Type>(inType);
    if (!isSupportedNonDistributedCastType(inputType, desiredOutputType)) {
        return std::nullopt;
    }
    return inputType;
}

//
// verify
//

mlir::LogicalResult vpux::VPUIP::NonDistributedCastOp::verify() {
    const auto op = getOperation();
    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };
    return mlir::success(isSupportedNonDistributedCastType(getInput().getType(), getOutput().getType(), logCb));
}
