//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// ViewLikeOpInterface
//

mlir::Value VPUIP::DistributedCastOp::getViewSource() {
    return getInput();
}

namespace {

bool isSupportedDistributedCastType(mlir::Type inputType, mlir::Type outputType, LogCb logCb = emptyLogCb) {
    if (auto inputSparseType = mlir::dyn_cast<VPUIP::SparseBufferType>(inputType)) {
        auto outputSparseType = mlir::dyn_cast<VPUIP::SparseBufferType>(outputType);
        if (outputSparseType == nullptr) {
            logCb(formatv("Mismatch between types for input and output. "
                          "If input is SparseBufferType then output must be of same type."));
            return false;
        }

        const auto inputData = mlir::dyn_cast<VPUIP::DistributedBufferType>(inputSparseType.getData());
        const auto outputData = mlir::dyn_cast<VPUIP::DistributedBufferType>(outputSparseType.getData());
        if (inputData == nullptr || outputData == nullptr) {
            logCb(formatv("DistributedCast requires distributed data buffers for sparse types."));
            return false;
        }

        return mlir::succeeded(VPU::isDistributedCastCompatible(inputData, outputData, logCb));
    }

    if (mlir::isa<VPUIP::SparseBufferType>(outputType)) {
        logCb(formatv("Mismatch between types for input and output. "
                      "If output is SparseBufferType then input must be of same type."));
        return false;
    }

    const auto inputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(inputType);
    const auto outputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(outputType);
    if (inputDistType == nullptr || outputDistType == nullptr) {
        logCb(formatv("Mismatch between types for input and output. DistributedCast requires distributed buffers."));
        return false;
    }

    return mlir::succeeded(VPU::isDistributedCastCompatible(inputDistType, outputDistType, logCb));
}

std::optional<VPU::DistributionInfoAttr> getDistributedCastTargetDistribution(mlir::Type type) {
    auto distributedType = mlir::dyn_cast<VPU::DistributedTypeInterface>(type);
    if (distributedType == nullptr || !distributedType.containsDistributedTypes()) {
        return std::nullopt;
    }

    auto distributedData = mlir::dyn_cast<VPUIP::DistributedBufferType>(distributedType.getDistributedTypes().front());
    if (distributedData == nullptr) {
        return std::nullopt;
    }

    auto distribution = distributedData.getDistribution();
    auto sparseType = mlir::dyn_cast<VPUIP::SparseBufferType>(type);
    if (sparseType == nullptr) {
        return distribution;
    }

    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distribution)) {
        return VPU::getExplicitDistrAttrForActualDataFromSparseType(sparseType);
    }

    const auto needsMetadataDerivedDistribution = sparseType.getIsWeights() != nullptr ||
                                                  sparseType.getSeAttr() != nullptr ||
                                                  sparseType.getStorageElementTable() != nullptr;
    return needsMetadataDerivedDistribution ? std::nullopt : std::optional<VPU::DistributionInfoAttr>(distribution);
}

std::optional<mlir::Type> changeDistributedCastType(mlir::Type baseType, VPU::DistributionInfoAttr targetDistribution) {
    auto baseNDType = mlir::dyn_cast<vpux::NDTypeInterface>(baseType);
    auto baseDistType = mlir::dyn_cast<VPU::DistributedTypeInterface>(baseType);
    if (baseNDType == nullptr || baseDistType == nullptr || !baseDistType.containsDistributedTypes()) {
        return std::nullopt;
    }

    const auto baseStrides = baseNDType.getStrides();
    auto typeComps = TypeComponents()
                             .setShape(baseNDType.getShape())
                             .setElementType(baseNDType.getElementType())
                             .setDimsOrder(baseNDType.getDimsOrder())
                             .setStrides(baseStrides)
                             .setMemSpace(baseNDType.getMemSpace());
    auto outType = baseDistType.changeTypeComponentsForExplicitDistribution(typeComps, targetDistribution);
    return mlir::cast<mlir::Type>(outType);
}

}  // namespace

//
// BackInferViewTypeOpInterface
//

std::optional<mlir::Type> VPUIP::DistributedCastOp::inferOutputTypeFromInput(mlir::Type newInputType) {
    if (!mlir::isa<vpux::NDTypeInterface>(newInputType)) {
        return std::nullopt;
    }

    auto targetDistribution = getDistributedCastTargetDistribution(getOutput().getType());
    if (!targetDistribution.has_value()) {
        return std::nullopt;
    }
    const auto outputType = changeDistributedCastType(newInputType, targetDistribution.value());
    if (!outputType.has_value()) {
        return std::nullopt;
    }
    if (!isSupportedDistributedCastType(newInputType, outputType.value())) {
        return std::nullopt;
    }
    return outputType.value();
}

std::optional<mlir::Type> VPUIP::DistributedCastOp::inferInputTypeFromOutput(mlir::Type desiredOutputType) {
    if (!mlir::isa<vpux::NDTypeInterface>(desiredOutputType)) {
        return std::nullopt;
    }

    auto targetDistribution = getDistributedCastTargetDistribution(getInput().getType());
    if (!targetDistribution.has_value()) {
        return std::nullopt;
    }
    const auto inputType = changeDistributedCastType(desiredOutputType, targetDistribution.value());
    if (!inputType.has_value()) {
        return std::nullopt;
    }
    if (!isSupportedDistributedCastType(inputType.value(), desiredOutputType)) {
        return std::nullopt;
    }
    return inputType.value();
}

//
// fold
//

mlir::OpFoldResult VPUIP::DistributedCastOp::fold(FoldAdaptor) {
    return getInput().getType() == getOutput().getType() ? getInput() : mlir::TypedValue<mlir::MemRefType>{nullptr};
}

//
// verify
//

mlir::LogicalResult vpux::VPUIP::DistributedCastOp::verify() {
    const auto op = getOperation();
    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };
    return mlir::success(isSupportedDistributedCastType(getInput().getType(), getOutput().getType(), logCb));
}
