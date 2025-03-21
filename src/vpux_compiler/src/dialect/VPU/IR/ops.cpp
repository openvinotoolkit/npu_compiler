//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"

using namespace vpux;

bool VPU::arePerClusterDistributionMemoryShapeAndOffsetsEqual(vpux::NDTypeInterface srcType,
                                                              VPU::DistributionInfo& sourceDistribution,
                                                              vpux::NDTypeInterface targetType,
                                                              VPU::DistributionInfo& targetDistribution) {
    // Ensure the memory view for the source and target distributions are the same,
    // no matter the attributes of the distribution.
    // For example, given:
    // sourceAttr = SEGMENTED across 2 clusters without uniformDistributedSegments
    // targetAttr = SEGMENTED across 2 clusters with uniformDistributedSegments
    // memory view will always be the same, so the distribution attrs are compatible.

    SmallVector<Shape> sourceMemoryOffsets{};
    SmallVector<Shape> targetMemoryOffsets{};
    SmallVector<Shape> sourceMemoryShapes{};
    SmallVector<Shape> targetMemoryShapes{};
    if (sourceDistribution.getMemoryShapes().size() == 0) {
        auto optionalMemoryShapes = VPU::getPerClusterMemoryShapes(srcType.getShape(), sourceDistribution);
        if (optionalMemoryShapes.has_value()) {
            sourceMemoryShapes = optionalMemoryShapes.value();
        }
    } else {
        for (auto& shape : sourceDistribution.getMemoryShapes()) {
            sourceMemoryShapes.push_back(Shape(shape));
        }
    }

    if (targetDistribution.getMemoryShapes().size() == 0) {
        auto optionalMemoryShapes = VPU::getPerClusterMemoryShapes(targetType.getShape(), targetDistribution);
        if (optionalMemoryShapes.has_value()) {
            targetMemoryShapes = optionalMemoryShapes.value();
        }
    } else {
        for (auto& shape : targetDistribution.getMemoryShapes()) {
            targetMemoryShapes.push_back(Shape(shape));
        }
    }

    if (sourceDistribution.getMemoryOffsets().size() == 0) {
        sourceMemoryOffsets = VPU::getPerClusterMemoryShapeOffsets(srcType.getShape(), sourceDistribution);
    } else {
        for (auto& shape : sourceDistribution.getMemoryOffsets()) {
            sourceMemoryOffsets.push_back(Shape(shape));
        }
    }

    if (targetDistribution.getMemoryOffsets().size() == 0) {
        targetMemoryOffsets = VPU::getPerClusterMemoryShapeOffsets(targetType.getShape(), targetDistribution);
    } else {
        for (auto& shape : targetDistribution.getMemoryOffsets()) {
            targetMemoryOffsets.push_back(Shape(shape));
        }
    }

    return (sourceMemoryOffsets == targetMemoryOffsets) && (sourceMemoryShapes == targetMemoryShapes);
}

mlir::LogicalResult VPU::areDistributionsCompatible(vpux::NDTypeInterface srcType, VPU::DistributionInfo& sourceAttr,
                                                    vpux::NDTypeInterface targetType, VPU::DistributionInfo& targetAttr,
                                                    const bool allowDifferentPerClusterMemoryView) {
    const auto inDistributionMode = sourceAttr.getDistributionMode();
    const auto outDistributionMode = targetAttr.getDistributionMode();

    if (inDistributionMode != outDistributionMode) {
        if (VPU::canTheDistributionModesBeCompatible(inDistributionMode, outDistributionMode).failed()) {
            return mlir::failure();
        }
    }

    const auto inDistributionNumClusters = sourceAttr.getNumClusters();
    const auto outDistributionNumClusters = targetAttr.getNumClusters();

    if (VPU::areDistributionNumClustersCompatible(inDistributionNumClusters, outDistributionNumClusters).failed()) {
        return mlir::failure();
    }

    if ((inDistributionMode == VPU::DistributionMode::SEGMENTED ||
         inDistributionMode == VPU::DistributionMode::OVERLAPPED) &&
        (outDistributionMode == VPU::DistributionMode::SEGMENTED ||
         outDistributionMode == VPU::DistributionMode::OVERLAPPED)) {
        const auto inDistributionNumTiles = sourceAttr.getNumTiles();
        const auto outDistributionNumTiles = targetAttr.getNumTiles();
        if (inDistributionNumTiles != outDistributionNumTiles) {
            return mlir::failure();
        }

        // When the source & target types are the types of an op's input & output, there is no generally applicable
        // way to verify the compatibility without having information about the op itself.
        // This util will indicate the types are compatible, with any extra checks having to be done at calling
        // location.
        if (allowDifferentPerClusterMemoryView) {
            return mlir::success();
        }

        // If source & target types are the type of a producer op's output and the type of a consumer op's input,
        // respectively, then as long as memory view is equal, the two distributed attributes are equivalent
        return arePerClusterDistributionMemoryShapeAndOffsetsEqual(srcType, sourceAttr, targetType, targetAttr)
                       ? mlir::success()
                       : mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult VPU::sameLayout(VPU::DistributedTensorType inDistributedType,
                                    VPU::DistributedTensorType outDistributedType, LogCb logCb) {
    if (inDistributedType.getOrder() != outDistributedType.getOrder()) {
        logCb(formatv("Mismatch between order for input ({0}) and output ({1}).", inDistributedType.getOrder(),
                      outDistributedType.getOrder()));
        return mlir::failure();
    }
    return mlir::success();
}

mlir::LogicalResult VPU::sameLayout(VPUIP::DistributedBufferType inDistributedType,
                                    VPUIP::DistributedBufferType outDistributedType, LogCb logCb) {
    auto isContinuousWithSameOrder = [&]() {
        const auto inStrideReqs = StrideReqs::compact(inDistributedType.getShape().size());
        const auto outStrideReqs = StrideReqs::compact(outDistributedType.getShape().size());
        auto inRes = inStrideReqs.checkStrides(inDistributedType);
        auto outRes = outStrideReqs.checkStrides(outDistributedType);
        return inRes && outRes && inDistributedType.getDimsOrder() == outDistributedType.getDimsOrder();
    };

    // The strides will be checked when comparing the layouts. So the function will return true if the layouts are equal
    // or the buffers are compact with same dim order
    if (inDistributedType.getLayout() != outDistributedType.getLayout() && !isContinuousWithSameOrder()) {
        logCb(formatv("Mismatch between order for input ({0}) and output ({1}).", inDistributedType.getLayout(),
                      outDistributedType.getLayout()));
        return mlir::failure();
    }
    return mlir::success();
}

//
// materializeConstant
//

mlir::Operation* vpux::VPU::VPUDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                            mlir::Type type, mlir::Location loc) {
    if (!mlir::isa<Const::ContentAttr>(value)) {
        (void)errorAt(loc, "Can't materialize VPU Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::RankedTensorType>()) {
        (void)errorAt(loc, "Can't materialize VPU Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type, mlir::cast<Const::ContentAttr>(value));
}

bool VPU::isNCEWithInt4Weights(mlir::Operation* op) {
    auto nceOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(op);
    if (nceOp == nullptr) {
        return false;
    }

    auto weights = nceOp.getWeightsOperand();
    if (weights == nullptr) {
        return false;
    }

    auto weightsElemType = weights.getType().cast<NDTypeInterface>().getElementType();
    if (const auto quantizedType = weightsElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        return quantizedType.getStorageTypeIntegralWidth() == 4;
    }

    return false;
}

bool VPU::isNCEWithSEPActivation(mlir::Operation* op) {
    auto nceOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(op);
    if (nceOp == nullptr) {
        return false;
    }
    auto sparseTensorActivation = nceOp->getOperand(0).getDefiningOp<VPU::GroupSparseTensorOp>();
    if (sparseTensorActivation == nullptr) {
        return false;
    }
    return sparseTensorActivation.getStorageElementTable() != nullptr;
}

std::optional<int64_t> VPU::getWeightsChannelsAutopad(mlir::Operation* op) {
    if (VPU::canAutopadOutput(op)) {
        return vpux::VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
    }
    return std::nullopt;
}

SmallVector<SmallVector<int64_t>> arrayOfArrayFromShape(ArrayRef<Shape> shape);

bool vpux::VPU::arePerClusterMemoryShapeAndOffsetsEqual(vpux::NDTypeInterface sourceType,
                                                        const VPU::DistributionInfo& sourceDistribution,
                                                        const VPU::DistributionInfo& targetDistribution) {
    // Ensure the memory view for the source type and target explicit distribution are the same,
    // no matter the attributes of the distribution.
    // For example, given:
    // sourceAttr = SEGMENTED across 2 clusters without uniformDistributedSegments
    // targetAttr = SEGMENTED across 2 clusters with uniformDistributedSegments
    // memory view will always be the same, so the distribution attrs are compatible.

    VPUX_THROW_WHEN(targetDistribution.getMemoryShapes().empty() || targetDistribution.getMemoryOffsets().empty(),
                    "Target distribution is not explicit = {0}", targetDistribution);

    SmallVector<SmallVector<int64_t>> srcMemoryOffsets{};
    auto explicitMemoryOffsets = sourceDistribution.getMemoryOffsets();
    if (explicitMemoryOffsets.empty()) {
        srcMemoryOffsets =
                arrayOfArrayFromShape(VPU::getPerClusterMemoryShapeOffsets(sourceType.getShape(), sourceDistribution));
    } else {
        srcMemoryOffsets.append(explicitMemoryOffsets.begin(), explicitMemoryOffsets.end());
    }
    auto targetMemoryOffsets = targetDistribution.getMemoryOffsets();

    SmallVector<SmallVector<int64_t>> srcMemoryShapes{};
    auto explicitMemoryShapes = sourceDistribution.getMemoryShapes();
    if (explicitMemoryShapes.empty()) {
        auto srcMemoryShapesOpt = VPU::getPerClusterMemoryShapes(sourceType.getShape(), sourceDistribution);
        if (!srcMemoryShapesOpt.has_value()) {
            return false;
        }
        srcMemoryShapes = arrayOfArrayFromShape(srcMemoryShapesOpt.value());
    } else {
        srcMemoryShapes.append(explicitMemoryShapes.begin(), explicitMemoryShapes.end());
    }

    auto targetMemoryShapes = targetDistribution.getMemoryShapes();

    return (srcMemoryOffsets == targetMemoryOffsets) && (srcMemoryShapes == targetMemoryShapes);
}
