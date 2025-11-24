//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/utils/core/error.hpp"

#include <algorithm>

using namespace vpux;

static constexpr auto MODE_AUTO = "auto";
static constexpr auto MODE_TRUE = "true";
static constexpr auto MODE_FALSE = "false";

VPU::EnableActivationSparsityMode VPU::getActSparsityMode(std::string strMode) {
    std::transform(strMode.begin(), strMode.end(), strMode.begin(), ::tolower);

    if (strMode == MODE_AUTO) {
        return VPU::EnableActivationSparsityMode::AUTO;
    } else if (strMode == MODE_TRUE) {
        return VPU::EnableActivationSparsityMode::TRUE;
    } else if (strMode == MODE_FALSE) {
        return VPU::EnableActivationSparsityMode::FALSE;
    }

    VPUX_THROW("Unknown value for the enable activation sparsity option: {0}", strMode);
}

VPU::EnableActivationSparsityMode VPU::getActSparsityMode(const StrOption& enableActivationSparsityOption) {
    auto strOption = convertToOptional(enableActivationSparsityOption);
    if (!strOption.has_value()) {
        return VPU::EnableActivationSparsityMode::AUTO;
    }
    return getActSparsityMode(strOption.value());
}

bool VPU::isActSparsityEnabled(const StrOption& enableActivationSparsityOption) {
    const auto actSparsityMode = getActSparsityMode(enableActivationSparsityOption);
    return actSparsityMode == VPU::EnableActivationSparsityMode::TRUE ||
           actSparsityMode == VPU::EnableActivationSparsityMode::AUTO;
}

// Get the largest storage element size that is compatible with the given number of channels
// The storage element size must be a multiple of 16
// Example: if the number of channels is 48 and the sparsity constraint is for the storage element size to be a power of
// two, the returned value will be 16
int64_t VPU::getSESize(int64_t channels, const VPU::SparsityConstraint& sparsityConstraint, bool isDepthwise) {
    auto checkDepthwiseLimitation = [&](int64_t seSize) {
        if (!isDepthwise) {
            return true;
        }
        // Only 16,32 and 64 is supported for depthwise
        return llvm::find(NCEInvariant::DEPTHWISE_WORKLOAD_SIZES, seSize) !=
               NCEInvariant::DEPTHWISE_WORKLOAD_SIZES.end();
    };
    constexpr int64_t maxDepthWiseSeSize = 64;
    int64_t seSize = isDepthwise ? std::min(maxDepthWiseSeSize, channels) : channels;
    for (; seSize >= 16; seSize -= 16) {
        if (channels % seSize == 0 && sparsityConstraint.areChannelsFitForSESize(seSize) &&
            checkDepthwiseLimitation(seSize)) {
            return seSize;
        }
    }
    VPUX_THROW("Failed to find se_size for '{0}' channels", channels);
}

//
// shouldRemoveOutputSparsity
//

VPU::SparsityRemovalFlag VPU::shouldRemoveOutputSparsity(mlir::Operation* op) {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return SparsityRemovalFlag::ClusteredOpInterfaceMissingFail;
    }

    const auto outputTensorType = mlir::cast<NDTypeInterface>(clusteredOp->getResult(0).getType());
    const auto sparseOutputType = mlir::dyn_cast<VPU::SparseTensorType>(outputTensorType);
    if (sparseOutputType == nullptr) {
        return SparsityRemovalFlag::SparseOutputMissingFail;
    }

    // First try to retrive the DistributedTensorType (it will be present in the passes post
    // MakeOpsWithDistributedTensorPass).
    auto distributedTensorType = mlir::dyn_cast_or_null<VPU::DistributedTensorType>(sparseOutputType.getData());
    if (distributedTensorType == nullptr) {
        if (!clusteredOp.getMultiClusterStrategy().has_value()) {
            return SparsityRemovalFlag::MultiClusterStrategyMissingFail;
        }

        const auto strategy = clusteredOp.getMultiClusterStrategy().value();
        if (strategy != VPU::MultiClusterStrategy::SplitOverKernel) {
            return SparsityRemovalFlag::SOKMissingFail;
        }

        VPUX_THROW_UNLESS(sparseOutputType.getSparsityMap() != nullptr, "Missing sparsity map from sparse type {0}",
                          sparseOutputType);
        VPUX_THROW_UNLESS(sparseOutputType.getStorageElementTable() == nullptr,
                          "Dynamically populated storage element table is not supported");

        const auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
        const auto distributedDataType = getDistributedOutputTypeFromOp(
                clusteredOp, sparseOutputType.getData(), numClusters,
                /*inputTypes*/ {}, /*tileInfo*/ TileInfo(ShapeRef()), /*hasExplicitDistributedAttr*/ false);

        distributedTensorType = mlir::cast<VPU::DistributedTensorType>(distributedDataType);
    }

    // Removes SOK layer's output sparsity if SOK layer has different split sizes on clusters excluding the last
    // one. For example, we need to split OC = 128 on 6 tiles, the tiled size will be {32, 32, 16, 16, 16, 16}.
    // If there's output sparsity, we need to split 32 into two pieces of 16 because we must have the same
    // workload channel excluding the last one. However, two workloads with 16 channels have much worse
    // performance than a workload with 32 channels. If there's no sparsity, we can keep the workload with 32
    // channels.
    if (distributedTensorType.getDistribution().getUniformDistributedSegments() != nullptr) {
        return SparsityRemovalFlag::Success;
    }

    // Removes SOK layer's output sparsity if SOK layer's output is used by `VPU.Concat`.
    //
    // Conv1_1 (OC = 256, SOK)  Conv1_2 (OC = 256, SOK)
    //       \                               /
    //                   Concat on C
    //                        |
    //                      Conv2
    //
    // Take above graph as an example, we need to split OC = 256 on 6 tiles, the tiled size will be {48, 48, 48,
    // 48, 48, 16}. After concatenation, the combined workloads will be {48, 48, 48, 48, 48, 16, 48, 48, 48, 48,
    // 48, 16}. If there's output sparsity for Conv1_1 and Conv1_2, we need to split 48 into three pieces of 16
    // because we must have the same workload channel excluding the last one. If there's no sparsity, we can
    // keep the workload with 48 channels.
    auto users = to_small_vector(clusteredOp->getUsers());
    if (llvm::find_if(users, [](mlir::Operation* op) {
            auto opToCheck = op;
            while (mlir::isa_and_nonnull<VPU::UnrolledTypeOp>(opToCheck)) {
                if (!opToCheck->hasOneUse()) {
                    break;
                }

                opToCheck = *opToCheck->getUsers().begin();
            }

            if (auto concatOp = mlir::dyn_cast_or_null<VPU::ConcatOp>(opToCheck)) {
                const auto outputType = mlir::cast<NDTypeInterface>(concatOp.getOutput().getType());
                const auto outputShape = outputType.getShape();
                const auto inputDataType = mlir::cast<NDTypeInterface>(concatOp.getInputs().front().getType());
                const auto inputShape = inputDataType.getShape();

                if (inputShape[Dims4D::Act::C] != outputShape[Dims4D::Act::C]) {
                    return true;
                }
            }

            return false;
        }) != users.end()) {
        return SparsityRemovalFlag::Success;
    }
    return SparsityRemovalFlag::CatchAllFail;
}

bool VPU::isSEOnlyWithoutSMSupported(config::ArchKind arch) {
    return arch != config::ArchKind::NPU37XX && arch != config::ArchKind::NPU40XX;
}

mlir::Type VPU::getEffectiveSparseOutputType(mlir::Type sparseType) {
    mlir::Type dataType;
    mlir::Type seTableType;
    if (auto sparseTensorType = mlir::dyn_cast<VPU::SparseTensorType>(sparseType)) {
        dataType = sparseTensorType.getData();
        seTableType = sparseTensorType.getStorageElementTable();
    } else if (auto sparseBufferType = mlir::dyn_cast<VPUIP::SparseBufferType>(sparseType)) {
        dataType = sparseBufferType.getData();
        seTableType = sparseBufferType.getStorageElementTable();
    } else {
        VPUX_THROW("Expected sparse type. Got {0}", sparseType);
    }

    if (seTableType == nullptr) {
        return dataType;
    }

    auto dataNDType = mlir::cast<NDTypeInterface>(dataType);
    auto seTableNDType = mlir::cast<NDTypeInterface>(seTableType);
    auto outShape = Shape(dataNDType.getShape());
    outShape[Dims4D::Act::H] = outShape[Dims4D::Act::H] != mlir::ShapedType::kDynamic
                                       ? seTableNDType.getShape()[Dims4D::Act::H]
                                       : mlir::ShapedType::kDynamic;
    outShape[Dims4D::Act::W] = outShape[Dims4D::Act::W] != mlir::ShapedType::kDynamic
                                       ? seTableNDType.getShape()[Dims4D::Act::W]
                                       : mlir::ShapedType::kDynamic;

    if (auto boundedData = mlir::dyn_cast<Core::BoundedTensorType>(dataType)) {
        auto bounds = SmallVector<int64_t>(boundedData.getBounds().raw());
        bounds[Dims4D::Act::H.ind()] = seTableNDType.getShape()[Dims4D::Act::H];
        bounds[Dims4D::Act::W.ind()] = seTableNDType.getShape()[Dims4D::Act::W];
        dataNDType = mlir::cast<NDTypeInterface>(boundedData.changeBounds(BoundsRef(bounds)));
    }

    auto distributedTypeIf = mlir::cast<VPU::DistributedTypeInterface>(sparseType);
    if (!distributedTypeIf.containsDistributedTypes()) {
        return dataNDType.changeShape(outShape);
    }

    auto getDistribution = [](mlir::Type componentType) -> VPU::DistributionInfoAttr {
        if (auto distributedTensor = mlir::dyn_cast<VPU::DistributedTensorType>(componentType)) {
            return distributedTensor.getDistribution();
        } else if (auto distributedBuffer = mlir::dyn_cast<VPUIP::DistributedBufferType>(componentType)) {
            return distributedBuffer.getDistribution();
        }

        VPUX_THROW("Sparse type's component is not distributed, component type = {0}", componentType);
    };

    auto dataDistribution = getDistribution(dataNDType);
    if (!VPU::isDistributedAttrWithExplicitShapesAndOffsets(dataDistribution)) {
        return dataNDType.changeShape(outShape);
    }

    auto distributionForEffectiveType = VPU::getExplicitDistrAttrForActualDataFromSparseType(sparseType);
    return mlir::cast<VPU::DistributedTypeInterface>(dataNDType)
            .changeShapeForExplicitDistribution(outShape, distributionForEffectiveType);
}

std::pair<SmallVector<int64_t>, SmallVector<int64_t>> VPU::getUpdatedSliceOffsetsAndShapesForSETable(
        const int64_t seDepth, mlir::ArrayAttr seSizeAttr, ArrayRef<int64_t> sliceOffsets,
        ArrayRef<int64_t> sliceSizes) {
    SmallVector<int64_t> seTableOffsets(sliceOffsets);
    SmallVector<int64_t> seTableSizes(sliceSizes);
    seTableOffsets[Dims4D::Act::N.ind()] = 0;
    seTableSizes[Dims4D::Act::N.ind()] = 1;

    if (seDepth == 1) {
        seTableOffsets[Dims4D::Act::C.ind()] = 0;
        seTableSizes[Dims4D::Act::C.ind()] = 1;
        return std::make_pair(seTableOffsets, seTableSizes);
    }

    auto seSizes = parseIntArrayAttr<int64_t>(seSizeAttr);
    auto offsetRange = irange(seSizes.size());
    auto offsetIter = llvm::find_if(offsetRange, [&](auto idx) {
        auto sum = std::accumulate(seSizes.begin(), seSizes.begin() + idx, 0);
        return sum == sliceOffsets[Dims4D::Act::C.ind()];
    });
    VPUX_THROW_WHEN(offsetIter == offsetRange.end(), "Slice over channels offset is not aligned with SE size");
    seTableOffsets[Dims4D::Act::C.ind()] = static_cast<int64_t>(*offsetIter);

    auto sizeRange = irange(*offsetIter, seSizes.size());
    auto sizeIter = llvm::find_if(sizeRange, [&](auto idx) {
        auto sum = std::accumulate(seSizes.begin() + *offsetIter, seSizes.begin() + idx + 1, 0);
        return sum == sliceSizes[Dims4D::Act::C.ind()];
    });
    VPUX_THROW_WHEN(sizeIter == sizeRange.end(), "Slice over channels size is not aligned with SE size");

    seTableSizes[Dims4D::Act::C.ind()] = static_cast<int64_t>(*sizeIter) - *offsetIter + 1;
    return std::make_pair(seTableOffsets, seTableSizes);
}
