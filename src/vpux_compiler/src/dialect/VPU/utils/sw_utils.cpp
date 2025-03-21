//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace VPU;

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::MultiClusterStrategy strategy) {
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeightOverlapped:
        return DistributionMode::OVERLAPPED;
    case VPU::MultiClusterStrategy::SplitOverWidth:
    case VPU::MultiClusterStrategy::SplitOverHeight:
    case VPU::MultiClusterStrategy::HKSwitch:
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::InterpolateOp interpolateOp,
                                                             VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    auto inType = interpolateOp.getInput().getType().cast<NDTypeInterface>();
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeightOverlapped:
        return (inType != inputType) ? DistributionMode::DUPLICATED : DistributionMode::OVERLAPPED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(LSTMSequenceOp lstmSequenceOp,
                                                             MultiClusterStrategy strategy, NDTypeInterface inputType) {
    VPUX_THROW_WHEN(
            strategy != MultiClusterStrategy::SplitOverBatch && strategy != MultiClusterStrategy::SplitOverKernel,
            "{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
            "activation tensor",
            strategy);

    if (strategy == MultiClusterStrategy::SplitOverBatch) {
        const auto reccurenceWeightsType = mlir::cast<NDTypeInterface>(lstmSequenceOp.getReccurenceWeights().getType());
        if (inputType == reccurenceWeightsType) {
            return DistributionMode::DUPLICATED;
        }
    }

    const auto syncBufferType = mlir::cast<NDTypeInterface>(lstmSequenceOp.getSyncBuffer().getType());
    if (inputType == syncBufferType) {
        return DistributionMode::DUPLICATED;
    }

    return DistributionMode::SEGMENTED;
}

VPU::DistributionMode vpux::VPU::getSWInputTensorDistributionMode(mlir::Operation* eltwiseOp,
                                                                  VPU::MultiClusterStrategy strategy,
                                                                  vpux::NDTypeInterface inputType) {
    auto isTileAtBroadCastAxis = [&](vpux::Dim tileAxis) {
        if (!eltwiseOp->hasAttr("auto_broadcast")) {
            return false;
        }
        const auto outputShape = getShape(eltwiseOp->getResult(0));
        const auto inputShape = inputType.getShape();
        VPUX_THROW_UNLESS(inputShape.size() == outputShape.size(),
                          "Input tensor rank {0} is mismatched with Output tensor rank {1}", inputShape.size(),
                          outputShape.size());
        return (outputShape[tileAxis] != inputShape[tileAxis]) && (inputShape[tileAxis] == 1);
    };

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return isTileAtBroadCastAxis(Dims4D::Act::W) ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return isTileAtBroadCastAxis(Dims4D::Act::H) ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return isTileAtBroadCastAxis(Dims4D::Act::C) ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::PReluOp preluOp, VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    // It is possible that two inputs has the same type.
    // For this case, cannot be completely sure if this type is a Slope input.
    // However, this does not conflict with the selection of DistributionMode.
    // The Slope input will only have at most one axis with size greater than 1 at channel
    // and this value equals to the input channel size.
    auto preluSlopeShape = preluOp.getNegativeSlope().getType().cast<NDTypeInterface>().getShape().toValues();
    const auto inputShape = inputType.getShape();
    // Skip checking the channel because the shape of the channel may be inconsistent
    // if the input tensor is tiled.
    preluSlopeShape[Dims4D::Act::C] = inputShape[Dims4D::Act::C];
    auto isPotentialSlopeInput = (inputShape == preluSlopeShape);

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return isPotentialSlopeInput ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return isPotentialSlopeInput ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::AccumulateOp accumulateOp,
                                                             VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    auto outType = accumulateOp.getOutput().getType().cast<NDTypeInterface>();
    // Scale must have 1xCx1x1 shape, which is different from NxCxHxW output shape.
    const auto isScale = (outType != inputType);

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverWidth:
    case VPU::MultiClusterStrategy::SplitOverHeight:
        // VPU.Accumulate(lhs=[N,C,H,W], rhs=[N,C,H,W], lhsScale=[1,C,1,1], rhsScale=[1,C,1,1])
        // Split over height, where Y = H / clusters
        // VPU.Accumulate(lhs=[N,C,Y,W], rhs=[N,C,Y,W], lhsScale=[1,C,1,1], rhsScale=[1,C,1,1])
        // Split over width, where X = W / clusters
        // VPU.Accumulate(lhs=[N,C,H,X], rhs=[N,C,H,X], lhsScale=[1,C,1,1], rhsScale=[1,C,1,1])
        // lhs and rhs are segmented, scales are duplicated.
        return isScale ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        // VPU.Accumulate(lhs=[N,C,H,W], rhs=[N,C,H,W], lhsScale=[1,C,1,1], rhsScale=[1,C,1,1])
        // Split over kernel, where Z = C / clusters
        // VPU.Accumulate(lhs=[N,Z,H,W], rhs=[N,Z,H,W], lhsScale=[1,Z,1,1], rhsScale=[1,Z,1,1])
        // All operands are segmented.
        return DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::DynamicDequantizeOp dynamicDequantizeOp,
                                                             VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    // Scale and Zp of DynamicDequantizeOp may need broadcast
    // Broadcast dim can't be segmented.
    auto isTileAtBroadCastAxis = [&](vpux::Dim tileAxis) {
        const auto outputShape = getShape(dynamicDequantizeOp->getResult(0));
        const auto inputShape = inputType.getShape();
        VPUX_THROW_UNLESS(inputShape.size() == outputShape.size(),
                          "Input tensor rank {0} is mismatched with Output tensor rank {1}", inputShape.size(),
                          outputShape.size());
        return (outputShape[tileAxis] != inputShape[tileAxis]) && (inputShape[tileAxis] == 1);
    };

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return isTileAtBroadCastAxis(Dims4D::Act::W) ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return isTileAtBroadCastAxis(Dims4D::Act::H) ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return isTileAtBroadCastAxis(Dims4D::Act::C) ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::DetectionOutputSortOp /*op*/,
                                                             VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface /*inputType*/) {
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return DistributionMode::SEGMENTED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::MatMulOp /*op*/, VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface /*inputType*/) {
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return DistributionMode::SEGMENTED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::LSTMGatesOp /*op*/,
                                                             VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface /*inputType*/) {
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::MVN1NormalizeOp op,
                                                             VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    auto meanVarType = op.getMeanVar().getType().cast<NDTypeInterface>();
    if (meanVarType == inputType) {
        if (strategy == VPU::MultiClusterStrategy::SplitOverKernel && !op.getAcrossChannels()) {
            return DistributionMode::SEGMENTED;
        }
        return DistributionMode::DUPLICATED;
    }

    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverWidth:
    case VPU::MultiClusterStrategy::SplitOverHeight:
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return DistributionMode::SEGMENTED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::MVN6Op op, VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    const auto curShape = inputType.getShape();
    const auto axesVec = parseIntArrayAttr<int64_t>(op.getAxesAttr());

    auto getDistribMode = [&](vpux::Dim dim) -> DistributionMode {
        if (std::find(axesVec.begin(), axesVec.end(), dim.ind()) != axesVec.end()) {
            // Cannot SEGMENT along a normalization axis
            return DistributionMode::DUPLICATED;
        }
        if (curShape[dim] == 1) {
            // Cannot SEGMENT a broadcast dim
            return DistributionMode::DUPLICATED;
        }
        return DistributionMode::SEGMENTED;
    };

    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return getDistribMode(Dims4D::Act::W);
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return getDistribMode(Dims4D::Act::H);
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return getDistribMode(Dims4D::Act::C);
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::GatherOp op, VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    // GatherOp with 4D Input, Indices, and Output. The data structures must adhere to the following configurations:
    //      Input: [BatchDimsRange, DataBeforeAxisRange, IndicesRange, DataAfterAxisRange]
    //      Indices: [BatchDimsRange, IndicesRange, 1, 1]
    // If 'SplitOverKernel' and 'SplitOverWidth' are enabled, the Input is split while Indices remain intact
    // If 'SplitOverHeight' is enabled, the Indices are split while the Input remains intact
    const auto isIndicesTensor = inputType.getShape() == getShape(op.getIndices());

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverKernel:
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return isIndicesTensor ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return isIndicesTensor ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::GatherElementsOp, VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface) {
    // GatherElementsOp with 4D Input, Indices, and Output. The data structures for inputs and indices must adhere to
    // the following configurations:
    // [1, DataBeforeAxisRange, IndicesRange, DataAfterAxisRange]
    // If 'SplitOverKernel' and 'SplitOverWidth' are enabled, the input and indices are split.
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverKernel:
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::RMSOp op, VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    const auto isInputTensor = inputType.getShape() == getShape(op.getInput());

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverKernel:
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return isInputTensor ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::RoPEOp op, VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    const auto isInputTensor = inputType.getShape() == getShape(op.getInput());

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return isInputTensor ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::TopKOp op, VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    const auto isInputTensor = inputType.getShape() == getShape(op.getInput());

    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverKernel:
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return isInputTensor ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::ClusteredOpInterface clusteredOp,
                                                             VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    return llvm::TypeSwitch<mlir::Operation*, VPU::DistributionMode>(clusteredOp.getOperation())
            .Case<VPU::InterpolateOp>([&](VPU::InterpolateOp interpolateOp) {
                return getSWInputTensorDistributionMode(interpolateOp, strategy, inputType);
            })
            .Case<VPU::MultiplyOp, VPU::DivideOp, VPU::PowerOp, VPU::MaximumOp, VPU::MinimumOp, VPU::GreaterOp,
                  VPU::LessOp, VPU::EqualOp, VPU::AndOp, VPU::SubtractOp, VPU::AddOp, VPU::FloorOp, VPU::CeilingOp,
                  VPU::FakeQuantizeOp, VPU::SelectOp, VPU::RoundOp, VPU::SinOp, VPU::CosOp, VPU::ExpOp, VPU::MishOp>(
                    [&](mlir::Operation* eltwiseOp) {
                        return getSWInputTensorDistributionMode(eltwiseOp, strategy, inputType);
                    })
            .Case<VPU::PReluOp>([&](VPU::PReluOp preluOp) {
                return getSWInputTensorDistributionMode(preluOp, strategy, inputType);
            })
            .Case<VPU::AccumulateOp>([&](VPU::AccumulateOp accumulateOp) {
                return getSWInputTensorDistributionMode(accumulateOp, strategy, inputType);
            })
            .Case<VPU::DetectionOutputSortOp>([&](VPU::DetectionOutputSortOp op) {
                return getSWInputTensorDistributionMode(op, strategy, inputType);
            })
            .Case<VPU::MatMulOp>([&](VPU::MatMulOp op) {
                return getSWInputTensorDistributionMode(op, strategy, inputType);
            })
            .Case<VPU::LSTMGatesOp>([&](VPU::LSTMGatesOp lstmGatesOp) {
                return getSWInputTensorDistributionMode(lstmGatesOp, strategy, inputType);
            })
            .Case<VPU::LSTMSequenceOp>([&](VPU::LSTMSequenceOp op) {
                return getSWInputTensorDistributionMode(op, strategy, inputType);
            })
            .Case<VPU::MVN1NormalizeOp>([&](VPU::MVN1NormalizeOp op) {
                return getSWInputTensorDistributionMode(op, strategy, inputType);
            })
            .Case<VPU::MVN6Op>([&](VPU::MVN6Op op) {
                return getSWInputTensorDistributionMode(op, strategy, inputType);
            })
            .Case<VPU::GatherOp>([&](VPU::GatherOp op) {
                return getSWInputTensorDistributionMode(op, strategy, inputType);
            })
            .Case<VPU::GatherElementsOp>([&](VPU::GatherElementsOp op) {
                return getSWInputTensorDistributionMode(op, strategy, inputType);
            })
            .Case<VPU::DynamicDequantizeOp>([&](VPU::DynamicDequantizeOp dynamicDequantizeOp) {
                return getSWInputTensorDistributionMode(dynamicDequantizeOp, strategy, inputType);
            })
            .Case<VPU::RMSOp>([&](VPU::RMSOp op) {
                return getSWInputTensorDistributionMode(op, strategy, inputType);
            })
            .Case<VPU::RoPEOp>([&](VPU::RoPEOp op) {
                return getSWInputTensorDistributionMode(op, strategy, inputType);
            })
            .Case<VPU::TopKOp>([&](VPU::TopKOp op) {
                return getSWInputTensorDistributionMode(op, strategy, inputType);
            })
            .Default([&](mlir::Operation*) {
                VPUX_THROW_UNLESS(clusteredOp->getOperands().size() == 1,
                                  "General method only support SW layer with one operand but got '{0}'",
                                  clusteredOp->getOperands().size());
                return getSWInputTensorDistributionMode(strategy);
            });
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy) {
    return getActivationTensorNumTiles(clusteredOp, numClustersAvailableForCompilation, strategy);
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::InterpolateOp interpolateOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(interpolateOp, strategy, inputType);

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeightOverlapped:
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    case VPU::MultiClusterStrategy::Clustering:
        return {1, 1, 1, 1};
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(LSTMSequenceOp lstmSequenceOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         MultiClusterStrategy strategy, NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(lstmSequenceOp, strategy, inputType);

    if (distributionMode == DistributionMode::SEGMENTED) {
        const auto inputData = mlir::cast<NDTypeInterface>(lstmSequenceOp.getInputData().getType());
        if (strategy == MultiClusterStrategy::SplitOverBatch) {
            const auto batchSize = inputData.getShape()[Dims4D::Act::N];
            const auto numClusters = std::min(batchSize, numClustersAvailableForCompilation);
            return {numClusters, 1, 1, 1};
        } else if (strategy == MultiClusterStrategy::SplitOverKernel) {
            const auto numDirections = inputData.getShape()[Dims4D::Act::C];
            const auto reccurenceWeights = mlir::cast<NDTypeInterface>(lstmSequenceOp.getReccurenceWeights().getType());
            if (inputType == reccurenceWeights) {
                return {numDirections, 1, 1, 1};
            } else {
                return {1, numDirections, 1, 1};
            }
        }
    }

    return {1, 1, 1, 1};
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::MVN1NormalizeOp normalizeOpOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(normalizeOpOp, strategy, inputType);

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    case VPU::MultiClusterStrategy::SplitOverKernel: {
        auto IC = inputType.getShape()[Dims4D::Act::C];
        int64_t numClustersToUseForLayer = std::min(numClustersAvailableForCompilation, IC);
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, numClustersToUseForLayer, 1, 1};
    }
    case VPU::MultiClusterStrategy::Clustering:
        return {1, 1, 1, 1};
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, 1, numClustersAvailableForCompilation};
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::MVN6Op op, int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, inputType);
    if (distributionMode == VPU::DistributionMode::DUPLICATED) {
        return SmallVector<int64_t>{1, 1, 1, 1};
    }

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverKernel: {
        auto C = inputType.getShape()[Dims4D::Act::C];
        int64_t clusters = std::min(numClustersAvailableForCompilation, C);
        return SmallVector<int64_t>{1, clusters, 1, 1};
    }
    case VPU::MultiClusterStrategy::SplitOverHeight: {
        auto H = inputType.getShape()[Dims4D::Act::H];
        int64_t clusters = std::min(numClustersAvailableForCompilation, H);
        return SmallVector<int64_t>{1, 1, clusters, 1};
    }
    case VPU::MultiClusterStrategy::SplitOverWidth: {
        auto W = inputType.getShape()[Dims4D::Act::W];
        int64_t clusters = std::min(numClustersAvailableForCompilation, W);
        return SmallVector<int64_t>{1, 1, 1, clusters};
    }
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(mlir::Operation* eltwiseOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(eltwiseOp, strategy, inputType);

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    case VPU::MultiClusterStrategy::SplitOverKernel: {
        auto IC = inputType.getShape()[Dims4D::Act::C];
        int64_t numClustersToUseForLayer = std::min(numClustersAvailableForCompilation, IC);
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, numClustersToUseForLayer, 1, 1};
    }
    case VPU::MultiClusterStrategy::Clustering:
        return {1, 1, 1, 1};
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, 1, numClustersAvailableForCompilation};
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::AccumulateOp accumulateOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(accumulateOp, strategy, inputType);
    if (distributionMode == VPU::DistributionMode::DUPLICATED) {
        return {1, 1, 1, 1};
    }

    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        return {1, 1, 1, 1};
    case VPU::MultiClusterStrategy::SplitOverHeight: {
        const auto height = inputType.getShape()[Dims4D::Act::H];
        const auto clusters = std::min(numClustersAvailableForCompilation, height);
        return {1, 1, clusters, 1};
    }
    case VPU::MultiClusterStrategy::SplitOverWidth: {
        const auto width = inputType.getShape()[Dims4D::Act::W];
        const auto clusters = std::min(numClustersAvailableForCompilation, width);
        return {1, 1, 1, clusters};
    }
    case VPU::MultiClusterStrategy::SplitOverKernel: {
        const auto channels = inputType.getShape()[Dims4D::Act::C];
        const auto clusters = std::min(numClustersAvailableForCompilation, channels);
        return {1, clusters, 1, 1};
    }
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::DynamicDequantizeOp dynamicDequantizeOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(dynamicDequantizeOp, strategy, inputType);
    if (distributionMode == VPU::DistributionMode::DUPLICATED) {
        return {1, 1, 1, 1};
    }

    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        return {1, 1, 1, 1};
    case VPU::MultiClusterStrategy::SplitOverHeight: {
        const auto height = inputType.getShape()[Dims4D::Act::H];
        const auto clusters = std::min(numClustersAvailableForCompilation, height);
        return {1, 1, clusters, 1};
    }
    case VPU::MultiClusterStrategy::SplitOverWidth: {
        const auto width = inputType.getShape()[Dims4D::Act::W];
        const auto clusters = std::min(numClustersAvailableForCompilation, width);
        return {1, 1, 1, clusters};
    }
    case VPU::MultiClusterStrategy::SplitOverKernel: {
        const auto channels = inputType.getShape()[Dims4D::Act::C];
        const auto clusters = std::min(numClustersAvailableForCompilation, channels);
        return {1, clusters, 1, 1};
    }
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::DetectionOutputSortOp /*op*/,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface /*inputType*/) {
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::MatMulOp /*op*/,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface /*inputType*/) {
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return SmallVector<int64_t>{1, numClustersAvailableForCompilation, 1, 1};
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::LSTMGatesOp /*op*/,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface /*inputType*/) {
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    case VPU::MultiClusterStrategy::Clustering:
        return {1, 1, 1, 1};
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::GatherOp op, int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    // GatherOp with 4D Input, Indices, and Output. The data structures must adhere to the following configurations:
    //      Input: [BatchDimsRange, DataBeforeAxisRange, IndicesRange, DataAfterAxisRange]
    //      Indices: [BatchDimsRange, IndicesRange, 1, 1]
    // If 'SplitOverKernel' and 'SplitOverWidth' are enabled:
    //      - The Input is split, maintaining consistency with the 'C' or 'W'
    //      - Indices remain intact
    // If 'SplitOverHeight' is enabled:
    //      - The Indices is split, specifically at the IndicesRange('C')
    //      - The Input remains intact
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, inputType);

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
    case VPU::MultiClusterStrategy::SplitOverKernel: {
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, numClustersAvailableForCompilation, 1, 1};
    }
    case VPU::MultiClusterStrategy::Clustering:
        return {1, 1, 1, 1};
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, 1, numClustersAvailableForCompilation};
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::GatherElementsOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface) {
    // GatherElementsOp with 4D Input, Indices, and Output. The data structures for inputs and indices must adhere to
    // the following configurations:
    // [1, DataBeforeAxisRange, IndicesRange, DataAfterAxisRange]
    // If 'SplitOverKernel' and 'SplitOverWidth' are enabled:
    //      - The Input/Indice are split, maintaining consistency with the 'C' or 'W'
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverKernel: {
        return SmallVector<int64_t>{1, numClustersAvailableForCompilation, 1, 1};
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return SmallVector<int64_t>{1, 1, 1, numClustersAvailableForCompilation};
    case VPU::MultiClusterStrategy::Clustering:
        return {1, 1, 1, 1};
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::RMSOp op, int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, inputType);

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight: {
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    }
    case VPU::MultiClusterStrategy::SplitOverKernel: {
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, numClustersAvailableForCompilation, 1, 1};
    }
    case VPU::MultiClusterStrategy::Clustering: {
        return SmallVector<int64_t>{1, 1, 1, 1};
    }
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::RoPEOp op, int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, inputType);

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight: {
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    }
    case VPU::MultiClusterStrategy::SplitOverKernel: {
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, numClustersAvailableForCompilation, 1, 1};
    }
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::TopKOp op, int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, inputType);

    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        return {1, 1, 1, 1};
    case VPU::MultiClusterStrategy::SplitOverHeight: {
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    }
    case VPU::MultiClusterStrategy::SplitOverKernel: {
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, numClustersAvailableForCompilation, 1, 1};
    }
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    return llvm::TypeSwitch<mlir::Operation*, SmallVector<int64_t>>(clusteredOp.getOperation())
            .Case<VPU::InterpolateOp>([&](VPU::InterpolateOp interpolateOp) {
                return getSWInputTensorNumTiles(interpolateOp, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::MultiplyOp, VPU::DivideOp, VPU::PowerOp, VPU::MaximumOp, VPU::MinimumOp, VPU::PReluOp,
                  VPU::GreaterOp, VPU::LessOp, VPU::EqualOp, VPU::AndOp, VPU::SubtractOp, VPU::AddOp, VPU::FloorOp,
                  VPU::CeilingOp, VPU::FakeQuantizeOp, VPU::SelectOp, VPU::RoundOp, VPU::SinOp, VPU::CosOp, VPU::ExpOp,
                  VPU::MishOp>([&](mlir::Operation* eltwiseOp) {
                return getSWInputTensorNumTiles(eltwiseOp, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::AccumulateOp>([&](VPU::AccumulateOp accumulateOp) {
                return getSWInputTensorNumTiles(accumulateOp, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::DetectionOutputSortOp>([&](VPU::DetectionOutputSortOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::MatMulOp>([&](VPU::MatMulOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::LSTMGatesOp>([&](VPU::LSTMGatesOp lstmGatesOp) {
                return getSWInputTensorNumTiles(lstmGatesOp, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::LSTMSequenceOp>([&](VPU::LSTMSequenceOp lstmSequenceOp) {
                return getSWInputTensorNumTiles(lstmSequenceOp, numClustersAvailableForCompilation, strategy,
                                                inputType);
            })
            .Case<VPU::MVN1NormalizeOp>([&](VPU::MVN1NormalizeOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::MVN6Op>([&](VPU::MVN6Op op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::GatherOp>([&](VPU::GatherOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::GatherElementsOp>([&](VPU::GatherElementsOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::DynamicDequantizeOp>([&](VPU::DynamicDequantizeOp dynamicDequantizeOp) {
                return getSWInputTensorNumTiles(dynamicDequantizeOp, numClustersAvailableForCompilation, strategy,
                                                inputType);
            })
            .Case<VPU::RMSOp>([&](VPU::RMSOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::RoPEOp>([&](VPU::RoPEOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::TopKOp>([&](VPU::TopKOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Default([&](mlir::Operation*) {
                VPUX_THROW_UNLESS(clusteredOp->getOperands().size() == 1,
                                  "General method only support SW layer with one operand but got '{0}'",
                                  clusteredOp->getOperands().size());
                return getSWInputTensorNumTiles(clusteredOp, numClustersAvailableForCompilation, strategy);
            });
}
