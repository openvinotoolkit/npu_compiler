//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
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
    case VPU::MultiClusterStrategy::SplitOverGroup:
        return DistributionMode::SEGMENTED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::InterpolateOp interpolateOp,
                                                             VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeightOverlapped:
        return (operand != interpolateOp.getInput()) ? DistributionMode::DUPLICATED : DistributionMode::OVERLAPPED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(LSTMSequenceOp lstmSequenceOp,
                                                             MultiClusterStrategy strategy, mlir::Value operand) {
    VPUX_THROW_WHEN(
            strategy != MultiClusterStrategy::SplitOverBatch && strategy != MultiClusterStrategy::SplitOverKernel,
            "{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
            "activation tensor",
            strategy);

    if (strategy == MultiClusterStrategy::SplitOverBatch) {
        if (operand == lstmSequenceOp.getReccurenceWeights()) {
            return DistributionMode::DUPLICATED;
        }

        if (operand == lstmSequenceOp.getBiases()) {
            return DistributionMode::DUPLICATED;
        }
    }

    if (operand == lstmSequenceOp.getSyncBuffer()) {
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
                                                             mlir::Value operand) {
    const auto isSlopeInput = (operand == preluOp.getNegativeSlope());

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return isSlopeInput ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return isSlopeInput ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
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
                                                             VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    // Scale must have 1xCx1x1 shape, which is different from NxCxHxW output shape.
    const auto isScale = (operand == accumulateOp.getLhsScale() || operand == accumulateOp.getRhsScale());

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
                                                             VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                                             vpux::NDTypeInterface inputType) {
    // Scale and Zp of DynamicDequantizeOp may need broadcast
    // Broadcast dim can't be segmented.
    auto isTileAtBroadCastAxis = [&](vpux::Dim tileAxis) {
        const auto inputShape = inputType.getShape();
        return (operand == dynamicDequantizeOp.getZp() || operand == dynamicDequantizeOp.getScale()) &&
               (inputShape[tileAxis] == 1);
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
                                                             VPU::MultiClusterStrategy strategy) {
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return DistributionMode::SEGMENTED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::MatMulOp /*op*/, VPU::MultiClusterStrategy strategy) {
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
                                                             VPU::MultiClusterStrategy strategy) {
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

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::GRUGatesOp op, VPU::MultiClusterStrategy strategy,
                                                             mlir::Value operand) {
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return (operand != op.getBiases()) ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::MVN1NormalizeOp op,
                                                             VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    if (operand == op.getMeanVar()) {
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
                                                             mlir::Value operand) {
    // GatherOp with 4D Input, Indices, and Output. The data structures must adhere to the following configurations:
    //      Input: [BatchDimsRange, DataBeforeAxisRange, IndicesRange, DataAfterAxisRange]
    //      Indices: [BatchDimsRange, IndicesRange, 1, 1]
    // If 'SplitOverKernel' and 'SplitOverWidth' are enabled, the Input is split while Indices remain intact
    // If 'SplitOverHeight' is enabled, the Indices are split while the Input remains intact
    const auto isIndicesTensor = operand == op.getIndices();

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

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::GatherNDOp gatherNDOp,
                                                             VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    // GatherNDOp with 4D Input, Indices, and Output. The data structures must adhere to the following configurations:
    //      Input Data: [fixed_dim, batch_dim_size, coord_dim_size, after_coord_dim_size]
    //      Indice:     [fixed_dim, batch_dim_size, indices_dim_size, coord_rank]
    // If 'SplitOverKernel' is enabled, the Input and Indices are split
    // If 'SplitOverHeight' is enabled, the Indices are split while the Input remains intact
    // If 'SplitOverWidth' is enabled, the Input is split while Indices remain intact
    const auto isIndicesTensor = operand == gatherNDOp.getIndices();

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return isIndicesTensor ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return isIndicesTensor ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::GridSampleOp op, VPU::MultiClusterStrategy strategy,
                                                             mlir::Value operand) {
    // For GridSampleOp, input is [N, C, H, W], grid is [N, H, W, 2], and output is [N, C, H, W]
    // If 'SplitOverKernel' is enabled, the input is segmented (over C) while the grid is duplicated (since grid cannot
    // be tiled this case)
    // If 'SplitOverHeight' or 'SplitOverWidth' is enabled, the input is duplicated(since input cannot be tiled this
    // case) while the grid is segmented (over H or over W)
    const auto isInputTensor = (operand == op.getInput());
    const auto isGridTensor = (operand == op.getGrid());
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return isInputTensor ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return isGridTensor ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return isGridTensor ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverBatch:
        return DistributionMode::SEGMENTED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::DeformableConvolutionOp op,
                                                             VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    /* For DeformableConvolutionOp, input is [N, C, H, W], offset is [N,
  deformable_group * kernel_Y * kernel_X * 2, H, W], kernel is [N, C, Y, X], mask
  is [N, deformable_group * kernel_Y * kernel_X, H, W], and output is [N, C, H, W]
  If 'SplitOverBatch' is enabled, input, offset and mask can be tiled on N and will
  be segmented. If 'SplitOverKernel' is enabled, input, offset and mask will be
  duplicated, and kernel will be tiled on N. If 'SplitOverHeight' or
  'SplitOverWidth' is enabled, input and kernel will be duplicated, offset and
  mask will be segmented. */

    const auto isInputTensor = (operand == op.getInput());
    const auto isOffsetTensor = (operand == op.getOffset());
    const auto isKernelTensor = (operand == op.getKernel());
    const auto isMaskTensor = (operand == op.getMask());

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverBatch:
        return (isInputTensor || isOffsetTensor || isMaskTensor) ? DistributionMode::SEGMENTED
                                                                 : DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverWidth:
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return (isOffsetTensor || isMaskTensor) ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return isKernelTensor ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster "
                   "strategy, unable to determine "
                   "the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::GatherElementsOp,
                                                             VPU::MultiClusterStrategy strategy) {
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

template <typename OpType>
DistributionMode getSWInputTensorDistributionModeImpl(OpType /*op*/, VPU::MultiClusterStrategy strategy) {
    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverWidth:
    case VPU::MultiClusterStrategy::SplitOverHeight:
    case VPU::MultiClusterStrategy::SplitOverKernel:
    case VPU::MultiClusterStrategy::SplitOverBatch:
        return DistributionMode::SEGMENTED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode", strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::ReverseOp op, VPU::MultiClusterStrategy strategy) {
    return getSWInputTensorDistributionModeImpl(op, strategy);
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::CumSumOp op, VPU::MultiClusterStrategy strategy) {
    return getSWInputTensorDistributionModeImpl(op, strategy);
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::RMSOp op, VPU::MultiClusterStrategy strategy,
                                                             mlir::Value operand) {
    const auto isInputTensor = operand == op.getInput();

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
                                                             mlir::Value operand) {
    const auto isInputTensor = operand == op.getInput();
    const auto isInputSinOrCosTensor = operand == op.getInputSin() || operand == op.getInputCos();

    const auto operandOrigShape = getShape(operand);
    const auto inputOrigShape = getShape(op.getInput());

    auto cOperand = operandOrigShape[Dims4D::Act::C];
    auto cIn = inputOrigShape[Dims4D::Act::C];
    auto isSameChannelsSinCos = cOperand == cIn && isInputSinOrCosTensor;

    auto hOperand = operandOrigShape[Dims4D::Act::H];
    auto hIn = inputOrigShape[Dims4D::Act::H];
    auto isSameHeightSinCos = hOperand == hIn && isInputSinOrCosTensor;

    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return isInputTensor || isSameHeightSinCos ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return isInputTensor || isSameChannelsSinCos ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

// Segmented over kernel is legal for Inputs: Q, K, V, 3D Mask, 3D Bias and DataStorage
// Segmented over height is legal for Inputs: Q, 2D Mask, 2D Bias and DataStorage
DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::SDPAOp op, VPU::MultiClusterStrategy strategy,
                                                             mlir::Value operand) {
    const auto isInputQTensor = operand == op.getInputQ();
    const auto isInputKTensor = operand == op.getInputK();
    const auto isInputVTensor = operand == op.getInputV();
    const auto isDataStorageTensor = operand == op.getDataStorage();

    // Optional inputs distribution mode
    auto is2DMaskTensor = operand == op.getInputMask() && getShape(op.getInputMask())[Dims4D::Act::H] != 1;
    auto is3DMaskTensor = operand == op.getInputMask() && getShape(op.getInputMask())[Dims4D::Act::C] != 1;
    auto is2DBiasTensor = operand == op.getInputBias() && getShape(op.getInputBias())[Dims4D::Act::H] != 1;
    auto is3DBiasTensor = operand == op.getInputBias() && getShape(op.getInputBias())[Dims4D::Act::C] != 1;

    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return isInputQTensor || isInputKTensor || isInputVTensor || isDataStorageTensor || is3DMaskTensor ||
                               is3DBiasTensor
                       ? DistributionMode::SEGMENTED
                       : DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return isInputQTensor || isDataStorageTensor || is2DMaskTensor || is2DBiasTensor ? DistributionMode::SEGMENTED
                                                                                         : DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::FlashSDPAOp op, VPU::MultiClusterStrategy strategy,
                                                             mlir::Value operand) {
    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverKernel: {
        const auto isMaskWithoutBatch =
                (operand == op.getAttentionMask()) && (getShape(op.getAttentionMask())[Dims4D::Act::C] == 1);
        const auto isScale = (operand == op.getScale());

        return (isMaskWithoutBatch || isScale) ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    }
    case VPU::MultiClusterStrategy::SplitOverHeight: {
        auto isSegmented = false;
        isSegmented |= (operand == op.getQuery());
        isSegmented |= (operand == op.getAuxBuffer());
        isSegmented |= (operand == op.getAttentionMask());
        isSegmented |= (operand == op.getInputRunningOutput());
        isSegmented |= (operand == op.getInputRunningMax());
        isSegmented |= (operand == op.getInputRunningSum());
        isSegmented |= (operand == op.getAttentionMask());

        return isSegmented ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    }
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::SDPAExtendedOp op, VPU::MultiClusterStrategy strategy,
                                                             mlir::Value operand) {
    const auto isInputQTensor = operand == op.getInputQ();
    const auto isInputKTensor = operand == op.getInputK();
    const auto isInputVTensor = operand == op.getInputV();
    const auto isDataStorageTensor = operand == op.getDataStorage();

    // Optional inputs distribution mode
    auto is2DMaskTensor = operand == op.getInputMask() && getShape(op.getInputMask())[Dims4D::Act::H] != 1;
    auto is3DMaskTensor = operand == op.getInputMask() && getShape(op.getInputMask())[Dims4D::Act::C] != 1;

    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return isInputQTensor || isInputKTensor || isInputVTensor || is3DMaskTensor ? DistributionMode::SEGMENTED
                                                                                    : DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return isInputQTensor || isDataStorageTensor || is2DMaskTensor ? DistributionMode::SEGMENTED
                                                                       : DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::RandomUniformOp /*op*/,
                                                             VPU::MultiClusterStrategy strategy) {
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
    case VPU::MultiClusterStrategy::SplitOverWidth:
    case VPU::MultiClusterStrategy::SplitOverKernel:
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::RollOp op, VPU::MultiClusterStrategy strategy,
                                                             mlir::Value operand) {
    const auto isDataTensor = (operand == op.getData());
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
    case VPU::MultiClusterStrategy::SplitOverWidth:
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return isDataTensor ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::TopKOp op, VPU::MultiClusterStrategy strategy,
                                                             mlir::Value operand) {
    const auto isInputTensor = operand == op.getInput();

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

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::DynamicQuantizeOp dqOp,
                                                             VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    const auto isInputTensor = operand == dqOp.getInput();
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverWidth:
    case VPU::MultiClusterStrategy::SplitOverHeight:
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return isInputTensor ? DistributionMode::SEGMENTED : DistributionMode::DUPLICATED;
    case VPU::MultiClusterStrategy::Clustering:
        return DistributionMode::DUPLICATED;
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode", strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::YuvToRgbOp /*op*/, VPU::MultiClusterStrategy strategy,
                                                             mlir::Value /*operand*/) {
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return DistributionMode::SEGMENTED;
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

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::ClusteredOpInterface clusteredOp,
                                                             VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                                             vpux::NDTypeInterface inputType) {
    return llvm::TypeSwitch<mlir::Operation*, VPU::DistributionMode>(clusteredOp.getOperation())
            .Case<VPU::InterpolateOp>([&](VPU::InterpolateOp interpolateOp) {
                return getSWInputTensorDistributionMode(interpolateOp, strategy, operand);
            })
            .Case<VPU::MultiplyOp, VPU::DivideOp, VPU::PowerOp, VPU::MaximumOp, VPU::MinimumOp, VPU::GreaterOp,
                  VPU::GreaterEqualOp, VPU::LessOp, VPU::EqualOp, VPU::NotEqualOp, VPU::LessEqualOp, VPU::AndOp,
                  VPU::SubtractOp, VPU::AddOp, VPU::FloorOp, VPU::CeilingOp, VPU::FakeQuantizeOp, VPU::SelectOp,
                  VPU::RoundOp, VPU::SinOp, VPU::CosOp, VPU::ExpOp, VPU::MishOp, VPU::NegativeOp, VPU::LogicalNotOp,
                  VPU::SoftPlusOp, VPU::BitwiseOrOp, VPU::BitwiseAndOp, VPU::BitwiseNotOp, VPU::BitwiseXorOp>(
                    [&](mlir::Operation* eltwiseOp) {
                        return getSWInputTensorDistributionMode(eltwiseOp, strategy, inputType);
                    })
            .Case<VPU::PReluOp>([&](VPU::PReluOp preluOp) {
                return getSWInputTensorDistributionMode(preluOp, strategy, operand);
            })
            .Case<VPU::AccumulateOp>([&](VPU::AccumulateOp accumulateOp) {
                return getSWInputTensorDistributionMode(accumulateOp, strategy, operand);
            })
            .Case<VPU::DetectionOutputSortOp>([&](VPU::DetectionOutputSortOp op) {
                return getSWInputTensorDistributionMode(op, strategy);
            })
            .Case<VPU::MatMulOp>([&](VPU::MatMulOp op) {
                return getSWInputTensorDistributionMode(op, strategy);
            })
            .Case<VPU::LSTMGatesOp>([&](VPU::LSTMGatesOp lstmGatesOp) {
                return getSWInputTensorDistributionMode(lstmGatesOp, strategy);
            })
            .Case<VPU::GRUGatesOp>([&](VPU::GRUGatesOp gruGatesOp) {
                return getSWInputTensorDistributionMode(gruGatesOp, strategy, operand);
            })
            .Case<VPU::LSTMSequenceOp>([&](VPU::LSTMSequenceOp op) {
                return getSWInputTensorDistributionMode(op, strategy, operand);
            })
            .Case<VPU::MVN1NormalizeOp>([&](VPU::MVN1NormalizeOp op) {
                return getSWInputTensorDistributionMode(op, strategy, operand);
            })
            .Case<VPU::MVN6Op>([&](VPU::MVN6Op op) {
                return getSWInputTensorDistributionMode(op, strategy, inputType);
            })
            .Case<VPU::GatherOp>([&](VPU::GatherOp op) {
                return getSWInputTensorDistributionMode(op, strategy, operand);
            })
            .Case<VPU::GatherNDOp>([&](VPU::GatherNDOp gatherNDOp) {
                return getSWInputTensorDistributionMode(gatherNDOp, strategy, operand);
            })
            .Case<VPU::GatherElementsOp>([&](VPU::GatherElementsOp op) {
                return getSWInputTensorDistributionMode(op, strategy);
            })
            .Case<VPU::GridSampleOp>([&](VPU::GridSampleOp op) {
                return getSWInputTensorDistributionMode(op, strategy, operand);
            })
            .Case<VPU::DeformableConvolutionOp>([&](VPU::DeformableConvolutionOp op) {
                return getSWInputTensorDistributionMode(op, strategy, operand);
            })
            .Case<VPU::DynamicDequantizeOp>([&](VPU::DynamicDequantizeOp dynamicDequantizeOp) {
                return getSWInputTensorDistributionMode(dynamicDequantizeOp, strategy, operand, inputType);
            })
            .Case<VPU::ReverseOp>([&](VPU::ReverseOp op) {
                return getSWInputTensorDistributionMode(op, strategy);
            })
            .Case<VPU::CumSumOp>([&](VPU::CumSumOp op) {
                return getSWInputTensorDistributionMode(op, strategy);
            })
            .Case<VPU::RMSOp>([&](VPU::RMSOp op) {
                return getSWInputTensorDistributionMode(op, strategy, operand);
            })
            .Case<VPU::RoPEOp>([&](VPU::RoPEOp op) {
                return getSWInputTensorDistributionMode(op, strategy, operand);
            })
            .Case<VPU::RandomUniformOp>([&](VPU::RandomUniformOp randomUniformOp) {
                return getSWInputTensorDistributionMode(randomUniformOp, strategy);
            })
            .Case<VPU::RollOp>([&](VPU::RollOp rollOp) {
                return getSWInputTensorDistributionMode(rollOp, strategy, operand);
            })
            .Case<VPU::TopKOp>([&](VPU::TopKOp op) {
                return getSWInputTensorDistributionMode(op, strategy, operand);
            })
            .Case<VPU::DynamicQuantizeOp>([&](VPU::DynamicQuantizeOp op) {
                return getSWInputTensorDistributionMode(op, strategy, operand);
            })
            .Case<VPU::SDPAOp>([&](VPU::SDPAOp op) {
                return getSWInputTensorDistributionMode(op, strategy, operand);
            })
            .Case<VPU::FlashSDPAOp>([&](VPU::FlashSDPAOp op) {
                return getSWInputTensorDistributionMode(op, strategy, operand);
            })
            .Case<VPU::YuvToRgbOp>([&](VPU::YuvToRgbOp op) {
                return getSWInputTensorDistributionMode(op, strategy, operand);
            })
            .Case<VPU::SDPAExtendedOp>([&](VPU::SDPAExtendedOp op) {
                return getSWInputTensorDistributionMode(op, strategy, operand);
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

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::MemPermuteOp mempermuteOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy) {
    SmallVector<int64_t> outputTensorNumTiles;
    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        outputTensorNumTiles = {1, 1, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        outputTensorNumTiles = {1, numClustersAvailableForCompilation, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        outputTensorNumTiles = {1, 1, numClustersAvailableForCompilation, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for mempermute ",
                   strategy);
    }

    // back infer for input tensor num tiles before permutation
    // outLogicDim -(map)-> outMemDim -(reversed Perm)-> inMemDim -(map)-> inLogicDim
    const auto memPerm = mempermuteOp.getMemPerm();
    const auto perm = DimsOrder::fromAffineMap(memPerm);
    const auto inType = mlir::cast<vpux::NDTypeInterface>(mempermuteOp.getInput().getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(mempermuteOp.getOutput().getType());
    const auto inOrder = inType.getDimsOrder();
    const auto outOrder = outType.getDimsOrder();
    SmallVector<int64_t> inputTensorNumTiles(outputTensorNumTiles.size(), 1);
    for (size_t outLogicInd = 0; outLogicInd < outputTensorNumTiles.size(); ++outLogicInd) {
        const auto outMemDim = outOrder.dimAt(outLogicInd);
        const auto inMemDim = perm.dimAt(outMemDim.ind());
        const auto inLogicDim = inOrder.dimAt(inMemDim.ind());
        inputTensorNumTiles[inLogicDim.ind()] = outputTensorNumTiles[outLogicInd];
    }
    return inputTensorNumTiles;
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::InterpolateOp interpolateOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(interpolateOp, strategy, operand);

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
                                                         MultiClusterStrategy strategy, mlir::Value operand) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(lstmSequenceOp, strategy, operand);

    if (distributionMode == DistributionMode::SEGMENTED) {
        const auto inputData = mlir::cast<NDTypeInterface>(lstmSequenceOp.getInputData().getType());
        if (strategy == MultiClusterStrategy::SplitOverBatch) {
            const auto batchSize = inputData.getShape()[Dims4D::Act::N];
            const auto numClusters = std::min(batchSize, numClustersAvailableForCompilation);
            return {numClusters, 1, 1, 1};
        } else if (strategy == MultiClusterStrategy::SplitOverKernel) {
            const auto numDirections = inputData.getShape()[Dims4D::Act::C];
            if (operand == lstmSequenceOp.getReccurenceWeights()) {
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
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(normalizeOpOp, strategy, operand);

    if (distributionMode == VPU::DistributionMode::DUPLICATED) {
        return SmallVector<int64_t>{1, 1, 1, 1};
    }

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    case VPU::MultiClusterStrategy::SplitOverKernel: {
        auto IC = inputType.getShape()[Dims4D::Act::C];
        int64_t numClustersToUseForLayer = std::min(numClustersAvailableForCompilation, IC);
        return SmallVector<int64_t>{1, numClustersToUseForLayer, 1, 1};
    }
    case VPU::MultiClusterStrategy::Clustering:
        return {1, 1, 1, 1};
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return SmallVector<int64_t>{1, 1, 1, numClustersAvailableForCompilation};
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
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(accumulateOp, strategy, operand);
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
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode =
            VPU::getSWInputTensorDistributionMode(dynamicDequantizeOp, strategy, operand, inputType);
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
                                                         VPU::MultiClusterStrategy strategy) {
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
                                                         VPU::MultiClusterStrategy strategy) {
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
                                                         VPU::MultiClusterStrategy strategy) {
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

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::GRUGatesOp op, int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    if (strategy == MultiClusterStrategy::SplitOverHeight) {
        if (operand == op.getBiases()) {
            return {1, 1, 1, 1};
        }
        return SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    }
    return {1, 1, 1, 1};
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::GatherOp op, int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    // GatherOp with 4D Input, Indices, and Output. The data structures must adhere to the following configurations:
    //      Input: [BatchDimsRange, DataBeforeAxisRange, IndicesRange, DataAfterAxisRange]
    //      Indices: [BatchDimsRange, IndicesRange, 1, 1]
    // If 'SplitOverKernel' and 'SplitOverWidth' are enabled:
    //      - The Input is split, maintaining consistency with the 'C' or 'W'
    //      - Indices remain intact
    // If 'SplitOverHeight' is enabled:
    //      - The Indices is split, specifically at the IndicesRange('C')
    //      - The Input remains intact
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, operand);

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

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::GatherNDOp gatherNDOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    // GatherOp with 4D Input, Indices, and Output. The data structures must adhere to the following configurations:
    //      Input Data: [fixed_dim, batch_dim_size, coord_dim_size, after_coord_dim_size]
    //      Indice:     [fixed_dim, batch_dim_size, indices_dim_size, coord_rank]
    // If 'SplitOverKernel' is enabled, the Input and Indices are split
    // If 'SplitOverHeight' is enabled, the Indices are split while the Input remains intact
    // If 'SplitOverWidth' is enabled, the Input is split while Indices remain intact
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(gatherNDOp, strategy, operand);

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return SmallVector<int64_t>{1, numClustersAvailableForCompilation, 1, 1};
    case VPU::MultiClusterStrategy::SplitOverHeight:
    case VPU::MultiClusterStrategy::SplitOverWidth: {
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    }
    case VPU::MultiClusterStrategy::Clustering:
        return {1, 1, 1, 1};
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::GatherElementsOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy) {
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

template <typename OpType>
SmallVector<int64_t> getSWInputTensorNumTilesImpl(OpType op, int64_t numClustersAvailableForCompilation,
                                                  VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy);
    if (distributionMode == VPU::DistributionMode::DUPLICATED) {
        return SmallVector<int64_t>{1, 1, 1, 1};
    }

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverBatch: {
        auto N = inputType.getShape()[Dims4D::Act::N];
        int64_t clusters = std::min(numClustersAvailableForCompilation, N);
        return SmallVector<int64_t>{clusters, 1, 1, 1};
    }
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
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles", strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::ReverseOp op, int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    return getSWInputTensorNumTilesImpl(op, numClustersAvailableForCompilation, strategy, inputType);
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::CumSumOp op, int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    return getSWInputTensorNumTilesImpl(op, numClustersAvailableForCompilation, strategy, inputType);
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::GridSampleOp op,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                                         vpux::NDTypeInterface inputType) {
    // GirdSample with 4D input, grid and output. The input always has the shape [N, C, H, W], grid has the shape [N,
    // H_out, W_out, 2], and output has the shape [N, C, H_out, W_out]. So for SplitOverHeight/SplitOverWidth, input
    // cannot be tiled over H/W, and for grid H and W is dim 1 and dim 2. And for SplitOverKernel, grid cannot be tiled,
    // input tiled as output.
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, operand);

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, numClustersAvailableForCompilation, 1, 1};
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, numClustersAvailableForCompilation, 1, 1};
    case VPU::MultiClusterStrategy::SplitOverBatch:
        return SmallVector<int64_t>{std::min(numClustersAvailableForCompilation, inputType.getShape()[Dims4D::Act::N]),
                                    1, 1, 1};
    case VPU::MultiClusterStrategy::Clustering:
        return SmallVector<int64_t>{1, 1, 1, 1};
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::DeformableConvolutionOp op,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    /* DeformableConvolutionOp with 4D input, offset, kernel, mask and
    output.
     Input always has the shape [N, C, H, W], offset has the shape [N,
     deformable_group * kernel_Y * kernel_X * 2, H, W], kernel has the
     shape [N, C, Y, X], mask has the shape [N, deformable_group * kernel_Y
     * kernel_X, H, W], and output has the shape [N, C, H, W].

     For 'SplitOverBatch' input, offset and mask can be tiled on N.
     For 'SplitOverHeight' or 'SplitOverWidth' input and kernel cannot be
     tiled, and offset and mask will be tiled as output. For
     'SplitOverKernel' only the kernel will be tiled on N. */

    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, operand);

    if (distributionMode != DistributionMode::SEGMENTED) {
        return {1, 1, 1, 1};
    }

    const auto inputDataType = mlir::cast<NDTypeInterface>(op.getInput().getType());
    const auto offsetDataType = mlir::cast<NDTypeInterface>(op.getOffset().getType());
    const auto kernelDataType = mlir::cast<NDTypeInterface>(op.getKernel().getType());

    const auto inputData = op.getInput();
    const auto offsetData = op.getOffset();
    const auto kernelData = op.getKernel();
    const auto maskData = op.getMask();

    if (strategy == MultiClusterStrategy::SplitOverBatch) {
        if (operand == inputData || operand == offsetData || operand == maskData) {
            const auto batchSize = inputDataType.getShape()[Dims4D::Act::N];
            const auto numClusters = std::min(batchSize, numClustersAvailableForCompilation);
            return {numClusters, 1, 1, 1};
        }
    } else if (strategy == MultiClusterStrategy::SplitOverHeight) {
        if (operand == offsetData || operand == maskData) {
            const auto heightSize = offsetDataType.getShape()[Dims4D::Act::H];
            const auto numClusters = std::min(heightSize, numClustersAvailableForCompilation);
            return {1, 1, numClusters, 1};
        }
    } else if (strategy == MultiClusterStrategy::SplitOverWidth) {
        if (operand == offsetData || operand == maskData) {
            const auto widthSize = offsetDataType.getShape()[Dims4D::Act::W];
            const auto numClusters = std::min(widthSize, numClustersAvailableForCompilation);
            return {1, 1, 1, numClusters};
        }
    } else if (strategy == MultiClusterStrategy::SplitOverKernel) {
        const auto batchSizeKernel = kernelDataType.getShape()[Dims4D::Act::N];
        const auto numClusters = std::min(batchSizeKernel, numClustersAvailableForCompilation);

        if (operand == kernelData) {
            return {numClusters, 1, 1, 1};
        }
    }

    return {1, 1, 1, 1};
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::RMSOp op, int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, operand);

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
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, operand);

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

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::SDPAOp op, int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, operand);
    if (distributionMode == VPU::DistributionMode::DUPLICATED) {
        return SmallVector<int64_t>{1, 1, 1, 1};
    }

    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering: {
        return SmallVector<int64_t>{1, 1, 1, 1};
    }
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
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::SDPAExtendedOp op,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, operand);
    if (distributionMode == VPU::DistributionMode::DUPLICATED) {
        return SmallVector<int64_t>{1, 1, 1, 1};
    }

    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering: {
        return SmallVector<int64_t>{1, 1, 1, 1};
    }
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
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::RandomUniformOp /*op*/,
                                                         int64_t /*numClustersAvailableForCompilation*/,
                                                         VPU::MultiClusterStrategy strategy) {
    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
    case VPU::MultiClusterStrategy::SplitOverWidth:
    case VPU::MultiClusterStrategy::SplitOverKernel:
    case VPU::MultiClusterStrategy::Clustering:
        return {1, 1, 1, 1};
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::RollOp op, int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, operand);

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
    case VPU::MultiClusterStrategy::SplitOverWidth: {
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, 1, numClustersAvailableForCompilation};
    }
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::TopKOp op, int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, operand);

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

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::DynamicQuantizeOp op,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, operand);
    if (distributionMode == VPU::DistributionMode::DUPLICATED) {
        return SmallVector<int64_t>{1, 1, 1, 1};
    }

    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        return {1, 1, 1, 1};
    case VPU::MultiClusterStrategy::SplitOverWidth: {
        return SmallVector<int64_t>{1, 1, 1, numClustersAvailableForCompilation};
    }
    case VPU::MultiClusterStrategy::SplitOverHeight: {
        return SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    }
    case VPU::MultiClusterStrategy::SplitOverKernel: {
        return SmallVector<int64_t>{1, numClustersAvailableForCompilation, 1, 1};
    }
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::FlashSDPAOp op,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, operand);
    if (distributionMode == VPU::DistributionMode::DUPLICATED) {
        return SmallVector<int64_t>{1, 1, 1, 1};
    }

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverKernel: {
        return SmallVector<int64_t>{1, numClustersAvailableForCompilation, 1, 1};
    }
    case VPU::MultiClusterStrategy::SplitOverHeight: {
        return SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    }
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::YuvToRgbOp op, int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand) {
    bool singlePlane = (op.getInput2() == nullptr);
    if (singlePlane && strategy != VPU::MultiClusterStrategy::Clustering) {
        VPUX_THROW("SinglePlane YuvToRgbOp is not supported for multi-cluster strategy {0}", strategy);
    }

    const auto distributionMode = VPU::getSWInputTensorDistributionMode(op, strategy, operand);
    if (distributionMode == VPU::DistributionMode::DUPLICATED) {
        return SmallVector<int64_t>{1, 1, 1, 1};
    }

    // We are always tiling based on the second input (UV plane in NV12 or U in i420)
    auto getTiledDimension = [&](Dim dimension) -> int64_t {
        auto input2Type = mlir::cast<vpux::NDTypeInterface>(op.getInput2().getType());
        return input2Type.getShape()[dimension];
    };

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverKernel: {
        int64_t tiledChannels = getTiledDimension(Dims4D::Act::C);
        int64_t clusters = std::min(numClustersAvailableForCompilation, tiledChannels);
        return SmallVector<int64_t>{1, clusters, 1, 1};
    }
    case VPU::MultiClusterStrategy::SplitOverHeight: {
        int64_t tiledHeight = getTiledDimension(Dims4D::Act::H);
        int64_t clusters = std::min(numClustersAvailableForCompilation, tiledHeight);
        return SmallVector<int64_t>{1, 1, clusters, 1};
    }
    case VPU::MultiClusterStrategy::Clustering: {
        return {1, 1, 1, 1};
    }
    default:
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                                         vpux::NDTypeInterface inputType) {
    return llvm::TypeSwitch<mlir::Operation*, SmallVector<int64_t>>(clusteredOp.getOperation())
            .Case<VPU::InterpolateOp>([&](VPU::InterpolateOp interpolateOp) {
                return getSWInputTensorNumTiles(interpolateOp, numClustersAvailableForCompilation, strategy, operand);
            })
            .Case<VPU::MultiplyOp, VPU::DivideOp, VPU::PowerOp, VPU::MaximumOp, VPU::MinimumOp, VPU::PReluOp,
                  VPU::GreaterOp, VPU::GreaterEqualOp, VPU::LessOp, VPU::EqualOp, VPU::NotEqualOp, VPU::LessEqualOp,
                  VPU::AndOp, VPU::SubtractOp, VPU::AddOp, VPU::FloorOp, VPU::CeilingOp, VPU::FakeQuantizeOp,
                  VPU::SelectOp, VPU::RoundOp, VPU::SinOp, VPU::CosOp, VPU::ExpOp, VPU::MishOp, VPU::NegativeOp,
                  VPU::LogicalNotOp, VPU::SoftPlusOp, VPU::BitwiseOrOp, VPU::BitwiseAndOp, VPU::BitwiseNotOp,
                  VPU::BitwiseXorOp>([&](mlir::Operation* eltwiseOp) {
                return getSWInputTensorNumTiles(eltwiseOp, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::AccumulateOp>([&](VPU::AccumulateOp accumulateOp) {
                return getSWInputTensorNumTiles(accumulateOp, numClustersAvailableForCompilation, strategy, operand,
                                                inputType);
            })
            .Case<VPU::DetectionOutputSortOp>([&](VPU::DetectionOutputSortOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy);
            })
            .Case<VPU::MatMulOp>([&](VPU::MatMulOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy);
            })
            .Case<VPU::MemPermuteOp>([&](VPU::MemPermuteOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy);
            })
            .Case<VPU::LSTMGatesOp>([&](VPU::LSTMGatesOp lstmGatesOp) {
                return getSWInputTensorNumTiles(lstmGatesOp, numClustersAvailableForCompilation, strategy);
            })
            .Case<VPU::GRUGatesOp>([&](VPU::GRUGatesOp gruGatesOp) {
                return getSWInputTensorNumTiles(gruGatesOp, numClustersAvailableForCompilation, strategy, operand);
            })
            .Case<VPU::LSTMSequenceOp>([&](VPU::LSTMSequenceOp lstmSequenceOp) {
                return getSWInputTensorNumTiles(lstmSequenceOp, numClustersAvailableForCompilation, strategy, operand);
            })
            .Case<VPU::MVN1NormalizeOp>([&](VPU::MVN1NormalizeOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, operand, inputType);
            })
            .Case<VPU::MVN6Op>([&](VPU::MVN6Op op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::GatherOp>([&](VPU::GatherOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, operand);
            })
            .Case<VPU::GatherNDOp>([&](VPU::GatherNDOp gatherNDOp) {
                return getSWInputTensorNumTiles(gatherNDOp, numClustersAvailableForCompilation, strategy, operand);
            })
            .Case<VPU::GridSampleOp>([&](VPU::GridSampleOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, operand, inputType);
            })
            .Case<VPU::DeformableConvolutionOp>([&](VPU::DeformableConvolutionOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, operand);
            })
            .Case<VPU::GatherElementsOp>([&](VPU::GatherElementsOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy);
            })
            .Case<VPU::DynamicDequantizeOp>([&](VPU::DynamicDequantizeOp dynamicDequantizeOp) {
                return getSWInputTensorNumTiles(dynamicDequantizeOp, numClustersAvailableForCompilation, strategy,
                                                operand, inputType);
            })
            .Case<VPU::RMSOp>([&](VPU::RMSOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, operand);
            })
            .Case<VPU::RoPEOp>([&](VPU::RoPEOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, operand);
            })
            .Case<VPU::RandomUniformOp>([&](VPU::RandomUniformOp randomUniformOp) {
                return getSWInputTensorNumTiles(randomUniformOp, numClustersAvailableForCompilation, strategy);
            })
            .Case<VPU::RollOp>([&](VPU::RollOp rollOp) {
                return getSWInputTensorNumTiles(rollOp, numClustersAvailableForCompilation, strategy, operand);
            })
            .Case<VPU::TopKOp>([&](VPU::TopKOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, operand);
            })
            .Case<VPU::ReverseOp>([&](VPU::ReverseOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy);
            })
            .Case<VPU::CumSumOp>([&](VPU::CumSumOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::SDPAOp>([&](VPU::SDPAOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, operand, inputType);
            })
            .Case<VPU::SDPAExtendedOp>([&](VPU::SDPAExtendedOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, operand, inputType);
            })
            .Case<VPU::DynamicQuantizeOp>([&](VPU::DynamicQuantizeOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, operand);
            })
            .Case<VPU::FlashSDPAOp>([&](VPU::FlashSDPAOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, operand);
            })
            .Case<VPU::YuvToRgbOp>([&](VPU::YuvToRgbOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, operand);
            })
            .Default([&](mlir::Operation*) {
                VPUX_THROW_UNLESS(clusteredOp->getOperands().size() == 1,
                                  "General method only support SW layer "
                                  "with one operand but got '{0}'",
                                  clusteredOp->getOperands().size());
                return getSWInputTensorNumTiles(clusteredOp, numClustersAvailableForCompilation, strategy);
            });
}
