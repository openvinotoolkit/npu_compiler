//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ReverseSequenceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ReverseSequenceOpAdaptor rev(operands, attrs, prop);
    if (mlir::failed(rev.verify(loc))) {
        return mlir::failure();
    }

    const auto dataType = mlir::cast<vpux::NDTypeInterface>(rev.getData().getType());
    const auto dataShape = dataType.getShape().raw();

    if (dataShape.size() < 2) {
        return errorAt(loc, "First input tensor's size should not be less than 2D. Got {0}D tensor", dataShape.size());
    }

    const auto seqShape = getShape(rev.getSeqLength());
    const auto dataDims = checked_cast<int64_t>(dataShape.size());
    const auto batchAxis = rev.getBatchAxis();

    if (batchAxis >= dataDims || batchAxis < -dataDims) {
        return errorAt(loc, "ReverseSequence Parameter batch axis {0} out of the tensor rank range [{1}, {2}].",
                       batchAxis, -dataDims, dataDims - 1);
    }

    const auto seqAxis = rev.getSeqAxis();

    if (seqAxis >= dataDims || seqAxis < -dataDims) {
        return errorAt(loc, "ReverseSequence Parameter sequence axis {0} out of the tensor rank range [{1}, {2}].",
                       seqAxis, -dataDims, dataDims - 1);
    }

    const auto batchAxisNorm = (batchAxis < 0) ? (batchAxis + dataDims) : batchAxis;
    const auto seqCheckAxis = (seqShape.size() == 1) ? Dim(0) : Dim(batchAxisNorm);
    if (static_cast<size_t>(seqCheckAxis.ind()) >= seqShape.size()) {
        return errorAt(loc, "Sequence-lengths axis check {0} exceeds sequence-lengths rank {1}", seqCheckAxis.ind(),
                       seqShape.size());
    }
    if (seqShape[seqCheckAxis] != dataShape[batchAxisNorm]) {
        return errorAt(loc, "Sequence lengths input size {0} is not equal to batch axis dimension of data input {1}",
                       seqShape[seqCheckAxis], dataShape[batchAxisNorm]);
    }

    const auto elementType = dataType.getElementType();
    if (!(elementType.isF16() || elementType.isF32() || elementType.isInteger(8))) {
        return errorAt(loc, "Reverse Sequence only support FP16, FP32, INT8 (I8/U8/SI8) data type");
    }

    auto outType = dataType.changeElemType(elementType);
    if (!mlir::isa<VPU::DistributedTensorType>(outType)) {
        outType = outType.changeShape(ShapeRef(dataShape));
    } else {
        outType = mlir::RankedTensorType::get(dataShape, dataType.getElementType(), createTensorAttrFromType(dataType));
    }

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::ReverseSequenceOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    const auto origSeqLengthShape = getShape(getSeqLength());
    const auto origBatchAxis = getBatchAxis();
    auto inTile = outputTile;
    TileInfo seqLengthTile(origSeqLengthShape);

    // if tiled axis is batch_axis, then seqLength should also be split along this axis;
    // if tiled axis is not batch_axis or seq_axis, then seqLength should not be split.
    if (outputTile.axis[Dim(origBatchAxis)] != 1) {
        seqLengthTile.shape[Dim(origBatchAxis)] = outputTile.shape[Dim(origBatchAxis)];
        seqLengthTile.offsets[Dim(origBatchAxis)] = outputTile.offsets[Dim(origBatchAxis)];
        seqLengthTile.axis[Dim(origBatchAxis)] = outputTile.axis[Dim(origBatchAxis)];
    }

    return TilingInfo{{std::move(inTile), std::move(seqLengthTile)}};
}

void vpux::VPU::ReverseSequenceOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::ReverseSequenceOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    const auto op = getOperation();
    const auto seq_axis = getSeqAxis();
    SmallVector<int64_t> axes = {seq_axis};
    SmallVector<int64_t> maxNumTiles = getMaxNumTilesWithAxesExclusion(op, axes);
    return getSWLayerTilingStrategy(op, tilingMode, log, maxNumTiles);
}

// Return a list with all dims that can be tiled, meaning the dims that are not in 'axes' list.
DimArr vpux::VPU::ReverseSequenceOp::getTileableDims() {
    const auto rank = mlir::cast<vpux::NDTypeInterface>(getData().getType()).getRank();
    VPUX_THROW_UNLESS(rank == 4, "Function valid only for 4D shape, got {0}D", rank);

    DimArr dims;
    const auto seq_axis = getSeqAxis();
    for (int64_t dimIdx = 0; dimIdx < rank; dimIdx++) {
        if (seq_axis != dimIdx) {
            dims.push_back(Dim(dimIdx));
        }
    }
    return dims;
}

//
// ClusteredOpInterface
//

bool vpux::VPU::ReverseSequenceOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto tileableDims = this->getTileableDims();

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverBatch) {
        return std::find(tileableDims.begin(), tileableDims.end(), Dims4D::Act::N) != tileableDims.end();
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return std::find(tileableDims.begin(), tileableDims.end(), Dims4D::Act::C) != tileableDims.end();
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return std::find(tileableDims.begin(), tileableDims.end(), Dims4D::Act::H) != tileableDims.end();
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
        return std::find(tileableDims.begin(), tileableDims.end(), Dims4D::Act::W) != tileableDims.end();
    }

    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::ReverseSequenceOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::ReverseSequenceOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3,
                      "ReverseSequenceOp requires 2 inputs and 1 output, but the number of buffers is {0} ",
                      buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::ReverseSequenceOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::ReverseSequenceOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::ReverseSequenceOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value data,
                                         ::mlir::Value seq_length, ::mlir::IntegerAttr seq_axis,
                                         ::mlir::IntegerAttr batch_axis) {
    build(builder, state, data, seq_length, seq_axis, batch_axis, {});
}
