//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/max_lstm_hidden_size_constant.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/shave_controls_dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LSTMSequenceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::LSTMSequenceOpAdaptor lstm(operands, attrs, prop);
    if (mlir::failed(lstm.verify(loc))) {
        return mlir::failure();
    }

    const auto inDataType = lstm.getInputData().getType();
    const auto inDataShape = mlir::cast<vpux::NDTypeInterface>(inDataType).getShape();

    const auto initialHiddenStateType = mlir::cast<vpux::NDTypeInterface>(lstm.getInitialHiddenState().getType());
    const auto initialHiddenStateShape = initialHiddenStateType.getShape();
    const auto elementType = initialHiddenStateType.getElementType();
    const auto tensorAttr = createTensorAttrFromType(initialHiddenStateType);

    const auto batchSize = initialHiddenStateShape[Dims4D::Act::N];
    const auto numDirections = initialHiddenStateShape[Dims4D::Act::C];
    const auto hiddenSize = initialHiddenStateShape.back();

    const auto lengthIndex = Dim(inDataShape.size() - 2);
    const auto sequenceLength = inDataShape[lengthIndex];

    const SmallVector<int64_t> outputHiddenValuesShape{batchSize, numDirections, sequenceLength, hiddenSize};

    auto outputHiddenValuesType = mlir::RankedTensorType::get(outputHiddenValuesShape, elementType, tensorAttr);
    const auto outputHiddenStateType = mlir::RankedTensorType::get(initialHiddenStateShape, elementType, tensorAttr);
    const auto outputCellStateType = mlir::RankedTensorType::get(initialHiddenStateShape, elementType, tensorAttr);

    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(inDataType)) {
        const auto inBounds = boundedType.getBounds();
        const auto outputHVRank = outputHiddenValuesShape.size();

        Bounds outHVBounds;
        outHVBounds.reserve(outputHVRank);
        for (size_t i = 0; i < outputHVRank; i++) {
            if (outputHiddenValuesShape[i] == mlir::ShapedType::kDynamic) {
                outHVBounds.push_back(inBounds[lengthIndex]);
            } else {
                outHVBounds.push_back(outputHiddenValuesShape[i]);
            }
        }

        auto boundedHiddenValuesType = Core::BoundedTensorType::get(outputHiddenValuesType, outHVBounds);
        inferredReturnTypes.push_back(boundedHiddenValuesType);

    } else {
        inferredReturnTypes.push_back(outputHiddenValuesType);
    }

    inferredReturnTypes.push_back(outputHiddenStateType);
    inferredReturnTypes.push_back(outputCellStateType);

    return mlir::success();
}

namespace {

static mlir::ModuleOp getModule(::mlir::OpBuilder& odsBuilder) {
    auto block = odsBuilder.getInsertionBlock();
    auto parentOp = block->getParentOp();
    while (parentOp && !llvm::isa<mlir::ModuleOp>(parentOp)) {
        parentOp = parentOp->getParentOp();
    }
    return llvm::cast<mlir::ModuleOp>(parentOp);
}

mlir::Value createSyncBuffer(mlir::OpBuilder& rewriter, ShapeRef shape) {
    const auto auxIndicesType = mlir::RankedTensorType::get(shape.raw(), getSInt32Type(rewriter.getContext()));
    return Const::createConst(rewriter,
                              appendLoc(mlir::UnknownLoc::get(rewriter.getContext()), "LSTMSequence_SyncBuffer"),
                              auxIndicesType, ArrayRef<int32_t>(0));
}

//
// Create a buffer that will contain all internal usage buffers for MatMull on DPU usage scope.
// Dpu usage buffers: [dpuInvariantSize + dpuVariantSize + dpuWeightTableSize + maxDpuStatsSize]
// lstmIntermediateMultiplicationBuffersize - is the outputs of MatMull between hidden and all 4 recurrence weights.
mlir::Value createIntermediateSumsBuffer(mlir::OpBuilder& rewriter, int64_t hiddenSize) {
    const auto module = getModule(rewriter);
    constexpr int32_t lstmNumberOfGates = 4;
    auto phaseSizeWithPadding = hiddenSize;
    const auto lstmIntermediateMultiplicationBuffersize =
            phaseSizeWithPadding * lstmNumberOfGates * sizeof(uint16_t);  // intermediate buffer size
    const auto dpuWeightTableSize = vpux::VPU::NCEInvariant::getWeightsTableSize(hiddenSize) * lstmNumberOfGates;

    int64_t size = dpuWeightTableSize.count() + lstmIntermediateMultiplicationBuffersize +
                   VPU::getDpuDebugDataSize(config::getArch(module)) +
                   VPU::getDPUVariantDataSize(config::getArch(module)) +
                   VPU::getDPUInvariantDataSize(config::getArch(module));

    size = size / sizeof(int32_t);  // int32_t type format

    auto tileOp = config::getTileExecutor(module);
    const auto numShavesPerTile = tileOp.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT).getCount();

    const auto shape = Shape{1, 1, numShavesPerTile, size};
    const auto auxIndicesType = mlir::RankedTensorType::get(shape.raw(), getSInt32Type(rewriter.getContext()));
    return Const::createConst(rewriter,
                              appendLoc(mlir::UnknownLoc::get(rewriter.getContext()), "LstmDpu_InternalSumsBuffer"),
                              auxIndicesType, ArrayRef<int32_t>(0));
}

bool isSupported(config::ArchKind arch, ShapeRef initialHiddenStateShape, bool useDpu) {
    auto maxHiddenSize = VPU::getMaxLstmSequenceHiddenSizeConstant(arch);

    // shave implementation allow reduced size. Bigger size can and are map on DPU.
    if (initialHiddenStateShape.back() > maxHiddenSize) {
        return false;
    }

    // shave asm implement just 32 element alignment hidden size. Except that, speed is low.
    int64_t alignmentRequired = 16;
    if (useDpu) {
        alignmentRequired = 32;
    }
    if (initialHiddenStateShape.back() % alignmentRequired != 0) {
        return false;
    }
    return true;
}

}  // namespace

void vpux::VPU::LSTMSequenceOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                      ::mlir::Value inputData, ::mlir::Value initialHiddenState,
                                      ::mlir::Value initialCellState, ::mlir::Value reccurenceWeights,
                                      ::mlir::Value biases, ::mlir::IntegerAttr sequenceLength,
                                      vpux::IE::RNNSequenceDirectionAttr direction, ::mlir::BoolAttr useDpu,
                                      vpux::VPU::MultiClusterStrategyAttr multiClusterStrategy) {
    const auto module = getModule(odsBuilder);
    auto tileOp = config::getTileExecutor(module);

    mlir::Value internalBuffer = nullptr;
    auto useDpuVal = useDpu ? useDpu.getValue() : false;
    if (useDpuVal) {
        const auto initialHiddenStateType = mlir::cast<vpux::NDTypeInterface>(initialHiddenState.getType());
        const auto initialHiddenStateShape = initialHiddenStateType.getShape();
        internalBuffer = createIntermediateSumsBuffer(odsBuilder, initialHiddenStateShape.back());
    } else {
        const auto numShavesPerTile = tileOp.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT).getCount();
        Shape shape{1, 1, 1, numShavesPerTile};
        internalBuffer = createSyncBuffer(odsBuilder, shape);
    }

    build(odsBuilder, odsState, inputData, initialHiddenState, initialCellState, reccurenceWeights, biases,
          internalBuffer, sequenceLength, direction, useDpu, multiClusterStrategy);
}

void vpux::VPU::LSTMSequenceOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                      ::mlir::Value inputData, ::mlir::Value initialHiddenState,
                                      ::mlir::Value initialCellState, ::mlir::Value reccurenceWeights,
                                      ::mlir::Value biases, ::mlir::IntegerAttr sequenceLength,
                                      vpux::IE::RNNSequenceDirectionAttr direction,
                                      vpux::VPU::MultiClusterStrategyAttr multiClusterStrategy) {
    const auto module = getModule(odsBuilder);
    auto useDpu = VPU::getShaveControlsDpu(config::getArch(module));
    // extra alignment condition should be meet in order to run on internal on dpu.
    useDpu = useDpu ? ::isSupported(config::getArch(module), getShape(initialHiddenState), useDpu) : useDpu;
    mlir::BoolAttr useDpuAttr(nullptr);
    useDpuAttr = useDpu ? mlir::BoolAttr::get(odsBuilder.getContext(), useDpu) : useDpuAttr;
    build(odsBuilder, odsState, inputData, initialHiddenState, initialCellState, reccurenceWeights, biases,
          sequenceLength, direction, useDpuAttr, multiClusterStrategy);
}

bool vpux::VPU::LSTMSequenceOp::isSupported(vpux::IE::LSTMSequenceOp op, bool useDpu) {
    if (op.getReccurenceWeights().getDefiningOp<Const::DeclareOp>() == nullptr) {
        return false;
    }
    return ::isSupported(config::getArch(op), getShape(op.getInitialHiddenState()), useDpu);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::LSTMSequenceOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto inputShape = getShape(getInputData());
    const auto numDirections = inputShape[Dims4D::Act::C];

    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel && numDirections == 2) {
        return true;
    }

    const auto batchSize = inputShape[Dims4D::Act::N];
    return strategy == VPU::MultiClusterStrategy::SplitOverBatch && batchSize > 1;
}

bool VPU::LSTMSequenceOp::isOperationSplitOverKernelCompatible(ShapeRef, ShapeRef, ShapeRef) {
    const auto numDirections = getShape(getInputData())[Dims4D::Act::C];
    return numDirections == 2;
}

bool VPU::LSTMSequenceOp::isOperationSplitOverBatchCompatible(ShapeRef) {
    const auto batchSize = getShape(getInputData())[Dims4D::Act::N];
    return batchSize > 1;
}

vpux::VPU::DistributionInfo vpux::VPU::LSTMSequenceOp::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::LSTMSequenceOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
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

bool vpux::VPU::LSTMSequenceOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::LSTMSequenceOp::supportCycleCostCalculation() {
    return false;
}
