//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinOps.h>
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

//
// inferReturnTypes
//

mlir::LogicalResult VPU::DetectionOutputSortOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputSortOpAdaptor sort(operands, attrs, prop);
    if (mlir::failed(sort.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(sort.getConfidence().getType());
    const auto inputShape = inputType.getShape();

    const auto numClasses = inputShape[Dims4D::Act::H];
    const auto numPriors = inputShape[Dims4D::Act::W];

    const auto outConfidenceShape = SmallVector<int64_t>{1, 1, numClasses, numPriors};
    const auto outIndicesShape = SmallVector<int64_t>{1, 1, numClasses, numPriors};
    const auto outSizesShape = SmallVector<int64_t>{1, 1, numClasses, 1};

    const auto outConfidenceType = mlir::RankedTensorType::get(outConfidenceShape, inputType.getElementType(),
                                                               createTensorAttrFromType(inputType));
    const auto outIndicesElemType = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
    const auto outIndicesType = mlir::RankedTensorType::get(outIndicesShape, outIndicesElemType);
    const auto outSizesType = mlir::RankedTensorType::get(outSizesShape, outIndicesElemType);

    inferredReturnTypes.push_back(outConfidenceType);
    inferredReturnTypes.push_back(outIndicesType);
    inferredReturnTypes.push_back(outSizesType);

    return mlir::success();
}

mlir::Value createIndicesAuxiliaryBuffer(mlir::OpBuilder& rewriter, ShapeRef shape) {
    VPUX_THROW_UNLESS(shape.size() == 4, "Class predictions tensor must be 4D");
    auto auxIndicesContent = std::vector<int32_t>(shape.totalSize());
    const auto width = shape[Dims4D::Act::W];
    const auto height = shape[Dims4D::Act::H];
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            auxIndicesContent[h * width + w] = w;
        }
    }

    const auto auxIndicesType = mlir::RankedTensorType::get(shape.raw(), getSInt32Type(rewriter.getContext()));
    return Const::createConst(rewriter, appendLoc(mlir::UnknownLoc::get(rewriter.getContext()), "sort_IndicesBuffer"),
                              auxIndicesType, ArrayRef(auxIndicesContent));
}

mlir::Value createSortingAuxiliaryBuffer(mlir::OpBuilder& rewriter, ShapeRef shape) {
    const auto auxIndicesType = mlir::RankedTensorType::get(shape.raw(), getSInt32Type(rewriter.getContext()));
    return Const::createConst(rewriter, appendLoc(mlir::UnknownLoc::get(rewriter.getContext()), "sort_SortingBuffer"),
                              auxIndicesType, ArrayRef<int32_t>(0));
}

static mlir::ModuleOp getModule(::mlir::OpBuilder& odsBuilder) {
    auto block = odsBuilder.getInsertionBlock();
    auto parentOp = block->getParentOp();
    while (parentOp && !llvm::isa<mlir::ModuleOp>(parentOp)) {
        parentOp = parentOp->getParentOp();
    }
    return llvm::cast<mlir::ModuleOp>(parentOp);
}

void vpux::VPU::DetectionOutputSortOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                             ::mlir::Value classPredictions, ::mlir::FloatAttr confidenceThreshold,
                                             ::mlir::IntegerAttr topK) {
    auto indicesBuffer = createIndicesAuxiliaryBuffer(odsBuilder, getShape(classPredictions));
    auto module = getModule(odsBuilder);
    auto numShaves = config::getTotalNumOfEngines(module, VPU::ExecutorKind::SHAVE_ACT);

    // 4 buffers of size 256 elements are required for counting sort
    // tensor has SEGMENTED distribution mode
    // multiply the buffer numShaves times to provide unique buffer for each shave
    Shape shape{1, 1, 4 * numShaves, 256};
    auto sortingBuffer = createSortingAuxiliaryBuffer(odsBuilder, shape);

    build(odsBuilder, odsState, classPredictions, indicesBuffer, sortingBuffer, confidenceThreshold, topK, nullptr);
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::DetectionOutputSortOp::backInferTileInfo(const vpux::TileInfo& firstOutputTile,
                                                                vpux::Logger /*log*/) {
    auto numShaves = config::getTotalNumOfEngines(getOperation(), VPU::ExecutorKind::SHAVE_ACT);
    return DetectionOutputSortOpInputTiling(firstOutputTile, numShaves);
}

void vpux::VPU::DetectionOutputSortOp::adjustAttrs(const TilingInfo&, const TileInfo&) {
    return;
}

mlir::FailureOr<OutputTiling> vpux::VPU::DetectionOutputSortOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

OutputTiling vpux::VPU::DetectionOutputSortOp::getOutputTiling(const vpux::TileInfo& firstOutputTile,
                                                               vpux::Logger /*log*/) {
    return DetectionOutputSortOpOutputTiling(firstOutputTile);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::DetectionOutputSortOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::SplitOverHeight;
}

vpux::VPU::DistributionInfo vpux::VPU::DetectionOutputSortOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

bool VPU::DetectionOutputSortOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& outputTile) {
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto tileOp = config::getTileExecutor(moduleOp);

    auto outputShape = ShapeRef(outputTile.shape);
    if (outputShape == ShapeRef()) {
        outputShape = getShape(getOutConfidence());
    }
    auto height = outputShape[Dims4D::Act::H];

    return height >= tileOp.getCount();
}

bool VPU::DetectionOutputSortOp::isOperationSplitOverWidthCompatible(ShapeRef, ShapeRef, ShapeRef) {
    return false;
}

bool VPU::DetectionOutputSortOp::isOperationSplitOverKernelCompatible(ShapeRef, ShapeRef, ShapeRef) {
    return false;
}

//
// SWOpInterface
//

bool vpux::VPU::DetectionOutputSortOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
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

bool vpux::VPU::DetectionOutputSortOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::DetectionOutputSortOp::supportCycleCostCalculation() {
    return false;
}
