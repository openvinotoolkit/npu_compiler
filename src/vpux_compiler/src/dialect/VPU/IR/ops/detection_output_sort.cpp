//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinOps.h>
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

SmallVector<mlir::Type> getAuxiliaryBufferTypes(mlir::ModuleOp moduleOp, mlir::Value confidence,
                                                std::vector<int32_t>& indicesBufferData) {
    const auto confidenceShape = getShape(confidence);
    VPUX_THROW_UNLESS(confidenceShape.size() == 4, "Class predictions tensor must be 4D");
    indicesBufferData.clear();
    indicesBufferData.resize(confidenceShape.totalSize());
    const auto width = confidenceShape[Dims4D::Act::W];
    const auto height = confidenceShape[Dims4D::Act::H];
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            indicesBufferData[h * width + w] = w;
        }
    }
    const auto indicesBufferType =
            mlir::RankedTensorType::get(confidenceShape.raw(), getSInt32Type(confidence.getContext()));

    // Four buffers of 256 elements are required for counting sort
    // tensor has SEGMENTED distribution mode
    // multiply the buffer numShaves times to provide unique buffer for each shave
    auto numShaves = config::getTotalNumOfEngines(moduleOp, VPU::ExecutorKind::SHAVE_ACT);
    Shape sortingBufferShape{1, 1, 4 * numShaves, 256};
    const auto sortingBufferType =
            mlir::RankedTensorType::get(sortingBufferShape.raw(), getSInt32Type(confidence.getContext()));

    return {indicesBufferType, sortingBufferType};
}

mlir::Value createIndicesBufferConstant(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type indicesBufferType,
                                        ArrayRef<int32_t> indicesBufferData) {
    return Const::createConst(builder, appendLoc(loc, "sort_IndicesBuffer"),
                              mlir::cast<mlir::RankedTensorType>(indicesBufferType), indicesBufferData);
}

void vpux::VPU::DetectionOutputSortOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                             mlir::Value confidence, mlir::FloatAttr confidenceThreshold,
                                             mlir::IntegerAttr topK) {
    auto block = odsBuilder.getInsertionBlock();
    const auto moduleOp = getModuleOp(block->getParentOp());

    std::vector<int32_t> indicesBufferData;
    auto auxBufferTypes = getAuxiliaryBufferTypes(moduleOp, confidence, indicesBufferData);
    VPUX_THROW_WHEN(auxBufferTypes.size() != 2, "Expected 2 auxiliary buffer types, got {0}", auxBufferTypes.size());
    auto indicesBuffer =
            createIndicesBufferConstant(odsBuilder, odsState.location, auxBufferTypes[0], indicesBufferData);
    auto sortingBuffer = VPU::createAuxiliaryBuffer(odsBuilder, odsState.location, auxBufferTypes[1]);

    build(odsBuilder, odsState, confidence, indicesBuffer, sortingBuffer, confidenceThreshold, topK, nullptr);
}

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

llvm::LogicalResult VPU::DetectionOutputSortOp::verify() {
    const auto moduleOp = getModuleOp(getOperation()->getParentOp());
    // The size of the auxiliary buffers depend on the number of available SHAVE executors
    // In case the IR is not populated with this information, skip the verification of the buffer sizes
    auto tileOp = config::getTileExecutor(moduleOp);
    if (tileOp == nullptr) {
        return mlir::success();
    }

    std::vector<int32_t> indicesBufferData;
    auto expectedAuxBuffTypes = getAuxiliaryBufferTypes(moduleOp, getConfidence(), indicesBufferData);
    if (expectedAuxBuffTypes.size() != 2) {
        return errorAt(getOperation(), "Expected two reference auxiliary buffer types, but got {0}",
                       expectedAuxBuffTypes.size());
    }
    auto loc = getOperation()->getLoc();
    if (mlir::failed(VPU::compareTypes(loc, getIndicesBuffer().getType(), expectedAuxBuffTypes[0]))) {
        return errorAt(getOperation(), "Invalid indices auxiliary buffer");
    }
    if (mlir::failed(VPU::compareTypes(loc, getSortingBuffer().getType(), expectedAuxBuffTypes[1]))) {
        return errorAt(getOperation(), "Invalid sorting auxiliary buffer");
    }
    return mlir::success();
}

SmallVector<mlir::Value> VPU::DetectionOutputSortOp::getAuxiliaryBuffers() {
    return {getIndicesBuffer(), getSortingBuffer()};
}
