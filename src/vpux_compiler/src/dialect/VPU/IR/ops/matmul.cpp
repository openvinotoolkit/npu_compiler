//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <algorithm>
#include <unordered_set>
#include <utility>

using namespace vpux;

mlir::LogicalResult vpux::VPU::MatMulOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MatMulOpAdaptor matMul(operands, attrs);
    if (mlir::failed(matMul.verify(loc))) {
        return mlir::failure();
    }

    const auto inType1 = mlir::cast<vpux::NDTypeInterface>(matMul.getInput1().getType());
    const auto inType2 = mlir::cast<vpux::NDTypeInterface>(matMul.getInput2().getType());
    const auto inShape1 = inType1.getShape();
    const auto inShape2 = inType2.getShape();

    const auto inRank1 = inShape1.size();
    const auto inRank2 = inShape2.size();
    const auto transA = matMul.getTransposeA();
    const auto transB = matMul.getTransposeB();

    // Rightmost two axes are row & col. Remaining left axes are batch
    constexpr int kRowColIdxRange = 2;

    SmallVector<int64_t> outShape;
    outShape.reserve(std::max(inRank1, inRank2));

    // Temporally transformed shapes
    auto inShape1Trans = to_small_vector(inShape1);
    auto inShape2Trans = to_small_vector(inShape2);
    std::reverse(inShape1Trans.begin(), inShape1Trans.end());
    std::reverse(inShape2Trans.begin(), inShape2Trans.end());

    // Apply transpose only when rank >= 2
    if (transA && (inRank1 > 1)) {
        std::swap(inShape1Trans[0], inShape1Trans[1]);
    }
    if (transB && (inRank2 > 1)) {
        std::swap(inShape2Trans[0], inShape2Trans[1]);
    }

    // Only use the dim when it is Mat
    if (inRank2 >= kRowColIdxRange) {
        outShape.push_back(inShape2Trans[0]);
    }
    if (inRank1 >= kRowColIdxRange) {
        outShape.push_back(inShape1Trans[1]);
    }

    // Process batch axes
    uint32_t idx1 = kRowColIdxRange;
    uint32_t idx2 = kRowColIdxRange;

    while (idx1 < inRank1 || idx2 < inRank2) {
        if (idx1 < inRank1 && idx2 < inRank2) {
            outShape.push_back(std::max(inShape1Trans[idx1], inShape2Trans[idx2]));
            ++idx1;
            ++idx2;
        } else if (idx2 >= inRank2) {
            outShape.push_back(inShape1Trans[idx1]);
            ++idx1;
        } else if (idx1 >= inRank1) {
            outShape.push_back(inShape2Trans[idx2]);
            ++idx2;
        }
    }
    std::reverse(std::begin(outShape), std::end(outShape));

    const auto outType =
            mlir::RankedTensorType::get(outShape, inType1.getElementType(), createTensorAttrFromType(inType1));
    inferredReturnTypes.push_back(outType);
    return mlir::success();
}

namespace {

bool isSupported(config::ArchKind arch, ShapeRef input1Shape, ShapeRef input2Shape, bool transposeA = false,
                 bool transposeB = false) {
    if (arch == config::ArchKind::NPU37XX) {  // All platforms except MTL can use this
        return false;
    }

    if (input1Shape.size() < 2 || input2Shape.size() < 2) {
        return false;
    }

    SmallVector<int64_t> input1HeightWidth{input1Shape[Dim(input1Shape.size() - 2)],
                                           input1Shape[Dim(input1Shape.size() - 1)]};
    SmallVector<int64_t> input2HeightWidth{input2Shape[Dim(input2Shape.size() - 2)],
                                           input2Shape[Dim(input2Shape.size() - 1)]};

    if (transposeA) {
        std::swap(input1HeightWidth[0], input1HeightWidth[1]);
    }

    if (transposeB) {
        std::swap(input2HeightWidth[0], input2HeightWidth[1]);
    }

    // The list of specialized configurations implemented in ASM.
    // The list is empty for now, as the operation has efficient DPU support.
    // If any patterns of more efficient Shave operations are identified then they have to be added here.
    static const std::unordered_set<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
            supportedHeightWidthConfigs{};

    return supportedHeightWidthConfigs.find({input1HeightWidth, input2HeightWidth}) !=
           supportedHeightWidthConfigs.end();
}

}  // namespace

//
// isSupported
//

bool vpux::VPU::MatMulOp::isSupported(vpux::IE::MatMulOp matmulOp) {
    return ::isSupported(config::getArch(matmulOp), getShape(matmulOp.getInput1()), getShape(matmulOp.getInput2()),
                         matmulOp.getTransposeA(), matmulOp.getTransposeB());
}

//
// verify
//

mlir::LogicalResult vpux::VPU::MatMulOp::verify() {
    if (getTransposeA() || getTransposeB()) {
        return mlir::failure();
    }

    const auto operation = getOperation();
    const auto arch = config::getArch(operation);
    if (::isSupported(arch, getShape(getInput1()), getShape(getInput2()))) {
        return mlir::success();
    }

    return mlir::failure();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::MatMulOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    const auto input1Shape = getShape(getInput1());
    const auto input2Shape = getShape(getInput2());
    TileInfo input1Tile(outputTile);
    TileInfo input2Tile(outputTile);

    input1Tile.shape[Dim(input1Tile.shape.size() - 2)] = input1Shape[Dim(input1Shape.size() - 2)];
    input1Tile.shape[Dim(input1Tile.shape.size() - 1)] = input1Shape[Dim(input1Shape.size() - 1)];

    input2Tile.shape[Dim(input2Tile.shape.size() - 2)] = input2Shape[Dim(input2Shape.size() - 2)];
    input2Tile.shape[Dim(input2Tile.shape.size() - 1)] = input2Shape[Dim(input2Shape.size() - 1)];

    return InputTiling{{std::move(input1Tile), std::move(input2Tile)}};
}

void vpux::VPU::MatMulOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::MatMulOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(getOperation(), tilingMode, std::move(log));
}

SmallVector<int64_t> vpux::VPU::MatMulOp::getMaxNumTiles() {
    auto op = this->getOperation();
    SmallVector<int64_t> maxNumTiles;

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto outputRank = outputType.getShape().size();

    SmallVector<int64_t> axes{checked_cast<int64_t>(outputRank - 2), checked_cast<int64_t>(outputRank - 1)};
    maxNumTiles = getMaxNumTilesWithAxesExclusion(op, axes);

    return vpux::getMaxNumTiles(op, false, false, maxNumTiles);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::MatMulOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t numTiles) {
    const auto input1Shape = mlir::cast<vpux::NDTypeInterface>(getInput1().getType()).getShape();
    const auto input2Shape = mlir::cast<vpux::NDTypeInterface>(getInput2().getType()).getShape();

    return strategy == VPU::MultiClusterStrategy::SplitOverKernel &&
           input1Shape[Dims4D::Act::C] >= checked_cast<int64_t>(numTiles) &&
           input2Shape[Dims4D::Act::C] >= checked_cast<int64_t>(numTiles);
}

vpux::VPU::DistributionInfo vpux::VPU::MatMulOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

bool vpux::VPU::MatMulOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3, "MatMulOp requires 2 inputs and 1 output, but the number of buffers is {0}",
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

bool vpux::VPU::MatMulOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::MatMulOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::MatMulOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input1,
                                ::mlir::Value input2, ::mlir::UnitAttr transpose_a, ::mlir::UnitAttr transpose_b,
                                /*optional*/ vpux::IE::PostOpAttr post_op) {
    build(builder, state, input1, input2, transpose_a, transpose_b, post_op, {});
}
