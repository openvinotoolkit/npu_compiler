//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

namespace vpux {
namespace VPU {

template <typename NCEOp>
SmallVector<vpux::NDTypeInterface> getTileTypes(NCEOp origOp, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles) {
    auto siblingsAnalysis = SiblingOpsAnalysis(origOp.getOperation());
    auto tileDistributions = getTileDistributions(origOp, siblingsAnalysis, outTile, inputTiles);
    SmallVector<vpux::NDTypeInterface> tileTypes;
    for (auto tileDistribution : tileDistributions) {
        auto tileType = getDistributedTypeFromDistributionMap(tileDistribution.first, tileDistribution.second);
        tileTypes.push_back(tileType);
    }

    return tileTypes;
}

// Convolution

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::ConvolutionOp origOp, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles = std::nullopt);

// GroupConvolution

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::GroupConvolutionOp origOp, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles = std::nullopt);

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::DequantizeOp origOp, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles = std::nullopt);

SmallVector<vpux::NDTypeInterface> getTileTypes(mlir::Operation* op, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles = std::nullopt);

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        mlir::Operation* op, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles = std::nullopt);

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        mlir::Operation* op, const TileInfo& outTile, const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMXForWeight(VPU::ConvolutionOp convOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMXForWeight(VPU::NCEConvolutionOp convOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::ConvolutionOp convOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::NCEConvolutionOp convOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::NCEConvolutionOp convOp, const SmallVector<NDTypeInterface>& types);

Byte getRequiredCMX(VPU::NCECompressConvolutionOp convOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::NCECompressConvolutionOp convOp, const SmallVector<NDTypeInterface>& types);

Byte getRequiredCMXForWeight(VPU::NCECompressConvolutionOp convOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMXForWeight(VPU::GroupConvolutionOp gConvOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMXForWeight(VPU::NCEDepthConvolutionOp gConvOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::GroupConvolutionOp gConvOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::NCEDepthConvolutionOp dConvOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::NCEDepthConvolutionOp dConvOp, const SmallVector<NDTypeInterface>& types);

Byte getRequiredCMX(VPU::NCEPermuteOp pqOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::NCEPermuteOp pqOp, const SmallVector<NDTypeInterface>& types);

Byte getRequiredCMXForWeight(VPU::NCEPermuteOp op, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMXForWeight(VPU::MaxPoolOp op, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMXForWeight(VPU::NCEMaxPoolOp op, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMXForWeight(VPU::NCEAveragePoolOp op, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::MaxPoolOp poolOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::NCEMaxPoolOp poolOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::NCEMaxPoolOp poolOp, const SmallVector<NDTypeInterface>& types);

Byte getRequiredCMX(VPU::NCEAveragePoolOp poolOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::NCEAveragePoolOp poolOp, const SmallVector<NDTypeInterface>& types);

Byte getEltwiseRequiredCMX(mlir::Operation* op, const vpux::TileInfo& tiling,
                           const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::AddOp op, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMXForWeight(VPU::AddOp op, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles = std::nullopt);
Byte getRequiredCMX(VPU::MultiplyOp op, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::MultiplyOp op, const SmallVector<NDTypeInterface>& types);

Byte getRequiredCMXForWeight(VPU::MultiplyOp op, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::SubtractOp op, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMXForWeight(VPU::SubtractOp op, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::NCEEltwiseOp op, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(VPU::NCEEltwiseOp op, const SmallVector<NDTypeInterface>& types);

Byte getRequiredCMX(VPU::NCEPermuteOp pqOp, const SmallVector<NDTypeInterface>& tileTypes);

Byte getRequiredCMX(VPU::MultiplyOp op, const SmallVector<NDTypeInterface>& tileTypes);

Byte getRequiredCMXForWeight(VPU::NCEEltwiseOp op, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMXForWeight(mlir::Operation* op, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMX(mlir::Operation* op, const vpux::TileInfo& tiling, Logger log,
                    const std::optional<InputTiling>& inputTiles = std::nullopt);

Byte getRequiredCMXSize(ArrayRef<vpux::NDTypeInterface> operands);

Byte getRequiredCMXSize(ArrayRef<std::pair<NDTypeInterface, TensorDistributionMap>> operands);

Byte getRequiredCMXSizeForNCEOps(ArrayRef<vpux::NDTypeInterface> operands, int64_t numChannels);

Byte getRequiredCMXSizeForNCEOps(ArrayRef<std::pair<NDTypeInterface, TensorDistributionMap>> operands,
                                 int64_t numChannels);

Byte getRequiredCMXSizeForDefaultOps(mlir::Operation* op);

Byte getRequiredCMX(mlir::Operation* op, const SmallVector<NDTypeInterface>& types);

OutputTiling getUniqueShapeTilingCandidates(mlir::Operation* op, const OutputTiling& origTiles, Logger log);

bool canSWLayerBeEvenlyUnrolled(mlir::Operation* op, const OutputTiling& tiles, Dim targetDim, Logger);
struct TileShapeCompare {
    bool operator()(const TileInfo& tile1, const TileInfo& tile2) const {
        return tile1.shape < tile2.shape;
    }
};

bool isDivisibleTile(mlir::Operation* op, ShapeRef tileAxis, Dim tileDim);

}  // namespace VPU
}  // namespace vpux
