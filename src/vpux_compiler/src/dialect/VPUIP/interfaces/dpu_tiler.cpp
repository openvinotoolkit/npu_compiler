//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/factors.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <numeric>
#include <set>

using namespace vpux;

namespace {

constexpr int64_t DEFAULT_ZTILE_VALUE = 16;
constexpr int64_t MIN_VALID_ZTILE_EXPONENT = 4;
constexpr int64_t MAX_VALID_ZTILE_EXPONENT = 8;

SmallVector<int64_t> getSplitsFromRange(int64_t maxSplitRange, int64_t maxLimit) {
    SmallVector<int64_t> splits;
    for (int64_t idx = 0; idx < std::log2(maxSplitRange); idx++) {
        auto powIdx = static_cast<int64_t>(std::pow(2, idx));
        auto splitCandidate = maxSplitRange / powIdx;
        if (maxSplitRange % powIdx == 0 && splitCandidate <= maxLimit) {
            splits.push_back(splitCandidate);
        }
    }
    return splits;
}

using MpeModeSize = std::pair<int64_t, int64_t>;

MpeModeSize getMpeModeSize(VPU::MPEMode mpeMode) {
    switch (mpeMode) {
    case VPU::MPEMode::MATRIX:
        return {4, 4};
    case VPU::MPEMode::VECTOR:
        return {1, 16};
    case VPU::MPEMode::VECTOR_FP16:
        return {1, 4};
    case VPU::MPEMode::CUBOID_16x16:
        return {16, 16};
    case VPU::MPEMode::CUBOID_8x16:
        return {8, 16};
    case VPU::MPEMode::CUBOID_4x16:
        return {4, 16};
    default:
        VPUX_THROW("Unsupported MPE mode {0}", mpeMode);
    }
}

/*
 * Split workloads on both dimension H and W evenly. Each workload will try to make shape value align with the mpe
 * mode on the splitting dimension. For example, output tensor is [1, 16, 12, 12],  mpe mode is 4x4, and we tried to
 * split workloads into 5x5 pieces. On dimension W, max width = 3 = 12/5 (width/splitNumber) doesn't align with mpe mode
 * value(4), so the max workload width will be changed to 4 (alignValUp(3,4)), and the split number will be 3 instead of
 * 5 then. So does for dimension H.
 *                            --------------------------------
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *                            --------------------------------
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|  --->   | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *                            --------------------------------
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 */

VPUIP::WorkloadSplit createWorkloadSplitOverHW(ShapeRef shape, int64_t widthFactor, int64_t heightFactor,
                                               VPU::MPEMode mpeMode) {
    const auto mpeModeSize = getMpeModeSize(mpeMode);

    auto width = shape[Dims4D::Act::W];
    auto height = shape[Dims4D::Act::H];

    auto maxWidth = divUp(width, widthFactor);
    maxWidth = alignValUp(maxWidth, mpeModeSize.second);
    auto actualWidthSplitsNum = divUp(width, maxWidth);
    auto maxHeight = divUp(height, heightFactor);
    maxHeight = alignValUp(maxHeight, mpeModeSize.first);
    auto actualHeightSplitsNum = divUp(height, maxHeight);

    VPUIP::WorkloadSplit split;

    auto remainedHeight = height;
    for (int64_t idx = 0; idx < actualHeightSplitsNum; idx++) {
        auto currentHeightStep = remainedHeight > maxHeight ? maxHeight : remainedHeight;

        auto remainedWidth = width;
        for (int64_t idy = 0; idy < actualWidthSplitsNum; idy++) {
            TileInfo outTile(shape);
            outTile.shape[Dims4D::Act::C] = shape[Dims4D::Act::C];
            outTile.shape[Dims4D::Act::H] = currentHeightStep;
            outTile.shape[Dims4D::Act::W] = remainedWidth > maxWidth ? maxWidth : remainedWidth;
            outTile.offsets[Dims4D::Act::H] = idx * maxHeight;
            outTile.offsets[Dims4D::Act::W] = idy * maxWidth;
            outTile.offsets[Dims4D::Act::C] = 0;
            outTile.axis[Dims4D::Act::H] = actualHeightSplitsNum;
            outTile.axis[Dims4D::Act::W] = actualWidthSplitsNum;

            remainedWidth -= outTile.shape[Dims4D::Act::W];

            split.emplace_back(std::move(outTile), mpeMode);
        }

        remainedHeight -= currentHeightStep;
    }

    return split;
}

}  // namespace

/// Helper function: split workload channel into supported channels
/// @Warning Numbers in supportedChannels should be reversed order from max to min
SmallVector<int64_t> vpux::splitWorkloadChannel(int64_t wlChannel, ArrayRef<int64_t> supportedChannels) {
    SmallVector<int64_t> newWorkloadChannels;
    for (auto supportedChannel : supportedChannels) {
        while (wlChannel >= supportedChannel) {
            newWorkloadChannels.push_back(supportedChannel);
            wlChannel -= supportedChannel;
        }
    }
    // Return empty list to tell user that an invalid wlChannel catched
    // let user to decide what behavior need to be applied
    if (wlChannel != 0) {
        Logger::global().error("Invalid remaining workload channel {0}", wlChannel);
        return {};
    }
    return newWorkloadChannels;
};

vpux::VPUIP::DpuTiler::DpuTiler(ShapeRef outShape, VPU::MPEMode mpeMode): _outShape(outShape.raw()), _mpeMode(mpeMode) {
    VPUX_THROW_WHEN(outShape.size() != 4 && outShape.size() != DimsGroups5D::Act::numDims,
                    "Only 4D and 5D shapes are supported, got '{0}'", outShape);
}

SmallVector<int64_t> vpux::VPUIP::DpuTiler::generateSplitNumberPool(int64_t numDPU, int64_t maxSplits) const {
    SmallVector<int64_t> validZTiles;

    // Note: refer the values from workload number pool implementation in presilicon setting by
    // ./blob/main/src/Controllers/WorkloadGen.py#L84
    // 2^4 equals to the CMX word size in bytes,  2^8 is an up bound to limit the number of splits
    for (int64_t i = MIN_VALID_ZTILE_EXPONENT; i < MAX_VALID_ZTILE_EXPONENT; ++i) {
        validZTiles.push_back(static_cast<int64_t>(std::pow(2, i)));
        validZTiles.push_back(validZTiles.back() + DEFAULT_ZTILE_VALUE);
    }

    SmallVector<int64_t> maxSplitsInZ;
    SmallVector<int64_t> maxSplitsInXY;
    const auto mpeModeSize = getMpeModeSize(_mpeMode);

    for (const auto& zTile : validZTiles) {
        maxSplitsInZ.push_back(static_cast<int64_t>(std::ceil(_outShape[Dims4D::Act::C] / static_cast<double>(zTile))));
    }

    maxSplitsInXY.push_back(
            static_cast<int64_t>(std::ceil(_outShape[Dims4D::Act::H] / static_cast<double>(mpeModeSize.first)) *
                                 std::ceil(_outShape[Dims4D::Act::W] / static_cast<double>(mpeModeSize.second))));

    auto maxZ = *std::max_element(maxSplitsInZ.begin(), maxSplitsInZ.end());
    auto maxXY = *std::max_element(maxSplitsInXY.begin(), maxSplitsInXY.end());

    maxSplits = std::min(maxSplits, std::max(maxZ, maxXY));
    VPUX_THROW_WHEN(maxSplits == 0, "Invalid max split number: {0}", maxSplits);

    std::set<int64_t> dpuMulSplits;
    for (auto i = numDPU; i < maxSplits + 1; i = i + numDPU) {
        dpuMulSplits.insert(static_cast<uint32_t>(i));
    }
    for (auto splitsZ : maxSplitsInZ) {
        auto zRanges = getSplitsFromRange(splitsZ, maxSplits);
        dpuMulSplits.insert(zRanges.begin(), zRanges.end());
    }
    for (auto splitsXY : maxSplitsInXY) {
        auto xyRanges = getSplitsFromRange(splitsXY, maxSplits);
        dpuMulSplits.insert(xyRanges.begin(), xyRanges.end());
    }
    dpuMulSplits.insert(1);
    return to_small_vector(dpuMulSplits);
}

void vpux::VPUIP::DpuTiler::tileOverH(int64_t numDPU, WorkloadSplitPool& splitPool) {
    const int64_t minTileSize = 1;

    const int64_t minTilesCount = 1;
    const int64_t maxTilesCount = numDPU;

    int64_t tilesCount = _outShape[Dims4D::Act::H] / minTileSize;
    tilesCount = std::min(std::max(tilesCount, minTilesCount), maxTilesCount);

    Shape nTilesOnDim(_outShape.size(), minTilesCount);
    nTilesOnDim[Dims4D::Act::H] = tilesCount;

    auto outTiles = fillDividedTiles(nTilesOnDim, _outShape);
    if (mlir::failed(outTiles)) {
        return;
    }

    VPUIP::WorkloadSplit split;
    for (auto& outTile : outTiles.value()) {
        split.emplace_back(std::move(outTile), _mpeMode);
    }

    splitPool.insert(std::move(split));
}

void vpux::VPUIP::DpuTiler::tileOverZ(int64_t splitNumber, WorkloadSplitPool& splitPool, bool requiresEqualZ) {
    VPUX_THROW_WHEN(splitNumber == 0, "Invalid split number: {0}", splitNumber);

    const auto C = _outShape[Dims4D::Act::C];
    auto maxChannelPerWL = divUp(C, splitNumber);
    maxChannelPerWL = alignValUp(maxChannelPerWL, DEFAULT_ZTILE_VALUE);
    if (maxChannelPerWL < DEFAULT_ZTILE_VALUE) {
        return;
    }
    if (requiresEqualZ && (maxChannelPerWL * splitNumber) != C) {
        return;
    }

    VPUIP::WorkloadSplit split;

    auto actualSplitNumber = divUp(C, maxChannelPerWL);
    auto remainedChannels = C;

    for (int64_t idx = 0; idx < actualSplitNumber; idx++) {
        TileInfo outTile(_outShape);
        outTile.shape[Dims4D::Act::C] = remainedChannels > maxChannelPerWL ? maxChannelPerWL : remainedChannels;
        outTile.offsets[Dims4D::Act::W] = 0;
        outTile.offsets[Dims4D::Act::H] = 0;
        outTile.offsets[Dims4D::Act::C] = idx * maxChannelPerWL;
        outTile.axis[Dims4D::Act::C] = actualSplitNumber;

        if (outTile.shape[Dims4D::Act::C] % DEFAULT_ZTILE_VALUE != 0) {
            return;
        }

        remainedChannels -= outTile.shape[Dims4D::Act::C];

        split.emplace_back(std::move(outTile), _mpeMode);
    }

    splitPool.insert(std::move(split));
}

void vpux::VPUIP::DpuTiler::tileOverHW(int64_t splitNumber, SplitDimension splitDimension,
                                       WorkloadSplitPool& splitPool) {
    VPUX_THROW_WHEN(splitNumber == 0, "Invalid split number: {0}", splitNumber);

    auto width = _outShape[Dims4D::Act::W];
    auto height = _outShape[Dims4D::Act::H];

    switch (splitDimension) {
    case SplitDimension::SPLIT_OVER_H: {
        splitPool.emplace(createWorkloadSplitOverHW(_outShape, 1, splitNumber, _mpeMode));
        break;
    }
    case SplitDimension::SPLIT_OVER_W: {
        splitPool.emplace(createWorkloadSplitOverHW(_outShape, splitNumber, 1, _mpeMode));
        break;
    }
    case SplitDimension::SPLIT_OVER_HW: {
        const auto factorList = getFactorsList(splitNumber);

        for (const auto factor : factorList) {
            // Map factor.first , factor.second -> width, height
            if (factor.first <= width && factor.second <= height) {
                splitPool.emplace(createWorkloadSplitOverHW(_outShape, factor.first, factor.second, _mpeMode));
            }

            // Map factor.second, factor.first -> width, height
            if (factor.second <= width && factor.first <= height) {
                splitPool.emplace(createWorkloadSplitOverHW(_outShape, factor.second, factor.first, _mpeMode));
            }
        }
        break;
    }
    default:
        VPUX_THROW("Unsupported split dimension {0}", splitDimension);
    }
}

void vpux::VPUIP::DpuTiler::tileOverHWMixedPrecision(WorkloadSplitPool& splitPool) {
    VPUX_THROW_WHEN(_mpeMode != VPU::MPEMode::VECTOR_FP16, "Invalid MPE mode, actual: {0}, expected: {1}", _mpeMode,
                    VPU::MPEMode::VECTOR_FP16);
    VPUIP::WorkloadSplit split;
    split.emplace_back(TileInfo(_outShape), _mpeMode);
    splitPool.insert(std::move(split));
}

VPUNN::Operation vpux::VPUIP::getOperationType(VPUIP::NCETaskType taskType) {
    switch (taskType) {
    case VPUIP::NCETaskType::CONV:
        // Here for a compressConv, we still map it to VPUNN CONVOLUTION for IC < 16
        return VPUNN::Operation::CONVOLUTION;
    case VPUIP::NCETaskType::DWCONV:
        return VPUNN::Operation::DW_CONVOLUTION;
    case VPUIP::NCETaskType::MAXPOOL:
        return VPUNN::Operation::MAXPOOL;
    case VPUIP::NCETaskType::AVEPOOL:
        return VPUNN::Operation::AVEPOOL;
    case VPUIP::NCETaskType::ELTWISE:
        return VPUNN::Operation::ELTWISE;
    case VPUIP::NCETaskType::REDUCEMEAN:
        return VPUNN::Operation::CONVOLUTION;
    case VPUIP::NCETaskType::REDUCESUM:
        return VPUNN::Operation::CONVOLUTION;
    // unsupported type for vpunn, use convolution as work around
    case VPUIP::NCETaskType::IDENTITY:
        return VPUNN::Operation::CONVOLUTION;
    default:
        VPUX_THROW("Unsupported operation type: '{0}'", taskType);
    }
}

StringLiteral vpux::VPUIP::stringifyEnum(SplitDimension splitDimension) {
    switch (splitDimension) {
    case SplitDimension::SPLIT_OVER_HW:
        return "SPLIT_OVER_HW";
    case SplitDimension::SPLIT_OVER_H:
        return "SPLIT_OVER_H";
    case SplitDimension::SPLIT_OVER_W:
        return "SPLIT_OVER_W";
    default:
        return "UNKNOWN";
    }
}
