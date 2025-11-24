//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include <set>
#include <tuple>

namespace vpux::VPUIP {
enum class NCETaskType : uint64_t;
}  // namespace vpux::VPUIP

namespace VPUNN {
enum class Operation;
}

namespace vpux {
namespace VPUIP {

struct SEPInfo {
    Shape sepTableShape;
    Shape sepActShape;
};

struct WorkloadCostParams {
    VPUIP::NCETaskType nceTaskType;
    mlir::Type inDataType;
    mlir::Type outDataType;
    std::optional<mlir::Type> weightsDataType = std::nullopt;
    DimsOrder inOrder;
    DimsOrder outOrder;
    config::ArchKind arch;
    Shape fullInputShape;
    Shape inputShape;
    Shape outputShape;
    PadInfo padInfo;
    int64_t numDPU;    // DPUs per tile
    int64_t numTiles;  // Store used CMX tiles, e.g., SOK may use partial nce clusters
    SmallVector<int64_t> kernelSize;
    SmallVector<int64_t> kernelStride;
    // Sparsity ratio calculation can refer the comments for getWeightsSparsityRatio()
    // The two items will pass to VPUNN for memory calculation
    bool isWeightsSparsityEnabled = false;
    float weightsSparsityRatio = 0.0;
    // SEP info when enabled
    std::optional<SEPInfo> sepInfo = std::nullopt;
    VPU::MultiClusterStrategy layerStrategy = VPU::MultiClusterStrategy::Clustering;
    VPU::PPEAttr ppeAttr = nullptr;
    bool isNcePermute = false;
    bool isNceCompressConv = false;
};

struct ShaveWorkloadCostParams {
    config::ArchKind arch;
    std::vector<mlir::Type> inDataTypes;
    std::vector<mlir::Type> outDataTypes;
    std::vector<Shape> inputShapes;
    std::vector<Shape> outputShapes;
    std::vector<DimsOrder> inOrders;
    std::vector<DimsOrder> outOrders;
    int64_t numSHV;    // SHAVEs per tile
    int64_t numTiles;  // Store used CMX tiles, e.g., SOK may use partial nce clusters
    VPU::MultiClusterStrategy layerStrategy = VPU::MultiClusterStrategy::Clustering;
};

enum class SplitDimension { SPLIT_OVER_H = 0, SPLIT_OVER_W = 1, SPLIT_OVER_HW = 2 };

StringLiteral stringifyEnum(SplitDimension splitDimension);

using WorkloadTile = std::tuple<TileInfo, VPU::MPEMode>;
using WorkloadSplit = SmallVector<WorkloadTile>;
using WorkloadSplitPool = std::set<WorkloadSplit>;

class DpuTiler final {
public:
    DpuTiler(ShapeRef outShape, VPU::MPEMode mpeMode);

public:
    SmallVector<int64_t> generateSplitNumberPool(int64_t numDPU, int64_t maxSplits) const;

    void tileOverH(int64_t numDPU, WorkloadSplitPool& splitPool);
    void tileOverZ(int64_t splitNumber, WorkloadSplitPool& splitPool, bool requiresEqualZ = false);
    void tileOverHW(int64_t splitNumber, SplitDimension splitDimension, WorkloadSplitPool& splitPool);
    void tileOverHWMixedPrecision(WorkloadSplitPool& splitPool);

private:
    Shape _outShape;
    VPU::MPEMode _mpeMode;
};

VPUNN::Operation getOperationType(VPUIP::NCETaskType taskType);

}  // namespace VPUIP

SmallVector<int64_t> splitWorkloadChannel(int64_t wlChannel, ArrayRef<int64_t> supportedChannels);

}  // namespace vpux
