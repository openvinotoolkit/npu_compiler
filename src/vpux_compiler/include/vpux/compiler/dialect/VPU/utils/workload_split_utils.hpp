//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

#include <vpu/layer_split_info.h>

namespace VPUNN {
class VPUCostModel;
}  // namespace VPUNN

namespace vpux::VPU {
/**
 * Generic method to split workload with heuristic MPE mode
 * Split pattern is decided by VPUNN L1 API cost
 */
mlir::LogicalResult genericNCEWorkloadSplit(VPU::NCEOpInterface nceOp, mlir::PatternRewriter& rewriter,
                                            config::ArchKind arch, int64_t numDPU,
                                            std::shared_ptr<VPUNN::VPUCostModel> costModel, Logger log);

/**
 * Check if the operation is supported by pre-split VPUNN API
 * Special cases still use generic VPUNN API because of inaccurate cost
 */
bool isSupportedPreSplitNCEOp(VPU::NCEOpInterface nceOp);

/**
 * Split nceOp onto workloads inplace according to the splitInfo from VPUNN
 * Used when pre-split is supported
 */
void splitWorkloadsWithInfo(VPU::NCEOpInterface nceOp, mlir::OpBuilder& builder, const VPUNN::LayerSplitInfo& splitInfo,
                            Logger log);

VPU::DistributedTensorType getDistributedTensor(const mlir::Value value);

/**
 * Map distributedType mode to VPUNN:ISIStrategy and outputWriteTiles
 *     DistributionMode         ISIStrategy      OutputWriteTiles
 * SEGMENTED|MULTICASTED        CLUSTERING          numTiles
 * SEGMENTED|DUPLICATED         SPLIT_OVER_K        numTiles
 *      else                    CLUSTERING              1
 */
template <typename TensorType, typename = std::enable_if_t<std::is_same_v<VPU::DistributedTensorType, TensorType> ||
                                                           std::is_same_v<VPUIP::DistributedBufferType, TensorType>>>
VPUNN::ISIStrategy getISIStrategyForType(TensorType type, unsigned int& outputWriteTiles) {
    const auto distributionAttr = type.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();
    auto numClusters = distributionAttr.getNumClusters().getInt();

    if (mode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED)) {
        outputWriteTiles = numClusters;
        return VPUNN::ISIStrategy::CLUSTERING;
    }
    if (mode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
        outputWriteTiles = numClusters;
        return VPUNN::ISIStrategy::SPLIT_OVER_K;
    }
    return VPUNN::ISIStrategy::CLUSTERING;
}
}  // namespace vpux::VPU
