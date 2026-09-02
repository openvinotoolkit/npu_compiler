//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/weight_table_offset_utils.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

#include <set>

namespace vpux::VPUIP {

int64_t DataPointerTableWtOffsetBuilder::getAlignment(int64_t zSize) const {
    return static_cast<int64_t>(VPU::NCESparsity::NewWeightsTableFormatMapper::getNewPointerTableAlignmentForWorkload(
            static_cast<int32_t>(zSize)));
}

int64_t ZeroPointTableWtOffsetBuilder::getAlignment(int64_t zSize) const {
    return static_cast<int64_t>(VPU::NCESparsity::NewWeightsTableFormatMapper::getZeroPointTableAlignmentForWorkload(
            _is4bit, static_cast<int32_t>(zSize)));
}

int64_t DisabledWtOffsetBuilder::getAlignment(int64_t) const {
    return 0;
}

std::unique_ptr<WtOffsetBuilder> WtOffsetBuilder::create(NCEClusterTaskOp nceOp, bool hasMultipleVariants) {
    if (!hasMultipleVariants) {
        return std::make_unique<DisabledWtOffsetBuilder>();
    }

    if (nceOp.getWeightTableDataPtr() != nullptr) {
        return std::make_unique<DataPointerTableWtOffsetBuilder>();
    }

    const auto weightZeroPoints = nceOp.getWeightZeroPoints();
    const auto weights = nceOp.getWeights();
    if (weightZeroPoints == nullptr || weights == nullptr) {
        return std::make_unique<DisabledWtOffsetBuilder>();
    }

    const auto weightsElementType = mlir::cast<NDTypeInterface>(weights.getType()).getElementType();
    const auto quantType = mlir::dyn_cast<mlir::quant::QuantizedType>(weightsElementType);
    if (quantType == nullptr) {
        return std::make_unique<DisabledWtOffsetBuilder>();
    }

    auto storageType = quantType.getStorageType();
    // Unwrap QuantileType to reach the inner integer storage type.
    if (const auto quantileType = mlir::dyn_cast<type::QuantileType>(storageType)) {
        storageType = quantileType.getStorageType();
    }

    const auto intStorageType = mlir::dyn_cast<mlir::IntegerType>(storageType);
    if (intStorageType == nullptr) {
        return std::make_unique<DisabledWtOffsetBuilder>();
    }

    return std::make_unique<ZeroPointTableWtOffsetBuilder>(intStorageType.getWidth() == 4);
}

std::unique_ptr<WtOffsetBuilder> WtOffsetBuilder::create(NCEClusterTaskOp nceOp, mlir::Region& workloads) {
    // More than one variant maps to the same cluster when a cluster id repeats across workloads.
    std::set<int64_t> seenClusterIds;
    for (auto dpuTaskOp : workloads.getOps<VPU::DPUWorkloadOp>()) {
        if (!seenClusterIds.insert(dpuTaskOp.getClusterIdAttr() ? dpuTaskOp.getClusterIdAttr().getInt() : 0).second) {
            return create(nceOp, /*hasMultipleVariants=*/true);
        }
    }

    return create(nceOp, /*hasMultipleVariants=*/false);
}

void WtOffsetBuilder::maybeSetWeightTableOffsetAttr(DPUTaskOp dpuTask, int64_t zStart, int64_t zEnd) {
    const int64_t clusterId = dpuTask.getClusterIdAttr() ? dpuTask.getClusterIdAttr().getInt() : 0;

    // Reset the cumulative offset at the start of each cluster.
    const auto isNewCluster = clusterId != _prevClusterId;
    if (isNewCluster) {
        _cumulativeWtOffset = 0;
        _prevZStart = -1;
        _prevZEnd = -1;
    } else {
        const auto isSameZRange = (zStart == _prevZStart) && (zEnd == _prevZEnd);
        VPUX_THROW_UNLESS((zStart == _prevZEnd + 1) || isSameZRange,
                          "Unexpected Z slice gap/overlap in cluster {0}: prev=[{1}, {2}], cur=[{3}, {4}]", clusterId,
                          _prevZStart, _prevZEnd, zStart, zEnd);

        _cumulativeWtOffset += getAlignment(_prevZEnd - _prevZStart + 1);
    }

    _prevClusterId = clusterId;
    _prevZStart = zStart;
    _prevZEnd = zEnd;

    dpuTask.setWeightTableOffsetAttr(mlir::IntegerAttr::get(getInt64Type(dpuTask.getContext()), _cumulativeWtOffset));
}

void DisabledWtOffsetBuilder::maybeSetWeightTableOffsetAttr(DPUTaskOp, int64_t, int64_t) {
}

}  // namespace vpux::VPUIP
