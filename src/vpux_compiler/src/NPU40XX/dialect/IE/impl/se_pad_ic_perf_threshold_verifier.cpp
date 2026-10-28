//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/IE/impl/se_pad_ic_perf_threshold_verifier.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux::IE::arch40xx;

namespace {

int64_t getMinTileCountAtThreshold(mlir::Operation* op) {
    const auto platform = vpux::config::getPlatform(op);
    const auto maxTiles = vpux::VPU::getMaxDPUClusterNum(platform);

    VPUX_THROW_WHEN(maxTiles == 0, "Invalid max DPU cluster number for platform: {0}", platform);
    return vpux::checked_cast<int64_t>(maxTiles - 1);
}

}  // namespace

//
// SEPadICPerfThresholdVerifier
//

bool SEPadICPerfThresholdVerifier::isBeneficialForPerformance(const Shape& newInputShape, mlir::Operation* op) const {
    const auto ic = newInputShape[Dims4D::Act::C];
    if (ic < SEP_PAD_IC_NUM_PERF_THRESHOLD) {
        return false;
    }

    // For 40XX SE Pad at the threshold boundary is beneficial only when the active tile count reaches the
    // platform-specific max-tile threshold (max DPU groups - 1).
    if (ic == SEP_PAD_IC_NUM_PERF_THRESHOLD) {
        auto module = op->getParentOfType<mlir::ModuleOp>();
        auto tileCount = config::getTileExecutor(module).getCount();
        return tileCount >= getMinTileCountAtThreshold(op);
    }

    return true;
}
