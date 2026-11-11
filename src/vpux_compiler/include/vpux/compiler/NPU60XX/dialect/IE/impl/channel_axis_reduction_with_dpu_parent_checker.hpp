//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/channel_axis_reduction_with_dpu_parent_checker.hpp"

namespace vpux::IE::arch60xx {

// Delegates to the structural util; preserving fusion of a C-axis Reduce
// onto its DPU-capable parent is beneficial starting from NPU60XX.
class ChannelAxisReductionWithDPUParentChecker final : public IE::ChannelAxisReductionWithDPUParentCheckerBase {
public:
    explicit ChannelAxisReductionWithDPUParentChecker(bool enableFuseReduceMinMaxToDpu)
            : _enableFuseReduceMinMaxToDpu(enableFuseReduceMinMaxToDpu) {
    }

    bool isChannelAxisReductionWithDPUParent(mlir::Operation* op, ArrayRef<int64_t> axes, Logger log) const override;

private:
    bool _enableFuseReduceMinMaxToDpu;
};

}  // namespace vpux::IE::arch60xx
