//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/channel_axis_reduction_with_dpu_parent_checker.hpp"

namespace vpux::IE::arch37xx {

// Reduce->Pooling conversion is unconditionally enabled on this platform family;
// the checker reports false so the structural Reduce pre-check never blocks the fusion.
class ChannelAxisReductionWithDPUParentChecker final : public IE::ChannelAxisReductionWithDPUParentCheckerBase {
public:
    bool isChannelAxisReductionWithDPUParent(mlir::Operation* op, ArrayRef<int64_t> axes, Logger log) const override;
};

}  // namespace vpux::IE::arch37xx
