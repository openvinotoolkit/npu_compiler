//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/channel_axis_reduction_with_dpu_parent_checker.hpp"

using namespace vpux;

bool IE::arch37xx::ChannelAxisReductionWithDPUParentChecker::isChannelAxisReductionWithDPUParent(mlir::Operation*,
                                                                                                 ArrayRef<int64_t>,
                                                                                                 Logger) const {
    return false;
}
