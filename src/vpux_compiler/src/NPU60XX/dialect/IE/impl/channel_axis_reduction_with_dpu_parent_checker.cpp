//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/IE/impl/channel_axis_reduction_with_dpu_parent_checker.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"

using namespace vpux;

bool IE::arch60xx::ChannelAxisReductionWithDPUParentChecker::isChannelAxisReductionWithDPUParent(mlir::Operation* op,
                                                                                                 ArrayRef<int64_t> axes,
                                                                                                 Logger log) const {
    if (!_enableFuseReduceMinMaxToDpu) {
        return false;
    }
    return IE::isChannelAxisReductionWithDPUParent(op, axes, log);
}
