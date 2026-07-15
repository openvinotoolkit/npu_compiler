//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Operation.h>

#include <cstdint>

namespace vpux::IE {

// Per-platform decision: should a channel-axis Reduce that has a DPU-capable
// parent be left untouched to keep the fusion opportunity?
class ChannelAxisReductionWithDPUParentCheckerBase {
public:
    virtual ~ChannelAxisReductionWithDPUParentCheckerBase() = default;

    virtual bool isChannelAxisReductionWithDPUParent(mlir::Operation* op, ArrayRef<int64_t> axes, Logger log) const = 0;
};

}  // namespace vpux::IE
