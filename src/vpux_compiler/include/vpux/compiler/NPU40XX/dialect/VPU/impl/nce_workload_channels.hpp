//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Types.h>

namespace vpux::VPU::arch40xx {

SmallVector<int64_t> getChannelsSupportedByKernelOptimization(ArrayRef<int64_t> workloadsChannels, int64_t maxSlotsSum);

}  // namespace vpux::VPU::arch40xx
