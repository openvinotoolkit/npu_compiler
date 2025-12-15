//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/max_kernel_size_constant.hpp"

namespace vpux::VPU::arch50xx {

constexpr int64_t maxKernelSize = 15;

struct MaxKernelSizeConstant final {
    int64_t getMaxKernelSize() const;
};

}  // namespace vpux::VPU::arch50xx
