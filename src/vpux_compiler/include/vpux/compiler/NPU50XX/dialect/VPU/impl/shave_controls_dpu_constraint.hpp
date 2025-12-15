//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace vpux::VPU::arch50xx {

constexpr bool shaveControlsDpuValue = true;

bool getShaveControlsDpuConstraint() {
    return shaveControlsDpuValue;
}

}  // namespace vpux::VPU::arch50xx
