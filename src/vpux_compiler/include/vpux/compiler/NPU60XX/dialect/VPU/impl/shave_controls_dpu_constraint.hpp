//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace vpux::VPU::arch60xx {

constexpr bool shaveControlsDpuValue = true;

constexpr bool getShaveControlsDpuConstraint() {
    return shaveControlsDpuValue;
}

}  // namespace vpux::VPU::arch60xx
