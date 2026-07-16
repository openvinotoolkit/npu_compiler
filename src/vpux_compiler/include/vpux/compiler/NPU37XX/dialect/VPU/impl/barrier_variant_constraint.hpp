//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/barrier_variant_constraint.hpp"

namespace vpux::VPU::arch37xx {

// TODO: E#78647 refactor to use api/vpu_cmx_info_{arch}.h
constexpr size_t maxBarrierUsersCount = 256;

struct PerBarrierSlotConstraint final {
    size_t getPerBarrierMaxSlotCount() const;
};

}  // namespace vpux::VPU::arch37xx
