//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/barrier_variant_constraint.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

#include "vpux/utils/core/numeric.hpp"

using namespace vpux::VPU::arch37xx;

size_t PerBarrierSlotConstraint::getPerBarrierMaxSlotCount() const {
    return static_cast<size_t>(maxBarrierUsersCount);
}
