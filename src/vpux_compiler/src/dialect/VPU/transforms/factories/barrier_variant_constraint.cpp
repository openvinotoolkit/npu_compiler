//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/barrier_variant_constraint.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/barrier_variant_constraint.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

VPU::PerBarrierSlotConstraint VPU::getPerBarrierSlotConstraint() {
    return VPU::arch37xx::PerBarrierSlotConstraint{};
}
