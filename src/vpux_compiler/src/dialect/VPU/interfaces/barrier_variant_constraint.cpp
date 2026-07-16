//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/barrier_variant_constraint.hpp"

using namespace vpux::VPU;

size_t PerBarrierSlotConstraint::getPerBarrierMaxSlotCount() const {
    return self->getPerBarrierMaxSlotCount();
}
