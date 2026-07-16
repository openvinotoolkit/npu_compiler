//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/barrier_variant_constraint.hpp"

#include <cstdint>

namespace vpux {
namespace VPU {

VPU::PerBarrierSlotConstraint getPerBarrierSlotConstraint();

}  // namespace VPU
}  // namespace vpux
