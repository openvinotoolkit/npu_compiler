//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/barrier_variant_constraint.hpp"

namespace vpux {
namespace VPU {

VPU::PerBarrierVariantConstraint getPerBarrierVariantConstraint(VPU::ArchKind arch, bool workloadManagementEnable);

}  // namespace VPU
}  // namespace vpux
