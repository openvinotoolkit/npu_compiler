//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/dpu_variant_invariant_constraint.hpp"

#include <cstdint>

namespace vpux::config {
enum class ArchKind : uint64_t;
}

namespace vpux {
namespace VPU {

VPU::DPUVariantInvariantConstraint getDPUVariantInvariantConstraint(config::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
