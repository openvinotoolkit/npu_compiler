//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/dpu_variant_invariant_constraint.hpp"
namespace vpux::VPU::arch37xx {

constexpr size_t maxNumberOfDpuVariantsPerInvariant = 256;

struct DpuVariantInvariantConstraint final {
    size_t getMaxNumberOfDpuVariantsPerInvariant() const;
};

}  // namespace vpux::VPU::arch37xx
