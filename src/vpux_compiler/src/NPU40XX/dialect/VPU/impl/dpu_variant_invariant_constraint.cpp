//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/impl/dpu_variant_invariant_constraint.hpp"

using namespace vpux::VPU::arch40xx;

size_t DpuVariantInvariantConstraint::getMaxNumberOfDpuVariantsPerInvariant() const {
    return maxNumberOfDpuVariantsPerInvariant;
}
