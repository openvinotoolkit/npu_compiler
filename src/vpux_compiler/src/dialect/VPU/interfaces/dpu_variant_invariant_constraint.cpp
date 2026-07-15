//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/dpu_variant_invariant_constraint.hpp"

namespace vpux::VPU {

size_t DPUVariantInvariantConstraint::getMaxNumberOfDpuVariantsPerInvariant() const {
    return self->getMaxNumberOfDpuVariantsPerInvariant();
}

}  // namespace vpux::VPU
