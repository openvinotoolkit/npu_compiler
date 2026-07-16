//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/dpu_variant_invariant_constraint.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/dpu_variant_invariant_constraint.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/dpu_variant_invariant_constraint.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

VPU::DPUVariantInvariantConstraint VPU::getDPUVariantInvariantConstraint(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX: {
        return VPU::arch37xx::DpuVariantInvariantConstraint{};
    }
    default: {
        return VPU::arch40xx::DpuVariantInvariantConstraint{};
    }
    }
    VPUX_THROW("Unable to get DPUVariantInvariantConstraint for arch {0}", arch);
}
