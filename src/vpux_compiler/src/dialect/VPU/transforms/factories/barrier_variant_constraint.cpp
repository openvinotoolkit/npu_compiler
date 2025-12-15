//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/barrier_variant_constraint.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/barrier_variant_constraint.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/impl/barrier_variant_constraint.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/barrier_variant_constraint.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

VPU::PerBarrierVariantConstraint VPU::getPerBarrierVariantConstraint(config::ArchKind arch,
                                                                     bool enableWorkloadManagement) {
    switch (arch) {
    case config::ArchKind::NPU37XX: {
        return VPU::arch37xx::PerBarrierVariantConstraint{};
    }
    case config::ArchKind::NPU40XX: {
        return VPU::arch40xx::PerBarrierVariantConstraint{enableWorkloadManagement};
    }
    default: {
        return VPU::arch50xx::PerBarrierVariantConstraint{enableWorkloadManagement};
    }
    }
    VPUX_THROW("Unable to get PerBarrierVariantConstraint for arch {0}", arch);
}
