//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/impl/barrier_variant_constraint.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

#include "vpux/utils/core/numeric.hpp"

using namespace vpux::VPU::arch40xx;

namespace {

double getBarrierMaxVariantSumRatio(bool enableWorkloadManagement) {
    return enableWorkloadManagement ? barrierMaxVariantSumRatioWithWLM : barrierMaxVariantSumRatio;
}
size_t getFirmwareVariantCount(bool enableWorkloadManagement) {
    return enableWorkloadManagement ? firmwareVariantCountWithWLM : firmwareVariantCount;
}

}  // namespace

size_t PerBarrierVariantConstraint::getPerBarrierMaxVariantSum() const {
    return static_cast<size_t>(getBarrierMaxVariantSumRatio(_workloadManagementEnable) *
                               getFirmwareVariantCount(_workloadManagementEnable));
}

size_t PerBarrierVariantConstraint::getPerBarrierMaxVariantCount() const {
    return static_cast<size_t>(barrierMaxVariantCountRatio * getFirmwareVariantCount(_workloadManagementEnable));
}
