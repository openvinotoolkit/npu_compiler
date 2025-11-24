//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/workload_size_constraint.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/workload_size_constraint.hpp"

using namespace vpux;

VPU::WorkloadSizeConstraint VPU::getWorkloadSizeConstraint([[maybe_unused]] config::ArchKind arch) {
    return VPU::arch37xx::WorkloadSizeConstraint{};
}
