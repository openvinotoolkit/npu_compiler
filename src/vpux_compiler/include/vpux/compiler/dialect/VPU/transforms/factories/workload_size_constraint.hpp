//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/workload_size_constraint.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

namespace vpux {
namespace VPU {

VPU::WorkloadSizeConstraint getWorkloadSizeConstraint(config::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
