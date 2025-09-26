//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/sparsity_constraint.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/sparsity_constraint.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/sparsity_constraint.hpp"

using namespace vpux;

VPU::SparsityConstraint VPU::getSparsityConstraint(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX: {
        return VPU::arch37xx::SparsityConstraint{};
    }
    default: {
        return VPU::arch40xx::SparsityConstraint{};
    }
    }
}
