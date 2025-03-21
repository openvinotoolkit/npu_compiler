//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/sparsity_constraint.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/sparsity_constraint.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/sparsity_constraint.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

VPU::SparsityConstraint VPU::getSparsityConstraint(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX: {
        return VPU::arch37xx::SparsityConstraint{};
    }
    default: {
        return VPU::arch40xx::SparsityConstraint{};
    }
    }
}
