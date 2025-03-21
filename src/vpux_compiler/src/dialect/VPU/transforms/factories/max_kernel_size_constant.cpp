//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/max_kernel_size_constant.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/max_kernel_size_constant.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

VPU::MaxKernelSizeConstant VPU::getMaxKernelSizeConstant(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
    case VPU::ArchKind::NPU40XX: {
        return VPU::arch37xx::MaxKernelSizeConstant{};
    }
    default: {
    }
    }
    VPUX_THROW("Unable to get MaxKernelSizeConstant for arch {0}", arch);
}
