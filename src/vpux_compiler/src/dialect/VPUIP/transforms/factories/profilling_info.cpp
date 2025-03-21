//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/impl/profiling_info.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/factories/profiling_info.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

VPUIP::TimestampTypeCb VPUIP::getTimestampTypeCb(VPU::ArchKind arch) {
    if (arch == VPU::ArchKind::NPU37XX) {
        return VPUIP::arch37xx::getTimestampType;
    }
    VPUX_THROW("Unexpected architecture {0}", arch);
}

VPUIP::SetWorkloadIdsCb VPUIP::setWorkloadsIdsCb(VPU::ArchKind arch) {
    if (arch >= VPU::ArchKind::NPU37XX) {
        return VPUIP::arch37xx::setWorkloadIds;
    }
    VPUX_THROW("Unexpected architecture {0}", arch);
}
