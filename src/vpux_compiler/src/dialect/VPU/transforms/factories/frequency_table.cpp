//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/frequency_table.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/utils/performance_metrics.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/utils/performance_metrics.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

VPU::FrequencyTableCb VPU::getFrequencyTable(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU40XX: {
        return VPU::arch40xx::getFrequencyTable;
    }
    default: {
        return VPU::arch37xx::getFrequencyTable;
    }
    }
}
