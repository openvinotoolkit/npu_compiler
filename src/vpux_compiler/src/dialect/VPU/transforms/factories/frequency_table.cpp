//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/frequency_table.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/utils/performance_metrics.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/utils/performance_metrics.hpp"

using namespace vpux;

VPU::FrequencyTableCb VPU::getFrequencyTable(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU40XX: {
        return VPU::arch40xx::getFrequencyTable;
    }
    case config::ArchKind::NPU37XX: {
        return VPU::arch37xx::getFrequencyTable;
    }
    default: {
        Logger::global().warning("Use default NPU_4 frequency table for {0}", arch);
        return VPU::arch40xx::getFrequencyTable;
    }
    }
}
