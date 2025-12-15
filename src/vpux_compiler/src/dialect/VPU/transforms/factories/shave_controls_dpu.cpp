//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/shave_controls_dpu.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/profiling/parser/hw.hpp"
namespace npu37xx {
#include <vpux/compiler/NPU37XX/dialect/NPUReg37XX/firmware_headers/details/api/vpu_nce_hw_37xx.h>
}
namespace npu40xx {
#include <vpux/compiler/NPU40XX/dialect/NPUReg40XX/firmware_headers/details/api/vpu_nce_hw_40xx.h>
}
#include "vpux/compiler/NPU50XX/dialect/VPU/impl/shave_controls_dpu_constraint.hpp"

using namespace vpux;

constexpr bool shaveControlsDpuValue = false;

bool VPU::getShaveControlsDpu(config::ArchKind arch) {
    if (arch == config::ArchKind::NPU50XX) {
        return VPU::arch50xx::getShaveControlsDpuConstraint();
    }
    return shaveControlsDpuValue;
}

size_t VPU::getDpuDebugDataSize(config::ArchKind /*arch*/) {
    return sizeof(HwpDpuIduOduData_t);
}

size_t VPU::getDPUInvariantDataSize(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return sizeof(npu37xx::nn_public::VpuDPUInvariantRegisters);
    case config::ArchKind::NPU40XX:
    case config::ArchKind::NPU50XX:
        return sizeof(npu40xx::nn_public::VpuDPUInvariantRegisters);
    default:
        VPUX_THROW("Unable to get DPUInvariantDataSize for arch {0}", arch);
    }
}

size_t VPU::getDPUVariantDataSize(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return sizeof(npu37xx::nn_public::VpuDPUVariantRegisters);
    case config::ArchKind::NPU40XX:
    case config::ArchKind::NPU50XX:
        return sizeof(npu40xx::nn_public::VpuDPUVariantRegisters);
    default:
        VPUX_THROW("Unable to get DPUVariantDataSize for arch {0}", arch);
    }
}
