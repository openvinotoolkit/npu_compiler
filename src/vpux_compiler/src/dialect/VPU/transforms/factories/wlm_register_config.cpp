//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/wlm_register_config.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/wlm_register_config.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/wlm_register_config.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

VPU::RegisterConfig VPU::getRegisterConfig(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX: {
        return VPU::arch37xx::RegisterConfig{};
    }
    case config::ArchKind::NPU40XX: {
        return VPU::arch40xx::RegisterConfig{};
    }
    default: {
    }
    }
    VPUX_THROW("Unable to get RegisterConfig for arch {0}", arch);
}
