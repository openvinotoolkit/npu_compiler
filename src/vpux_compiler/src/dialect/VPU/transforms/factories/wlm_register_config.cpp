//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/wlm_register_config.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/wlm_register_config.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/impl/wlm_register_config.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPU/impl/wlm_register_config.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/wlm_register_config.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include "vpux/utils/core/error.hpp"

#include <mlir/IR/Operation.h>

using namespace vpux;

VPU::RegisterConfig VPU::getRegisterConfig(config::ArchKind arch, mlir::Operation* op) {
    std::ignore = op;
    switch (arch) {
    case config::ArchKind::NPU37XX: {
        return VPU::arch37xx::RegisterConfig{};
    }
    case config::ArchKind::NPU40XX: {
        return VPU::arch40xx::RegisterConfig{};
    }
    case config::ArchKind::NPU50XX: {
        return VPU::arch50xx::RegisterConfig{};
    }
    case config::ArchKind::NPU60XX: {
        return VPU::arch60xx::RegisterConfig{};
    }
    default: {
        return VPU::arch60xx::RegisterConfig{};
    }
    }
    VPUX_THROW("Unable to get RegisterConfig for arch {0}", arch);
}
