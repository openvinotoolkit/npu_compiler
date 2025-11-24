//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/wlm_register_config.hpp"

namespace vpux {
namespace VPU {

VPU::RegisterConfig getRegisterConfig(config::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
