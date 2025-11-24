//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <cstdint>

namespace vpux {
namespace VPU {

uint32_t getDefaultTaskListCount(VPU::TaskType taskType, config::ArchKind archKind);

}  // namespace VPU
}  // namespace vpux
