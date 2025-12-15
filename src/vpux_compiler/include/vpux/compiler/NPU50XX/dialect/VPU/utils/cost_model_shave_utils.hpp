//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/VPU/utils/cost_model_shave_utils.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/cost_model_shave_utils.hpp"

#include <map>
#include <string>

namespace vpux {
namespace VPU {
namespace arch50xx {

// @brief uses the same MAP as the 4.0
using Shave50NamingMap = arch40xx::Shave40NamingMap;
// generates based on template the specific class with the needed map
using CostModelShaveUtil = SHAVECMUtilsBase<Shave50NamingMap>;
}  // namespace arch50xx
}  // namespace VPU
}  // namespace vpux
