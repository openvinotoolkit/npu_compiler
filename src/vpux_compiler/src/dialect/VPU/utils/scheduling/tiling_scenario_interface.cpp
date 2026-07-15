//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/tiling_scenario_interface.hpp"

namespace vpux::VPU {

// Out-of-line destructor anchors the vtable in a single translation unit and provides a
// user-defined definition required by the rule of five.
TilingScenarioInterface::~TilingScenarioInterface() = default;

}  // namespace vpux::VPU
