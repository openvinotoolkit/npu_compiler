//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/options.hpp"

#include <memory>

namespace vpux::VPU::VF::v2 {

class VFCostStrategy;
class VFCostStrategyFactory final {
public:
    static std::shared_ptr<const VFCostStrategy> create(VFMergeCostMode costMode);
};

}  // namespace vpux::VPU::VF::v2
