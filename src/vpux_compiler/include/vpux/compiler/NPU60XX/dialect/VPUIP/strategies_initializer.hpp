//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/init/hw_strategy_registry.hpp"

namespace vpux::VPUIP {

//
// Initializing architecture-specific strategies in context
//

class StrategiesInitializer60XX final : public IStrategiesInitializer {
public:
    void initialize(mlir::MLIRContext* context) override;
};

}  // namespace vpux::VPUIP
