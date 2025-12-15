//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/constraints_initializer.hpp"

namespace vpux::config {

//
// Initializing architecture-specific constraints in context
//

class ConstraintsInitializer40XX final : public IConstraintsInitializer {
public:
    void initialize(mlir::MLIRContext* context) override;
};

}  // namespace vpux::config
