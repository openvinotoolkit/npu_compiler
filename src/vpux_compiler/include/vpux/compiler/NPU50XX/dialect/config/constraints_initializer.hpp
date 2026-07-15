//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/constraints_initializer.hpp"

namespace vpux::config {

//
// Initializing platform-specific constraints in context
//

class ConstraintsInitializer50XX final : public IConstraintsInitializer {
public:
    explicit ConstraintsInitializer50XX(config::Platform platform);
    void initialize(mlir::MLIRContext* context) override;

private:
    config::Platform _platform;
};

}  // namespace vpux::config
