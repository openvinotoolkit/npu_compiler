//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"  // ensure config::getArch(Platform) overload prevails

#include <mlir/IR/MLIRContext.h>

namespace vpux::config {

//
// IConstraintsInitializer
//

class IConstraintsInitializer {
public:
    virtual void initialize(mlir::MLIRContext* context) = 0;
    virtual ~IConstraintsInitializer() = default;
};

}  // namespace vpux::config
