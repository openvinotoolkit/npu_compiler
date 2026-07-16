//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <mlir/IR/DialectRegistry.h>

namespace vpux {
class IStrategiesInitializer {
public:
    virtual void initialize(mlir::MLIRContext* context) = 0;
    virtual ~IStrategiesInitializer();
};
}  // namespace vpux

namespace vpux::config {

//
// registerConstraints
//

void registerConstraints(mlir::DialectRegistry& registry, config::Platform platform);

}  // namespace vpux::config

namespace vpux::IE {

void registerStrategies(mlir::DialectRegistry& registry, config::Platform platform);

}  // namespace vpux::IE

namespace vpux::VPU {

void registerStrategies(mlir::DialectRegistry& registry, config::Platform platform);

}  // namespace vpux::VPU

namespace vpux::VPUIP {

void registerStrategies(mlir::DialectRegistry& registry, config::Platform platform);

}  // namespace vpux::VPUIP
