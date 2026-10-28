//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/config/constraints.hpp"
#include "vpux/compiler/core/interfaces/dialect_cache.hpp"
#include "vpux/compiler/dialect/config/IR/dialect.hpp"

#include <mlir/IR/MLIRContext.h>

#include <cassert>

using vpux::SmallVector;
using namespace vpux::config;

SmallVector<NPUConstraints::RoPEConfig> vpux::config::getDefaultLegalRoPEConfigs() {
    return {{32, 128}, {1, 128}, {40, 128}, {8, 256}, {1, 256}, {4, 256}, {1, 64}, {64, 64}, {16, 128}, {2, 128}};
}

void vpux::config::setNPUConstraints(mlir::MLIRContext* context, const NPUConstraints& constraint) {
    auto& registeredInterface = getCache<ConfigCache, ConfigDialect>(context);
    registeredInterface.setConstraints(constraint);
}

const NPUConstraints& vpux::config::getNPUConstraints(mlir::MLIRContext* context) {
    auto& registeredInterface = getCache<ConfigCache, ConfigDialect>(context);
    return registeredInterface.getConstraints();
}

const NPUConstraints& vpux::config::updateMaxKernelSize(mlir::MLIRContext* context, uint32_t maxKernelSize) {
    auto& registeredInterface = getCache<ConfigCache, ConfigDialect>(context);
    auto& npuConstraints = registeredInterface._npuConstraints;
    npuConstraints.maxKernelSize = maxKernelSize;
    return npuConstraints;
}
