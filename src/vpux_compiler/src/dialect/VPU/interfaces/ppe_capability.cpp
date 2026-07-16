//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/ppe_capability.hpp"
#include "vpux/compiler/core/interfaces/dialect_cache.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

const vpux::VPU::IPPECapability& vpux::VPU::getPPECapability(mlir::MLIRContext* context) {
    return getCache<vpux::VPU::IPPECapability, vpux::VPU::VPUDialect>(context);
}

const vpux::VPU::IPPECapability* vpux::VPU::tryGetPPECapability(mlir::MLIRContext* context) {
    if (context == nullptr) {
        return nullptr;
    }
    auto* dialect = context->getOrLoadDialect<vpux::VPU::VPUDialect>();
    if (dialect == nullptr) {
        return nullptr;
    }
    return dialect->getRegisteredInterface<vpux::VPU::IPPECapability>();
}
