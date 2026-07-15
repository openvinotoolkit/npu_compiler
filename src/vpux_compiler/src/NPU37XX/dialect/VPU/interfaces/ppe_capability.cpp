//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/ppe_capability.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

void vpux::VPU::arch37xx::registerPPECapabilityInterface(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext*, vpux::VPU::VPUDialect* dialect) {
        dialect->addInterfaces<vpux::VPU::arch37xx::PPECapability>();
    });
}
