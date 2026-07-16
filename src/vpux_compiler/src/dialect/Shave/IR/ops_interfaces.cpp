//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/Shave/IR/ops_interfaces.hpp"

#include "vpux/compiler/dialect/Shave/IR/dialect.hpp"
#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

using namespace vpux;

void Shave::registerShaveOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, Shave::ShaveDialect*) {
        Shave::ExternalKernelOp::attachInterface<VPU::LayerOpInterface>(*ctx);
    });
}

//
// Generated
//

#include <vpux/compiler/dialect/Shave/ops_interfaces.cpp.inc>
