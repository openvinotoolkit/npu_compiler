//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/ops_interfaces.hpp"

#include <mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h>
#include <mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h>

namespace vpux::ShaveCodeGen {

void registerShaveCodeGenOpInterfaces(mlir::DialectRegistry& registry) {
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
    mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::tensor::registerTilingInterfaceExternalModels(registry);
}

}  // namespace vpux::ShaveCodeGen
