//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/ops_interfaces.hpp"

#include <mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h>
#include <mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h>
#include <mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h>
#include <mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h>
#include <mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h>

namespace vpux::ShaveCodeGen {

void registerShaveCodeGenOpInterfaces(mlir::DialectRegistry& registry) {
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
    mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::tensor::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::tensor::registerTilingInterfaceExternalModels(registry);
    mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
    mlir::affine::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
}

}  // namespace vpux::ShaveCodeGen
