//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/make_ops_with_distributed_tensor_strategy_getter.hpp"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/make_ops_with_distributed_tensor_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/make_ops_with_distributed_tensor_strategy.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

std::unique_ptr<IGreedilyPassStrategy> VPU::createMakeOpsWithDistributedTensorStrategy(
        mlir::func::FuncOp funcOp, const llvm::DenseMap<mlir::OpResult, vpux::NDTypeInterface>& typeLookup,
        const llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>>& inputTypeLookup,
        bool enableExplicitDistributionInfoAttr) {
    const auto arch = config::getArch(funcOp);

    switch (arch) {
    case config::ArchKind::NPU37XX: {
        return std::make_unique<arch37xx::MakeOpsWithDistributedTensorStrategy>(typeLookup, inputTypeLookup,
                                                                                enableExplicitDistributionInfoAttr);
    }
    default: {
        return std::make_unique<arch40xx::MakeOpsWithDistributedTensorStrategy>(typeLookup, inputTypeLookup);
    }
    }
}
