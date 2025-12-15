//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/impl/make_ops_with_distributed_tensor_strategy.hpp"

#include "vpux/compiler/dialect/VPU/interfaces/common_rewriters/make_ops_with_distributed_tensor.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

using namespace vpux;

//
// MakeOpsWithDistributedTensorStrategy
//

void VPU::arch40xx::MakeOpsWithDistributedTensorStrategy::addPatterns(mlir::RewritePatternSet& patterns,
                                                                      Logger& log) const {
    auto ctx = patterns.getContext();
    patterns.add<VPU::ClusteredOpRewriter>(
            ctx, _typeLookup, _inputTypeLookup,
            [](VPU::ClusteredOpInterface op) {
                return !mlir::isa<VPU::NCEEltwiseOp>(op);
            },
            log);
    patterns.add<VPU::NCEEltwiseRewriter>(ctx, _typeLookup, _inputTypeLookup, log);
}
