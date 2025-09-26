//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"

namespace vpux {
namespace VPU {

constexpr std::array<int64_t, 3> supportedChannelsDW = {64, 32, 16};

class WorkloadSplitter {
public:
    WorkloadSplitter(mlir::func::FuncOp funcOp, ArrayRef<int64_t> supportedChannelsForDW, vpux::Logger log);

    void correctInvalidWorkload(const VPU::SparsityConstraint& sparsityConstraint);

    mlir::DenseSet<mlir::Operation*> findInvalidSparseOps(VPU::NCEOpInterface nceOp,
                                                          const VPU::SparsityConstraint& sparsityConstraint);

    SmallVector<int64_t> getSupportedChannels(const mlir::DenseSet<mlir::Operation*>& nceOps,
                                              const VPU::SparsityConstraint& sparsityConstraint);

protected:
    static bool isDepthwiseOp(mlir::Operation* op);

    SmallVector<Shape> getPerClusterShapesWhenSOK(VPU::NCEOpInterface nceOp);
    mlir::DenseSet<int64_t> getWorkloadsChannels(const mlir::DenseSet<mlir::Operation*>& nceOps,
                                                 bool skipLastWorkload = false);
    mlir::DenseSet<mlir::Operation*> findConsumerOps(mlir::Value value);
    mlir::DenseSet<mlir::Operation*> findProducerNCEOps(mlir::Value value);
    mlir::DenseSet<mlir::Operation*> findProducersForConsumers(
            mlir::Value value, mlir::DenseSet<mlir::Operation*> processedConsumerOps = {});

    mlir::DenseSet<mlir::Operation*> findInvalidDepthwiseOps(const mlir::DenseSet<mlir::Operation*>& nceOps,
                                                             ArrayRef<int64_t> supportedChannels);
    mlir::DenseSet<mlir::Operation*> findInvalidNCEPermuteOps(const mlir::DenseSet<mlir::Operation*>& nceOps);

    void splitWorkload(VPU::DPUWorkloadOp dpuWorkloadOp, ArrayRef<int64_t> supportedChannels,
                       const bool isInvalidNCEPermuteOp, int64_t channelPadding,
                       bool isNCEPermuteOffsetsCorrectionNeeded, Logger log);

private:
    mlir::func::FuncOp _funcOp;
    SmallVector<int64_t> _supportedChannelsForDW;

protected:
    vpux::Logger _log;
};

}  // namespace VPU
}  // namespace vpux
