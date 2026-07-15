//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/transforms/passes/unroll_distributed_ops.hpp"

namespace vpux {
namespace VPUIP {
namespace arch37xx {

//
// ClusterSWRewriter
//

class ClusterSWRewriter {
public:
    ClusterSWRewriter(mlir::MLIRContext* ctx, mlir::ModuleOp module, Logger log)
            : _log(log), _ctx(ctx), _module(module) {
    }

    void matchAndRewrite(VPUIP::SwKernelOp swTask, mlir::OpBuilder& builder) const;

private:
    Logger _log;
    mlir::MLIRContext* _ctx;
    mlir::ModuleOp _module;
};

//
// ClusterNCERewriter
//

class ClusterNCERewriter final : public ClusterNCEBaseRewriter {
public:
    ClusterNCERewriter(mlir::MLIRContext* ctx, Logger log): ClusterNCEBaseRewriter(ctx, log) {
    }

private:
    void getOutputBuffers(SmallVector<mlir::Value>& parentOutputBuffs, SmallVector<mlir::Value>& outputBuffs,
                          SmallVector<mlir::Value>& parentOutputSparsityMap,
                          SmallVector<mlir::Value>& outputSparsityMapBuffs,
                          SmallVector<SmallVector<mlir::Value>>& outputItiBuffs,
                          SmallVector<mlir::Value>& maxPerXyBuffs, SmallVector<mlir::Value>& minPerXyBuffs,
                          SmallVector<SmallVector<mlir::Value>>& minMaxPerTensorBuffs, mlir::Location loc,
                          VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters,
                          mlir::OpBuilder& builder) const override;

    void getInputBuffers(SmallVector<mlir::Value>& parentInputBuffs, SmallVector<mlir::Value>& inputBuffs,
                         SmallVector<mlir::Value>& parentInputSparsityMap,
                         SmallVector<mlir::Value>& inputSparsityMapBuffs, SmallVector<mlir::Value>& parentInputSETable,
                         SmallVector<mlir::Value>& inputSETableBuffs, mlir::Location loc,
                         VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters,
                         mlir::OpBuilder& builder) const override;

    bool isSegmentedNCETask(VPUIP::DistributedBufferType inputType) const override;
};

}  // namespace arch37xx
}  // namespace VPUIP
}  // namespace vpux
