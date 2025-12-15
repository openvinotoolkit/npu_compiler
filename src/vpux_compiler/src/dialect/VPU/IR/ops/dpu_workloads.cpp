//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"

using namespace vpux;

//
// DPUTaskOp
//

void vpux::VPU::DPUWorkloadOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ArrayAttr outOffsets,
                                     mlir::ArrayAttr outSizes, VPU::PaddingAttr pad, VPU::MPEMode mpeMode) {
    build(builder, state, outOffsets, outSizes, /*inOffsets=*/nullptr, /*inSizes=*/nullptr, pad, mpeMode,
          /*cluster_id=*/nullptr);
}

void vpux::VPU::DPUWorkloadOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ArrayAttr outOffsets,
                                     mlir::ArrayAttr outSizes, VPU::PaddingAttr pad, VPU::MPEModeAttr mpeMode,
                                     mlir::IntegerAttr clusterId) {
    build(builder, state, outOffsets, outSizes, /*inOffsets=*/nullptr, /*inSizes=*/nullptr, pad, mpeMode, clusterId);
}

void vpux::VPU::DPUWorkloadOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ArrayAttr outOffsets,
                                     mlir::ArrayAttr outSizes, VPU::PaddingAttr pad, VPU::MPEMode mpeMode,
                                     mlir::IntegerAttr clusterId) {
    build(builder, state, outOffsets, outSizes, /*inOffsets=*/nullptr, /*inSizes=*/nullptr, pad, mpeMode, clusterId);
}
