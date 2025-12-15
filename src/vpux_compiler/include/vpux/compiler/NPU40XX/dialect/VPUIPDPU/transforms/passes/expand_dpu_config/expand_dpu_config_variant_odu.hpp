//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux::VPUIPDPU::arch40xx::ODU {

mlir::LogicalResult buildODUOutSubtensor(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                         const SmallVector<int64_t>&& start, const SmallVector<int64_t>&& end);

mlir::LogicalResult buildODUHaloRegionOp(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                         mlir::Block* varBlock, mlir::ArrayAttr haloRegions, bool outSparsityEnabled);

}  // namespace vpux::VPUIPDPU::arch40xx::ODU
