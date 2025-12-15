//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux::VPUIPDPU::arch50xx {

mlir::LogicalResult buildDPUInvariantIDU(VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, const Logger& log,
                                         mlir::Block* invBlock,
                                         const std::unordered_map<BlockArg, size_t>& invBlockArgsPos);

mlir::LogicalResult buildDPUInvariantMPE(VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder,
                                         mlir::Block* invBlock,
                                         const std::unordered_map<BlockArg, size_t>& invBlockArgsPos);

mlir::LogicalResult buildDPUInvariantPPE(VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, const Logger& log,
                                         mlir::Block* invBlock,
                                         const std::unordered_map<BlockArg, size_t>& invBlockArgsPos);

}  // namespace vpux::VPUIPDPU::arch50xx
