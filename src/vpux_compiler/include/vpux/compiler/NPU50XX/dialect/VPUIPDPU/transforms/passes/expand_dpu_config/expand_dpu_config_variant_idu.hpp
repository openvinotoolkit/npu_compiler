//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant_idu.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

namespace vpux::VPUIPDPU::arch50xx::IDU {

mlir::LogicalResult buildIDUWeightSet(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                      int64_t inStartZ, int64_t inEndZ, int64_t outStartZ, int64_t outEndZ,
                                      std::optional<int64_t> outChannelOffset, VPUIP::NCETaskType taskType,
                                      const vpux::NDTypeInterface& inActType, const vpux::NDTypeInterface& outActType,
                                      std::optional<mlir::ArrayAttr> kernelSize);

mlir::LogicalResult buildIDUSEOnly(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger&, bool seOnly);

mlir::LogicalResult buildIDUPerOutputChannelScaling(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger&,
                                                    VPUIP::NCETaskType taskType, bool weightTableProvided,
                                                    bool weightsSparse);

}  // namespace vpux::VPUIPDPU::arch50xx::IDU
