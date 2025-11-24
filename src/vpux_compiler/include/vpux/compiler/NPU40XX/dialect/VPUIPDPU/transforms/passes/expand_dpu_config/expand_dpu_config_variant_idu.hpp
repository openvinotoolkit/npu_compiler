//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux::VPUIPDPU::arch40xx::IDU {

mlir::LogicalResult buildIDUWorkloadSet(mlir::OpBuilder& builder, const mlir::Location& loc,
                                        const SmallVector<int64_t>&& inStart, const SmallVector<int64_t>&& inEnd);

mlir::LogicalResult buildIDUWeightSet(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                      int64_t inStartZ, int64_t inEndZ, int64_t outStartZ, int64_t outEndZ,
                                      std::optional<int64_t> outChannelOffset, VPUIP::NCETaskType taskType,
                                      const vpux::NDTypeInterface& inActType, const vpux::NDTypeInterface& outActType,
                                      std::optional<mlir::ArrayAttr> kernelSize);

mlir::LogicalResult buildIDUPadding(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger&,
                                    VPU::PaddingAttr pad);

mlir::LogicalResult buildIDUActSwizzle(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                       std::optional<int64_t> inSwizzling);

mlir::LogicalResult buildIDUWeightSwizzle(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                          std::optional<int64_t> weightsSwizzling);

mlir::LogicalResult buildIDUSEDense(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger&, bool seDense);

mlir::LogicalResult buildIDUConvContinue(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger&,
                                         std::optional<bool> isContinued);

mlir::LogicalResult buildIDUNthwNtk(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                    VPU::MPEMode mpeFrequentMode, VPUIP::NCETaskType dpuTaskType);

}  // namespace vpux::VPUIPDPU::arch40xx::IDU
