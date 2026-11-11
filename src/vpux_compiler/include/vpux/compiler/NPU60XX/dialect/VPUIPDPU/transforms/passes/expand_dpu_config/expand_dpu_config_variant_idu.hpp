//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant_idu.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

namespace vpux::VPUIPDPU::arch60xx::IDU {

mlir::LogicalResult buildIDUNthwNtk(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                    VPU::MPEMode& mpeFrequentMode, VPUIP::NCETaskType dpuTaskType,
                                    bool allowExtension = false);

mlir::LogicalResult buildIDUPalletLutREAD(mlir::OpBuilder& builder, VPUASM::DPUVariantOp origVarOp);

mlir::LogicalResult buildIDUWeightsTableZpBitwidth(mlir::OpBuilder& builder, const mlir::Location& loc,
                                                   const Logger& log, const vpux::NDTypeInterface& weightsType);

mlir::LogicalResult buildIDUWeightSet(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                      int64_t inStartZ, int64_t inEndZ, int64_t outStartZ, int64_t outEndZ,
                                      std::optional<int64_t> outChannelOffset, VPUIP::NCETaskType taskType,
                                      const vpux::NDTypeInterface& inActType, const vpux::NDTypeInterface& outActType,
                                      std::optional<mlir::ArrayAttr> kernelSize, bool hasWeightTable);

mlir::LogicalResult buildIDUGroupConv(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                      int64_t inStartZ, int64_t inEndZ, const vpux::NDTypeInterface& inActType,
                                      vpux::VPUIP::NCETaskType taskType);

mlir::LogicalResult buildIDUWeightTableOffsets(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger&,
                                               const std::optional<int64_t>& weightTableOffset, bool hasWeights,
                                               bool hasWeightTableDataPtr, bool hasWeightTableSpPtr,
                                               std::optional<mlir::MemRefType> weightTableScaleType,
                                               std::optional<mlir::MemRefType> weightTableBiasType,
                                               const vpux::NDTypeInterface& outActType, int64_t outStartZ,
                                               std::optional<int64_t> outChannelOffset);
}  // namespace vpux::VPUIPDPU::arch60xx::IDU
