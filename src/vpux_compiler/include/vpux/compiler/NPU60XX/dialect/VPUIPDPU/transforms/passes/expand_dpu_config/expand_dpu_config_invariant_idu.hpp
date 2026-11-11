//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant_idu.hpp"

#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux::VPUIPDPU::arch60xx::IDU {

struct IDUConfig : public arch50xx::IDU::IDUConfig {
    struct WeightTableSplit {
        WtSplitMode wtSplitMode = WtSplitMode::NO_WT_READS;
        std::optional<PPEScaleDataType> scaleDtype;
    } weightTableSplit;
    struct WeightTableLegacyZp {
        WtLegacyZpMode wtLegacyZpMode = WtLegacyZpMode::WEIGHTS_PTR;
    } weightTableLegacyZp;
    struct SubtractMpeWtBias {
        bool subtractMpeWtBias = false;
    } subtractMpeWeightsBias;
};

mlir::LogicalResult configureWorkload(const Logger& log, IDUConfig::WorkloadCfg& config, VPUIP::NCETaskType taskType,
                                      int64_t kernelX, int64_t kernelY, mlir::Type inActType, mlir::Type outActType,
                                      VPU::MPEMode mpeFrequentMode, bool isContinued);

mlir::LogicalResult configureWeights(const Logger& log, IDUConfig::Weights& config, VPUIP::NCETaskType taskType,
                                     mlir::Type inActType, mlir::Type weightsType, bool wtSparse);

mlir::LogicalResult configurePalletization(const Logger& log, IDUConfig::Weights& config, mlir::Type weightsType);

mlir::LogicalResult configureDepthWiseCfg(const Logger&, IDUConfig::DepthWiseCfg& config, VPUIP::NCETaskType taskType,
                                          bool smallKernelOptimization);

mlir::LogicalResult configureWeightTableSplit([[maybe_unused]] mlir::Operation* op, const Logger& log,
                                              IDUConfig::WeightTableSplit& config, bool wtLegacy, bool wtDataPtr,
                                              bool wtSpPtr, bool wtZp, bool perChZp, mlir::Type scalesElementType,
                                              VPUIP::NCETaskType taskType);

mlir::LogicalResult configureWeightTableLegacyZp([[maybe_unused]] mlir::Operation* op,
                                                 IDUConfig::WeightTableLegacyZp& config, mlir::Type weightsType,
                                                 bool wtLegacy);

mlir::LogicalResult configureSubtractMpeWtBias(IDUConfig::SubtractMpeWtBias& config, mlir::Type inActType,
                                               mlir::Type weightsType);

mlir::LogicalResult configureIDU(mlir::Operation* op, const Logger& log, IDU::IDUConfig& config, mlir::Type inActType,
                                 mlir::Type weightsElementType, mlir::Type outActType, VPUIP::NCETaskType taskType,
                                 std::optional<int64_t> spPattern, std::optional<bool> inChannelsCompression,
                                 bool smallKernelOptimization, bool inActSparse, bool weightsSparse,
                                 std::optional<mlir::ArrayAttr> kernelSize,
                                 std::optional<mlir::ArrayAttr> kernelStrides, std::optional<int64_t> seSize,
                                 std::optional<VPU::EltwiseType> eltwiseType, const arch50xx::IDU::PPETask& ppeTask,
                                 bool wtLegacy, bool wtDataPtr, bool wtSpPtr, bool wtZp, bool perChZp,
                                 mlir::Type scalesElementType, VPU::MPEMode mpeFrequentMode, bool isContinued);

mlir::LogicalResult buildIDUConfig(mlir::OpBuilder& builder, const mlir::Location& loc, const IDUConfig& config,
                                   mlir::Value inAct, bool wtLegacy, VPUIP::NCETaskType taskType,
                                   bool useWorkloadTypeInWT = false, bool isZpInUint4Range = false);

}  // namespace vpux::VPUIPDPU::arch60xx::IDU
