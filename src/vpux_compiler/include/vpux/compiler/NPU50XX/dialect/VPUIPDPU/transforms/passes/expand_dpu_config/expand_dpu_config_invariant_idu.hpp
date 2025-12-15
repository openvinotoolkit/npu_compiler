//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant_idu.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux::VPUIPDPU::arch50xx::IDU {

struct IDUConfig : public arch40xx::IDU::IDUConfig {
    struct EltwiseMode {
        bool eltwiseModeOp = false;
        IDUEltwiseType eltwiseType = IDUEltwiseType::ADD;
    } eltwiseMode;
};

struct PPETask {
    std::optional<SmallVector<float>> in1MultFp;
    std::optional<SmallVector<float>> in2MultFp;
    std::optional<IDUEltwiseType> eltwiseType;
};

PPETask evalPPETasks(mlir::Region& ppeRegion, std::optional<VPUIP::NCETaskType> taskType);

mlir::LogicalResult configureIDU(const Logger& log, VPUIPDPU::arch50xx::IDU::IDUConfig& config,
                                 const vpux::NDTypeInterface& inActType, mlir::Type weightsElementType,
                                 VPUIP::NCETaskType taskType, std::optional<int64_t> spPattern,
                                 std::optional<bool> inChannelsCompression, std::optional<bool> smallKernelOptimization,
                                 bool inActSparse, bool weightsSparse, std::optional<mlir::ArrayAttr> kernelSize,
                                 std::optional<mlir::ArrayAttr> kernelStrides, std::optional<int64_t> seSize,
                                 std::optional<VPU::EltwiseType> eltwiseType, const PPETask& ppeTask);

mlir::LogicalResult buildIDUConfig(mlir::OpBuilder& builder, const mlir::Location& loc,
                                   const VPUIPDPU::arch50xx::IDU::IDUConfig& config, mlir::Value inAct);
mlir::LogicalResult configureEltwiseMode(VPUIPDPU::arch50xx::IDU::IDUConfig::EltwiseMode& config,
                                         VPUIP::NCETaskType taskType, std::optional<VPU::EltwiseType> eltwiseType);
mlir::LogicalResult configureEltwiseCfg(const Logger& log, VPUIPDPU::arch50xx::IDU::IDUConfig::EltWiseCfg& config,
                                        VPUIP::NCETaskType taskType, mlir::Type inActType, mlir::Type weightsType,
                                        const PPETask& ppeTask);
mlir::LogicalResult configureWorkload(const Logger& log, VPUIPDPU::arch50xx::IDU::IDUConfig::WorkloadCfg& config,
                                      VPUIP::NCETaskType taskType, int64_t kernelX, int64_t kernelY);
mlir::LogicalResult configureStorageElement(const Logger& log,
                                            VPUIPDPU::arch50xx::IDU::IDUConfig::StorageElement& config,
                                            VPUIP::NCETaskType taskType, const NDTypeInterface& inActType,
                                            bool inSparsityEnabled, std::optional<int64_t> seSize);
mlir::LogicalResult configureWeights(const Logger& log, VPUIPDPU::arch50xx::IDU::IDUConfig::Weights& config,
                                     VPUIP::NCETaskType taskType, mlir::Type inActType, mlir::Type weightsType,
                                     bool wtSparse);

mlir::LogicalResult getInQuantConfig(const Logger& log, mlir::Type in1Type, mlir::Type in2Type, const PPETask& ppeTask,
                                     SmallVector<float>& in1MultFp, SmallVector<float>& in2MultFp);

}  // namespace vpux::VPUIPDPU::arch50xx::IDU
