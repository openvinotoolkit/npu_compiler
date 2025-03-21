//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"

namespace vpux {

constexpr StringLiteral DPUCost = "minimumHardwareExecutionCost";
constexpr StringLiteral cycleCostAttrName = "cycleCost";
constexpr StringLiteral cycleBegin = "cycleBegin";
constexpr StringLiteral cycleEnd = "cycleEnd";

size_t getDMACost(mlir::Value input, mlir::Value output, VPU::ArchKind archKind,
                  const std::shared_ptr<VPUNN::VPUCostModel>& costModel);
size_t getDMACost(vpux::NDTypeInterface tensorType, VPUNN::VPUDevice vpuDevice,
                  const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts);
size_t getDPUCost(mlir::Operation* op);
size_t getAsyncExecuteCycleBegin(mlir::async::ExecuteOp op);
size_t getAsyncExecuteCycleEnd(mlir::async::ExecuteOp op);
VPUNN::DPUWorkload getDPUWorkload(VPUIP::DPUTaskOp dpuTaskOp, VPU::ArchKind arch);
size_t calculateCopyCycles(mlir::Operation* innerOp, VPU::ArchKind archKind,
                           const std::shared_ptr<VPUNN::VPUCostModel>& costModel);
size_t calculateShaveActCycles(VPUIP::SwKernelOp swKernelOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                               VPU::ArchKind arch);
std::vector<std::pair<int64_t, size_t>> calculateNceVariantCycles(VPUIP::NCEClusterTaskOp nceOp,
                                                                  const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                                                                  VPU::ArchKind arch, vpux::Logger log);
size_t calculateNceCycles(VPUIP::NCEClusterTaskOp nceOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                          VPU::ArchKind arch, vpux::Logger log, int64_t numDPU = 1);
vpux::Byte getSwKernelRunTotalAllocSize(VPUIP::SwKernelRun swKernelRun, ArrayRef<mlir::Value> inputs,
                                        ArrayRef<mlir::Value> outputBuffs, SmallVector<mlir::Value>& inputsForKernelRun,
                                        SmallVector<mlir::Value>& outputsForKernelRun);
std::unique_ptr<VPUNN::SWOperation> getVPUNNSWKernelOp(VPUIP::SwKernelOp swKernelOp);
std::unique_ptr<VPUNN::SWOperation> getVPUNNSWKernelOp(VPU::SWOpInterface operation);
std::unique_ptr<VPUNN::SWOperation> getVPUNNSWKernelOp(VPU::SWOpInterface operation, vpux::NDTypeInterface outputNDType,
                                                       ArrayRef<vpux::NDTypeInterface> inputTiles);
size_t getDPUTaskOpCost(VPUIP::DPUTaskOp dpuTaskOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                        VPU::ArchKind arch, vpux::Logger log);

VPUNN::MemoryLocation getMemoryLocation(mlir::Type type);
VPUNN::ActivationFunction getVPUNNActivationFunction(VPU::PPEAttr ppeAttr);

}  // namespace vpux
