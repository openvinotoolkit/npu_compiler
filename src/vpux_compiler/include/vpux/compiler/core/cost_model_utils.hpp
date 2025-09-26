//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/Dialect/Async/IR/Async.h>

#include <vpu/dma_types.h>
#include <vpu/dpu_types.h>
#include <vpu/vpu_tiling_strategy.h>

namespace vpux::VPU {
class SWOpInterface;
}  // namespace vpux::VPU
namespace vpux::VPUIP {
class DPUTaskOp;
class NCEClusterTaskOp;
class SwKernelOp;
class SwKernelRun;
}  // namespace vpux::VPUIP
namespace VPUNN {
class VPUCostModel;
struct SWOperation;
struct DPUWorkload;
}  // namespace VPUNN

namespace vpux {

constexpr StringLiteral DPUCost = "minimumHardwareExecutionCost";
constexpr StringLiteral cycleCostAttrName = "cycleCost";
constexpr StringLiteral cycleBegin = "cycleBegin";
constexpr StringLiteral cycleEnd = "cycleEnd";

size_t getDMACost(mlir::Value input, mlir::Value output, config::ArchKind archKind,
                  const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts = 1);
size_t getDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType, VPUNN::VPUDevice vpuDevice,
                  const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts);
size_t getDMACost(vpux::NDTypeInterface tensorType, VPUNN::VPUDevice vpuDevice,
                  const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts);
size_t getDPUCost(mlir::Operation* op);
size_t getAsyncExecuteCycleBegin(mlir::async::ExecuteOp op);
size_t getAsyncExecuteCycleEnd(mlir::async::ExecuteOp op);
VPUNN::DPUWorkload getDPUWorkload(VPUIP::DPUTaskOp dpuTaskOp, config::ArchKind arch);
size_t calculateCopyCycles(mlir::Operation* innerOp, config::ArchKind archKind,
                           const std::shared_ptr<VPUNN::VPUCostModel>& costModel);
size_t calculateShaveActCycles(VPUIP::SwKernelOp swKernelOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                               config::ArchKind arch);
std::vector<std::pair<int64_t, size_t>> calculateNceVariantCycles(VPUIP::NCEClusterTaskOp nceOp,
                                                                  const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                                                                  config::ArchKind arch, vpux::Logger log);
size_t calculateNceCycles(VPUIP::NCEClusterTaskOp nceOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                          config::ArchKind arch, vpux::Logger log, int64_t numDPU = 1);
vpux::Byte getSwKernelRunTotalAllocSize(VPUIP::SwKernelRun swKernelRun, ArrayRef<mlir::Value> inputs,
                                        ArrayRef<mlir::Value> outputBuffs, SmallVector<mlir::Value>& inputsForKernelRun,
                                        SmallVector<mlir::Value>& outputsForKernelRun);
std::unique_ptr<VPUNN::SWOperation> getVPUNNSWKernelOp(VPUIP::SwKernelOp swKernelOp);
std::unique_ptr<VPUNN::SWOperation> getVPUNNSWKernelOp(VPU::SWOpInterface operation);
std::unique_ptr<VPUNN::SWOperation> getVPUNNSWKernelOp(VPU::SWOpInterface operation, vpux::NDTypeInterface outputNDType,
                                                       ArrayRef<vpux::NDTypeInterface> inputTiles);
size_t getDPUTaskOpCost(VPUIP::DPUTaskOp dpuTaskOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                        config::ArchKind arch, vpux::Logger log);

VPUNN::MemoryLocation getMemoryLocation(mlir::Type type);
VPUNN::Swizzling getVPUNNSwizzlingKey(mlir::Type type);
VPUNN::ActivationFunction getVPUNNActivationFunction(VPU::PPEAttr ppeAttr);
VPUNN::SEPModeInfo getSEPModeInfo(VPUIP::SEPInfo sepInfo);

std::string stringifyVPUNNStrategy(VPUNN::VPUTilingStrategy strategy);
}  // namespace vpux
