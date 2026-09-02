//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/dma_cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_fwd.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/Dialect/Async/IR/Async.h>

namespace vpux::VPU {
class SWOpInterface;
}  // namespace vpux::VPU
namespace VPUNN {
class VPUCostModel;
struct SWOperation;
struct DPUWorkload;
class VPUTensor;
enum class VPUTilingStrategy;
enum class VPUDevice;
class SHAVEWorkload;
enum class Swizzling;
enum class ActivationFunction;
class SEPModeInfo;
enum class ISIStrategy;
}  // namespace VPUNN

namespace vpux {

constexpr StringLiteral DPUCost = "minimumHardwareExecutionCost";
constexpr StringLiteral cycleCostAttrName = "cycleCost";
constexpr StringLiteral cycleBegin = "cycleBegin";
constexpr StringLiteral cycleEnd = "cycleEnd";

size_t getDPUCost(mlir::Operation* op);
size_t getAsyncExecuteCycleBegin(mlir::async::ExecuteOp op);
size_t getAsyncExecuteCycleEnd(mlir::async::ExecuteOp op);
VPUNN::DPUWorkload getDPUWorkload(VPUIP::DPUTaskOp dpuTaskOp, [[maybe_unused]] config::ArchKind arch);
size_t calculateShaveActCycles(VPUIP::SwKernelOp swKernelOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel);
std::vector<std::pair<int64_t, size_t>> calculateNceVariantCycles(VPUIP::NCEClusterTaskOp nceOp,
                                                                  const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                                                                  config::ArchKind arch, vpux::Logger log);
size_t calculateNceCycles(VPUIP::NCEClusterTaskOp nceOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                          config::ArchKind arch, vpux::Logger log, int64_t numDPU = 1);
vpux::Byte getSwKernelRunTotalAllocSize(VPUIP::SwKernelRun swKernelRun, ArrayRef<mlir::Value> inputs,
                                        ArrayRef<mlir::Value> outputBuffs, SmallVector<mlir::Value>& inputsForKernelRun,
                                        SmallVector<mlir::Value>& outputsForKernelRun);
std::unique_ptr<VPUNN::SHAVEWorkload> getVPUNNSWKernelOp(VPUIP::SwKernelOp swKernelOp);
std::unique_ptr<VPUNN::SHAVEWorkload> getVPUNNSWKernelOp(VPU::SWOpInterface operation);
std::unique_ptr<VPUNN::SHAVEWorkload> getVPUNNSWKernelOp(VPU::SWOpInterface operation,
                                                         ArrayRef<vpux::NDTypeInterface> outputTypes,
                                                         ArrayRef<vpux::NDTypeInterface> inputTiles);
std::unique_ptr<VPUNN::SHAVEWorkload> getVPUNNSWKernelOp(VPU::SWOpInterface operation,
                                                         const std::vector<VPUNN::VPUTensor>& outputTensors,
                                                         const std::vector<VPUNN::VPUTensor>& inputTensors);
size_t getDPUTaskOpCost(VPUIP::DPUTaskOp dpuTaskOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                        config::ArchKind arch, vpux::Logger log);

VPUNN::Swizzling getVPUNNSwizzlingKey(mlir::Type type);
VPUNN::ActivationFunction getVPUNNActivationFunction(VPU::PPEAttr ppeAttr);
VPUNN::SEPModeInfo getSEPModeInfo(const VPUIP::SEPInfo& sepInfo);

std::string stringifyVPUNNStrategy(VPUNN::VPUTilingStrategy strategy);
uint64_t addSaturating(uint64_t a, uint64_t b);
}  // namespace vpux
