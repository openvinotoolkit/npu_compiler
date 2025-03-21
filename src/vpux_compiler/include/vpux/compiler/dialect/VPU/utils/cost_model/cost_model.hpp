//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"

#include <vpu_cost_model.h>
#include <vpu_layer_cost_model.h>

#include <memory>

namespace vpux {

float getWeightsSparsityRatio(vpux::NDTypeInterface weightsType, int64_t compressedSize);

namespace VPU {
static constexpr uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
// The top 100 maximum UINT32 vals for error codes
static constexpr uint32_t NO_COST = 0;
static constexpr uint32_t UNIT_COST = 1;
static constexpr uint32_t INVALID_COST_BASE = MAX_VAL - 100;
static constexpr uint32_t ERROR_INPUT_TOO_BIG = MAX_VAL - 0;

std::shared_ptr<VPUNN::VPUCostModel> createCostModel(ArchKind arch);
std::shared_ptr<VPUNN::VPULayerCostModel> createLayerCostModel(ArchKind arch);
uint32_t checkAndReturnCost(const VPUNN::CyclesInterfaceType& cost, vpux::Logger log, bool beSilent = false);
void printVPUNNLayerConfig(const VPUNN::DPULayer& layer, const VPUNN::VPULayerStrategy& strategy, vpux::Logger log);
void printVPUNNWorkloadConfig(const VPUNN::DPUWorkload& wl, LogCb logCb = globalLogCb);
float getWeightsSparsityRatio(mlir::Value weights);
VPUNN::VPUDevice getVPUDeviceType(VPU::ArchKind archKind);
bool isVPUNNSupportedElementType(mlir::Type type);
std::optional<VPUNN::DataType> getVPUNNElementType(mlir::Type type);
VPUNN::Layout getVPUNNLayout(VPUIPDPU::ODUPermuteDataMode oduPermutation);
VPUNN::VPUTensor getVPUTensor(ShapeRef shape, mlir::Type elemType,
                              VPUIPDPU::ODUPermuteDataMode oduPermutation = VPUIPDPU::ODUPermuteDataMode::PERMUTE_ZXY);
VPUNN::ExecutionMode getExecutionMode(VPU::MPEMode mpeMode);
VPUNN::VPULayerStrategy getVPULayerStrategy(VPU::MultiClusterStrategy mcStrategy, size_t nDPUs, size_t nTiles,
                                            ArchKind arch, size_t nSHVs = 1, bool prefetching = false,
                                            VPU::DistributionMode distributionMode = DistributionMode::NONE);
VPUNN::DPULayer getDPULayer(const VPUIP::WorkloadCostParams& params);
VPUNN::DPUWorkload getDPUWorkload(const VPUIP::WorkloadCostParams& tileParams, const VPUIP::WorkloadTile& wl);
VPUIP::WorkloadCostParams getWorkloadCostParam(VPU::NCEOpInterface nceOp, VPU::ArchKind arch, int64_t numDPU,
                                               int64_t numTiles = 1);

}  // namespace VPU
}  // namespace vpux
