//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/Pass/AnalysisManager.h>

#include <vpu/cycles_interface_types.h>
#include <vpu/layer.h>
#include <vpu/layer_split_info.h>

#include <memory>

namespace vpux::VPU {
class NCEOpInterface;
}  // namespace vpux::VPU
namespace VPUNN {
class VPUCostModel;
class VPULayerCostModel;
struct VPULayerStrategy;
}  // namespace VPUNN

namespace vpux {

float getWeightsSparsityRatio(vpux::NDTypeInterface weightsType, int64_t compressedSize);

namespace VPU {

/**
 * Analyzes the VPUNN Layer Cost Model with a custom `isInvalidated` function.
 * This analysis object remains preserved once constructed, until `invalidate` is called
 * or the AnalysisManager is cleared.
 */
class LayerCostModelAnalysis {
public:
    explicit LayerCostModelAnalysis(mlir::ModuleOp moduleOp);
    std::shared_ptr<VPUNN::VPULayerCostModel> getVPUNNLayerCostModel();

    // Used by AnalysisManager to check if this analysis should be preserved.
    // If return false, preserve this analysis. Otherwise release it.
    bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses&);

    // Invalidate this analysis
    // The resource will be destroyed after pass
    void invalidate();

    // If the input analysis is empty, create a layer cost model instance and return it.
    // Otherwise, return the cached layer cost model instance.
    static std::shared_ptr<VPUNN::VPULayerCostModel> getOrCreateLayerCostModel(
            std::optional<std::reference_wrapper<LayerCostModelAnalysis>> analysis, config::ArchKind arch,
            Logger log = Logger::global().nest("layer-cost-model-analysis"));

private:
    std::shared_ptr<VPUNN::VPULayerCostModel> _layerCostModel;

    // Flag indicating whether the analysis is preserved. Initialized to true.
    bool _preserved = true;
};

/**
 * Analyzes the VPUNN L1 Cost Model with a custom `isInvalidated` function.
 * This analysis object remains preserved once constructed, until `invalidate` is called
 * or the AnalysisManager is cleared.
 */
class CostModelAnalysis {
public:
    explicit CostModelAnalysis(mlir::ModuleOp moduleOp);
    std::shared_ptr<VPUNN::VPUCostModel> getVPUNNCostModel();

    // Determines if the analysis is invalidated based on preserved analyses.
    // Returns true if the analysis is invalidated, false otherwise.
    bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses&);

    // Invalidate this analysis
    // The resource will be destroyed after pass
    void invalidate();

    // If the input analysis is empty, create a cost model instance and return it.
    // Otherwise, return the cached cost model instance.
    static std::shared_ptr<VPUNN::VPUCostModel> getOrCreateCostModel(
            std::optional<std::reference_wrapper<CostModelAnalysis>> analysis, config::ArchKind arch,
            Logger log = Logger::global().nest("cost-model-analysis"));

private:
    std::shared_ptr<VPUNN::VPUCostModel> _costModel;

    // Flag indicating whether the analysis is preserved. Initialized to true.
    bool _preserved = true;
};

static constexpr uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
// The top 100 maximum UINT32 vals for error codes
static constexpr uint32_t NO_COST = 0;
static constexpr uint32_t UNIT_COST = 1;
static constexpr uint32_t INVALID_COST_BASE = MAX_VAL - 100;
static constexpr uint32_t ERROR_INPUT_TOO_BIG = MAX_VAL - 0;

constexpr StringRef VPUNN_PRE_SPLIT = "VPU.EnableVPUNNPreSplit";
bool hasVPUNNPreSplit(mlir::Operation* op);
uint32_t checkAndReturnCost(const VPUNN::CyclesInterfaceType& cost, vpux::Logger log, bool beSilent = false);
void printVPUNNLayerConfig(const VPUNN::DPULayer& layer, const VPUNN::VPULayerStrategy& strategy, vpux::Logger log);
void printVPUNNLayers(ArrayRef<VPUNN::DPULayer> layers, vpux::Logger log);
void printVPUNNWorkloadConfig(const VPUNN::DPUWorkload& wl, LogCb logCb = globalLogCb);
void printLayerSplitInfo(const VPUNN::LayerSplitInfo& info, const Logger& log);
VPU::MPEMode getMPEMode(VPUNN::ExecutionMode executionMode);

float getWeightsSparsityRatio(mlir::Value weights);
VPUNN::VPUDevice getVPUDeviceType(config::ArchKind archKind);
bool isVPUNNSupportedElementType(mlir::Type type);
std::optional<VPUNN::DataType> getVPUNNElementType(mlir::Type type);
VPUNN::Layout getVPUNNLayout(vpux::DimsOrder vpuxLayout);
VPUNN::VPUTensor getVPUTensor(ShapeRef shape, mlir::Type elemType, vpux::DimsOrder layout = vpux::DimsOrder::NHWC);
VPUNN::ExecutionMode getExecutionMode(VPU::MPEMode mpeMode);
VPUNN::VPULayerStrategy getVPULayerStrategy(VPU::MultiClusterStrategy mcStrategy, size_t nDPUs, size_t nTiles,
                                            config::ArchKind arch, size_t nSHVs = 1, bool prefetching = false,
                                            VPU::DistributionMode distributionMode = DistributionMode::NONE,
                                            mlir::Operation* op = nullptr);
VPUNN::DPULayer getDPULayer(const VPUIP::WorkloadCostParams& params);
std::vector<VPUNN::DPULayer> getPerClusterDPULayers(VPU::NCEOpInterface nceOp, const VPUIP::WorkloadCostParams& params,
                                                    Logger log);
VPUNN::DPUWorkload getDPUWorkload(const VPUIP::WorkloadCostParams& tileParams, const VPUIP::WorkloadTile& wl);
VPUIP::WorkloadCostParams getWorkloadCostParam(VPU::NCEOpInterface nceOp, config::ArchKind arch, int64_t numDPU,
                                               int64_t numTiles = 1);
vpux::VPU::ICostModelUtilsInterface* getICostModelUtilsInterface(mlir::MLIRContext* ctx);

}  // namespace VPU
}  // namespace vpux
