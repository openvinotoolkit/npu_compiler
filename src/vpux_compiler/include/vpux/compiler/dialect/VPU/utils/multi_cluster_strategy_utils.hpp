//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/vpu_tiling_strategy.h>
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/operation_strategies.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"
#include "vpux/utils/core/dense_map.hpp"

namespace VPUNN {
class VPULayerCostModel;
struct VPULayerStrategy;
}  // namespace VPUNN

namespace vpux::VPU {

struct HwLayerTilingStrategyCosts {
    double costWithoutPrefetching = 0;
    double costWithPrefetching = 0;
};

struct ComputeAndDMATimeCost {
    SmallVector<uint32_t> actCosts;
    SmallVector<uint32_t> weightsCosts;
    SmallVector<uint32_t> computeCosts;
    SmallVector<uint32_t> outputCosts;
};

//
// LayerCostModel for layer cost estimation given by different strategies
//
class LayerCostModel final {
public:
    struct SpillingCost {
        double writeCost;
        double readCost;
    };

    struct MultiClusterStrategyInfo {
        double cost = COST_MAX;
        bool fitsIntoCMX = false;
        bool usesFullTiles = true;
    };

    // Bucket #     - Inclusion Criteria            - Evaluation Method
    //
    // Bucket 0     - valid VPUNN cost, fits cmx    - lowest cost
    // Bucket 1     - valid VPUNN cost, no fits cmx - lowest cost
    // Bucket 2     - fits cmx, uses all tiles      - prefer activation or channel split based on tensor sizes
    // Bucket 3     - compatible                    - prefer activation or channel split based on tensor sizes
    enum class PriorityBucket {
        COST_BASED_FITS_CMX,
        COST_BASED_NOT_FITS_CMX,
        HEURISTIC_FITS_CMX_FULL_TILES,
        HEURISTIC_COMPATIBLE_ONLY
    };
    using StrategyInfoPair = std::pair<VPU::MultiClusterStrategy, MultiClusterStrategyInfo>;

    explicit LayerCostModel(mlir::func::FuncOp func, bool enablePrefetchTiling, Logger log,
                            SiblingOpsAnalysis& siblingsOpsAnalysis);
    explicit LayerCostModel(mlir::func::FuncOp func, bool enablePrefetchTiling, SiblingOpsAnalysis& siblingsOpsAnalysis,
                            std::shared_ptr<VPUNN::VPULayerCostModel> layerCostModelPtr, Logger log);
    explicit LayerCostModel(mlir::func::FuncOp func, bool enablePrefetchTiling, bool enableTilingFullSearchSpace,
                            SiblingOpsAnalysis& siblingsOpsAnalysis,
                            std::shared_ptr<VPUNN::VPULayerCostModel> layerCostModelPtr, Logger log);
    ~LayerCostModel() = default;

    /* Use layer costs from VPUNN with assumed tiling, and MC preference heuristics to get optimal starting MC strategy
     * for subgraph optimization*/
    VPU::MultiClusterStrategy getOptimalLayerStrategy(VPU::ClusteredOpInterface clusteredOp);

    /* Functions for getting VPUNN cost of layers.
     * Note: MC strategy will be temporarily set in IR, not suitable for multi-threading */
    double getLayerCost(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy,
                        bool useTimeBasedCost = true);

    HwLayerTilingStrategyCosts getDPUandDMATimeCostWithCustomTiling(VPU::NCEOpInterface nceOp,
                                                                    VPU::MultiClusterStrategy strategy,
                                                                    const OutputTiling& outTiles) const;
    ComputeAndDMATimeCost getComputeAndDataCosts(mlir::Operation* op, std::optional<VPU::MultiClusterStrategy> strategy,
                                                 const OutputTiling& outTiles) const;

    std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTiledOpOperands(
            mlir::Operation* op, const TileInfo& outTile, std::optional<VPU::MultiClusterStrategy> customStrategy);

    SpillingCost calculateSpillingCost(VPU::ClusteredOpInterface parentOp, VPU::ClusteredOpInterface userOp,
                                       VPU::MultiClusterStrategy parentStrategy,
                                       VPU::MultiClusterStrategy userStrategy) const;

    /* DMA spills cost */
    double getSpillingDMACost(vpux::NDTypeInterface srcTensorType) const;
    double getSpillingDMACost(vpux::NDTypeInterface srcTensorType, const TensorDistributionMap& distributions) const;

    /* Helper functions for subgraph optimization of MC strategy*/
    bool hasMultiClusterStrategy(mlir::Operation* op) const;
    VPU::MultiClusterStrategy getMultiClusterStrategyValue(VPU::ClusteredOpInterface clusteredOp) const;

    bool hasSpilling(VPU::ClusteredOpInterface /*clusteredOp*/,
                     std::pair<mlir::Type, TensorDistributionMap>& srcTensorType,
                     std::pair<mlir::Type, TensorDistributionMap>& dstTensorType) const;
    bool hasSpilling(VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy origOpStrategy,
                     VPU::ClusteredOpInterface userOp, VPU::MultiClusterStrategy userOpStrategy) const;

    bool doesLayerRequireTiling(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy) const;
    bool doesLayerHaveVPUNNSupportedTypes(VPU::ClusteredOpInterface clusteredOp) const;

    std::pair<mlir::Type, TensorDistributionMap> getInputWithDistribution(
            VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp,
            VPU::MultiClusterStrategy specifiedStrategy) const;
    std::pair<mlir::Type, TensorDistributionMap> getInputWithDistribution(
            VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp, VPU::MultiClusterStrategy specifiedStrategy,
            mlir::ArrayRef<int64_t> customAlignment) const;

    std::pair<mlir::Type, TensorDistributionMap> getOutputWithDistribution(
            VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy specifiedStrategy) const;

    bool isUnderSubgraphOpt() const;
    void setUnderSubgraphOpt(bool underSubgraphOpt);

    double static constexpr COST_MAX = std::numeric_limits<double>::infinity();
    void resetNNCacheCounter();
    void printNNCacheStatistics() const;

private:
    /* Cost calculation - time based or efficiency based */
    double getNCELayerCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy, bool useTimeBasedCost = true);
    double getSWLayerCost(VPU::SWOpInterface swOp, VPU::MultiClusterStrategy strategy);
    double getDPUandDMATimeCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy);
    double getEfficiencyCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;

    SpillingCost getSpillingCost(vpux::NDTypeInterface srcTensorType, const TensorDistributionMap& srcDistribution,
                                 vpux::NDTypeInterface dstTensorType, const TensorDistributionMap& dstDistribution,
                                 VPU::ClusteredOpInterface parentOp, VPU::ClusteredOpInterface userOp) const;
    SpillingCost getSpillingCost(vpux::NDTypeInterface srcTensorType, vpux::NDTypeInterface dstTensorType,
                                 VPU::ClusteredOpInterface parentOp, VPU::ClusteredOpInterface userOp) const;

    bool hasSpilling(VPU::ClusteredOpInterface origOp, vpux::NDTypeInterface srcTensorType,
                     vpux::NDTypeInterface dstTensorType) const;

    std::pair<vpux::NDTypeInterface, vpux::NDTypeInterface> getDistributionTypesWithStrategy(
            VPU::ClusteredOpInterface parentOp, VPU::MultiClusterStrategy parentStrategy,
            VPU::ClusteredOpInterface userOp, VPU::MultiClusterStrategy userStrategy) const;
    std::pair<std::pair<mlir::Type, TensorDistributionMap>, std::pair<mlir::Type, TensorDistributionMap>>
    getDistributionsWithStrategy(VPU::ClusteredOpInterface parentOp, VPU::MultiClusterStrategy parentStrategy,
                                 VPU::ClusteredOpInterface userOp, VPU::MultiClusterStrategy userStrategy) const;

    /* DMA cost calculation helpers */
    double getDMACostOfType(vpux::NDTypeInterface srcTensorType) const;
    double getDMACostOfType(vpux::NDTypeInterface srcType, const DistributionInfo& distribution) const;

    /* Greedy strategy helpers */
    bool preferChannelSplitting(VPU::ClusteredOpInterface clusteredOp);
    bool isCompatible(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy);
    bool usesFullTiles(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy);
    bool useSOKWhenMVNSideOrFitCMX(VPU::ClusteredOpInterface clusteredOp, ArrayRef<StrategyInfoPair> bucket);

    /* Efficiency cost helpers */
    double computeSplitEfficiency(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    double calculateMPEVolume(VPU::MPEMode mpeMode, Shape shape) const;

    // CostCache has two-levels mappings:
    // The first-level mapping is from NCE op to op costs, op costs are represent by SmallVector<double>.
    // The second-level mapping is from op costs to op stratey cost value.
    using CostCache = DenseMap<VPU::NCEOpInterface, SmallVector<double>>;
    CostCache _costCache;

    const double _DDRLatency = 100;  // DDR latency is ~100 cycles per dma
    double _DMABandwidth = 0.0;      // Transition Bytes per cycle
    int64_t _numTiles = 0;           // Number of Tiles
    int64_t _numDPUs = 0;            // Number of DPUs per cluster
    int64_t _numShaveActs = 0;       // Number of ACT_SHVs per cluster
    int64_t _numDMAPorts = 1;        // Number of the DMA ports
    config::ArchKind _arch;
    VPUNN::VPUDevice _vpuDeviceType;
    std::shared_ptr<VPUNN::VPULayerCostModel> _layerCostModel;
    mlir::func::FuncOp _func;
    bool _enablePrefetchTiling;
    Logger _log;
    SiblingOpsAnalysis& _siblingsOpsAnalysis;
    bool _underSubgraphOpt = false;
    bool _enableTilingFullSearchSpace = false;
};
vpux::VPU::StrategyCost correctSwOpCost(VPU::SWOpInterface swOp, ArrayRef<vpux::NDTypeInterface> tiledInputTypes,
                                        vpux::VPU::StrategyCost cost);

std::optional<VPU::MultiClusterStrategy> getDefaultLayerStrategy(VPU::ClusteredOpInterface clusteredOp);

bool isStrategyCompatibleShape(VPU::ClusteredOpInterface clusteredOp, const vpux::TileInfo& outputTile,
                               VPU::MultiClusterStrategy strategy, Logger log);

bool isStrategySOXCompatible(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy,
                             size_t numTiles);

SmallVector<uint32_t> getDPUCostForNCEOp(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy mcStrategy,
                                         const OutputTiling& outTiles, const VPUIP::WorkloadCostParams& costParams,
                                         VPUNN::VPULayerStrategy vpunnStrategy,
                                         const std::shared_ptr<VPUNN::VPULayerCostModel>& vpunnCostModel, Logger log);

/*
 * Get DPU cost with LayersPreSplit L2 API
 * Compiler splits per cluster before feeding into cost model
 */
SmallVector<uint32_t> getDPUCostForNCEOpPreSplit(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                                 const VPUIP::WorkloadCostParams& costParams,
                                                 VPUNN::VPUTilingStrategy vpunnTilingStrategy,
                                                 const std::shared_ptr<VPUNN::VPULayerCostModel>& vpunnCostModel,
                                                 int64_t numDPU, Logger log);

SmallVector<uint32_t> getSHAVECostForSwOpPreSplit(VPU::SWOpInterface swOp, const OutputTiling& outTiles,
                                                  const VPUIP::ShaveWorkloadCostParams& costParams,
                                                  const std::shared_ptr<VPUNN::VPULayerCostModel>& vpunnCostModel,
                                                  int64_t numSHV, Logger log);
SmallVector<uint32_t> getSHAVECostForSwOpPreSplit(VPU::SWOpInterface swOp,
                                                  std::optional<VPU::MultiClusterStrategy> strategy,
                                                  const OutputTiling& outTiles,
                                                  const VPUIP::ShaveWorkloadCostParams& costParams,
                                                  const std::shared_ptr<VPUNN::VPULayerCostModel>& vpunnCostModel,
                                                  int64_t numSHV, Logger log);

SmallVector<uint32_t> getPerTileWeightsDMACosts(
        VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy, SiblingOpsAnalysis& siblingsAnalysis,
        ArrayRef<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes,
        std::function<uint32_t(NDTypeInterface, const TensorDistributionMap& distributions)> getSpillingReadCostFunc);

SmallVector<uint32_t> getPerTileActivationDMACosts(
        VPU::NCEOpInterface nceOp, ArrayRef<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes,
        std::function<uint32_t(NDTypeInterface, const TensorDistributionMap& distributions)> getSpillingReadCostFunc,
        VPU::MultiClusterStrategy strategy, int64_t numTiles);

SmallVector<uint32_t> getPerTileOutputDMACosts(
        VPU::NCEOpInterface nceOp, ArrayRef<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes,
        std::function<uint32_t(NDTypeInterface, const TensorDistributionMap& distributions)> getSpillingReadCostFunc);

std::pair<uint32_t, uint32_t> getWeightsDMACostForNCEOp(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                                        SmallVector<uint32_t>& layerDPUCosts,
                                                        ArrayRef<uint32_t> layerDMACosts, bool enablePrefetchTiling,
                                                        vpux::Logger log);

uint32_t getActivationDMACostForNCEOp(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                      SmallVector<uint32_t>& layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                      bool enablePrefetchTiling, vpux::Logger log);

uint32_t getOutputDMACostForNCEOp(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                  SmallVector<uint32_t>& layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                  bool enablePrefetchTiling, vpux::Logger log);

size_t getNumNonConstantOperands(mlir::Operation* op);

bool hasLayerWithMultipleInputs(mlir::Operation* op);

bool isSingleBatchRequired(mlir::Operation* op);

bool setSOKForRuntimeDequantConvolution(VPU::NCEOpInterface nceOp, LayerCostModel& costModel);

bool alignStrategyWithParentRuntimeDequant(VPU::ClusteredOpInterface clusteredOp, LayerCostModel& costModel);

bool isNCEOpUsedAsConvScale(mlir::Operation* op);

std::optional<VPU::MultiClusterStrategy> getMultiClusterStrategyFromOp(mlir::Operation* op);
}  // namespace vpux::VPU
