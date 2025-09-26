//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/factories/cost_model_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/op_tiling_cache.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/sparsity.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include <vpu/shave/layers.h>
#include <vpu_layer_cost_model.h>

using namespace vpux;
using namespace VPU;

namespace {

// Stride DMA cost is inaccurate by cost model, so use this variable to help correct the cost value
// TODO: Ticket E#171462, remove this variable when stride DMA cost is accurate by VPUNN cost model
constexpr double strideDMACorrectionThresholdInBitsV1 = 512;

double getSpillingCostForNonMultiCluster(vpux::NDTypeInterface tensorType, const VPU::DistributionInfo&,
                                         SpillingType /*spillingType*/, double ddrLatency, double ddrBandwidth,
                                         double /*cmxLatency*/, double /*cmxBandwidth*/, int64_t /*numDMAPorts*/) {
    // calculate the data byte size need copy from cmx to ddr or vice versa
    const auto totalSize = static_cast<double>(tensorType.getTotalAllocSize().count());
    return ddrLatency + totalSize / ddrBandwidth;
}

double getSpillingCostForDuplicated(vpux::NDTypeInterface tensorType, const VPU::DistributionInfo& distribution,
                                    SpillingType /*spillingType*/, double ddrLatency, double ddrBandwidth,
                                    double /*cmxLatency*/, double /*cmxBandwidth*/, int64_t /*numDMAPorts*/) {
    TensorDistributionMap distributionMap;
    distributionMap.insert(std::make_pair(tensorType, distribution));
    const auto totalSize = getTotalAllocSizeWithDistribution(tensorType, distributionMap);
    return ddrLatency + totalSize.count() / ddrBandwidth;
}

double getSpillingCostForSegmented(vpux::NDTypeInterface tensorType, const VPU::DistributionInfo& distribution,
                                   SpillingType, double ddrLatency, double ddrBandwidth, double, double,
                                   int64_t numDMAPorts) {
    SmallVector<Shape> perClusterMemShapes{};
    if (distribution.getMemoryShapes().size() == 0) {
        auto optionalPerClusterMemoryShapes = VPU::getPerClusterMemoryShapes(tensorType.getShape(), distribution);
        VPUX_THROW_UNLESS(optionalPerClusterMemoryShapes.has_value(),
                          "Cannot get per cluster memory shapes. Shape {0}, Unsupported distribution: {1}",
                          tensorType.getShape(), distribution);
        perClusterMemShapes = optionalPerClusterMemoryShapes.value();
    } else {
        for (auto& shape : distribution.getMemoryShapes()) {
            perClusterMemShapes.push_back(Shape(shape));
        }
    }

    // Aggregate the total size which needs to be transfered on each DMA port
    auto totalSizeOnPorts = SmallVector<int64_t>(numDMAPorts, 0);
    for (size_t i = 0; i < perClusterMemShapes.size(); ++i) {
        totalSizeOnPorts[i % numDMAPorts] += perClusterMemShapes[i].totalSize();
    }
    // Considering multiple ports used in parallel, only take into account the largest size to transfer
    auto totalSize = *std::max_element(totalSizeOnPorts.begin(), totalSizeOnPorts.end());

    const Bit elemSize = tensorType.getElemTypeSize();
    totalSize = alignMemSize(elemSize * totalSize, Byte(1)).to<Byte>().count();
    return ddrLatency + static_cast<double>(totalSize) / ddrBandwidth;
}

using GetSpillingCostCB = double (*)(vpux::NDTypeInterface, const VPU::DistributionInfo& distribution, SpillingType,
                                     double ddrLatency, double ddrBandwidth, double cmxLatency, double cmxBandwidth,
                                     int64_t numDMAPorts);
const EnumMap<DistributionMode, GetSpillingCostCB> spillingCostMap{
        // using  DistributionMode::NONE for single clustering case
        {DistributionMode::NONE, getSpillingCostForNonMultiCluster},
        {DistributionMode::DUPLICATED, getSpillingCostForDuplicated},
        {DistributionMode::SEGMENTED, getSpillingCostForSegmented},
        {DistributionMode::OVERLAPPED, getSpillingCostForSegmented},
        {DistributionMode::MULTICASTED, getSpillingCostForDuplicated},
        {DistributionMode::DUPLICATED | DistributionMode::SEGMENTED, getSpillingCostForDuplicated},
        {DistributionMode::MULTICASTED | DistributionMode::SEGMENTED, getSpillingCostForDuplicated},
};

mlir::Value getInputFromClusteredOp(VPU::ClusteredOpInterface clusteredOp, mlir::Operation* parentOp) {
    for (auto operand : clusteredOp->getOperands()) {
        auto parent = operand.getDefiningOp();
        if (parent == parentOp) {
            return operand;
        }
        while (mlir::isa_and_nonnull<VPU::DistributedCastOpInterface, VPU::ShapeCastOp, VPU::GroupSparseTensorOp>(
                parent)) {
            // propagate cast ops
            parent = parent->getOperand(0).getDefiningOp();
            if (parent == parentOp) {
                return operand;
            }
        }
    }

    VPUX_THROW("Cannot find input from op: {0}, parent op: {1}", clusteredOp, parentOp);
}

bool useSOKWhenMVNSide(VPU::ClusteredOpInterface clusteredOp) {
    // This check is a W/A for NPU40 or lower arch, where VPUNN cost is not accurate or not available (error code)
    // For newer platforms, a more accurate VPUNN cost model is provided so that this check should be dropped
    auto arch = config::getArch(clusteredOp.getOperation());
    if (arch > config::ArchKind::NPU40XX) {
        return false;
    }

    if (!mlir::isa<mlir::BlockArgument>(clusteredOp->getOperand(0)) &&
        mlir::isa<VPU::MVNOp>(clusteredOp->getOperand(0).getDefiningOp())) {
        // MVN producer
        return true;
    }
    for (auto* user : clusteredOp->getResult(0).getUsers()) {
        // MVN consumer
        if (mlir::isa<VPU::MVNOp>(user)) {
            return true;
        }
    }
    return false;
}

bool isSOHAlignmentCompatibleOrAdjustedCompatible(std::pair<mlir::Type, VPU::DistributionInfo>& srcTypeDistribution,
                                                  std::pair<mlir::Type, VPU::DistributionInfo>& dstTypeDistribution) {
    auto srcType = mlir::cast<vpux::NDTypeInterface>(srcTypeDistribution.first);
    auto dstType = mlir::cast<vpux::NDTypeInterface>(dstTypeDistribution.first);

    if (srcType.getShape() != dstType.getShape()) {
        return false;
    }
    if (srcType.getDimsOrder() != dstType.getDimsOrder() || srcType.getDimsOrder() != DimsOrder::NHWC) {
        return false;
    }

    auto srcDistribution = srcTypeDistribution.second;
    auto dstDistribution = dstTypeDistribution.second;
    if ((srcDistribution.getDistributionMode() != DistributionMode::SEGMENTED) ||
        (dstDistribution.getDistributionMode() != DistributionMode::SEGMENTED) ||
        (srcDistribution.getNumTiles() != dstDistribution.getNumTiles())) {
        return false;
    }

    return true;
}

bool isDistributedCastCompatible(std::pair<mlir::Type, VPU::DistributionInfo>& srcTypeDistribution,
                                 std::pair<mlir::Type, VPU::DistributionInfo>& dstTypeDistribution) {
    auto srcType = mlir::cast<vpux::NDTypeInterface>(srcTypeDistribution.first);
    auto dstType = mlir::cast<vpux::NDTypeInterface>(dstTypeDistribution.first);
    auto srcDistribution = srcTypeDistribution.second;
    auto dstDistribution = dstTypeDistribution.second;
    if (srcType.getShape() != dstType.getShape()) {
        return false;
    }

    if (areDistributionElementTypesCompatible(srcType.getElementType(), dstType.getElementType()).failed()) {
        return false;
    }

    if (srcType.getMemSpace() != dstType.getMemSpace()) {
        return false;
    }

    if (srcType.getDimsOrder() != dstType.getDimsOrder()) {
        return false;
    }

    if (areDistributionsCompatible(srcType, srcDistribution, dstType, dstDistribution).failed()) {
        return false;
    }

    return true;
}

bool isTargetTensorTypeCompatible(std::pair<mlir::Type, VPU::DistributionInfo>& srcType,
                                  std::pair<mlir::Type, VPU::DistributionInfo>& dstType) {
    const auto& srcTypeDistribution = srcType.second;
    const auto& dstTypeDistribution = dstType.second;
    const auto srcTypeIsDistributed = srcTypeDistribution.getDistributionMode() != DistributionMode::NONE;
    const auto dstTypeIsDistributed = dstTypeDistribution.getDistributionMode() != DistributionMode::NONE;
    if (srcTypeIsDistributed ^ dstTypeIsDistributed) {
        return false;
    }
    if (srcTypeIsDistributed && dstTypeIsDistributed) {
        if (!isDistributedCastCompatible(srcType, dstType)) {
            return false;
        }
    }
    return true;
}

uint32_t getPrefetchDMACostOverlappsWithPreviousDPU(SmallVector<uint32_t>& layerDPUCosts,
                                                    ArrayRef<uint32_t> layerDMACosts, bool isDMAOverlapsWithDPU) {
    VPUX_THROW_UNLESS(layerDPUCosts.size() == layerDMACosts.size(), "Size of DPU and DMA costs should be equal.");
    VPUX_THROW_WHEN(layerDPUCosts.empty(), "DPU costs should not be empty.");

    uint32_t totalDMACost = 0;
    if (isDMAOverlapsWithDPU) {
        for (size_t tileIdx = 0; tileIdx < layerDMACosts.size() - 1; ++tileIdx) {
            if (layerDMACosts[tileIdx + 1] > layerDPUCosts[tileIdx]) {
                // Prefetched and all the DPU cycles are overlapped with DMA
                totalDMACost += (layerDMACosts[tileIdx + 1] - layerDPUCosts[tileIdx]);
                layerDPUCosts[tileIdx] = 0;
            } else {
                // Prefetched and some DPU cycles are still not overlapped
                layerDPUCosts[tileIdx] -= layerDMACosts[tileIdx + 1];
            }
        }
    }
    totalDMACost += layerDMACosts[0];
    return totalDMACost;
}

uint32_t getOutputDMACostOverlappsWithNextDPU(SmallVector<uint32_t>& layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                              bool isDMAOverlapsWithDPU) {
    VPUX_THROW_UNLESS(layerDPUCosts.size() == layerDMACosts.size(), "Size of DPU and DMA costs should be equal.");
    VPUX_THROW_WHEN(layerDPUCosts.empty(), "DPU costs should not be empty.");

    uint32_t totalDMACost = 0;
    if (isDMAOverlapsWithDPU) {
        for (size_t tileIdx = 0; tileIdx < layerDMACosts.size() - 1; ++tileIdx) {
            if (layerDMACosts[tileIdx] > layerDPUCosts[tileIdx + 1]) {
                // Prefetched and all the DPU cycles are overlapped with DMA
                totalDMACost += (layerDMACosts[tileIdx] - layerDPUCosts[tileIdx + 1]);
                layerDPUCosts[tileIdx + 1] = 0;
            } else {
                // Prefetched and some DPU cycles are still not overlapped
                layerDPUCosts[tileIdx + 1] -= layerDMACosts[tileIdx];
            }
        }
    }
    totalDMACost += layerDMACosts.back();
    return totalDMACost;
}

bool isTiledOnLowestDim(const TileInfo& outTile, DimsOrder dimOrder) {
    const auto lowestDim = dimOrder.dimAt(dimOrder.numDims() - 1);
    for (auto item : outTile.axis | indexed) {
        const auto index = item.index();
        const auto axis = item.value();
        if (axis > 1 && Dim(index) == lowestDim) {
            return true;
        }
    }
    return false;
}

}  // namespace

LayerCostModel::LayerCostModel(mlir::func::FuncOp func, bool enablePrefetchTiling, Logger log,
                               SiblingOpsAnalysis& siblingsOpsAnalysis)
        : _func(func),
          _enablePrefetchTiling(enablePrefetchTiling),
          _log(log),
          _siblingsOpsAnalysis(siblingsOpsAnalysis) {
    auto module = func->getParentOfType<mlir::ModuleOp>();

    if (auto tileOp = config::getTileExecutor(module)) {
        auto dpuExec = tileOp.getSubExecutor(VPU::ExecutorKind::DPU);
        _NCEFrequency = tileOp.getProcessorFrequency().getValueAsDouble();
        _numTiles = tileOp.getCount();
        _numDPUs = dpuExec.getCount();
        _NCEThroughput = getNCEThroughput();
        _DMABandwidth = getDMABandwidth(config::getArch(tileOp), config::getRevisionID(module));
        if (auto shaveActExec = tileOp.getSubExecutor(ExecutorKind::SHAVE_ACT)) {
            _numShaveActs = shaveActExec.getCount();
        }
    }
    _numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();
    _arch = config::getArch(module);
    _vpuDeviceType = VPU::getVPUDeviceType(_arch);
    _layerCostModel = VPU::CostModelConfig::createLayerCostModel(_arch);
}

LayerCostModel::LayerCostModel(mlir::func::FuncOp func, bool enablePrefetchTiling,
                               SiblingOpsAnalysis& siblingsOpsAnalysis,
                               std::shared_ptr<VPUNN::VPULayerCostModel> layerCostModelPtr, Logger log)
        : _layerCostModel(std::move(layerCostModelPtr)),
          _func(func),
          _enablePrefetchTiling(enablePrefetchTiling),
          _log(log),
          _siblingsOpsAnalysis(siblingsOpsAnalysis) {
    auto module = func->getParentOfType<mlir::ModuleOp>();

    if (auto tileOp = config::getTileExecutor(module)) {
        auto dpuExec = tileOp.getSubExecutor(VPU::ExecutorKind::DPU);
        _NCEFrequency = tileOp.getProcessorFrequency().getValueAsDouble();
        _numTiles = tileOp.getCount();
        _numDPUs = dpuExec.getCount();
        _NCEThroughput = getNCEThroughput();
        _DMABandwidth = getDMABandwidth(config::getArch(tileOp), config::getRevisionID(module));
        if (auto shaveActExec = tileOp.getSubExecutor(ExecutorKind::SHAVE_ACT)) {
            _numShaveActs = shaveActExec.getCount();
        }
    }
    _numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();
    _arch = config::getArch(module);
    _vpuDeviceType = VPU::getVPUDeviceType(_arch);
}

void LayerCostModel::resetNNCacheCounter() {
    _layerCostModel->getDPUPreloadedCacheCounter().reset();
}

void LayerCostModel::printNNCacheStatistics() const {
    _log.info("[NN Cache statistics]  {0}", _layerCostModel->getDPUPreloadedCacheCounter().printString());
}

vpux::NDTypeInterface LayerCostModel::getNormalInputType(VPU::ClusteredOpInterface origOp,
                                                         mlir::Operation* parentOp) const {
    auto input = getInputFromClusteredOp(origOp, parentOp);
    return mlir::dyn_cast<vpux::NDTypeInterface>(input.getType());
}

vpux::NDTypeInterface LayerCostModel::getNormalOutputType(VPU::ClusteredOpInterface origOp) const {
    auto output = origOp->getResult(0);
    return mlir::dyn_cast<vpux::NDTypeInterface>(output.getType());
}

std::pair<mlir::Type, TensorDistributionMap> LayerCostModel::getInputWithDistribution(
        VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp,
        VPU::MultiClusterStrategy specifiedStrategy) const {
    auto input = getInputFromClusteredOp(origOp, parentOp);
    auto numClustersAttr = VPU::getOptimalNumClusters(
            origOp, mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType()).getShape(), specifiedStrategy);
    if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation())) {
        auto isFilter = nceOp->getNumOperands() > 1 && input == nceOp->getOperand(1) &&
                        !mlir::isa<VPU::NCEEltwiseOp>(origOp.getOperation());
        if (isFilter) {
            return std::make_pair(input.getType(), getFilterDistributionAttrFromOp(nceOp, input.getType(),
                                                                                   numClustersAttr, specifiedStrategy));
        }
    }
    return std::make_pair(input.getType(),
                          getActivationDistributionAttrFromOp(origOp, input, input.getType(), numClustersAttr,
                                                              specifiedStrategy, _siblingsOpsAnalysis));
}

std::pair<mlir::Type, TensorDistributionMap> LayerCostModel::getInputWithDistribution(
        VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp, VPU::MultiClusterStrategy specifiedStrategy,
        mlir::ArrayRef<int64_t> customAlignment) const {
    auto input = getInputFromClusteredOp(origOp, parentOp);
    auto numClustersAttr = VPU::getOptimalNumClusters(
            origOp, mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType()).getShape(), specifiedStrategy);
    return std::make_pair(input.getType(), getActivationDistributionAttrFromOp(origOp, input, input.getType(),
                                                                               numClustersAttr, specifiedStrategy,
                                                                               _siblingsOpsAnalysis, customAlignment));
}

VPU::DistributedTypeInterface LayerCostModel::getDistributedInputType(
        VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp,
        VPU::MultiClusterStrategy specifiedStrategy) const {
    auto [type, distribution] = getInputWithDistribution(origOp, parentOp, specifiedStrategy);
    return mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(
            getDistributedTypeFromDistributionMap(type, distribution));
}

VPU::DistributedTypeInterface LayerCostModel::getDistributedInputType(VPU::ClusteredOpInterface origOp,
                                                                      mlir::Operation* parentOp,
                                                                      VPU::MultiClusterStrategy specifiedStrategy,
                                                                      mlir::ArrayAttr customAlignment) const {
    auto customAlignmentArr = parseIntArrayAttr<int64_t>(customAlignment);
    auto [type, distribution] = getInputWithDistribution(origOp, parentOp, specifiedStrategy, customAlignmentArr);
    return mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(
            getDistributedTypeFromDistributionMap(type, distribution));
}

VPU::DistributedTypeInterface LayerCostModel::getDistributedOutputType(
        VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy specifiedStrategy) const {
    auto numClusters = VPU::getOptimalNumClusters(
            origOp, mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType()).getShape(), specifiedStrategy);
    return VPU::getDistributedOutputTypeFromOp(origOp, origOp->getResult(0).getType(), numClusters, specifiedStrategy);
}

std::pair<mlir::Type, TensorDistributionMap> LayerCostModel::getOutputWithDistribution(
        VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy specifiedStrategy) const {
    auto numClustersAttr = VPU::getOptimalNumClusters(
            origOp, mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType()).getShape(), specifiedStrategy);
    return std::make_pair(origOp->getResult(0).getType(),
                          getOutputDistributionAttrFromOp(origOp, origOp->getResult(0).getType(), numClustersAttr,
                                                          specifiedStrategy, _siblingsOpsAnalysis));
}

/*
 * Get the spilling cost
 * srcTensorType is the output of parent op (current op)
 * dstTensorType is the input of child op
 * return spilling write cost and spilling read cost
 */
LayerCostModel::SpillingCost LayerCostModel::getSpillingCost(vpux::NDTypeInterface srcTensorType,
                                                             vpux::NDTypeInterface dstTensorType,
                                                             VPU::ClusteredOpInterface parentOp,
                                                             VPU::ClusteredOpInterface userOp) const {
    // Concat is on DDR memory if there's spilling. So we don't need copy from CMX to DDR if Concat is parent. Also we
    // don't need copy from DDR to CMX if Concat is user.
    if (mlir::isa<VPU::ConcatOp>(parentOp)) {
        return {0.0, getSpillingReadCost(dstTensorType)};
    }

    if (mlir::isa<VPU::ConcatOp>(userOp)) {
        return {getSpillingWriteCost(srcTensorType), 0.0};
    }

    return {getSpillingWriteCost(srcTensorType), getSpillingReadCost(dstTensorType)};
}

LayerCostModel::SpillingCost LayerCostModel::getSpillingCost(vpux::NDTypeInterface srcTensorType,
                                                             const TensorDistributionMap& srcDistribution,
                                                             vpux::NDTypeInterface dstTensorType,
                                                             const TensorDistributionMap& dstDistribution,
                                                             VPU::ClusteredOpInterface parentOp,
                                                             VPU::ClusteredOpInterface userOp) const {
    // Concat is on DDR memory if there's spilling. So we don't need copy from CMX to DDR if Concat is parent. Also we
    // don't need copy from DDR to CMX if Concat is user.
    if (mlir::isa<VPU::ConcatOp>(parentOp)) {
        return {0.0, getSpillingReadCost(dstTensorType, dstDistribution)};
    }

    if (mlir::isa<VPU::ConcatOp>(userOp)) {
        return {getSpillingWriteCost(srcTensorType, srcDistribution), 0.0};
    }

    return {getSpillingWriteCost(srcTensorType, srcDistribution), getSpillingReadCost(dstTensorType, dstDistribution)};
}

double LayerCostModel::getDMACostOfType(vpux::NDTypeInterface srcType, SpillingType spillingType) const {
    auto distributedSrcType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(srcType);
    auto srcMode = distributedSrcType != nullptr ? distributedSrcType.getDistribution().getMode().getValue()
                                                 : VPU::DistributionMode::NONE;

    if (_arch == config::ArchKind::NPU37XX) {
        return static_cast<double>(getDMACost(srcType, _vpuDeviceType,
                                              _layerCostModel->get_TheoreticalDMA_cost_model_shared(), _numDMAPorts));
    }

    auto spillingReadCostFunc = spillingCostMap.at(srcMode);
    auto distribution = distributedSrcType != nullptr
                                ? DistributionInfo::getClassFromAttr(distributedSrcType.getDistribution())
                                : DistributionInfo();
    return spillingReadCostFunc(srcType, distribution, spillingType, _DDRLatency, _DMABandwidth, _CMXLatency,
                                _CMXMulticastBandwidth, _numDMAPorts);
}

double LayerCostModel::getSpillingDMACost(vpux::NDTypeInterface srcTensorType, SpillingType spillingType) const {
    if (auto sparseTensorType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(srcTensorType)) {
        srcTensorType = mlir::cast<vpux::NDTypeInterface>(sparseTensorType.getData());
    }
    return getDMACostOfType(srcTensorType, spillingType);
}

double LayerCostModel::getSpillingReadCost(vpux::NDTypeInterface srcTensorType) const {
    return getSpillingDMACost(srcTensorType, SpillingType::SPILL_READ);
}

double LayerCostModel::getSpillingWriteCost(vpux::NDTypeInterface srcTensorType) const {
    return getSpillingDMACost(srcTensorType, SpillingType::SPILL_WRITE);
}

double LayerCostModel::getDMACostOfType(vpux::NDTypeInterface srcType, const VPU::DistributionInfo& distribution,
                                        SpillingType spillingType) const {
    if (_arch == config::ArchKind::NPU37XX) {
        TensorDistributionMap distributionMap;
        distributionMap.insert(std::make_pair(srcType, distribution));
        auto distributedType = getDistributedTypeFromDistributionMap(srcType, distributionMap);
        return static_cast<double>(getDMACost(distributedType, _vpuDeviceType,
                                              _layerCostModel->get_TheoreticalDMA_cost_model_shared(), _numDMAPorts));
    }
    auto srcMode = distribution.getDistributionMode();
    auto spillingReadCostFunc = spillingCostMap.at(srcMode);
    return spillingReadCostFunc(srcType, distribution, spillingType, _DDRLatency, _DMABandwidth, _CMXLatency,
                                _CMXMulticastBandwidth, _numDMAPorts);
}

double LayerCostModel::getSpillingDMACost(vpux::NDTypeInterface srcTensorType,
                                          const TensorDistributionMap& distributions, SpillingType spillingType) const {
    if (auto sparseTensorType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(srcTensorType)) {
        srcTensorType = mlir::cast<vpux::NDTypeInterface>(sparseTensorType.getData());
    }
    VPU::DistributionInfo distribution{};
    if (distributions.contains(srcTensorType)) {
        distribution = distributions.at(srcTensorType);
    }
    return getDMACostOfType(srcTensorType, distribution, spillingType);
}

double LayerCostModel::getSpillingReadCost(vpux::NDTypeInterface srcTensorType,
                                           const TensorDistributionMap& distributions) const {
    return getSpillingDMACost(srcTensorType, distributions, SpillingType::SPILL_READ);
}

double LayerCostModel::getSpillingWriteCost(vpux::NDTypeInterface srcTensorType,
                                            const TensorDistributionMap& distributions) const {
    return getSpillingDMACost(srcTensorType, distributions, SpillingType::SPILL_WRITE);
}

// The function computes the actual output tensor volume (i.e. computation that is performed)
// given the stratey and the MPE mode
double LayerCostModel::calculateMPEVolume(VPU::MPEMode mpeMode, Shape shape) const {
    int64_t mpeHeight;
    int64_t mpeWidth;
    switch (mpeMode) {
    case VPU::MPEMode::VECTOR: {
        mpeHeight = 1;
        mpeWidth = 16;
        break;
    }
    case VPU::MPEMode::VECTOR_FP16: {
        mpeHeight = 1;
        mpeWidth = 4;
        break;
    }
    case VPU::MPEMode::MATRIX:
    // These different mpe modes on VPUX37XX have impact on the reuse of activation and weights. We can't estimate reuse
    // cost with current cost equation. In the future we will integrate VPUNN to estimate the cost.
    case VPU::MPEMode::CUBOID_4x16:
    case VPU::MPEMode::CUBOID_8x16:
    case VPU::MPEMode::CUBOID_16x16: {
        mpeHeight = 4;
        mpeWidth = 4;
        break;
    }
    default:
        VPUX_THROW("Unsupported mpeMode '{0}'", mpeMode);
    }

    return static_cast<double>(_numDPUs * divUp((mpeHeight * divUp(shape[Dims4D::Act::H], mpeHeight) * mpeWidth *
                                                 divUp(shape[Dims4D::Act::W], mpeWidth) * _numChannelAlignment *
                                                 divUp(shape[Dims4D::Act::C], _numChannelAlignment)),
                                                _numDPUs));
}

// The efficiency calculation that is being performed here can be described as follows.
// A ratio of the real output tensor volume to the actual computation that occurs on the
// hardware for each MPE Mode 4x4x16 and 16x1x16 is computed and the maximum is selected.
double LayerCostModel::computeSplitEfficiency(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp.checkStrategyCompatibility(strategy, _numTiles) == true,
                      "Unsupported multi-cluster strategy '{0}' for layer type '{1}'", strategy, nceOp->getName());
    auto numClusters = getOptimalNumClusters(
            clusteredOp, mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType()).getShape(), strategy);
    const auto distributedOutputTensorType = VPU::getDistributedOutputTypeFromOp(
            clusteredOp, mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType()), numClusters, strategy);

    VPUX_THROW_UNLESS(distributedOutputTensorType.containsDistributedTypes(), "Missing output distributed types");
    const auto distributedOutputDataType =
            mlir::cast<vpux::VPU::DistributedTensorType>(distributedOutputTensorType.getDistributedTypes().front());
    const auto perClusterShape = distributedOutputDataType.getLargestCompactShape();
    const auto perClusterOutputTensorVolume =
            perClusterShape[Dims4D::Act::H] * perClusterShape[Dims4D::Act::W] * perClusterShape[Dims4D::Act::C];

    return std::max({static_cast<double>(perClusterOutputTensorVolume) /
                             calculateMPEVolume(VPU::MPEMode::CUBOID_4x16, perClusterShape),
                     static_cast<double>(perClusterOutputTensorVolume) /
                             calculateMPEVolume(VPU::MPEMode::CUBOID_8x16, perClusterShape),
                     static_cast<double>(perClusterOutputTensorVolume) /
                             calculateMPEVolume(VPU::MPEMode::CUBOID_16x16, perClusterShape)});
}

// Returns the duration in cycles for the execution of a NCE task
double LayerCostModel::clusterComputeTime(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp.checkStrategyCompatibility(strategy, _numTiles) == true,
                      "Unsupported multi-cluster strategy '{0}' for layer type '{1}'", strategy, nceOp->getName());

    double clusterOpsPerCycle = _NCEThroughput / _NCEFrequency / _numTiles;
    double clusterEff = computeSplitEfficiency(nceOp, strategy);
    auto largestClusterOutShape = getLargestClusterOutputShape(clusteredOp, strategy);

    auto kernelSize = nceOp.getKernelSizeVal();
    auto op = nceOp.getOperation();
    int64_t baseKernelCost = kernelSize[Dims4D::Kernel::Y.ind()] * kernelSize[Dims4D::Kernel::X.ind()];
    if (mlir::isa<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp, VPU::NCEInterpolateOp>(op)) {
        int64_t IC = getShape(
                op->getOperand(0))[Dims4D::Act::C];  // Get input channel (already channel-alignment in previous pass)
        baseKernelCost = IC * baseKernelCost;

    } else if (mlir::isa<VPU::NCEEltwiseOp>(op)) {
        baseKernelCost = 1;
    } else if (!mlir::isa<VPU::NCEMaxPoolOp>(op) && !mlir::isa<VPU::NCEAveragePoolOp>(op) &&
               !mlir::isa<VPU::NCEDepthConvolutionOp>(op)) {
        VPUX_THROW("Invalid NCE operation type: '{0}'", op->getName());
    }

    // Here calculating the total basic operation number for the largest cluster output
    // And also we can reduce formula like:
    // basicOperationVolume = clusterOutShapeSize * baseKernelCost / Efficiency
    //                      = clusterOutShapeSize * baseKernelCost / (clusterOutShapeSize / MPEVolume) =
    //                      = MPEVolume * baseKernelCost
    // So that MPEVolume * baseKernelCost is the final result, and then we can divide frequency to get final cycles
    double basicOperationVolume =
            (static_cast<double>(largestClusterOutShape.totalSize() * baseKernelCost)) / clusterEff;
    double clusterComputeCycles = basicOperationVolume / clusterOpsPerCycle;
    return clusterComputeCycles;
}

/// @brief Returns total number of cycles required to weights DMA and CMX broadcast
/// for a layer with given strategy
/// @details Data transferring cost is modeled as (latency + size / bandwidth)
double LayerCostModel::totalDMATime(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp.checkStrategyCompatibility(strategy, _numTiles) == true,
                      "Unsupported multi-cluster strategy '{0}' for layer type '{1}'", strategy, nceOp->getName());
    double totalWeightCycles = 0.0;
    double totalWeightsTableCycles = 0.0;
    double outputCycles = 0.0;
    const auto op = nceOp.getOperation();
    const int64_t IC = (mlir::isa<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp, VPU::NCEInterpolateOp>(op))
                               ? getShape(op->getOperand(0))[Dims4D::Act::C]
                               : 1;
    const int64_t OC = getShape(op->getResult(0))[Dims4D::Act::C];
    const auto kernelSize = nceOp.getKernelSizeVal();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, getShape(op->getResult(0)), strategy);

    /// Weights cost
    /// Weights and weightTable are Segmented mode under SOK (only including ddr -> cmx cost),
    /// SOK may use less clusters to avoid alignment
    /// So it's not proper to estimate total weightsSize by "clusterWeightsSize * _numTiles" simply
    /// Using distributed tensor for SOK to get accurate total size
    if (mlir::isa<VPU::NCEConvolutionOp, VPU::NCEDepthConvolutionOp, VPU::NCECompressConvolutionOp,
                  VPU::NCEInterpolateOp>(op)) {
        auto weights = op->getOperand(1);

        SmallVector<int64_t> numElemsPerOC;
        int64_t weightSetAlignment = _cmxAddressAlignment;
        if (auto sparseWeightsType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(weights.getType())) {
            VPU::SparsityCompressionAttr sparsityCompression = sparseWeightsType.getSparsityCompression();
            if (sparsityCompression != nullptr && sparsityCompression.getAxis() != nullptr) {
                auto axis = sparsityCompression.getAxis().getInt();
                VPUX_THROW_UNLESS(axis == Dims4D::Filter::OC.ind(),
                                  "SplitOverK is only compatible with compression over OC");
                numElemsPerOC = to_small_vector(sparsityCompression.getNumElems().getValues<int64_t>());
                if (sparsityCompression.getAlignment() != nullptr) {
                    weightSetAlignment = sparsityCompression.getAlignment().getInt();
                }

                auto weightsShape = sparseWeightsType.getShape();
                VPUX_THROW_UNLESS(
                        static_cast<int64_t>(numElemsPerOC.size()) == weightsShape[Dims4D::Filter::OC],
                        "Different number of output channels {0} compared to compression scheme with {1} elements",
                        weightsShape[Dims4D::Filter::OC], numElemsPerOC.size());
            }
        }

        if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            int64_t totalWeightsSize = 0;
            auto distributedWeightsTensorType =
                    VPU::getDistributedFilterTypeFromOp(nceOp, weights.getType(), numClusters, strategy);

            for (auto type : distributedWeightsTensorType.getDistributedTypes() | indexed) {
                auto distributedWeightsType = mlir::cast<vpux::VPU::DistributedTensorType>(type.value());
                auto tiledWeightsShapes = distributedWeightsType.getPerClusterMemoryShapes();
                auto tiledWeightsOffsets = distributedWeightsType.getPerClusterMemoryShapeOffsets();

                const Bit elemBitSize = getElemTypeSize(distributedWeightsType);
                int64_t weightsByteSize = 0;

                if (type.index() == 0 && !numElemsPerOC.empty()) {
                    for (auto p : zip(tiledWeightsShapes, tiledWeightsOffsets)) {
                        const auto tileShape = std::get<0>(p);
                        const auto tileOffsets = std::get<1>(p);
                        const auto startOC = tileOffsets[Dims4D::Filter::OC];
                        const auto endOC = startOC + tileShape[Dims4D::Filter::OC];
                        for (auto idx = startOC; idx < endOC; ++idx) {
                            const auto numElems = *(numElemsPerOC.begin() + idx);
                            weightsByteSize +=
                                    alignMemSize(elemBitSize * numElems, Byte(weightSetAlignment)).to<Byte>().count();
                        }
                    }
                } else {
                    for (auto& clusterWeightsShape : tiledWeightsShapes) {
                        weightsByteSize +=
                                alignMemSize(elemBitSize * clusterWeightsShape.totalSize(), Byte(1)).to<Byte>().count();
                    }
                }

                totalWeightsSize += weightsByteSize;
            }

            const double weightCycles = static_cast<double>(totalWeightsSize) / _DMABandwidth;
            totalWeightCycles = _DDRLatency + weightCycles;
        } else {
            // Duplicated mode in other strategies has ddr->cmx cost
            // The weight set size (IC * KX * KY * BytesPerElement) needs to be aligned to 16B for kernels

            auto ndWeightsType = mlir::cast<vpux::NDTypeInterface>(weights.getType());
            const Bit elemBiSize = ndWeightsType.getElemTypeSize();
            const int64_t weightSetSize =
                    IC * kernelSize[Dims4D::Kernel::Y.ind()] * kernelSize[Dims4D::Kernel::X.ind()];

            int64_t clusterWeightsSize = 0;
            if (!numElemsPerOC.empty()) {
                for (auto numElems : numElemsPerOC) {
                    clusterWeightsSize +=
                            alignMemSize(elemBiSize * numElems, Byte(_cmxAddressAlignment)).to<Byte>().count();
                }
            } else {
                clusterWeightsSize +=
                        OC * alignMemSize(elemBiSize * weightSetSize, Byte(_cmxAddressAlignment)).to<Byte>().count();
            }

            if (mlir::isa<vpux::VPU::SparseTensorType>(weights.getType())) {
                const int64_t weightSetBitAlignment = 128;
                const int64_t sparsityMapSize = (OC * alignValUp<int64_t>(weightSetSize, weightSetBitAlignment));
                const int64_t sparsityMapByteSize = sparsityMapSize / CHAR_BIT;
                clusterWeightsSize += sparsityMapByteSize;
            }

            const double weightCycles = static_cast<double>(clusterWeightsSize) / _DMABandwidth;
            totalWeightCycles = _DDRLatency + weightCycles;
        }
    }

    /// WeightsTable cost
    /// WeightTable has OC entries, each entry includes sparsity/weights pointer, bias and mult/shfit quantized
    /// params. The total size for each entry is 16 Bytes
    if (nceOp.getWeightsTableOperand() != nullptr) {
        auto largestClusterOutShape = getLargestClusterOutputShape(clusteredOp, strategy);
        int64_t alignedClusterOutChannels = largestClusterOutShape[Dims4D::Act::C];
        int64_t clusterWeightTableSize = NCEInvariant::getWeightsTableSize(alignedClusterOutChannels).count();

        if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            totalWeightsTableCycles =
                    _DDRLatency + static_cast<double>(clusterWeightTableSize * numClusters) / _DMABandwidth;
        } else {
            totalWeightsTableCycles = _DDRLatency + (static_cast<double>(clusterWeightTableSize) / _DMABandwidth);
        }
    }

    // Total cost for single layer
    return totalWeightCycles + totalWeightsTableCycles + outputCycles;
}

/// @brief A switcher to select time-cost or efficiency-cost for greedy strategy assignment
/// @details Time-cost includes an extra input spilling cost to be more accurate
double LayerCostModel::getNCELayerCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy,
                                       bool useTimeBasedCost) {
    if (!useTimeBasedCost) {
        return getEfficiencyCost(nceOp, strategy);
    }

    double basicDPUandDMACost = COST_MAX;

    const auto it = _costCache.find(nceOp);
    if (it == _costCache.end()) {
        // Case 1 - Op costs are not found in cache:
        // 1.Calculate cost value with VPUNN
        // 2.Create new op costs
        // 3.Store the new op costs in cost cache
        basicDPUandDMACost = getDPUandDMATimeCost(nceOp, strategy);
        SmallVector newOpCosts((getMaxEnumValForMultiClusterStrategy() + 1), COST_MAX);
        newOpCosts[static_cast<uint64_t>(strategy)] = basicDPUandDMACost;
        _costCache.insert({nceOp, newOpCosts});
    } else {
        auto strategyCostIt = it->second.begin() + static_cast<uint64_t>(strategy);
        if (strategyCostIt != nullptr && *strategyCostIt != COST_MAX) {
            // Case 2 - Strategy cost is found in cache:
            // Retrieve the cost value directly
            basicDPUandDMACost = *strategyCostIt;
        } else {
            // Case 3 - Op costs are found but op strategy cost is not found:
            // 1.Calculate cost value with VPUNN
            // 2.Update op strategy cost value in cache
            basicDPUandDMACost = getDPUandDMATimeCost(nceOp, strategy);
            *strategyCostIt = basicDPUandDMACost;
        }
    }

    return basicDPUandDMACost;
}

/// @brief Time-cost : return the shave computation time (cycles)
/// @details use vpunn cost model to get the shave cost of sw layer
double LayerCostModel::getSWLayerCost(VPU::SWOpInterface swOp, VPU::MultiClusterStrategy strategy) const {
    const auto allTypesSupported = [](mlir::ValueRange values) -> bool {
        return llvm::all_of(values, [](mlir::Value value) {
            const auto valueType = mlir::cast<vpux::NDTypeInterface>(value.getType());
            const auto nnType = VPU::getVPUNNElementType(valueType.getElementType());
            return nnType.has_value();
        });
    };

    if (!allTypesSupported(swOp->getOperands()) || !allTypesSupported(swOp->getResults())) {
        _log.warning("'{0}': Unable to compute SW layer cost due to unsupported data type", swOp->getLoc());
        return 0.0;
    }

    auto getVPUTensors = [&](mlir::ValueRange values) -> std::vector<VPUNN::VPUTensor> {
        std::vector<VPUNN::VPUTensor> tensors;
        std::transform(values.begin(), values.end(), std::back_inserter(tensors), [](mlir::Value value) {
            auto valueType = mlir::cast<vpux::NDTypeInterface>(value.getType());
            return VPU::getVPUTensor(valueType.getShape(), valueType.getElementType(), valueType.getDimsOrder());
        });
        return tensors;
    };

    const auto device = VPU::getVPUDeviceType(_arch);
    const auto inputTensors = getVPUTensors(swOp->getOperands());
    const auto outputTensors = getVPUTensors(swOp->getResults());

    std::shared_ptr<VPUNN::SWOperation> vpunnLayer;
    llvm::TypeSwitch<mlir::Operation*, void>(swOp.getOperation())
            .Case<VPU::TanhOp>([&](VPU::TanhOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVTanh>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::MVNOp>([&](VPU::MVNOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVMVN>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::SoftMaxOp>([&](VPU::SoftMaxOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVSoftmax>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::SwishOp>([&](VPU::SwishOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVSwish>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::HSwishOp>([&](VPU::HSwishOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVHardSwish>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::MishOp>([&](VPU::MishOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVMish>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::FloorOp>([&](VPU::FloorOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVFloor>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::CeilingOp>([&](VPU::CeilingOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVCeiling>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::RoundOp>([&](VPU::RoundOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVRound>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::SinOp>([&](VPU::SinOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVSin>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::CosOp>([&](VPU::CosOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVCos>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::ExpOp>([&](VPU::ExpOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVExp>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::GatherOp>([&](VPU::GatherOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVGather>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::PowerOp>([&](VPU::PowerOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVPower>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::DivideOp>([&](VPU::DivideOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVDivide>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::GreaterOp>([&](VPU::GreaterOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVGreater>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::GreaterEqualOp>([&](VPU::GreaterEqualOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVGreaterEqual>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::LessOp>([&](VPU::LessOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVLess>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::EqualOp>([&](VPU::EqualOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVEqual>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::NotEqualOp>([&](VPU::NotEqualOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVNotEqual>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::LessEqualOp>([&](VPU::LessEqualOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVNotEqual>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::SoftPlusOp>([&](VPU::SoftPlusOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVSoftPlus>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::AndOp>([&](VPU::AndOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVAnd>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::SubtractOp>([&](VPU::SubtractOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVSubtract>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::AddOp>([&](VPU::AddOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVAdd>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::NegativeOp>([&](VPU::NegativeOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVNegative>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::LogicalNotOp>([&](VPU::LogicalNotOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVLogicalNot>(device, inputTensors, outputTensors.front());
            })
            .Default([&](mlir::Operation* op) {
                VPUX_THROW("SW op {0} has no VPUNN support", op->getName());
            });

    auto vpunnStrategy = VPU::getVPULayerStrategy(strategy, _numDPUs, _numTiles, _arch, _numShaveActs, false);
    return _layerCostModel->Layer(*vpunnLayer, vpunnStrategy);
}

/// @brief get computation cost
/// @details Time-cost includes an extra input spilling cost to be more accurate
double LayerCostModel::getLayerCost(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy,
                                    bool useTimeBasedCost) {
    if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation())) {
        return getNCELayerCost(nceOp, strategy, useTimeBasedCost);
    } else if (auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(clusteredOp.getOperation())) {
        return getSWLayerCost(swOp, strategy);
    } else if (mlir::isa<VPU::ConcatOp>(clusteredOp.getOperation())) {
        // Concat has no computation cost
        return 0.0;
    } else if (mlir::isa<VPU::GatherDMAOp>(clusteredOp.getOperation())) {
        // GatherDMAOp has no computation cost
        return 0.0;
    } else {
        VPUX_THROW("Unsupported op type {0} at {1}", clusteredOp->getName(), clusteredOp->getLoc());
    }
}

// For sw op, if strategy is set for it's parent or user nce op, check if sw op can avoid spilling with the op
bool isSoftwareOpCustomStrategyIncompatibleWithOtherNceOp(mlir::Operation* swOp, VPU::MultiClusterStrategy strategy,
                                                          int64_t numTiles) {
    if (!mlir::isa_and_nonnull<VPU::SWOpInterface>(swOp)) {
        return false;
    }

    // Rank need to be 4 when checking strategy compatibility
    for (const auto& input : swOp->getOperands()) {
        const auto inputShape = mlir::cast<vpux::NDTypeInterface>(input.getType()).getShape();
        if (inputShape.size() != RANK_REQUIRED_FOR_TILING) {
            return false;
        }
    }
    for (const auto& output : swOp->getResults()) {
        const auto outputShape = mlir::cast<vpux::NDTypeInterface>(output.getType()).getShape();
        if (outputShape.size() != RANK_REQUIRED_FOR_TILING) {
            return false;
        }
    }

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        // For Clustering, Sw op can always set Clustering to avoid spilling
        return false;
    }

    if (auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(swOp)) {
        // If SOH liked strategy can not be assigned for SW op, there will be spilling between them
        // If SOK strategy can not be assigned for SW op, the only way to avoid spilling is setting Clustering. But
        // Clustering has bad compute efficiency, so we can assume there's spilling.
        if (strategy == VPU::MultiClusterStrategy::HKSwitch) {
            strategy = VPU::MultiClusterStrategy::SplitOverHeight;
        }
        return !clusteredOp.checkStrategyCompatibility(strategy, numTiles);
    }

    return false;
}

HwLayerTilingStrategyCosts LayerCostModel::getDPUandDMATimeCostWithCustomTiling(VPU::NCEOpInterface nceOp,
                                                                                VPU::MultiClusterStrategy strategy,
                                                                                const OutputTiling& outTiles) const {
    // Types for each tile
    std::vector<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes;

    _log.trace("[Cost Analysis] {0} [{1}]  Get VPUNN cost for {2} {3}", nceOp.getLoc(), nceOp->getName(), strategy,
               outTiles[0].axis);

    const auto costParams = VPU::getWorkloadCostParam(nceOp, _arch, _numDPUs, _numTiles);

    auto distributionMode = DistributionMode::NONE;
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    if (clusteredOp != nullptr) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());
        auto tiledOutputType = outputType.extractDenseTile(outTiles[0].offsets, outTiles[0].shape);
        distributionMode = getOutputTensorDistributionMode(clusteredOp, strategy, tiledOutputType);
    }

    SmallVector<uint32_t> vpunnLayerDPUCosts;
    SmallVector<uint32_t> vpunnOriginalLayerDPUCosts;
    const auto vpunnStrategy =
            VPU::getVPULayerStrategy(strategy, _numDPUs, _numTiles, _arch, 1, true, distributionMode, nceOp);

    const auto enableVPUNNPreSplit = hasVPUNNPreSplit(nceOp);
    if (enableVPUNNPreSplit && !isActSparseOp(nceOp)) {
        // Track E#160972. Activation sparse op accuracy issue
        vpunnLayerDPUCosts = getDPUCostForNCEOpPreSplit(nceOp, outTiles, costParams, vpunnStrategy.tiling_strategy,
                                                        _layerCostModel, _numDPUs, _log);
    } else {
        vpunnLayerDPUCosts =
                getDPUCostForNCEOp(nceOp, strategy, outTiles, costParams, vpunnStrategy, _layerCostModel, _log);
    }
    if (vpunnLayerDPUCosts.empty()) {
        return {COST_MAX, COST_MAX};
    }
    _log.trace("VPUNN DPU layer costs {0}", vpunnLayerDPUCosts);
    vpunnOriginalLayerDPUCosts = vpunnLayerDPUCosts;

    const auto getSpillingReadCost = [&](NDTypeInterface srcType,
                                         const TensorDistributionMap& distributions) -> uint32_t {
        return checked_cast<uint32_t>(this->getSpillingReadCost(srcType, distributions));
    };

    const auto getSpillingWriteCost = [&](NDTypeInterface srcType,
                                          const TensorDistributionMap& distributions) -> uint32_t {
        return checked_cast<uint32_t>(this->getSpillingWriteCost(srcType, distributions));
    };

    VPUX_THROW_WHEN(outTiles.empty(), "Empty output tiles");
    const auto inOrder = DimsOrder::fromValue(nceOp->getOperand(0));
    const auto outOrder = DimsOrder::fromValue(nceOp->getResult(0));
    const auto inputTiledOnLowestDim = isTiledOnLowestDim(outTiles[0], inOrder);
    const auto outputTiledOnLowestDim = isTiledOnLowestDim(outTiles[0], outOrder);
    bool correctedStrideActDMACost = false;
    bool correctedStrideOutputDMACost = false;

    // When weights are tiled over channel, the activation input has to be copied with a strided DMA. The cost of a
    // strided DMA is not accurate in VPUNN. It should be addressed by NNDMA cost model. Without accurate strided DMA
    // costs the total DMA cost (for weights and activation) appears lowest when tiling over channel compared to height
    // and width. But in fact the total DMA and DPU cost when tiling over height is the lowest when the the strided dma
    // cost is correct.
    const auto correctStrideDMACost =
            [&](ArrayRef<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes,
                const std::function<NDTypeInterface(ArrayRef<std::pair<NDTypeInterface, TensorDistributionMap>>)>&
                        tileTypeGetter,
                SmallVector<uint32_t>& dmaCost, bool isStridedDMA) -> bool {
        if (!isStridedDMA) {
            return false;
        }
        VPUX_THROW_WHEN(dmaCost.size() != tilesTypes.size(), "DMA costs size mismatches with tiled types");
        for (auto tileId : irange(tilesTypes.size())) {
            auto currentTileType = tileTypeGetter(tilesTypes[tileId]);
            const auto dimOrder = currentTileType.getDimsOrder();
            const auto lowestDim = dimOrder.dimAt(dimOrder.numDims() - 1);
            const Bit elemSize = currentTileType.getElemTypeSize();
            if (auto sparseTensorType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(currentTileType)) {
                currentTileType = mlir::cast<vpux::NDTypeInterface>(sparseTensorType.getData());
            }
            Bit continuousBitsOnLowestDim;
            if (auto distributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(currentTileType)) {
                continuousBitsOnLowestDim = distributedType.getLargestCompactShape()[lowestDim] * elemSize;
            } else {
                continuousBitsOnLowestDim = currentTileType.getShape()[lowestDim] * elemSize;
            }
            auto curStrideDMACorrectionThreshold = getStrideDMACorrectionThresholdByArch(config::getArch(nceOp));
            if (continuousBitsOnLowestDim.count() < curStrideDMACorrectionThreshold) {
                auto factor = curStrideDMACorrectionThreshold / continuousBitsOnLowestDim.count();
                dmaCost[tileId] = checked_cast<uint32_t>(std::floor(factor * dmaCost[tileId]));
            }
        }
        return true;
    };
    auto activationTileTypeGetter = [](ArrayRef<std::pair<NDTypeInterface, TensorDistributionMap>> tilesType) {
        auto [srcType, distributionMap] = tilesType.front();
        return getDistributedTypeFromDistributionMap(srcType, distributionMap);
    };

    auto outputTileTypeGetter = [](ArrayRef<std::pair<NDTypeInterface, TensorDistributionMap>> tilesType) {
        auto [srcType, distributionMap] = tilesType.back();
        return getDistributedTypeFromDistributionMap(srcType, distributionMap);
    };

    // Accumulate all the DPU costs
    double cost = std::accumulate(vpunnLayerDPUCosts.begin(), vpunnLayerDPUCosts.end(), 0.0);
    double costWithPrefetching = cost;
    auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
    for (auto& outTile : outTiles) {
        auto inTiles = tilingBuilderOp.backInferTileInfo(outTile, _log);
        tilesTypes.push_back(getTileDistributions(nceOp.getOperation(), _siblingsOpsAnalysis, outTile, inTiles));
    }
    _log.trace(" VPUNN accumulated DPU cost {0}", cost);

    // Add weights DMA costs, which considers DMA prefetching and pipelining and only accumulates extra DMA costs based
    // on DPU cost
    auto vpunnLayerWeightsCosts =
            getPerTileWeightsDMACosts(nceOp, _siblingsOpsAnalysis, tilesTypes, getSpillingReadCost);
    _log.trace("VPUNN weights DMA costs {0}", vpunnLayerWeightsCosts);
    _log.trace("vpunnLayerDPUCosts {0}", vpunnLayerDPUCosts);
    const auto [weightsCost, weightsCostWithPrefetching] = getWeightsDMACostForNCEOp(
            nceOp, outTiles, vpunnLayerDPUCosts, vpunnLayerWeightsCosts, _enablePrefetchTiling, _log);
    _log.trace(" VPUNN accumulated weights DMA cost {0} with prefetchTiling option {1}", weightsCost,
               _enablePrefetchTiling);
    cost += weightsCost;
    costWithPrefetching += weightsCostWithPrefetching;
    _log.trace("Include Weights DMA cost {0}, Full layer cost now {1}", weightsCost, cost);
    _log.trace("Include Weights Prefetch DMA cost {0}, Full Prefetch layer cost now {1}", weightsCostWithPrefetching,
               costWithPrefetching);

    auto getParentOp = [&]() -> mlir::Operation* {
        mlir::Operation* parentOp = nceOp->getOperand(0).getDefiningOp();
        while (parentOp && isPureViewOp(parentOp)) {
            parentOp = parentOp->getOperand(0).getDefiningOp();
        }
        return parentOp;
    };
    // Add activation DMA costs
    // Subgraph optimization will calculate input spilling cost instead of activation dma cost
    SmallVector<uint32_t> vpunnLayerActCosts;
    if (!isUnderSubgraphOpt() || getParentOp() == nullptr) {
        vpunnLayerActCosts = getPerTileActivationDMACosts(nceOp, tilesTypes, getSpillingReadCost, strategy, _numTiles);
        // TODO: Ticket E#135490, remove this after stride DMA cost is accurate
        correctedStrideActDMACost =
                correctStrideDMACost(tilesTypes, activationTileTypeGetter, vpunnLayerActCosts, inputTiledOnLowestDim);
        _log.trace("VPUNN activation DMA costs {0}", vpunnLayerActCosts);
        _log.trace("vpunnLayerDPUCosts {0}", vpunnLayerDPUCosts);
        const auto actCost = getActivationDMACostForNCEOp(nceOp, outTiles, vpunnLayerDPUCosts, vpunnLayerActCosts,
                                                          _enablePrefetchTiling, _log);
        cost += actCost;
        costWithPrefetching += actCost;
        _log.trace("Include Activation DMA cost {0}, Full layer cost now {1}", actCost, cost);
        _log.trace("Include Activation DMA cost {0}, Prefetched full layer cost now {1}", actCost, costWithPrefetching);
    }

    // Add output spilling cost
    // for non clusteredOp, must be ops that requires tiling
    VPUX_THROW_WHEN(clusteredOp == nullptr, "NCE op at '{0}' is not a clustered op", nceOp.getLoc());
    SmallVector<uint32_t> vpunnLayerOutputCosts;
    if (!clusteredOp.doesLayerFitIntoCMX(strategy, _siblingsOpsAnalysis, /*reservedMem=*/Byte(0))) {
        // Consider output spilling pipelining with the next tile's DPU
        // Might be inaccurate when the DPU time is smaller than the sum of DMA time (input + weights + output)
        vpunnLayerOutputCosts = getPerTileOutputDMACosts(nceOp, tilesTypes, getSpillingWriteCost);
        // TODO: Ticket E#135490, remove this after stride DMA cost is accurate
        correctedStrideOutputDMACost =
                correctStrideDMACost(tilesTypes, outputTileTypeGetter, vpunnLayerOutputCosts, outputTiledOnLowestDim);
        _log.trace("VPUNN output DMA costs {0}", vpunnLayerOutputCosts);
        _log.trace("vpunnLayerDPUCosts {0}", vpunnLayerDPUCosts);
        const auto outCost = getOutputDMACostForNCEOp(nceOp, outTiles, vpunnLayerDPUCosts, vpunnLayerOutputCosts,
                                                      _enablePrefetchTiling, _log);
        cost += outCost;
        costWithPrefetching += outCost;
        _log.trace("Include Output DMA cost {0}, Full layer cost now {1}", outCost, cost);
        _log.trace("Include Output DMA cost {0}, Prefetched full layer cost now {1}", outCost, costWithPrefetching);
    }

    _log.nest(2).trace("[Cost Analysis] Full DPU costs {0}", vpunnOriginalLayerDPUCosts);
    _log.nest(2).trace("[Cost Analysis] Weights costs {0}", vpunnLayerWeightsCosts);
    _log.nest(2).trace("[Cost Analysis] Activation costs {0}", vpunnLayerActCosts);
    _log.nest(2).trace("[Cost Analysis] Output costs {0}", vpunnLayerOutputCosts);
    _log.nest(2).trace("[Cost Analysis] Non-Pipelined DPU cost remainder {0}", vpunnLayerDPUCosts);
    if (correctedStrideActDMACost || correctedStrideOutputDMACost) {
        _log.nest(4).trace(
                "[Cost Analysis] Note: DMA costs have been corrected for stride DMA. Activation {0}, Output {1}",
                correctedStrideActDMACost, correctedStrideOutputDMACost);
    }
    _log.trace("[Cost Analysis] Final Full Cost with No Prefetch: {0}", cost);
    _log.trace("[Cost Analysis] Final Full Cost with Prefetching: {0}", costWithPrefetching);

    return {cost, costWithPrefetching};
}

/// @brief Time-cost : return a sum of layer DPU time and weights DMA time (cycles)
/// @details DPU time calculation also considers the impact of workloads split efficiency
double LayerCostModel::getDPUandDMATimeCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "NCE op {0} at {1} should be a clustered op", nceOp->getName(),
                        nceOp.getLoc());

        // Set customized strategy to the op to get corresponding output tiles when tiling
        // Save and restore original strategy if needed
        auto origStrategy = clusteredOp.getMultiClusterStrategy();
        clusteredOp.setMultiClusterStrategy(strategy);

        auto restoreOrigStrategy = [&]() {
            // Restore original strategy or remove temporary strategy
            if (origStrategy.has_value()) {
                clusteredOp.setMultiClusterStrategy(origStrategy.value());
            } else {
                clusteredOp->removeAttr(multiClusterStrategy);
            }
        };

        // Output tiling for each tile
        OutputTiling outTiles({TileInfo(getShape(nceOp->getResult(0)))});

        // Check CMX memory as VPUNN works with layer which fits CMX memory
        // If not, tiling big layer to fit into CMX
        if (!(clusteredOp.doesLayerFitIntoCMX(strategy, _siblingsOpsAnalysis, /*reservedMem=*/Byte(0)))) {
            _log.trace("Tiling op {0} to fit into cmx before passing to VPUNN Layer API", nceOp.getLoc());
            auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
            VPUX_THROW_WHEN(tilingBuilderOp == nullptr, "NCE op {0} at {1} should be a tiling op", nceOp->getName(),
                            nceOp.getLoc());
            auto costModel = std::make_shared<vpux::VPU::LayerCostModel>(*this);
            auto tiles = getHWLayerTilingStrategy(tilingBuilderOp, _enablePrefetchTiling, costModel, _log);

            if (mlir::failed(tiles)) {
                _log.trace("Invalid tiling strategy for {0}", nceOp->getName());
                restoreOrigStrategy();
                return COST_MAX;
            }
            outTiles = tiles.value();
        }

        // If the output size of nceOp on dimension C exceeds VPU_DIMENSION_LIMIT,
        // we tile it again to avoid potential VPUNN errors in MC strategy assignment.
        // Tracking Number: E#140892
        int tileNum = outTiles.size();
        for (int tileIdx = 0; tileIdx < tileNum; tileIdx++) {
            ShapeRef shape = outTiles[tileIdx].shape;
            if (shape[Dims4D::Act::C] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
                _log.debug("Tiled Op size {0} is still over VPU_DIMENSION_LIMIT on C, tile again to avoid VPUNN error.",
                           shape);
                // tile the oversized ops into 16-aligned tiles under VPU_DIMENSION_LIMIT
                int shapeOnC = shape[Dims4D::Act::C];
                int oversizeFactor = shapeOnC / VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
                if (shapeOnC % VPU::NCEInvariant::VPU_DIMENSION_LIMIT != 0) {
                    oversizeFactor += 1;
                }
                int tiledShapeOnC = shapeOnC / oversizeFactor;
                tiledShapeOnC += (tiledShapeOnC % 16 == 0) ? 0 : 16 - tiledShapeOnC % 16;
                int remainderOnC = shapeOnC - (oversizeFactor - 1) * tiledShapeOnC;

                // update the original tile
                outTiles[tileIdx].shape[Dims4D::Act::C] = remainderOnC;
                for (int i = 1; i < oversizeFactor; i++) {
                    TileInfo newTile = TileInfo(shape);
                    newTile.shape[Dims4D::Act::C] = tiledShapeOnC;
                    outTiles.push_back(newTile);
                }
            }
        }

        auto [costWithoutPrefetching, costWithPrefetching] =
                getDPUandDMATimeCostWithCustomTiling(nceOp, strategy, outTiles);
        auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nceOp.getOperation());
        double cost = 0;
        if (_enablePrefetchTiling && tilingInfoOp != nullptr &&
            tilingInfoOp.isSupportedTiling(outTiles, vpux::TilingMode::PREFETCHING, _log)) {
            cost = costWithPrefetching;
        } else {
            cost = costWithoutPrefetching;
        }

        _log.trace("VPUNN total layer cost for {0} and {1} is {2}", strategy, outTiles[0].axis, cost);

        restoreOrigStrategy();

        return cost;
    }
}

///@brief Effi-cost : A simple cost considering DPU computing efficiency
double LayerCostModel::getEfficiencyCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    return 1.0 / computeSplitEfficiency(nceOp, strategy);
}

bool LayerCostModel::hasMultiClusterStrategy(mlir::Operation* op) const {
    if (auto clusteringOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op)) {
        return clusteringOp.getMultiClusterStrategy().has_value();
    }

    return false;
}

VPU::MultiClusterStrategy LayerCostModel::getMultiClusterStrategyValue(VPU::ClusteredOpInterface clusteredOp) const {
    auto strategy = clusteredOp.getMultiClusterStrategy();
    if (!strategy.has_value()) {
        VPUX_THROW("NCE operation {0} doesn't have multiClusterStrategy attribute", clusteredOp->getLoc());
    }

    return strategy.value();
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface /*clusteredOp*/,
                                 std::pair<mlir::Type, TensorDistributionMap>& srcTensorType,
                                 std::pair<mlir::Type, TensorDistributionMap>& dstTensorType) const {
    auto getActivationTypeFromSparseType = [](std::pair<mlir::Type, TensorDistributionMap>& tensorType) {
        VPU::DistributionInfo distribution{};
        if (auto sparseTensorType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tensorType.first)) {
            if (tensorType.second.contains(sparseTensorType.getData())) {
                distribution = tensorType.second.at(sparseTensorType.getData());
            }
            // interested in activation spills so use data for compatibility
            return std::make_pair(sparseTensorType.getData(), distribution);
        }
        if (tensorType.second.contains(tensorType.first)) {
            distribution = tensorType.second.at(tensorType.first);
        }
        return std::make_pair(tensorType.first, distribution);
    };

    auto srcTensor = getActivationTypeFromSparseType(srcTensorType);
    auto dstTensor = getActivationTypeFromSparseType(dstTensorType);

    if (isTargetTensorTypeCompatible(srcTensor, dstTensor) ||
        isSOHAlignmentCompatibleOrAdjustedCompatible(srcTensor, dstTensor)) {
        return false;
    }
    return true;
}

/// Anywhere if you need to judge spilling existing, please call me!
/// srcTensorType is the output of parent origOp
/// dstTensorType is the input of child NCE op
bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface op, vpux::NDTypeInterface srcTensorType,
                                 vpux::NDTypeInterface dstTensorType) const {
    auto srcTensorDistributionMap =
            std::make_pair(mlir::cast<mlir::Type>(srcTensorType), getDistributionMapFromDistributedType(srcTensorType));
    auto dstTensorDistributionMap =
            std::make_pair(mlir::cast<mlir::Type>(dstTensorType), getDistributionMapFromDistributedType(dstTensorType));
    return hasSpilling(op, srcTensorDistributionMap, dstTensorDistributionMap);
}

std::pair<std::pair<mlir::Type, TensorDistributionMap>, std::pair<mlir::Type, TensorDistributionMap>>
LayerCostModel::getDistributionsWithStrategy(VPU::ClusteredOpInterface parentOp,
                                             VPU::MultiClusterStrategy parentStrategy, VPU::ClusteredOpInterface userOp,
                                             VPU::MultiClusterStrategy userStrategy) const {
    // Set the custom strategy to the op to get the accurate distributed type
    // The distribution mode depends on the neighboring op's strategy
    // e.g., Conv (SOK) -> SW (SOK), the output of the Conv would be SEGMENTED
    // Conv (SOK) -> SW (Clustering), the output of the Conv would be DUPLICATED|SEGMENTED
    // The DistributedType is decided by the ops multiCluster strategy attributes
    auto greedyStrategyParentOp = getMultiClusterStrategyValue(parentOp);
    auto greedyStrategyUserOp = getMultiClusterStrategyValue(userOp);
    parentOp.setMultiClusterStrategy(parentStrategy);
    userOp.setMultiClusterStrategy(userStrategy);
    auto targetOutput = getOutputWithDistribution(parentOp, parentStrategy);
    auto targetInput = getInputWithDistribution(userOp, parentOp, userStrategy);
    parentOp.setMultiClusterStrategy(greedyStrategyParentOp);
    userOp.setMultiClusterStrategy(greedyStrategyUserOp);

    VPU::DistributionInfo outputDistribution{};
    auto outputType = mlir::isa<VPU::SparseTensorType>(targetOutput.first)
                              ? mlir::cast<vpux::VPU::SparseTensorType>(targetOutput.first).getData()
                              : targetOutput.first;
    if (targetOutput.second.contains(outputType)) {
        outputDistribution = targetOutput.second.at(outputType);
    }

    VPU::DistributionInfo inputDistribution{};
    auto inputType = mlir::isa<VPU::SparseTensorType>(targetInput.first)
                             ? mlir::cast<vpux::VPU::SparseTensorType>(targetInput.first).getData()
                             : targetInput.first;
    if (targetInput.second.contains(inputType)) {
        inputDistribution = targetInput.second.at(inputType);
    }

    // Adjust inputType alignment for SW op
    // e.g., Conv (SOK) -> SW (SOK), the input of SW can have a same alignment with Conv
    // to avoid spilling

    auto parentOutAlignment = outputDistribution.getAlignment();
    auto UserInAlignment = inputDistribution.getAlignment();
    if (!parentOutAlignment.empty() && UserInAlignment.empty() &&
        mlir::isa<VPU::SWOpInterface>(userOp.getOperation()) &&
        isSWOpChannelAlignmentCompatible(userOp, targetInput.first,
                                         mlir::cast<vpux::NDTypeInterface>(userOp->getResult(0).getType()))) {
        targetInput = getInputWithDistribution(userOp, parentOp, userStrategy, parentOutAlignment);
    }
    return std::make_pair(targetOutput, targetInput);
}

std::pair<vpux::NDTypeInterface, vpux::NDTypeInterface> LayerCostModel::getDistributionTypesWithStrategy(
        VPU::ClusteredOpInterface parentOp, VPU::MultiClusterStrategy parentStrategy, VPU::ClusteredOpInterface userOp,
        VPU::MultiClusterStrategy userStrategy) const {
    auto [outDist, inDist] = getDistributionsWithStrategy(parentOp, parentStrategy, userOp, userStrategy);
    return std::make_pair(getDistributedTypeFromDistributionMap(outDist.first, outDist.second),
                          getDistributedTypeFromDistributionMap(inDist.first, inDist.second));
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface clustered, VPU::ClusteredOpInterface userOp) const {
    auto targetOutputType = hasMultiClusterStrategy(clustered)
                                    ? mlir::cast<vpux::NDTypeInterface>(getDistributedOutputType(
                                              clustered, getMultiClusterStrategyValue(clustered)))
                                    : getNormalOutputType(clustered);

    auto targetInputType = hasMultiClusterStrategy(userOp)
                                   ? mlir::cast<vpux::NDTypeInterface>(getDistributedInputType(
                                             userOp, clustered, getMultiClusterStrategyValue(userOp)))
                                   : getNormalInputType(userOp, clustered);
    return hasSpilling(clustered, targetOutputType, targetInputType);
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy origOpStrategy,
                                 VPU::ClusteredOpInterface userOp) const {
    auto targetOutputType = mlir::cast<vpux::NDTypeInterface>(getDistributedOutputType(origOp, origOpStrategy));
    auto targetInputType = hasMultiClusterStrategy(userOp)
                                   ? mlir::cast<vpux::NDTypeInterface>(getDistributedInputType(
                                             userOp, origOp, getMultiClusterStrategyValue(origOp)))
                                   : getNormalInputType(userOp, origOp);
    if (hasMultiClusterStrategy(userOp)) {
        std::tie(targetOutputType, targetInputType) =
                getDistributionTypesWithStrategy(origOp, origOpStrategy, userOp, getMultiClusterStrategyValue(userOp));
    }
    return hasSpilling(origOp, targetOutputType, targetInputType);
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface origOp, VPU::ClusteredOpInterface userOp,
                                 VPU::MultiClusterStrategy userOpStrategy) const {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    auto targetOutputType = hasMultiClusterStrategy(origOp)
                                    ? mlir::cast<vpux::NDTypeInterface>(getDistributedOutputType(
                                              origOp, getMultiClusterStrategyValue(clusteredOp)))
                                    : getNormalOutputType(origOp);
    auto targetInputType = mlir::cast<vpux::NDTypeInterface>(getDistributedInputType(userOp, origOp, userOpStrategy));
    if (hasMultiClusterStrategy(origOp)) {
        std::tie(targetOutputType, targetInputType) =
                getDistributionTypesWithStrategy(origOp, getMultiClusterStrategyValue(origOp), userOp, userOpStrategy);
    }
    return hasSpilling(origOp, targetOutputType, targetInputType);
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy origOpStrategy,
                                 VPU::ClusteredOpInterface userOp, VPU::MultiClusterStrategy userOpStrategy) const {
    auto targetTypes = getDistributionsWithStrategy(origOp, origOpStrategy, userOp, userOpStrategy);
    auto targetOutputType = targetTypes.first;
    auto targetInputType = targetTypes.second;
    return hasSpilling(origOp, targetOutputType, targetInputType);
}

bool LayerCostModel::doesLayerRequireTiling(VPU::ClusteredOpInterface clusteredOp,
                                            VPU::MultiClusterStrategy strategy) const {
    return !(clusteredOp.doesLayerFitIntoCMX(strategy, _siblingsOpsAnalysis, /*reservedMem=*/Byte(0)));
}

bool LayerCostModel::doesLayerHaveVPUNNSupportedTypes(VPU::ClusteredOpInterface clusteredOp) const {
    const bool hasSupportedOperandTypes = llvm::all_of(clusteredOp->getOperands(), [](const mlir::Value val) {
        return vpux::VPU::isVPUNNSupportedElementType(
                mlir::cast<vpux::NDTypeInterface>(val.getType()).getElementType());
    });
    const bool hasSupportedResultTypes = llvm::all_of(clusteredOp->getResults(), [](const mlir::Value val) {
        return vpux::VPU::isVPUNNSupportedElementType(
                mlir::cast<vpux::NDTypeInterface>(val.getType()).getElementType());
    });
    return hasSupportedOperandTypes && hasSupportedResultTypes;
}

LayerCostModel::SpillingCost LayerCostModel::calculateSpillingCost(VPU::ClusteredOpInterface parentOp,
                                                                   VPU::ClusteredOpInterface userOp,
                                                                   VPU::MultiClusterStrategy parentStrategy,
                                                                   VPU::MultiClusterStrategy userStrategy) const {
    auto [targetOutput, targetInput] = getDistributionsWithStrategy(parentOp, parentStrategy, userOp, userStrategy);
    return getSpillingCost(targetOutput.first, targetOutput.second, targetInput.first, targetInput.second, parentOp,
                           userOp);
}

VPU::MultiClusterStrategy LayerCostModel::getOptimalLayerStrategy(VPU::ClusteredOpInterface clusteredOp) {
    double splitOverHeightCost = COST_MAX;
    double splitOverKernelCost = COST_MAX;
    auto splitOverHeightFitIntoCMX = false;
    auto splitOverKernelFitIntoCMX = false;
    auto splitOverHeightIsCompatible = false;
    auto splitOverKernelIsCompatible = false;

    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverHeight, _numTiles) &&
        clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()))) {
        splitOverHeightIsCompatible = true;
        splitOverHeightCost = getLayerCost(clusteredOp, VPU::MultiClusterStrategy::SplitOverHeight);
        splitOverHeightFitIntoCMX =
                clusteredOp.doesLayerFitIntoCMX(VPU::MultiClusterStrategy::SplitOverHeight, _siblingsOpsAnalysis,
                                                /*reservedMem=*/Byte(0));
        _log.nest().debug("SplitOverHeight cost is {0}. Fit into CMX is {1}", splitOverHeightCost,
                          splitOverHeightFitIntoCMX);
    }

    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverKernel, _numTiles) &&
        clusteredOp.isOperationSplitOverKernelCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                         /*axis=*/ShapeRef())) {
        splitOverKernelIsCompatible = true;
        splitOverKernelCost = getLayerCost(clusteredOp, VPU::MultiClusterStrategy::SplitOverKernel);
        splitOverKernelFitIntoCMX =
                clusteredOp.doesLayerFitIntoCMX(VPU::MultiClusterStrategy::SplitOverKernel, _siblingsOpsAnalysis,
                                                /*reservedMem=*/Byte(0));
        _log.nest().debug("SplitOverKernel cost is {0}. Fit into CMX is {1}", splitOverKernelCost,
                          splitOverKernelFitIntoCMX);
    }

    const auto optimalHeightTiling = [&](void) {
        return (mlir::isa<vpux::VPU::NCECompressConvolutionOp, vpux::VPU::NCEPermuteOp>(clusteredOp))
                       ? VPU::MultiClusterStrategy::SplitOverHeightOverlapped
                       : VPU::MultiClusterStrategy::SplitOverHeight;
    };

    // Check if SplitOverHeight is the only strategy which fits into CMX
    if (splitOverHeightFitIntoCMX && (!splitOverKernelFitIntoCMX)) {
        return optimalHeightTiling();
    }

    // Compute amount of clusters so that SOK is compatible
    const auto outputChannels =
            mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType()).getShape()[Dims4D::Act::C];
    auto uniformDistributedSegments = VPU::isUniformDistributedSegmentsSupported(clusteredOp);
    const auto sokOptimalClusters =
            getNumberOfClustersForSOKToAvoidAlignment(outputChannels, _numTiles, uniformDistributedSegments);

    // Check if SplitOverKernel is the only strategy which fits into CMX and utilize full clusters
    if ((!splitOverHeightFitIntoCMX) && splitOverKernelFitIntoCMX && (sokOptimalClusters == _numTiles)) {
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }

    if ((splitOverHeightCost != COST_MAX) && (splitOverKernelCost != COST_MAX) && (sokOptimalClusters == _numTiles)) {
        if (!useSOKWhenMVNSide(clusteredOp) && splitOverHeightCost <= splitOverKernelCost) {
            return optimalHeightTiling();
        }
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }
    // SOH is P1 option as SOK may not utilize full clusters
    // However, it is still more optimal than clustering
    if (splitOverHeightCost != COST_MAX) {
        return optimalHeightTiling();
    }

    if (splitOverKernelCost != COST_MAX) {
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }

    // Handle the case when VPUNN is failing to return valid strategy cost due to unsupported parameters
    if (splitOverHeightIsCompatible && splitOverKernelIsCompatible) {
        if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation())) {
            if (nceOp.getWeightsOperand() != nullptr) {
                auto activationSize =
                        mlir::cast<vpux::NDTypeInterface>(nceOp->getOperand(0).getType()).getTotalAllocSize();
                activationSize += mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType()).getTotalAllocSize();
                auto weightSize =
                        mlir::cast<vpux::NDTypeInterface>(nceOp.getWeightsOperand().getType()).getTotalAllocSize();

                // use size of weights vs size of activations for determining strategy
                if (activationSize >= weightSize) {
                    return VPU::MultiClusterStrategy::SplitOverHeight;
                }

                if (activationSize < weightSize) {
                    return VPU::MultiClusterStrategy::SplitOverKernel;
                }
            }
        }
    }

    // Return SOH or SOK if any one is compatible, it's more optimal than clustering
    if (splitOverHeightIsCompatible) {
        return VPU::MultiClusterStrategy::SplitOverHeight;
    }

    if (splitOverKernelIsCompatible) {
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }

    return VPU::MultiClusterStrategy::Clustering;
}

bool LayerCostModel::isUnderSubgraphOpt() const {
    return _underSubgraphOpt;
}
void LayerCostModel::setUnderSubgraphOpt(bool underSubgraphOpt) {
    _underSubgraphOpt = underSubgraphOpt;
}

bool vpux::VPU::isStrategySOXCompatible(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy,
                                        size_t numTiles) {
    if (clusteredOp == nullptr) {
        return false;
    }

    if (clusteredOp.checkStrategyCompatibility(strategy, numTiles)) {
        switch (strategy) {
        case VPU::MultiClusterStrategy::SplitOverHeight:
        case VPU::MultiClusterStrategy::SplitOverHeightOverlapped:
            return clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()));
        case VPU::MultiClusterStrategy::SplitOverKernel:
            return clusteredOp.isOperationSplitOverKernelCompatible(/*outputShape=*/ShapeRef(),
                                                                    /*offset=*/ShapeRef(),
                                                                    /*axis=*/ShapeRef());
        case VPU::MultiClusterStrategy::SplitOverWidth:
            return clusteredOp.isOperationSplitOverWidthCompatible(/*outputShape=*/ShapeRef(),
                                                                   /*offset=*/ShapeRef(),
                                                                   /*axis=*/ShapeRef());
        default:
            return false;
        }
    }

    return false;
}

// For a clustered op which doesn't support cycle cost calculation the priority for strategies is parent strategy >
// SOH/SOHOverlapped > SOK > SOW > Clustering
// Note: parent strategy will take precedence, but it is delayed just before subgraph optimization to allow
// multi-threading of this pass in the future
std::optional<VPU::MultiClusterStrategy> vpux::VPU::getDefaultLayerStrategy(VPU::ClusteredOpInterface clusteredOp) {
    auto module = clusteredOp->getParentOfType<mlir::ModuleOp>();
    auto tileOp = config::getTileExecutor(module);
    const auto numTiles = tileOp.getCount();

    // Only the highest dimension is prioritized
    // Need to investigate if the complete layout order could be optimal
    // Track E#124146
    auto strategyOrder = SmallVector(
            {VPU::MultiClusterStrategy::SplitOverHeight, VPU::MultiClusterStrategy::SplitOverHeightOverlapped,
             VPU::MultiClusterStrategy::SplitOverBatch, VPU::MultiClusterStrategy::SplitOverKernel,
             VPU::MultiClusterStrategy::SplitOverWidth});
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());
    const auto outputShape = getBoundedShape(outputType);
    const auto highestDim = vpux::getHighestNonTrivialDim(outputShape, outputType.getDimsOrder()).value_or(Dim(0));
    const auto numActShaves = config::getTotalNumOfEngines(clusteredOp, VPU::ExecutorKind::SHAVE_ACT);

    DenseMap<int64_t, SmallVector<VPU::MultiClusterStrategy>> dimToStrategyMap{
            {Dims4D::Act::N.ind(), {VPU::MultiClusterStrategy::SplitOverBatch}},
            {Dims4D::Act::H.ind(),
             {VPU::MultiClusterStrategy::SplitOverHeight, VPU::MultiClusterStrategy::SplitOverHeightOverlapped}},
            {Dims4D::Act::W.ind(), {VPU::MultiClusterStrategy::SplitOverWidth}},
            {Dims4D::Act::C.ind(), {VPU::MultiClusterStrategy::SplitOverKernel}},
    };

    auto updateStrategyOrder = [&](Dim priorityDim) {
        const auto& strategies = dimToStrategyMap[priorityDim.ind()];
        for (const auto& strategy : strategies) {
            strategyOrder.erase(llvm::find(strategyOrder, strategy));
        }
        strategyOrder.insert(strategyOrder.begin(), strategies.begin(), strategies.end());
    };

    // For EltwiseOp and DynamicQuantizeOp, any layout is acceptable. Prioritize the highest dimension for MC and MS
    // tiling
    const auto isEltwiseSwOp =
            clusteredOp->hasTrait<VPU::EltwiseOp>() && mlir::isa<VPU::SWOpInterface>(clusteredOp.getOperation());
    const auto isDQ = mlir::isa_and_nonnull<VPU::DynamicQuantizeOp>(clusteredOp.getOperation());
    if ((isEltwiseSwOp || isDQ) && outputShape[highestDim] >= numActShaves) {
        updateStrategyOrder(highestDim);
    }

    // For DequantizeOp with layout [OC, IC, KY, KX], SOK segmentation is on OC
    const auto isWeightDequant =
            mlir::isa_and_nonnull<VPU::DequantizeOp>(clusteredOp.getOperation()) && highestDim == Dims4D::Filter::OC;
    if (highestDim == Dims4D::Act::C || isWeightDequant) {
        updateStrategyOrder(Dims4D::Act::C);
    }

    auto findDimCanSupportShaveTiling = [&]() -> std::optional<Dim> {
        if (!isEltwiseSwOp) {
            return std::nullopt;
        }
        const auto dimOrder = outputType.getDimsOrder();
        if (dimOrder != DimsOrder::NHWC && dimOrder != DimsOrder::NCHW) {
            return std::nullopt;
        }
        if (dimOrder.dimPos(highestDim) == dimOrder.numDims() - 1) {
            return std::nullopt;
        }

        if (outputShape[highestDim] >= numActShaves) {
            return std::nullopt;
        }
        auto allInputsHaveSameShape = llvm::all_of(clusteredOp->getOperands(), [&](auto operand) {
            auto inputType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
            return inputType.getShape() == outputType.getShape();
        });
        if (!allInputsHaveSameShape) {
            return std::nullopt;
        }

        auto dimRange = irange(dimOrder.dimPos(highestDim) + 1, dimOrder.numDims());
        auto shaveTilingSupportDim = llvm::find_if(dimRange, [&](size_t dimIdx) {
            return outputShape[dimOrder.dimAt(dimIdx)] >= numActShaves;
        });
        if (shaveTilingSupportDim == dimRange.end()) {
            return std::nullopt;
        }
        const auto tileDim = dimOrder.dimAt(*shaveTilingSupportDim);

        // Sigmoid with small data size are not performant with shave tiling, so we need to add related check here. It's
        // supposed to be same one in TileActShaveKernelTaskPass.
        if (auto sigmoidOp = mlir::dyn_cast<VPU::SigmoidOp>(clusteredOp.getOperation())) {
            // E#92211: Measurements for the performance profiling, see this ticket for details.
            auto tileOp = config::getTileExecutor(clusteredOp->getParentOfType<mlir::ModuleOp>());
            auto dpuCount = tileOp.getCount();
            VPUX_THROW_WHEN(dpuCount <= 0, "Invalid number of clusters: {0}", dpuCount);
            auto outType = mlir::cast<vpux::NDTypeInterface>(sigmoidOp.getOutput().getType());
            Shape tiledOutShape = Shape(outputShape);
            tiledOutShape[tileDim] = tiledOutShape[tileDim] / dpuCount;
            auto tiledOutType = outType.changeShape(tiledOutShape);
            auto tiledSize = tiledOutType.getTotalAllocSize() / dpuCount;
            if (tiledSize < VPUIP::SIGMOID_SW_KERNEL_TILING_THRESHOLD) {
                return std::nullopt;
            }
        }
        return tileDim;
    };
    auto shaveTilingDim = findDimCanSupportShaveTiling();
    if (shaveTilingDim.has_value()) {
        // The compiler tend to use strategy which also supports shave tiling, since the gain of multi shaves usually is
        // greater than the loss of stride DMA.
        updateStrategyOrder(shaveTilingDim.value());
    }

    // #E179535: Experimental results show SOW may bring regression for EqualOp with small width
    const int64_t MIN_WIDTH_FOR_SOW_EQUAL = 256;

    for (auto strategy : strategyOrder) {
        if (!clusteredOp.checkStrategyCompatibility(strategy, numTiles)) {
            continue;
        }
        if (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
            strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
            if (clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()))) {
                return strategy;
            }
        }
        if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            if (clusteredOp.isOperationSplitOverKernelCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                                 /*axis=*/ShapeRef())) {
                return strategy;
            }
        }
        if (strategy == VPU::MultiClusterStrategy::SplitOverBatch) {
            if (clusteredOp.isOperationSplitOverBatchCompatible(/*outputShape=*/ShapeRef())) {
                return strategy;
            }
        }
        if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
            if (mlir::isa<VPU::SoftMaxOp, VPU::DepthToSpaceOp, VPU::PadOp, VPU::MVN1NormalizeOp, VPU::SwishOp,
                          VPU::MultiplyOp, VPU::SelectOp, VPU::DynamicDequantizeOp, VPU::DynamicQuantizeOp,
                          VPU::GreaterEqualOp, VPU::MaximumOp>(clusteredOp.getOperation()) &&
                clusteredOp.isOperationSplitOverWidthCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                                /*axis=*/ShapeRef())) {
                return strategy;
            }
            if (mlir::isa<VPU::EqualOp>(clusteredOp.getOperation()) &&
                outputShape[Dims4D::Act::W] > MIN_WIDTH_FOR_SOW_EQUAL &&
                clusteredOp.isOperationSplitOverWidthCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                                /*axis=*/ShapeRef())) {
                return strategy;
            }
        }
    }

    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::Clustering, numTiles)) {
        return VPU::MultiClusterStrategy::Clustering;
    }

    return std::nullopt;
}

bool vpux::VPU::isStrategyCompatibleShape(VPU::ClusteredOpInterface clusteredOp, const vpux::TileInfo& outputTile,
                                          VPU::MultiClusterStrategy strategy, Logger log) {
    auto shape = ShapeRef(outputTile.shape);

    if (shape.size() != RANK_REQUIRED_FOR_TILING && shape.size() != DimsGroups5D::Act::numDims) {
        log.trace("Operation '{0}' at '{1}' has output rank {2} and cannot be tiled. Expected rank: {3} or {4}.",
                  clusteredOp->getName(), clusteredOp->getLoc(), shape.size(), RANK_REQUIRED_FOR_TILING,
                  DimsGroups5D::Act::numDims);
        return false;
    }
    switch (strategy) {
    case MultiClusterStrategy::SplitOverHeight:
    case MultiClusterStrategy::SplitOverHeightOverlapped:
    case MultiClusterStrategy::HKSwitch: {
        return clusteredOp.isOperationSplitOverHeightCompatible(outputTile);
    }
    case MultiClusterStrategy::SplitOverHeightKernel: {
        return clusteredOp.isOperationSplitOverHeightCompatible(outputTile) &&
               clusteredOp.isOperationSplitOverKernelCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::SplitOverWidth: {
        return clusteredOp.isOperationSplitOverWidthCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::SplitOverKernel: {
        return clusteredOp.isOperationSplitOverKernelCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::SplitOverHeightWidth: {
        return clusteredOp.isOperationSplitOverHeightCompatible(outputTile) &&
               clusteredOp.isOperationSplitOverWidthCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::SplitOverBatch: {
        return clusteredOp.isOperationSplitOverBatchCompatible(outputTile.shape);
    }
    case MultiClusterStrategy::Clustering: {
        return true;
    }
    case MultiClusterStrategy::SplitOverGroup: {
        return clusteredOp.isOperationSplitOverGroupCompatible(outputTile);
    }
    default: {
        VPUX_THROW("Unknown multi cluster strategy {0}", strategy);
    }
    }
}

SmallVector<uint32_t> vpux::VPU::getDPUCostForNCEOpPreSplit(
        VPU::NCEOpInterface nceOp, const OutputTiling& outTiles, const VPUIP::WorkloadCostParams& costParams,
        VPUNN::VPUTilingStrategy vpunnTilingStrategy, const std::shared_ptr<VPUNN::VPULayerCostModel>& vpunnCostModel,
        int64_t numDPU, Logger log) {
    auto& cache = VPU::OpTilingCache::instance();
    std::optional<llvm::hash_code> opHash = std::nullopt;
    const auto useCache = cache.isCacheSupported();
    if (useCache) {
        opHash = cache.calculateOpHash(nceOp.getOperation(), std::nullopt, outTiles);
        auto cachedCost = cache.getOpDpuCost(opHash.value());
        if (cachedCost.has_value()) {
            return cachedCost.value();
        }
    }

    std::vector<std::vector<VPUNN::DPULayer>> preSplitVPUNNLayers;

    // E#160175 For not supported SEP layer costs - optimize activation spills
    if (VPU::isNCEWithSEPActivation(nceOp.getOperation())) {
        return SmallVector<uint32_t>(outTiles.size(), 1);
    }

    if (!outTiles.empty()) {
        auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
        VPUX_THROW_WHEN(tilingBuilderOp == nullptr, "NCE op {0} at {1} should be a tiling op", nceOp->getName(),
                        nceOp.getLoc());
        const auto tilingVPUNNLayer =
                [&](const VPUIP::WorkloadCostParams& basicCostParams) -> std::vector<std::vector<VPUNN::DPULayer>> {
            std::vector<std::vector<VPUNN::DPULayer>> tiledVpunnLayers;
            tiledVpunnLayers.reserve(outTiles.size());
            for (auto& outTile : outTiles) {
                auto inTiles = tilingBuilderOp.backInferTileInfo(outTile, log);
                auto& inputTile = inTiles.tiles.front();
                auto inPad = inTiles.pads;
                auto curCostParams = basicCostParams;
                curCostParams.inputShape = inputTile.shape;
                curCostParams.outputShape = outTile.shape;
                if (inPad.has_value()) {
                    curCostParams.padInfo = {
                            static_cast<unsigned int>(inPad->left), static_cast<unsigned int>(inPad->right),
                            static_cast<unsigned int>(inPad->top), static_cast<unsigned int>(inPad->bottom)};
                }
                tiledVpunnLayers.push_back(VPU::getPerClusterDPULayers(nceOp, curCostParams, log));
            }
            return tiledVpunnLayers;
        };
        preSplitVPUNNLayers = tilingVPUNNLayer(costParams);
    } else {
        preSplitVPUNNLayers = {VPU::getPerClusterDPULayers(nceOp, costParams, log)};
    }

    // Exclude multiClusterStrategy from hash code for VPUNN statistic
    // to make sure that one operation's hash code won't be changed by temporal multiClusterStrategy attribute
    const auto opHashWoStrategy = VPU::OpTilingCache::instance().calculateOpHashIncludingTilingExcludingAttr(
            nceOp, multiClusterStrategy, std::nullopt, outTiles);
    auto getVPUNNLayersCostFromCache = [&](const std::vector<VPUNN::DPULayer>& vpunnLayers,
                                           VPUNN::LayerSplitInfo& layerSplitInfo) {
        if (!useCache) {
            return checkAndReturnCost(
                    vpunnCostModel->LayersPreSplit(vpunnLayers, numDPU, /*input_in_ddr=*/false,
                                                   /*output_in_ddr=*/false, /*prefetching=*/true, layerSplitInfo,
                                                   static_cast<size_t>(opHashWoStrategy), vpunnTilingStrategy),
                    log);
        }
        auto hash = cache.calculateVPUNNLayersHash(vpunnLayers);
        auto cachedCost = cache.getVPUNNLayerCost(hash);
        if (cachedCost.has_value()) {
            return cachedCost.value();
        }
        auto cost = checkAndReturnCost(
                vpunnCostModel->LayersPreSplit(vpunnLayers, numDPU, /*input_in_ddr=*/false,
                                               /*output_in_ddr=*/false, /*prefetching=*/true, layerSplitInfo,
                                               static_cast<size_t>(opHashWoStrategy), vpunnTilingStrategy),
                log);
        // Track E#158789, cache layerSplitInfo
        cache.updateVPUNNLayerCost(hash, cost);
        return cost;
    };

    SmallVector<uint32_t> layerDPUCosts;
    for (auto& vpunnLayers : preSplitVPUNNLayers) {
        VPUNN::LayerSplitInfo layerSplitInfo;
        auto cost = getVPUNNLayersCostFromCache(vpunnLayers, layerSplitInfo);
        log.debug("Called L2 API with hash {0} strategy {1}", static_cast<size_t>(opHashWoStrategy),
                  stringifyVPUNNStrategy(vpunnTilingStrategy));
        if (cost >= VPU::INVALID_COST_BASE) {
            log.warning("Cost is invalid");
            printVPUNNLayers(vpunnLayers, log);
        }
        layerDPUCosts.push_back(cost);
    }
    if (useCache) {
        cache.updateOpDPUCost(opHash.value(), layerDPUCosts);
    }
    return layerDPUCosts;
}

SmallVector<uint32_t> vpux::VPU::getDPUCostForNCEOp(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy mcStrategy,
                                                    const OutputTiling& outTiles,
                                                    const VPUIP::WorkloadCostParams& costParams,
                                                    VPUNN::VPULayerStrategy vpunnStrategy,
                                                    const std::shared_ptr<VPUNN::VPULayerCostModel>& vpunnCostModel,
                                                    Logger log) {
    // E#160175 Apply workaround only for VPUX4XXX architecture
    if (costParams.arch == config::ArchKind::NPU40XX && VPU::isNCEWithSEPActivation(nceOp.getOperation())) {
        return SmallVector<uint32_t>(outTiles.size(), 1);
    }

    std::vector<VPUNN::DPULayer> vpunnLayers{VPU::getDPULayer(costParams)};

    auto& cache = VPU::OpTilingCache::instance();
    std::optional<llvm::hash_code> opHash = std::nullopt;
    const auto useCache = cache.isCacheSupported();
    if (useCache) {
        opHash = cache.calculateOpHash(nceOp.getOperation(), std::nullopt, outTiles);
        auto cachedCost = cache.getOpDpuCost(opHash.value());
        if (cachedCost.has_value()) {
            return cachedCost.value();
        }
    }

    if (!outTiles.empty()) {
        auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
        VPUX_THROW_WHEN(tilingBuilderOp == nullptr, "NCE op {0} at {1} should be a tiling op", nceOp->getName(),
                        nceOp.getLoc());
        const auto tilingVPUNNLayer =
                [&](const VPUIP::WorkloadCostParams& basicCostParams) -> std::vector<VPUNN::DPULayer> {
            std::vector<VPUNN::DPULayer> vpunnLayers;
            vpunnLayers.reserve(outTiles.size());
            for (auto& outTile : outTiles) {
                auto inTiles = tilingBuilderOp.backInferTileInfo(outTile, log);
                auto& inputTile = inTiles.tiles.front();
                auto inPad = inTiles.pads;
                auto curCostParams = basicCostParams;
                curCostParams.inputShape = inputTile.shape;
                curCostParams.outputShape = outTile.shape;
                if (inPad.has_value()) {
                    curCostParams.padInfo = {
                            static_cast<unsigned int>(inPad->left), static_cast<unsigned int>(inPad->right),
                            static_cast<unsigned int>(inPad->top), static_cast<unsigned int>(inPad->bottom)};
                }
                if (VPU::isNCEWithSEPActivation(nceOp.getOperation())) {
                    auto input = nceOp->getOperand(0);
                    auto inputSparseTensorOp = input.getDefiningOp<VPU::GroupSparseTensorOp>();
                    auto tilingViewOp = mlir::cast<VPU::TilingViewLikeOpInterface>(inputSparseTensorOp.getOperation());
                    auto sepInTiles = tilingViewOp.backInferTileInfo(inputTile, log);
                    auto sepDataTile = sepInTiles.tiles.front();
                    auto sepTableTile = sepInTiles.tiles.back();
                    curCostParams.sepInfo = VPUIP::SEPInfo{sepTableTile.shape, sepDataTile.shape};
                }
                vpunnLayers.push_back(VPU::getDPULayer(curCostParams));
            }
            return vpunnLayers;
        };
        vpunnLayers = tilingVPUNNLayer(costParams);
    }

    auto getVPUNNLayerCostFromCache = [&](VPUNN::DPULayer& vpunnLayer) {
        if (!useCache) {
            return checkAndReturnCost(vpunnCostModel->Layer(vpunnLayer, vpunnStrategy), log);
        }
        auto hash = cache.calculateVPUNNLayerHash(vpunnLayer, vpunnStrategy);
        auto cachedCost = cache.getVPUNNLayerCost(hash);
        if (cachedCost.has_value()) {
            return cachedCost.value();
        }
        auto cost = checkAndReturnCost(vpunnCostModel->Layer(vpunnLayer, vpunnStrategy), log);
        cache.updateVPUNNLayerCost(hash, cost);
        return cost;
    };

    SmallVector<uint32_t> layerDPUCosts;
    for (auto& vpunnLayer : vpunnLayers) {
        auto cost = getVPUNNLayerCostFromCache(vpunnLayer);
        if (cost >= VPU::INVALID_COST_BASE) {
            printVPUNNLayerConfig(vpunnLayer, vpunnStrategy, log);
            // A workaround to resolve new introduced INPUT_TOO_BIG error code after vpunn software update
            // We should set correct SOK & SOK_NO_BROADCAST when real nn model is updated in future
            if ((cost == VPU::ERROR_INPUT_TOO_BIG) &&
                (vpunnStrategy.tiling_strategy == VPUNN::VPUTilingStrategy::SOK)) {
                auto clusteredOpNceOp = mlir::cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
                auto outputType = mlir::cast<NDTypeInterface>(clusteredOpNceOp->getResult(0).getType());
                auto distributionMode = getOutputTensorDistributionMode(clusteredOpNceOp, mcStrategy, outputType);
                if (distributionMode == DistributionMode::SEGMENTED) {
                    vpunnStrategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK_NO_BROADCAST;
                    cost = checkAndReturnCost(vpunnCostModel->Layer(vpunnLayer, vpunnStrategy), log);
                    vpunnStrategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;
                }
            }

            if (cost == VPU::ERROR_INPUT_TOO_BIG && !layerDPUCosts.empty()) {
                log.trace(" Use the first availabe layer cost to estimate the layer with ERROR_INPUT_TOO_BIG");
                cost = layerDPUCosts.front();
            } else if (cost >= VPU::INVALID_COST_BASE) {
                layerDPUCosts.clear();
                break;
            } else {
                cost = cost * SOK_NO_BROADCAST_DPU_COST_RATIO;
                log.trace(" Use SOK_NO_BROADCAST to estimate the layer with ERROR_INPUT_TOO_BIG, get cost {0}", cost);
            }
        }

        if (mlir::isa<VPU::NCEEltwiseOp>(nceOp.getOperation()) &&
            (mcStrategy == VPU::MultiClusterStrategy::Clustering)) {
            // The VPUNN cost of NCEEltwiseOp is inaccurate
            // Multiply a ratio to correct the cost
            // Track [E#98656]
            log.trace("Using NCEELTWISE_DPU_COST_RATIO for DPU cost");
            cost *= NCEELTWISE_DPU_COST_RATIO;
        }

        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
        const auto arch = config::getArch(clusteredOp);
        if (mlir::isa<VPU::NCEConvolutionOp>(nceOp.getOperation()) && arch == config::ArchKind::NPU40XX) {
            if (mcStrategy == VPU::MultiClusterStrategy::SplitOverKernel && outTiles.size() > 1) {
                auto nTiles = vpunnStrategy.nTiles;
                auto outShape = mlir::cast<vpux::NDTypeInterface>(clusteredOp.getOperation()->getResult(0).getType())
                                        .getShape();
                auto outputChannelsDim = outShape[Dims4D::Act::C];
                auto perTileChannels = static_cast<double>(outputChannelsDim) / nTiles;
                auto maxSpatialDim = std::max(outShape[Dims4D::Act::H], outShape[Dims4D::Act::W]);
                if (perTileChannels / static_cast<double>(maxSpatialDim) < NCECONV_DPU_SOK_OC_TO_SPATIAL_RATIO) {
                    cost *= NCECONV_DPU_SOK_COST_RATIO;
                }
            }
        }
        if (mlir::isa<VPU::NCEDepthConvolutionOp>(nceOp.getOperation())) {
            auto nTiles = vpunnStrategy.nTiles;
            if ((mcStrategy == VPU::MultiClusterStrategy::SplitOverKernel) && (!config::isArchVPUX3XXX(arch))) {
                auto modeIn = VPU::getActivationTensorDistributionMode(clusteredOp, mcStrategy);
                auto modeOut = VPU::getOutputTensorDistributionMode(clusteredOp, mcStrategy, nullptr);

                // DUP -> DUP case
                if ((VPU::bitEnumContainsAny(modeIn, VPU::DistributionMode::DUPLICATED) ||
                     VPU::bitEnumContainsAny(modeIn, VPU::DistributionMode::MULTICASTED)) &&
                    ((VPU::bitEnumContainsAny(modeOut, VPU::DistributionMode::DUPLICATED) ||
                      VPU::bitEnumContainsAny(modeOut, VPU::DistributionMode::MULTICASTED)))) {
                    VPUX_THROW_WHEN(nTiles <= 0, "nTiles should be positive but got {0}", nTiles);
                    auto outputChannels =
                            mlir::cast<vpux::NDTypeInterface>(clusteredOp.getOperation()->getResult(0).getType())
                                    .getShape()[Dims4D::Act::C];
                    size_t perTileCh = outputChannels / nTiles;
                    if (perTileCh <= 32) {
                        // The VPUNN cost of NCEDWCONV is inaccurate
                        // Multiply a ratio to correct the cost
                        // Track [E#117314]
                        log.trace("Using NCEDWCONV_DPU_COST_RATIO for SOK DPU cost");
                        cost *= NCEDWCONV_DPU_COST_RATIO;
                    }
                }
            } else if (mcStrategy == VPU::MultiClusterStrategy::HKSwitch) {
                auto outputHeight =
                        mlir::cast<vpux::NDTypeInterface>(clusteredOp.getOperation()->getResult(0).getType())
                                .getShape()[Dims4D::Act::H];
                size_t perTileH = outputHeight / nTiles;
                if (perTileH < 6) {
                    // The VPUNN cost of NCEDWCONV is inaccurate
                    // Multiply a ratio to correct the cost
                    // Track [E#144661]
                    log.trace("Using NCEDWCONV_HK_DPU_COST_RATIO for HK DPU cost");
                    cost *= NCEDWCONV_HK_DPU_COST_RATIO;
                }
            }
        }

        if ((mlir::dyn_cast<vpux::VPU::SparseTensorType>(nceOp->getOperand(0).getType()) ||
             mlir::dyn_cast<vpux::VPU::SparseTensorType>(nceOp->getResult(0).getType())) &&
            (mcStrategy == VPU::MultiClusterStrategy::SplitOverKernel)) {
            // op with SEP activation should not use this ratio
            if (!VPU::isNCEWithSEPActivation(nceOp.getOperation()) && !config::isArchVPUX3XXX(arch)) {
                // The VPUNN cost of ACT-SPARSITY is inaccurate
                // Multiply a ratio to correct the cost
                // Track [E#117195]
                log.trace("Using ACTSPARSE_DPU_COST_RATIO for DPU cost");
                cost *= ACTSPARSE_DPU_COST_RATIO;
            }
        }

        layerDPUCosts.push_back(cost);
    }
    if (useCache) {
        cache.updateOpDPUCost(opHash.value(), layerDPUCosts);
    }
    return layerDPUCosts;
}

SmallVector<uint32_t> vpux::VPU::getPerTileWeightsDMACosts(
        VPU::NCEOpInterface nceOp, SiblingOpsAnalysis& siblingsAnalysis,
        ArrayRef<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes,
        std::function<uint32_t(NDTypeInterface, const TensorDistributionMap& distributions)> getSpillingReadCostFunc) {
    auto weightsOperand = nceOp.getWeightsOperand();
    if (weightsOperand == nullptr) {
        return SmallVector<uint32_t>(std::max<size_t>(tilesTypes.size(), 1), 0);
    }

    const auto inferredTileTypes =
            std::vector<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>>{getTileDistributions(
                    nceOp.getOperation(), siblingsAnalysis, TileInfo(getBoundedShape(nceOp->getResult(0))))};
    ArrayRef<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> inferredTileTypesRef{inferredTileTypes};
    const auto typesList = tilesTypes.empty() ? inferredTileTypesRef : tilesTypes;

    SmallVector<uint32_t> perTileWeightsCosts;
    for (const auto& tileTypes : typesList) {
        VPUX_THROW_UNLESS(tileTypes.size() > 1,
                          "NCEOp {0} at {1} has invalid number of tile types, got {2}, expected >1", nceOp->getName(),
                          nceOp->getLoc(), tileTypes.size());
        auto weightsDMACost = checked_cast<uint32_t>(getSpillingReadCostFunc(tileTypes[1].first, tileTypes[1].second));
        perTileWeightsCosts.push_back(weightsDMACost);
    }

    return perTileWeightsCosts;
}

SmallVector<uint32_t> vpux::VPU::getPerTileActivationDMACosts(
        VPU::NCEOpInterface nceOp, ArrayRef<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes,
        std::function<uint32_t(NDTypeInterface, const TensorDistributionMap& distributions)> getSpillingReadCostFunc,
        VPU::MultiClusterStrategy strategy, int64_t numTiles) {
    bool permuteCastInMiddle = false;
    auto getParentOp = [&]() -> mlir::Operation* {
        mlir::Operation* parentOp = nceOp->getOperand(0).getDefiningOp();
        while (parentOp && isPureViewOp(parentOp)) {
            // PermuteCast op will change MC axis so that we assume there must be a spill between parent sw op and
            // current nce op. And activation dma cost should be calculated. More general solution need
            // backInferTilingDim interface implementation for permuteCast op. See E#106960
            if (mlir::isa<VPU::PermuteCastOp>(parentOp)) {
                permuteCastInMiddle = true;
            }
            parentOp = parentOp->getOperand(0).getDefiningOp();
        }
        return parentOp;
    };
    auto parentOp = getParentOp();

    // If nceOp can fit into CMX and it has a parent op, we assume that act spilling can be avoided by adjusting
    // strategy of the parent op, hence we return a per tile activation DMA cost of zero
    if ((parentOp != nullptr) && (tilesTypes.size() == 1)) {
        if (!mlir::isa<VPU::SWOpInterface>(parentOp)) {
            return SmallVector<uint32_t>(std::max<size_t>(tilesTypes.size(), 1), 0);
        }

        if (!isSoftwareOpCustomStrategyIncompatibleWithOtherNceOp(parentOp, strategy, numTiles)) {
            if (!permuteCastInMiddle) {
                return SmallVector<uint32_t>(std::max<size_t>(tilesTypes.size(), 1), 0);
            }
        }
    }

    bool isEltwiseOpWithDiffInputs =
            (mlir::isa<VPU::NCEEltwiseOp>(nceOp) && nceOp->getOperand(0) != nceOp->getOperand(1));

    SmallVector<uint32_t> perTileActCosts;
    for (const auto& tileTypes : tilesTypes) {
        VPUX_THROW_UNLESS(tileTypes.size() > 1,
                          "NCEOp {0} at {1} has invalid number of tile types, got {2}, expected >1", nceOp->getName(),
                          nceOp->getLoc(), tileTypes.size());
        auto actDMACost = checked_cast<uint32_t>(getSpillingReadCostFunc(tileTypes[0].first, tileTypes[0].second));
        if (isEltwiseOpWithDiffInputs) {
            actDMACost += checked_cast<uint32_t>(getSpillingReadCostFunc(tileTypes[1].first, tileTypes[1].second));
        }
        perTileActCosts.push_back(actDMACost);
    }

    return perTileActCosts;
}

SmallVector<uint32_t> vpux::VPU::getPerTileOutputDMACosts(
        VPU::NCEOpInterface nceOp, ArrayRef<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes,
        std::function<uint32_t(NDTypeInterface, const TensorDistributionMap& distributions)> getSpillingWriteCostFunc) {
    SmallVector<uint32_t> perTileOutputCosts;
    for (const auto& tileTypes : tilesTypes) {
        VPUX_THROW_UNLESS(tileTypes.size() > 1,
                          "NCEOp {0} at {1} has invalid number of tile types, got {2}, expected > 1", nceOp->getName(),
                          nceOp->getLoc(), tileTypes.size());
        auto outputDMACost =
                checked_cast<uint32_t>(getSpillingWriteCostFunc(tileTypes.back().first, tileTypes.back().second));
        perTileOutputCosts.push_back(outputDMACost);
    }

    return perTileOutputCosts;
}

std::pair<uint32_t, uint32_t> vpux::VPU::getWeightsDMACostForNCEOp(VPU::NCEOpInterface nceOp,
                                                                   const OutputTiling& outTiles,
                                                                   SmallVector<uint32_t>& layerDPUCosts,
                                                                   ArrayRef<uint32_t> layerDMACosts,
                                                                   bool enablePrefetchTiling, vpux::Logger log) {
    VPUX_THROW_WHEN(layerDPUCosts.empty() || layerDPUCosts.size() != layerDMACosts.size(),
                    "Layer DPU costs must be non-empty and equal to DMA costs in size");

    const auto outShape = getShape(nceOp->getResult(0));
    auto tiles = outTiles.empty() ? OutputTiling({TileInfo(outShape)}) : outTiles;
    auto tilingStrategy = tiles.front().axis;
    const auto isWeightsSharedNestedTiling = isWeightsFirstNestedTiling(nceOp.getOperation(), tilingStrategy);
    SmallVector<uint32_t> filteredDMACosts;
    SmallVector<uint32_t> filteredDPUCosts;
    if (isWeightsSharedNestedTiling) {
        log.trace("[Cost Analysis] Assumption: Weights First nested tiling");
        // Unroll channel first
        // weights are partially shared. Every tile_H * tile_W weights are shared
        const auto temporalSize = tilingStrategy[Dims4D::Act::C];
        for (auto i = 0; i < temporalSize; i++) {
            filteredDPUCosts.push_back(layerDPUCosts[i]);
            filteredDMACosts.push_back(layerDMACosts[i]);
        }
    } else {
        filteredDMACosts = SmallVector<uint32_t>(layerDMACosts);
        filteredDPUCosts = layerDPUCosts;
    }

    auto weightsOperand = nceOp.getWeightsOperand();
    bool isWeightsDMASplitOnEachTile = (weightsOperand != nullptr && tilingStrategy[Dims4D::Act::C] > 1);

    auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nceOp.getOperation());
    // If the DMA will overlap with DPU from the second tile on
    bool isDMAOverlappedWithDPU =
            enablePrefetchTiling ? tilingInfoOp != nullptr &&
                                           tilingInfoOp.isSupportedTiling(tiles, vpux::TilingMode::PIPELINING, log)
                                 : false;
    uint32_t totalDMACost = 0;

    if (isDMAOverlappedWithDPU) {
        // Weights DMA from second tile on will be overlapped with DPU of previous tile
        log.trace("[Cost Analysis] Assumption: Weights DMA will pipeline with DPU");
        totalDMACost += getPrefetchDMACostOverlappsWithPreviousDPU(filteredDPUCosts, ArrayRef(filteredDMACosts),
                                                                   isWeightsDMASplitOnEachTile);
    } else {
        // When DMA not overlapped with DPU
        //  - If weights DMA will be copied on each tile, we need to accumulate all the DMA costs
        //  - If weights DMA will be shared for all tiles, we only add the first DMA cost
        totalDMACost += isWeightsDMASplitOnEachTile
                                ? std::accumulate(filteredDMACosts.begin(), filteredDMACosts.end(), 0U)
                                : filteredDMACosts.front();
    }

    return std::make_pair(totalDMACost, totalDMACost - filteredDMACosts.front());
}

uint32_t vpux::VPU::getActivationDMACostForNCEOp(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                                 SmallVector<uint32_t>& layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                                 bool enablePrefetchTiling, vpux::Logger log) {
    VPUX_THROW_WHEN(layerDPUCosts.empty() || layerDPUCosts.size() != layerDMACosts.size(),
                    "Layer DPU costs must be non-empty and equal to DMA costs in size");

    const auto outShape = getShape(nceOp->getResult(0));
    auto tiles = outTiles.empty() ? OutputTiling({TileInfo(outShape)}) : outTiles;

    auto tilingStrategy = tiles.front().axis;
    [[maybe_unused]] const auto isActSharedNestedTiling =
            isSpatialFirstNestedTiling(nceOp.getOperation(), tilingStrategy);
    SmallVector<uint32_t> filteredDMACosts;
    SmallVector<uint32_t> filteredDPUCosts;
    auto useFilteredCosts = false;
    const auto temporalSize = tilingStrategy[Dims4D::Act::C];
    if (!useFilteredCosts) {
        filteredDMACosts = SmallVector<uint32_t>(layerDMACosts);
        filteredDPUCosts = layerDPUCosts;
    }

    auto weightsOperand = nceOp.getWeightsOperand();
    // If the activation needs to be split:
    //      no weights operand - like Eltwise
    //      tiling on spatial dimension
    //      or DepthConvolution, the activation input is always split for tile, and the DMAs should be accumulated
    const auto nonOneDims = getNonOneDim(tiles.front().axis);
    const auto actSplit = llvm::any_of(nonOneDims, [](Dim tileDim) {
        return tileDim != Dims4D::Act::C;
    });
    bool isActDMASplitOnEachTile =
            (weightsOperand == nullptr || actSplit || mlir::isa<VPU::NCEDepthConvolutionOp>(nceOp.getOperation()));

    auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nceOp.getOperation());
    // The DMA will overlap with DPU from the second tile on
    bool isDMAOverlappedWithDPU =
            enablePrefetchTiling ? tilingInfoOp != nullptr &&
                                           tilingInfoOp.isSupportedTiling(tiles, vpux::TilingMode::PIPELINING, log)
                                 : false;

    uint32_t totalDMACost = 0;

    if (isDMAOverlappedWithDPU) {
        // Act DMA from second tile on will be overlapped with DPU of previous tile
        log.trace("[Cost Analysis] Assumption: Activation DMA will pipeline with DPU");
        totalDMACost += getPrefetchDMACostOverlappsWithPreviousDPU(filteredDPUCosts, ArrayRef(filteredDMACosts),
                                                                   isActDMASplitOnEachTile);
    } else {
        // When DMA not overlapped with DPU
        //  - If act DMA will be copied on each tile, we need to accumulate all the DMA costs
        //  - If act DMA will be shared for all tiles, we only add the first DMA cost
        totalDMACost += isActDMASplitOnEachTile ? std::accumulate(filteredDMACosts.begin(), filteredDMACosts.end(), 0U)
                                                : filteredDMACosts.front();
    }

    // overlapped part should be removed from DPU cost list
    // because it can't be overlapped with other DMA costs anymore
    if (!useFilteredCosts) {
        layerDPUCosts = std::move(filteredDPUCosts);
    } else {
        for (size_t i = 0; i < tiles.size(); i += temporalSize) {
            layerDPUCosts[i] = filteredDPUCosts[i / temporalSize];
        }
    }

    return totalDMACost;
}

uint32_t vpux::VPU::getOutputDMACostForNCEOp(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                             SmallVector<uint32_t>& layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                             bool enablePrefetchTiling, vpux::Logger log) {
    VPUX_THROW_WHEN(layerDPUCosts.empty() || layerDPUCosts.size() != layerDMACosts.size(),
                    "Layer DPU costs must be non-empty and equal to DMA costs in size");

    const auto outShape = getShape(nceOp->getResult(0));
    auto tiles = outTiles.empty() ? OutputTiling({TileInfo(outShape)}) : outTiles;

    auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nceOp.getOperation());
    // The DMA of the current tile will overlap with DPU of the next tile
    nceOp->setAttr(outputPipelining, mlir::BoolAttr::get(nceOp->getContext(), true));
    bool isDMAOverlappedWithDPU =
            enablePrefetchTiling ? tilingInfoOp != nullptr &&
                                           tilingInfoOp.isSupportedTiling(tiles, vpux::TilingMode::PIPELINING, log)
                                 : false;
    nceOp->removeAttr(outputPipelining);

    uint32_t totalDMACost = 0;

    if (isDMAOverlappedWithDPU) {
        // Output DMA expect for the last tile will be overlapped with DPU of the next tile
        log.trace("[Cost Analysis] Assumption: Output DMA will pipeline with DPU");
        totalDMACost += getOutputDMACostOverlappsWithNextDPU(layerDPUCosts, layerDMACosts, true);
    } else {
        totalDMACost += std::accumulate(layerDMACosts.begin(), layerDMACosts.end(), 0U);
    }

    return totalDMACost;
}

size_t vpux::VPU::getNumNonConstantOperands(mlir::Operation* op) {
    return std::count_if(op->operand_begin(), op->operand_end(), [](mlir::Value operand) {
        return !mlir::isa_and_nonnull<Const::DeclareOp>(operand.getDefiningOp());
    });
}

bool vpux::VPU::hasLayerWithMultipleInputs(mlir::Operation* op) {
    return std::any_of(op->user_begin(), op->user_end(), [](mlir::Operation* user) {
        return getNumNonConstantOperands(user) > 1 || hasLayerWithMultipleInputs(user);
    });
}

bool vpux::VPU::isSingleBatchRequired(mlir::Operation* op) {
    return !mlir::isa<VPU::MVNOp, VPU::MVN1NormalizeOp, VPU::MVN1SumOp, VPU::LSTMSequenceOp, VPU::DequantizeOp,
                      VPU::ReverseOp, VPU::GridSampleOp, VPU::MemPermuteOp>(op);
}

bool vpux::VPU::setSOKForRuntimeDequantConvolution(VPU::NCEOpInterface nceOp, LayerCostModel& costModel) {
    // change to SOK if depends on SOK dequantize and does not fit into CMX
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    if (!clusteredOp) {
        return false;
    }
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType());
    if (!clusteredOp.isOperationSplitOverKernelCompatible(outputType.getShape(), ShapeRef(), ShapeRef())) {
        return false;
    }
    // If Tiling is not possible with SOK , cost model returns COST_MAX, avoid assigning SOK when tiling cant be done.
    // Heuristic should be improved by considering SOH + DMA cost vs SOK when costs are more reliable : E#163827
    auto SOKCost = costModel.getLayerCost(clusteredOp, VPU::MultiClusterStrategy::SplitOverKernel);
    if (SOKCost == costModel.COST_MAX) {
        return false;
    }

    // check if weights fit into CMX (force SOK only for large weights)
    auto numClusters = VPU::getOptimalNumClusters(nceOp.getOperation(), outputType.getShape(),
                                                  VPU::MultiClusterStrategy::SplitOverKernel);
    auto weightType = nceOp.getWeightsOperand().getType();
    auto filterSize = VPU::getTotalAllocSizeWithDistribution(
                              weightType, getFilterDistributionAttrFromOp(nceOp, weightType, numClusters,
                                                                          VPU::MultiClusterStrategy::SplitOverKernel))
                              .count();
    auto totalAvailableCMXSize = VPU::getTotalCMXSize(nceOp.getOperation()).count();

    if (filterSize <= totalAvailableCMXSize) {
        return false;
    }

    clusteredOp.setMultiClusterStrategy(MultiClusterStrategy::SplitOverKernel);
    return true;
}

bool vpux::VPU::alignStrategyWithParentRuntimeDequant(VPU::ClusteredOpInterface clusteredOp,
                                                      LayerCostModel& costModel) {
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation());
    if (!nceOp) {
        return false;
    }
    auto weights = nceOp.getWeightsOperand();
    if (!weights) {
        return false;
    }
    auto dequantizeOp = weights.getDefiningOp<VPU::DequantizeOp>();
    if (!dequantizeOp) {
        return false;
    }
    auto opStrategy = clusteredOp.getMultiClusterStrategy();
    auto dequantizeOpStrategy = dequantizeOp.getMultiClusterStrategy();

    if (!opStrategy.has_value() || !dequantizeOpStrategy.has_value()) {
        return false;
    }
    if (opStrategy != MultiClusterStrategy::SplitOverHeight && opStrategy != MultiClusterStrategy::SplitOverWidth &&
        opStrategy != MultiClusterStrategy::SplitOverHeightOverlapped && opStrategy != MultiClusterStrategy::HKSwitch) {
        return false;
    }
    if (dequantizeOpStrategy != MultiClusterStrategy::SplitOverKernel) {
        return false;
    }

    return setSOKForRuntimeDequantConvolution(nceOp, costModel);
}

double vpux::VPU::getStrideDMACorrectionThresholdByArch([[maybe_unused]] config::ArchKind arch) {
    return strideDMACorrectionThresholdInBitsV1;
}
