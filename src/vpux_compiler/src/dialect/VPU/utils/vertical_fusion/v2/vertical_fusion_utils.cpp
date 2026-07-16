//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include <mlir/Support/LLVM.h>
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_case.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduler_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduling_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_axis_increment.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/profiling/reports/api.hpp"

namespace vpux::VPU::VF::v2 {

constexpr double ELTWISE_PERFORMANT_RATIO_FOR_MULTI_DIM_TILING = 0.5;

bool isCmxOperation(mlir::Operation* operation, const bool checkTilingType) {
    if (!mlir::isa_and_nonnull<VPU::TilingInfoOpInterface, VPU::VerticalFusionOp>(operation)) {
        return false;
    }

    if (!operation->hasAttr(tilingStrategy)) {
        return true;
    }

    auto tiling = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy)));
    auto hasTiling = llvm::any_of(tiling, [](auto value) {
        return value > 1;
    });

    if (!hasTiling) {
        return true;
    }

    if (checkTilingType) {
        if (isSpatialTiling(tiling)) {
            return false;
        }

        const auto checkNCEFunc = [](mlir::Operation* oper) {
            auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(oper);
            auto hasWeights = nceOp != nullptr && nceOp.getWeightsOperand() != nullptr;
            auto needFullInput = true;
            if (auto vfInterface = mlir::dyn_cast<VPU::VerticalFusionOpInterface>(oper)) {
                auto restrictedAxes = vfInterface.restrictedFusionAxes();
                needFullInput =
                        !restrictedAxes.empty() && llvm::find(restrictedAxes, Dims4D::Act::C) != restrictedAxes.end();
            }
            return hasWeights && needFullInput;
        };

        if (auto vfUser = mlir::dyn_cast<VPU::VerticalFusionOp>(operation)) {
            auto userConfig = VFConfig(vfUser);
            return userConfig.getOperationsForTiling().size() == 1 &&
                   llvm::all_of(userConfig.getInputs(), checkNCEFunc);
        }
        return checkNCEFunc(operation);
    }

    const auto outputSize = mlir::cast<NDTypeInterface>(operation->getResult(0).getType()).getTotalAllocSize();

    if (outputSize > VPU::getTotalCMXSize(operation)) {
        return false;
    }
    return !isSpatialTiling(tiling);
}

// Using queue to traverse all ops in VF block and back-infer their tiling dims
// The data structure pattern - {(op, tilingDim)...}
void traverseVFOps(VFConfig& config, Dim outputDim, llvm::function_ref<void(mlir::Operation*, Dim)> visitor) {
    std::queue<std::pair<mlir::Operation*, Dim>> opQueue;

    VPUX_THROW_WHEN(config.getOutputs().empty(), "VF has no output operations");
    auto* lastOp = config.getOutputs().back();
    opQueue.push({lastOp, outputDim});
    auto operations = config.getVFOperations().getArrayRef();

    while (!opQueue.empty()) {
        auto curOp = opQueue.front().first;
        auto curAxis = opQueue.front().second;
        opQueue.pop();

        visitor(curOp, curAxis);

        for (auto input : curOp->getOperands()) {
            auto* parentOp = input.getDefiningOp();
            if (parentOp == nullptr || llvm::find(operations, parentOp) == operations.end()) {
                continue;
            }
            // Back infer the tiling dim for the producer op. For view like op, it can be inferred by the interface
            // backInferTilingDim. For other ops, currently it lacks the interface to back-infer tiling dim for the
            // input. So here we need to handle the input tiling dim case by case.
            auto inputAxis = curAxis;
            if (auto tilingViewLikeOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(curOp)) {
                inputAxis = tilingViewLikeOp.backInferTilingDim(inputAxis);
            } else if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(curOp)) {
                if (nceOp.getWeightsOperand() != nullptr && nceOp.getWeightsOperand().getDefiningOp() == parentOp) {
                    auto is5DTensor = getShape(curOp->getResult(0)).size() == 5;
                    auto isTilingOnChannel =
                            is5DTensor ? inputAxis == DimsGroups5D::Act::C : inputAxis == Dims4D::Act::C;
                    if (!isTilingOnChannel) {
                        continue;
                    }
                    inputAxis = is5DTensor ? DimsGroups5D::Filter::OC : Dims4D::Filter::OC;
                }
            }
            opQueue.push({parentOp, inputAxis});
        }
    }
}

mlir::DenseMap<mlir::Operation*, Dim> buildOpDimMap(VFConfig& config, Dim outputDim) {
    mlir::DenseMap<mlir::Operation*, Dim> dimMap;
    traverseVFOps(config, outputDim, [&](mlir::Operation* op, Dim dim) {
        dimMap.insert({op, dim});
    });
    return dimMap;
}

int64_t getTilingLimit(Dim axis, VFConfig& config, bool multiDimTiling) {
    SmallVector<int64_t> axisLengthsOfNonChannelAlignedOps;
    SmallVector<int64_t> axisLengthsOfChannelAlignedOps;
    auto hasChannelAxis = axis == Dims4D::Act::C;

    traverseVFOps(config, axis, [&](mlir::Operation* curOp, Dim curAxis) {
        hasChannelAxis = hasChannelAxis || curAxis == Dims4D::Act::C;

        auto limit = getMaxNumTiles(curOp)[curAxis.ind()];
        if (curAxis.ind() >= Dims4D::Act::getSpatialDim(0).ind()) {
            limit = multiDimTiling ? divUp(limit, (MINIMUM_LENGTH_TILING * MINIMUM_LENGTH_TILING))
                                   : std::max(limit / MINIMUM_LENGTH_TILING, int64_t(1));
        } else if (curAxis.ind() == Dims4D::Act::C.ind() && multiDimTiling) {
            limit = divUp(limit, (MINIMUM_LENGTH_TILING * MINIMUM_LENGTH_TILING));
        }
        limit = std::min(limit, VPU::NCEInvariant::VPU_DIMENSION_LIMIT / MINIMUM_LENGTH_TILING);
        if (mlir::isa<IE::AlignedChannelsOpInterface>(curOp) && curAxis == Dims4D::Act::C) {
            axisLengthsOfChannelAlignedOps.emplace_back(limit);
        } else {
            axisLengthsOfNonChannelAlignedOps.emplace_back(limit);
        }
    });

    auto axisIncrement = getVFAxisIncrement(axis);
    if (hasChannelAxis && axis != Dims4D::Act::C) {
        // If there exists channel tiling, use the channel axis increment logic to get divisible factors
        // otherwise, use the default axis increment
        axisIncrement = getVFAxisIncrement(Dims4D::Act::C);
    }
    VPUX_THROW_WHEN(axisIncrement == nullptr, "Cannot get functions to get values for axis {0}", axis);

    return axisIncrement->getLimitValue(axisLengthsOfChannelAlignedOps, axisLengthsOfNonChannelAlignedOps);
}

std::optional<Dim> getNonTiledDimForVFOptimization(const VFSplit& vfSplit) {
    auto dim = llvm::find_if(vfSplit, [](const auto& kv) {
        return !kv.second.has_value();
    });

    if (dim == vfSplit.end()) {
        return std::nullopt;
    }

    return dim->first;
}

mlir::FailureOr<TilingStorage> calculateTilingRegions(VFConfig& config, ArrayRef<int64_t> tilingStrategy, Logger log,
                                                      const TilingOperationStorage::UPtr& opStorage) {
    auto outputOp = config.getSubgraph() != nullptr ? config.getSubgraph() : config.getOutputs().back();
    const auto outputShape = getBoundedShape(outputOp->getResult(0));
    const auto strategy = Shape(tilingStrategy);

    // returns true for output operations as only them should be aligned dynamically
    const auto dynAlignmentFunction = [&](mlir::Operation* op) {
        return llvm::is_contained(config.getOutputs(), op);
    };

    const auto tiles = fillDividedTiles(config.getOutputs().back(), config.getVFOperations().getArrayRef(), strategy,
                                        outputShape, dynAlignmentFunction);
    if (mlir::failed(tiles)) {
        return mlir::failure();
    }

    return calculateTilingRegions(config.getOutputs().back(), tiles.value(), log, opStorage, config.getVFOperations());
}

// get a valid tiling strategy for VF block between the given range of tiling strategy
// it returns mlir::failure() if all tiling strategies in this range can't be supported by all operations or operations
// can't fit in CMX
// otherwise, return the valid strategy that is close to the lower or upper boundary according to closeToUpperLimit
// parameter
mlir::FailureOr<SmallVector<int64_t>> getValidTilingStrategyFromRange(
        VFConfig& config, ArrayRef<int64_t> lowerTilingStrategy, ArrayRef<int64_t> upperTilingStrategy,
        bool closeToUpperLimit, Dim tilingAxis, TilingOperationStorage::UPtr& opStorage, Logger log) {
    SmallVector<int64_t> validTilingStrategy =
            closeToUpperLimit ? to_small_vector(upperTilingStrategy) : to_small_vector(lowerTilingStrategy);

    auto notBeyondBoundary = [](int64_t value, int64_t lowerLimit, int64_t upperLimit, bool closeToUpperLimit) {
        return closeToUpperLimit ? value >= lowerLimit : value <= upperLimit;
    };

    auto axisIncrement = VPU::getVFAxisIncrement(tilingAxis);
    VPUX_THROW_WHEN(axisIncrement == nullptr, "Cannot get functions to get values for axis {0}", tilingAxis);

    SmallVector<int64_t> dimValues;
    auto dimValue = validTilingStrategy[tilingAxis.ind()];
    const auto lowerDimLimit = lowerTilingStrategy[tilingAxis.ind()];
    const auto upperDimLimit = upperTilingStrategy[tilingAxis.ind()];

    while (notBeyondBoundary(dimValue, lowerDimLimit, upperDimLimit, closeToUpperLimit)) {
        auto currentDimValue = dimValue;
        dimValues.push_back(currentDimValue);

        if (closeToUpperLimit) {
            axisIncrement->decreasedValue(dimValue, lowerDimLimit);
        } else {
            axisIncrement->increasedValue(dimValue, upperDimLimit);
        }

        if (currentDimValue == dimValue) {
            break;
        }
    }

    std::vector<TilingOperationStorage::UPtr> candidateOpStorages(dimValues.size());
    auto ctx = config.getVFOperations().front()->getContext();
    auto result = vpux::parallel_find_index(ctx, static_cast<size_t>(dimValues.size()), [&](size_t valueIdx) {
        auto tilingStrategyCandidate = validTilingStrategy;
        tilingStrategyCandidate[tilingAxis.ind()] = dimValues[valueIdx];

        auto curOpStorage = std::make_unique<TilingOperationStorage>();
        auto tilingRegions = calculateTilingRegions(config, tilingStrategyCandidate, log, curOpStorage);
        if (mlir::failed(tilingRegions)) {
            return false;
        }
        candidateOpStorages[valueIdx] = std::move(curOpStorage);
        return true;
    });
    if (mlir::failed(result)) {
        // No valid strategy has been found
        return mlir::failure();
    }

    const auto strategyIdx = result.value();
    auto tilingStrategy = std::move(validTilingStrategy);
    tilingStrategy[tilingAxis.ind()] = dimValues[strategyIdx];

    opStorage = std::move(candidateOpStorages[strategyIdx]);
    VPUX_THROW_WHEN(opStorage == nullptr, "Expected tiling strategy to have cached storage");
    return tilingStrategy;
}

// get a maximal valid tiling strategy for VF block between the given range of tiling strategy
// it returns mlir::failure() if all tiling strategies in this range can't be supported by all operations or operations
// can't fit in CMX
mlir::FailureOr<SmallVector<int64_t>> getMaximalValidTilingStrategyFromRange(
        VFConfig& config, ArrayRef<int64_t> lowerTilingStrategy, ArrayRef<int64_t> upperTilingStrategy, Dim tilingAxis,
        TilingOperationStorage::UPtr& opStorage, Logger log) {
    return getValidTilingStrategyFromRange(config, lowerTilingStrategy, upperTilingStrategy, true, tilingAxis,
                                           opStorage, log);
}

// get a minimal valid tiling strategy for VF block between the given range of tiling strategy
// it returns mlir::failure() if all tiling strategies in this range can't be supported by all operations or operations
// can't fit in CMX
mlir::FailureOr<SmallVector<int64_t>> getMinimalValidTilingStrategyFromRange(
        VFConfig& config, ArrayRef<int64_t> lowerTilingStrategy, ArrayRef<int64_t> upperTilingStrategy, Dim tilingAxis,
        TilingOperationStorage::UPtr& opStorage, Logger log) {
    return getValidTilingStrategyFromRange(config, lowerTilingStrategy, upperTilingStrategy, false, tilingAxis,
                                           opStorage, log);
}

bool hasBeforeDDRUsers(mlir::Operation* prevOp, mlir::Operation* nextOp) {
    // check if previous operation has more than 1 users apart from nextOp
    // and all of them are in DDR
    auto uses = findUses(prevOp);
    if (uses.size() == 1) {
        return false;
    }

    const auto checkUser = [&](auto* use) {
        auto* user = use->getOwner();
        return user != nextOp && user->isBeforeInBlock(nextOp) && !v2::isCmxOperation(use->getOwner(), true);
    };

    return llvm::any_of(uses, checkUser);
}

namespace {
bool isDataTiledOnSameAxisWithMCStrategy(VPU::DistributedTensorType dataType, ArrayRef<int64_t> tiling) {
    if (dataType == nullptr) {
        return false;
    }
    auto mode = dataType.getDistribution().getMode().getValue();
    if (mode != VPU::DistributionMode::SEGMENTED && mode != VPU::DistributionMode::OVERLAPPED) {
        return false;
    }
    auto tilingScheme = parseIntArrayAttr<int64_t>(dataType.getDistribution().getNumTiles());
    VPUX_THROW_WHEN(tilingScheme.size() != tiling.size(), "Unmatched tiling scheme and tiling size");
    auto axis = getDistributedTilingAxis(tilingScheme);
    VPUX_THROW_UNLESS(checked_cast<size_t>(axis) < tilingScheme.size(), "Invalid tiling axis");
    return tiling[axis] != 1;
}
}  // namespace

bool hasOutputSpilledForDifferentDataSizeUses(mlir::Operation* op) {
    auto outElementSize = getShape(op->getResult(0)).totalSize();
    auto usedBySizeChangedViewOps = llvm::all_of(op->getUsers(), [&](auto user) {
        if (!isPureViewOp(user)) {
            return false;
        }
        auto elementSize = getShape(user->getResult(0)).totalSize();
        return outElementSize != elementSize;
    });
    return usedBySizeChangedViewOps;
}

bool outputTileAxisIsSameAsMultiClusterStrategy(mlir::Operation* op) {
    if (!isOpTiled(op)) {
        return false;
    }
    const auto tilingDim = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(op->getAttr(vpux::tilingStrategy)));
    auto distributedType =
            mlir::dyn_cast_if_present<VPU::DistributedTensorType>(getDistributedOutputType(op, op->getResult(0)));
    return isDataTiledOnSameAxisWithMCStrategy(distributedType, tilingDim);
}

bool inputTileAxisIsSameAsMultiClusterStrategy(mlir::Operation* op, mlir::Value operand) {
    if (!isOpTiled(op)) {
        return false;
    }
    const auto tilingDim = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(op->getAttr(vpux::tilingStrategy)));
    auto distributedType = mlir::dyn_cast_if_present<VPU::DistributedTensorType>(getDistributedInputType(op, operand));
    return isDataTiledOnSameAxisWithMCStrategy(distributedType, tilingDim);
}

bool isSpatialDim(Dim dim) {
    return dim.ind() >= static_cast<int32_t>(Dims4D::Act::numSpatialDims);
}

bool isMultiDimTilingPerformant(VFConfig& config, const VFSplit& currentVFSplit) {
    auto outputOp = config.getOutputs().back();
    auto isInplaceEltwise = [](mlir::Operation* op) {
        return mlir::isa<VPU::NCEOpInterface>(op) && op->hasTrait<VPU::EltwiseOp>() && op->getNumOperands() > 1 &&
               op->hasAttr(VPU::isInPlace);
    };
    auto hasInplaceEltwiseOutputWithViewOpInput =
            isInplaceEltwise(outputOp) && llvm::any_of(outputOp->getOperands(), [](mlir::Value operand) {
                return mlir::isa_and_present<VPU::TilingViewLikeOpInterface>(operand.getDefiningOp());
            });
    auto hasMultiDimTiling = currentVFSplit.size() > 1;
    auto inplaceEltwiseOpCount = llvm::count_if(config.getOperationsForTiling(), isInplaceEltwise);
    auto nceOpCount = llvm::count_if(config.getOperationsForTiling(), [](mlir::Operation* op) {
        return mlir::isa<VPU::NCEOpInterface>(op);
    });
    return hasMultiDimTiling || !hasInplaceEltwiseOutputWithViewOpInput ||
           inplaceEltwiseOpCount < ELTWISE_PERFORMANT_RATIO_FOR_MULTI_DIM_TILING * nceOpCount;
}

SmallVector<VFSplit> getSplitFromDimArr(DimArrRef dimsToCheck, DimArrRef allowedDims, VFConfig& config,
                                        bool enableMultiDimTiling) {
    SmallVector<VFSplit> splits;
    for (auto dim : dimsToCheck) {
        VFSplit singleSplit = {{dim, std::nullopt}};
        splits.emplace_back(singleSplit);
        for (auto otherDim : allowedDims) {
            if (enableMultiDimTiling && dim.ind() > otherDim.ind()) {
                const auto isSpatialTilingForOther = isSpatialDim(otherDim);
                auto outerDimLimit = getTilingLimit(otherDim, config, true);
                if (outerDimLimit > 1 || isSpatialTilingForOther) {
                    VFSplit doubleSplit = {{otherDim, outerDimLimit}, {dim, std::nullopt}};
                    splits.emplace_back(doubleSplit);
                }
            }
        }
    }
    return splits;
}

SmallVector<int64_t> restoreTilingBySplit(int64_t rank, const VFSplit& split) {
    SmallVector<int64_t> tilingStrategy(rank, 1);
    for (auto& [dim, dimValue] : split) {
        if (dimValue.has_value()) {
            tilingStrategy[dim.ind()] = dimValue.value();
        }
    }

    return tilingStrategy;
}

VFSplit getVFTilingSplit(ArrayRef<int64_t> tilingStrategy) {
    VFSplit vfSplit;

    for (auto value : tilingStrategy | indexed) {
        if (value.value() > 1) {
            vfSplit[Dim(value.index())] = value.value();
        }
    }

    return vfSplit;
}

int64_t getVFTilesLen(const VFSplit& vfSplit) {
    SmallVector<int64_t> splitValues;
    splitValues.reserve(vfSplit.size());
    llvm::transform(vfSplit, std::back_inserter(splitValues), [](auto& kv) {
        return kv.second.value_or(1);
    });

    return std::accumulate(splitValues.begin(), splitValues.end(), 1, std::multiplies<int64_t>());
}

// return the cube root of the max tile
std::optional<int64_t> getCbrtMaxTileCandidate(int64_t minTile, int64_t maxTile) {
    auto cbrtMaxTile = static_cast<int64_t>(std::floor(std::cbrt(maxTile)));
    if (cbrtMaxTile > minTile) {
        return cbrtMaxTile;
    }
    return std::nullopt;
}

bool isOperandSharedWeightsForTiling(mlir::Operation* op, mlir::Value operand, const TileInfo& tileInfo) {
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    if (nceOp == nullptr || operand != nceOp.getWeightsOperand()) {
        return false;
    }

    // Sparse weights are duplicated for each tile, referring to the OptimizeParallelCopies pass
    if (mlir::isa<VPU::SparseTensorType>(operand.getType())) {
        return false;
    }

    if (getShape(operand) == tileInfo.shape) {
        return true;
    }
    auto tileAxis = tileInfo.axis;
    if (tileAxis.size() == DimsGroups5D::Filter::numDims) {
        tileAxis[DimsGroups5D::Filter::OC] = 1;
        tileAxis[DimsGroups5D::Filter::G] = 1;
    } else {
        tileAxis[Dims4D::Filter::OC] = 1;
    }

    const auto tileOnSpatialDim = llvm::any_of(tileAxis, [](auto dimSize) {
        return dimSize > 1;
    });
    return tileOnSpatialDim;
}

namespace {
vpux::profiling::TaskInfo makeTaskInfo(const vpux::VPU::TimelineInterval& interval, vpux::Logger log) {
    vpux::profiling::TaskInfo taskInfo = {};
    switch (interval._mExecutor) {
    case vpux::config::ExecutorKind::DMA_NN:
        taskInfo.exec_type = vpux::profiling::TaskInfo::ExecType::DMA;
        break;
    case vpux::config::ExecutorKind::DPU:
        taskInfo.exec_type = vpux::profiling::TaskInfo::ExecType::DPU;
        taskInfo.isSubtask = false;
        break;
    case vpux::config::ExecutorKind::SHAVE_ACT:
        taskInfo.exec_type = vpux::profiling::TaskInfo::ExecType::SW;
        taskInfo.clusterId = 0;
        break;
    default:
        log.warning("Not supported executor type - '{0}'", interval._mExecutor);
        taskInfo.exec_type = vpux::profiling::TaskInfo::ExecType::NONE;
        break;
    }

    taskInfo.name = llvm::formatv("{0}/{1}", vpux::stringifyPrimaryLocation(interval._mLoc), interval._mIndex);
    taskInfo.layer_type = vpux::getLayerTypeFromLocation(interval._mLoc);
    taskInfo.start_time_ns = interval._mBegin;
    taskInfo.duration_ns = interval._mEnd - interval._mBegin;
    return taskInfo;
}
}  // namespace

void printVFSchedulingTrace(mlir::func::FuncOp funcOp, const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                            Logger log) {
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    auto tileOp = config::getTileExecutor(moduleOp);
    VPUX_THROW_WHEN(tileOp == nullptr, "Cannot get tile executor");
    auto freqInMHz = tileOp.getProcessorFrequency().getValueAsDouble();

    auto vfOps = funcOp.getOps<VPU::VerticalFusionOp>() | filtered([](auto vfOp) {
                     return vfOp.getScenario().has_value();
                 });
    VFSchedulingFactory vfFactory(true);
    for (auto item : vfOps | indexed) {
        auto vfOp = item.value();
        auto idx = item.index();
        auto fileName = llvm::formatv("scheduling_trace_vf_{0}.json", idx);
        log.trace("Dumping scheduling trace for VF {0} to file {1}", vfOp->getLoc(), fileName);

        VFConfig config(vfOp);
        auto type = vfOp.getScenario().value();
        auto tilingDims = parseIntArrayAttr<int64_t>(vfOp.getTilingStrategy());
        auto tileLen = std::accumulate(tilingDims.begin(), tilingDims.end(), 1, std::multiplies<int64_t>());
        auto vfScheduling = std::dynamic_pointer_cast<VPU::VF::v2::VFScheduling>(vfFactory.createVFScenario(type, log));
        VPUX_THROW_WHEN(vfScheduling == nullptr, "Cannot create VF scheduling for scenario '{0}'", type);

        auto vfTilingStorage = std::make_unique<TilingOperationStorage>();
        auto tilingStorage = calculateTilingRegions(vfOp, tilingDims, log, vfTilingStorage);
        VPUX_THROW_WHEN(mlir::failed(tilingStorage), "Cannot get tiling regions for {0} and {1} tiles", vfOp->getLoc(),
                        tilingDims);

        auto timeIntervals = vfScheduling->getTimeIntervals(config, tileLen, vfTilingStorage, costFunction);
        std::vector<profiling::TaskInfo> taskInfos;
        taskInfos.reserve(timeIntervals.size());
        llvm::transform(timeIntervals, std::back_inserter(taskInfos), [&](auto& interval) {
            return makeTaskInfo(interval, log);
        });

        auto layers = getLayerInfo(taskInfos);
        std::ofstream outStream(fileName.str());
        VPUX_THROW_UNLESS(outStream.good(), "File for schedule traces not created correctly");
        printProfilingAsTraceEvent(taskInfos, layers, /*dpuFreq=*/{freqInMHz, profiling::FreqStatus::SIM}, outStream,
                                   log);
    }
}

std::optional<Dim> getVFOptimizedDim(const VFSplit& vfSplit) {
    auto dim = llvm::find_if(vfSplit, [](const auto& kv) {
        return !kv.second.has_value();
    });

    if (dim == vfSplit.end()) {
        return std::nullopt;
    }

    return dim->first;
}

bool cmxSizeExceedForEltwiseOpWithSwOpUser(VFConfig& currentConfig, ArrayRef<mlir::Operation*> parents, Logger log) {
    /*
        Check the pattern below:
                         ParentVF0
                           /   \
                 EltwiseOp    SiblingOp
                     |
                   SWOp
                     |
    The execution order of EltWiseOp, SiblingOp and SWOp will be EltwiseOp -> SwOp -> SiblingOp, in which SiblingOp is
    expected to be overlapped with SwOp, but if may result in dynamic spilling when the cmx size of EltwiseOp and SWOp
    is greater than the available CMX Size.
    */
    auto currentVFOp = currentConfig.getSubgraph();
    auto uses = findUses(currentVFOp);
    if (uses.size() != 1) {
        return false;
    }
    auto userVFOp = mlir::dyn_cast_or_null<VPU::VerticalFusionOp>((*uses.begin())->getOwner());
    if (userVFOp == nullptr) {
        return false;
    }
    VFConfig userConfig(userVFOp, true);
    auto swOpUser = mlir::dyn_cast<VPU::SWOpInterface>(userConfig.getOperationsForTiling().front());
    if (swOpUser == nullptr) {
        return false;
    }
    auto parentHasMultiUses = llvm::any_of(parents, [&](auto* parent) {
        auto parentUses = findUses(parent);
        auto otherUserCount = llvm::count_if(parentUses, [&](auto* use) {
            auto userOp = use->getOwner();
            return userOp != nullptr && userOp != currentVFOp && VF::v2::isCmxOperation(userOp, false);
        });
        return otherUserCount > 0;
    });
    if (!parentHasMultiUses) {
        return false;
    }

    const auto currentTiling = parseIntArrayAttr<int64_t>(currentVFOp.getTilingStrategy());
    const auto userTiling = parseIntArrayAttr<int64_t>(userVFOp.getTilingStrategy());
    auto hasTiling = [&](const auto& tiling) {
        return llvm::any_of(tiling, [](auto i) {
            return i != 1;
        });
    };
    if (hasTiling(currentTiling) || hasTiling(userTiling)) {
        // Skipp this complex scenario
        return false;
    }

    auto eltwiseOp = currentConfig.getOperationsForTiling().front();
    auto getUsedSize = [&](mlir::Operation* operation) {
        auto usedSize = vpux::VPU::getRequiredCMX(operation, TileInfo(getShape(operation->getResult(0))), log);
        return usedSize;
    };
    auto strategy = vpux::VPU::getMultiClusterStrategyFromOp(eltwiseOp);
    auto types = vpux::VPU::getTileTypes(eltwiseOp, TileInfo(getShape(eltwiseOp->getResult(0))), strategy);
    auto sharedInputSize = types.front().getTotalAllocSize();
    auto totalAvailableCMXSize = getTotalCMXVFPipelineFragmentationAwareSize(currentVFOp);
    // Caculate the required size for the eltwise op and swOp user
    auto usedSize = getUsedSize(swOpUser) + sharedInputSize;
    return usedSize > totalAvailableCMXSize;
}

mlir::BlockArgument getVFBlockArgument(mlir::Value operand) {
    if (operand == nullptr) {
        return nullptr;
    }
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
        return blockArg;
    }
    auto definingOp = operand.getDefiningOp();
    while (definingOp != nullptr && VPU::isPureViewOp(definingOp)) {
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(definingOp->getOperand(0))) {
            return blockArg;
        }
        definingOp = definingOp->getOperand(0).getDefiningOp();
    }
    return nullptr;
}

bool supportMultiClusterStrategyAdjustmentInVF([[maybe_unused]] mlir::Operation* op) {
    if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op)) {
        // Cost model is not accurate for SEP ops, disable the adjustment for now
        if (mlir::isa<VPU::SparseTensorType>(op->getOperand(0).getType())) {
            return false;
        }
    }
    return true;
}

std::optional<std::pair<mlir::Operation*, int64_t>> findFirstNonViewUser(mlir::Operation* operation) {
    while (operation != nullptr && !operation->use_empty()) {
        const auto use = operation->use_begin();
        auto user = use->getOwner();
        if (mlir::isa<VPU::TilingViewLikeOpInterface>(user)) {
            operation = user;
        } else {
            return std::make_pair(user, use->getOperandNumber());
        }
    }
    return std::nullopt;
}
}  // namespace vpux::VPU::VF::v2
