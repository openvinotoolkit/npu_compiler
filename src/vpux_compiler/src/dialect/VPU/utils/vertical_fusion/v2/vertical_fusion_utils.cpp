//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduler_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduling_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_axis_increment.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/profiling/reports/api.hpp"

namespace vpux::VPU::VF::v2 {

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

int64_t getTilingLimit(Dim axis, VFConfig& config, bool tilingOnHW) {
    SmallVector<int64_t> axisLengthsOfNonChannelAlignedOps;
    SmallVector<int64_t> axisLengthsOfChannelAlignedOps;
    auto hasChannelAxis = axis == Dims4D::Act::C;
    // Using queue to traverse all ops in VF block and back-infer their tiling dims
    // The data structure pattern - {(op, tilingDim)...}
    std::queue<std::pair<mlir::Operation*, Dim>> opQueue;

    VPUX_THROW_WHEN(config.getOutputs().empty(), "VF has no output operations");
    auto* lastOp = config.getOutputs().back();
    opQueue.push({lastOp, axis});
    auto operations = config.getVFOperations().getArrayRef();
    while (!opQueue.empty()) {
        auto curOp = opQueue.front().first;
        auto curAxis = opQueue.front().second;
        opQueue.pop();

        auto limit = getMaxNumTiles(curOp)[curAxis.ind()];
        if (curAxis.ind() >= Dims4D::Act::getSpatialDim(0).ind()) {
            if (tilingOnHW) {
                limit = divUp(limit, (MINIMUM_LENGTH_TILING * MINIMUM_LENGTH_TILING));
            } else {
                limit = limit / MINIMUM_LENGTH_TILING;
            }
        }
        limit = std::min(limit, VPU::NCEInvariant::VPU_DIMENSION_LIMIT / MINIMUM_LENGTH_TILING);
        if (mlir::isa<IE::AlignedChannelsOpInterface>(curOp)) {
            axisLengthsOfChannelAlignedOps.emplace_back(limit);
        } else {
            axisLengthsOfNonChannelAlignedOps.emplace_back(limit);
        }

        // Get the next parent ops in this VF region
        for (auto input : curOp->getOperands()) {
            auto parentOp = input.getDefiningOp();
            if (parentOp != nullptr && llvm::find(operations, parentOp) != operations.end()) {
                if (auto tilingViewLikeOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(curOp)) {
                    curAxis = tilingViewLikeOp.backInferTilingDim(curAxis);
                    hasChannelAxis = hasChannelAxis || curAxis == Dims4D::Act::C;
                }
                opQueue.push({parentOp, curAxis});
            }
        }
    }

    auto axisIncrement = getVFAxisIncrement(axis);
    if (hasChannelAxis && axis != Dims4D::Act::C) {
        // If there exists channel tiling, use the channel axis increment logic to get divisible factors
        // otherwise, use the default axis increment
        axisIncrement = getVFAxisIncrement(Dims4D::Act::C);
    }
    VPUX_THROW_WHEN(axisIncrement == nullptr, "Cannot get functions to get values for axis {0}", axis);

    return axisIncrement->getLimitValue(axisLengthsOfChannelAlignedOps, axisLengthsOfNonChannelAlignedOps);
}

mlir::FailureOr<TilingStorage> calculateTilingRegions(VFConfig& config, ArrayRef<int64_t> tilingStrategy, Logger log,
                                                      const TilingOperationStorage::UPtr& opStorage) {
    auto outputOp = config.getSubgraph() != nullptr ? config.getSubgraph() : config.getOutputs().back();
    const auto outputShape = getBoundedShape(outputOp->getResult(0));
    const auto strategy = Shape(tilingStrategy);

    const auto tiles = fillDividedTiles(config.getVFOperations().getArrayRef(), strategy, outputShape);
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

    while (notBeyondBoundary(validTilingStrategy[tilingAxis.ind()], lowerTilingStrategy[tilingAxis.ind()],
                             upperTilingStrategy[tilingAxis.ind()], closeToUpperLimit)) {
        auto curOpStorage = std::make_unique<TilingOperationStorage>();
        auto tilingRegions = calculateTilingRegions(config, validTilingStrategy, log, curOpStorage);
        if (!mlir::failed(tilingRegions)) {
            // a valid strategy is found
            opStorage.reset(curOpStorage.release());
            return validTilingStrategy;
        }

        auto currentValue = validTilingStrategy[tilingAxis.ind()];

        if (closeToUpperLimit) {
            axisIncrement->decreasedValue(validTilingStrategy[tilingAxis.ind()], lowerTilingStrategy[tilingAxis.ind()]);
        } else {
            axisIncrement->increasedValue(validTilingStrategy[tilingAxis.ind()], upperTilingStrategy[tilingAxis.ind()]);
        }

        if (currentValue == validTilingStrategy[tilingAxis.ind()]) {
            return mlir::failure();
        }
    }

    // no valid strategy can be found
    return mlir::failure();
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
    auto distributedType = mlir::dyn_cast_or_null<VPU::DistributedTensorType>(getDistributedOutputType(op));
    return isDataTiledOnSameAxisWithMCStrategy(distributedType, tilingDim);
}

bool inputTileAxisIsSameAsMultiClusterStrategy(mlir::Operation* op, mlir::Value operand) {
    if (!isOpTiled(op)) {
        return false;
    }
    const auto tilingDim = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(op->getAttr(vpux::tilingStrategy)));
    auto distributedType = mlir::dyn_cast_or_null<VPU::DistributedTensorType>(getDistributedInputType(op, operand));
    return isDataTiledOnSameAxisWithMCStrategy(distributedType, tilingDim);
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

bool isOperandSharedWeightsForTiling(mlir::Operation* op, mlir::Value operand, ShapeRef tiledShape) {
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    if (nceOp == nullptr || operand != nceOp.getWeightsOperand()) {
        return false;
    }

    // Sparse weights are duplicated for each tile, referring to the OptimizeParallelCopies pass
    if (mlir::isa<VPU::SparseTensorType>(operand.getType())) {
        return false;
    }
    return getShape(operand) == tiledShape;
}

namespace {
vpux::profiling::TaskInfo makeTaskInfo(const vpux::VPU::TimelineInterval& interval, vpux::Logger log) {
    vpux::profiling::TaskInfo taskInfo = {};
    switch (interval._mExecutor) {
    case vpux::VPU::ExecutorKind::DMA_NN:
        taskInfo.exec_type = vpux::profiling::TaskInfo::ExecType::DMA;
        break;
    case vpux::VPU::ExecutorKind::DPU:
        taskInfo.exec_type = vpux::profiling::TaskInfo::ExecType::DPU;
        taskInfo.isSubtask = false;
        break;
    case vpux::VPU::ExecutorKind::SHAVE_ACT:
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
}  // namespace vpux::VPU::VF::v2
