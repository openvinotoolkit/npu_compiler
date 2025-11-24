//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vf_axis_increment.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/TypeSwitch.h>

#include <queue>

using namespace vpux;
using namespace VPU;

TilingStorage vpux::VPU::restoreTilingRegions(VPU::VerticalFusionOp vfOp, Logger log,
                                              const TilingOperationStorage::UPtr& opStorage) {
    auto storage = calculateTilingRegions(
            vfOp, ArrayRef(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(vfOp.getTilingStrategy()))), log,
            opStorage);

    VPUX_THROW_WHEN(mlir::failed(storage), "Restored tiling {0} of operation {1} is incorrect",
                    vfOp.getTilingStrategy(), vfOp);

    return storage.value();
}

mlir::FailureOr<TilingStorage> vpux::VPU::calculateTilingRegions(VPU::VerticalFusionOp vfOp, const OutputTiling& tiles,
                                                                 Logger log,
                                                                 const TilingOperationStorage::UPtr& opStorage) {
    auto termination = vfOp.getBody()->getTerminator();

    if (termination == nullptr) {
        return mlir::failure();
    }

    if (termination->getNumOperands() == 0) {
        return mlir::failure();
    }

    auto lastOp = termination->getOperands().back().getDefiningOp();

    if (lastOp == nullptr) {
        return mlir::failure();
    }

    return calculateTilingRegions(lastOp, tiles, log, opStorage);
}

mlir::FailureOr<TilingStorage> vpux::VPU::calculateTilingRegions(VPU::VerticalFusionOp vfOp,
                                                                 ArrayRef<int64_t> tilingStrategy, Logger log,
                                                                 const TilingOperationStorage::UPtr& opStorage) {
    const auto outputShape = getShape(vfOp->getResult(0));
    const auto strategy = Shape(tilingStrategy);

    const auto tiles = fillDividedTiles(vfOp, strategy, outputShape);
    if (mlir::failed(tiles)) {
        return mlir::failure();
    }

    return calculateTilingRegions(vfOp, tiles.value(), log, opStorage);
}

mlir::FailureOr<TilingStorage> vpux::VPU::calculateTilingRegions(mlir::Operation* operation, const OutputTiling& tiles,
                                                                 Logger log,
                                                                 const TilingOperationStorage::UPtr& opStorage,
                                                                 const llvm::SetVector<mlir::Operation*>& fusedOps) {
    TilingStorage storage;

    // Work queue of (operation, tile, tileNumber)
    using WorkItem = std::tuple<mlir::Operation*, TileInfo, size_t>;
    std::queue<WorkItem> workQueue;

    // Initialize the queue with the starting operation and its tiles
    for (const auto& item : tiles | indexed) {
        auto& tile = item.value();
        const auto tileNumber = item.index();
        workQueue.push(std::make_tuple(operation, tile, tileNumber));
    }

    // Process all operations in the queue
    while (!workQueue.empty()) {
        auto workItem = workQueue.front();
        auto& currentOp = std::get<0>(workItem);
        auto& tile = std::get<1>(workItem);
        auto& tileNumber = std::get<2>(workItem);
        workQueue.pop();

        auto inputTiling = TilingInfo(ArrayRef({tile}));
        try {
            if (auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(currentOp)) {
                inputTiling = tilingBuilderOp.backInferTileInfo(tile, log);
                if (opStorage != nullptr && !inputTiling.tiles.empty()) {
                    auto& allValues = opStorage->gatherValue(currentOp);
                    const auto sameTile = [&](auto& item) {
                        auto& tiling = item.second;
                        return tiling.second.shape == tile.shape &&
                               tiling.first.tiles[0].shape == inputTiling.tiles[0].shape;
                    };
                    if (llvm::none_of(allValues, sameTile)) {
                        if (auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(currentOp)) {
                            if (!isMultiClusterCompatibleForTiling(currentOp, {tile}, log) ||
                                !tilingInfoOp.isSupportedTiling({tile}, TilingMode::ISOLATED, log)) {
                                return mlir::failure();
                            }
                        }
                    }
                }
            } else if (auto tilingViewLikeOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(currentOp)) {
                if (!tilingViewLikeOp.isSupportedOutTile(tile)) {
                    return mlir::failure();
                }
                inputTiling = tilingViewLikeOp.backInferTileInfo(tile, log);
            } else {
                VPUX_THROW("Unsupported operation type {0} for VF", currentOp->getName());
            }
        } catch (Exception&) {
            return mlir::failure();
        }

        // Store the tiling info for the current operation
        if (opStorage != nullptr) {
            opStorage->insert(currentOp, tileNumber, std::make_pair(inputTiling, tile));
            log.trace("TileInfo inserted for operation at loc {0} tile {1}, {2}", currentOp->getLoc(), tileNumber,
                      tile);
        }

        // Process each operand of the current operation
        for (const auto& op : currentOp->getOperands() | indexed) {
            const auto operand = op.value();
            const auto indexOp = op.index();

            if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                // Store block argument info
                storage.insert(arg.getArgNumber(), tileNumber, inputTiling.tiles[indexOp]);
                log.trace("TileInfo inserted for argument {0} tile {1}, {2}", arg.getArgNumber(), tileNumber,
                          inputTiling.tiles[indexOp]);
                continue;
            }

            if (!fusedOps.empty() && !fusedOps.contains(operand.getDefiningOp())) {
                continue;
            }

            // Create the tile for the operand and add it to the work queue
            auto& oneTile = inputTiling.tiles[indexOp];
            auto inputTile = TileInfo(oneTile.shape, oneTile.offsets, tile.axis, tile.isCompletedTile);
            workQueue.push(std::make_tuple(operand.getDefiningOp(), inputTile, tileNumber));
        }
    }

    return storage;
}

int64_t vpux::VPU::getTilingLimit(Dim axis, ArrayRef<mlir::Operation*> operations, bool tilingOnHW) {
    SmallVector<int64_t> axisLengthsOfNonChannelAlignedOps;
    SmallVector<int64_t> axisLengthsOfChannelAlignedOps;
    auto hasChannelAxis = axis == Dims4D::Act::C;
    // Using queue to traverse all ops in VF block and back-infer their tiling dims
    // The data structure pattern - {(op, tilingDim)...}
    std::queue<std::pair<mlir::Operation*, Dim>> opQueue;
    opQueue.push({operations.back(), axis});
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
            if (parentOp != nullptr && llvm::is_contained(operations, parentOp)) {
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

std::optional<Dim> vpux::VPU::getVFTilingDim(ArrayRef<int64_t> tilingStrategy) {
    auto maxTiledLen = std::max_element(tilingStrategy.begin(), tilingStrategy.end());
    if (maxTiledLen != tilingStrategy.end() && *maxTiledLen != 1) {
        return Dim(std::distance(tilingStrategy.begin(), maxTiledLen));
    }
    return std::nullopt;
}

mlir::FailureOr<Dim> vpux::VPU::getVFTilingDim(ArrayRef<int64_t> tilingStrategy,
                                               ArrayRef<mlir::Operation*> operations) {
    auto dim = getVFTilingDim(tilingStrategy);
    if (dim.has_value()) {
        return dim.value();
    }

    auto allowedDims = getAllowedDims(operations, Logger::global());
    if (allowedDims.empty()) {
        return mlir::failure();
    }

    return allowedDims.front();
}

DimArr vpux::VPU::getAllowedDims(ArrayRef<mlir::Operation*> operations, Logger log) {
    DimArr allowedDims;
    auto outputOp = operations.back();
    auto outputTilingDims = getTileDimOrder(outputOp, TilingMode::ISOLATED, log);

    for (auto outputDim : outputTilingDims) {
        std::queue<std::pair<mlir::Operation*, Dim>> opQueue;
        opQueue.push({outputOp, outputDim});
        bool isAllowed = true;
        while (!opQueue.empty()) {
            auto curOp = opQueue.front().first;
            auto curAxis = opQueue.front().second;
            opQueue.pop();

            auto curAllowedDims = getTileDimOrder(curOp, TilingMode::ISOLATED, log);
            if (llvm::find(curAllowedDims, curAxis) == curAllowedDims.end()) {
                isAllowed = false;
                break;
            }

            if (auto tilingViewLikeOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(curOp)) {
                curAxis = tilingViewLikeOp.backInferTilingDim(curAxis);
            }

            for (auto input : curOp->getOperands()) {
                auto parentOp = input.getDefiningOp();
                if (parentOp != nullptr && llvm::find(operations, parentOp) != operations.end()) {
                    opQueue.push({parentOp, curAxis});
                }
            }
        }

        if (isAllowed) {
            allowedDims.push_back(outputDim);
        }
    }

    return allowedDims;
}

// Back-infer the tiling for the given operation according to infer strategy
template <typename ArgType, typename ResultType>
ResultType vpux::VPU::backInfer(VPU::TilingViewLikeOpInterface opIf, ArgType tiling, VPU::BackInferStrategy strategy) {
    std::variant<std::function<vpux::Dim(vpux::Dim)>,
                 std::function<mlir::SmallVector<int64_t>(mlir::ArrayRef<int64_t>)>>
            variantFunc;

    switch (strategy) {
    case VPU::BackInferStrategy::TILING_DIM:
        variantFunc = std::bind(&VPU::TilingViewLikeOpInterface::backInferTilingDim, opIf, std::placeholders::_1);
        break;
    case VPU::BackInferStrategy::TILING_STRATEGY:
        variantFunc = std::bind(&VPU::TilingViewLikeOpInterface::backInferTilingStrategy, opIf, std::placeholders::_1);
        break;
    default:
        VPUX_THROW("Unsupported back-infer strategy algorithm");
    }

    return std::get<std::function<ResultType(ArgType)>>(variantFunc)(tiling);
}

bool isLegalTilingDim(VPU::TilingViewLikeOpInterface opIf, Dim tiling) {
    return opIf.isSupportedTilingDim(tiling);
}

bool isLegalTilingDim(VPU::TilingViewLikeOpInterface opIf, ArrayRef<int64_t> tiling) {
    const auto tiles = fillDividedTiles(opIf, ShapeRef(tiling), getShape(opIf->getResult(0)));
    if (mlir::failed(tiles)) {
        return false;
    }
    const auto uniqueTiles = VPU::getUniqueShapeTilingCandidates(opIf.getOperation(), tiles.value(), Logger::global());
    auto allTilesAreLegal = llvm::all_of(uniqueTiles, [&](const TileInfo& outputTile) {
        return opIf.isSupportedOutTile(outputTile);
    });
    return allTilesAreLegal;
}

// Template method for inputs tiling dim (or strategy) back-infer
// Infer logic is decided by the passed strategy
// opTilingMap - record all ops in VF block and their tiling dims (or strategies) when back-infer given outputTiling
template <typename ArgType, typename ResultType, typename VFConfigType>
mlir::FailureOr<SmallVector<ResultType>> vpux::VPU::backInferVFTiling(
        VFConfigType& vfConfig, ArgType outputTiling, BackInferStrategy strategy,
        std::unordered_map<mlir::Operation*, ResultType>& opTilingMap) {
    // Vector index is the operand index in the VF op
    SmallVector<ResultType> inputTilings(vfConfig.getSubgraph().getNumOperands(), ResultType(outputTiling));
    auto vfOps = vfConfig.getVFOperations();
    VPUX_THROW_WHEN(vfConfig.getOutputs().empty(), "No output operation in VF block {0}", vfConfig.getSubgraph());
    auto outputOp = vfConfig.getOutputs().front();

    // Using queue to traverse all ops in VF block and back-infer their tiling dims (strategies)
    // The data structure pattern like {(op, tilingDims)...}
    std::queue<std::pair<mlir::Operation*, ResultType>> opQueue;
    opQueue.push({outputOp, ResultType(outputTiling)});
    while (!opQueue.empty()) {
        auto curOp = opQueue.front().first;
        auto curTiling = opQueue.front().second;
        opQueue.pop();

        opTilingMap[curOp] = curTiling;

        if (auto tilingViewLikeOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(curOp)) {
            curTiling = VPU::backInfer<ArgType, ResultType>(tilingViewLikeOp, curTiling, strategy);
            if (!isLegalTilingDim(tilingViewLikeOp, curTiling)) {
                return mlir::failure();
            }
        }

        for (auto operand : curOp->getOperands()) {
            if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                inputTilings[arg.getArgNumber()] = curTiling;
            }

            auto parentOp = operand.getDefiningOp();
            if (llvm::find(vfOps, parentOp) != vfOps.end()) {
                opQueue.push({parentOp, curTiling});
            }
        }
    }

    return inputTilings;
}

template <typename VFConfigType>
mlir::FailureOr<SmallVector<SmallVector<int64_t>>> vpux::VPU::backInferVFTilingStrategy(
        VFConfigType& config, ArrayRef<int64_t> tilingStrategy,
        std::unordered_map<mlir::Operation*, SmallVector<int64_t>>& opStrategyMap) {
    return backInferVFTiling<mlir::ArrayRef<int64_t>, mlir::SmallVector<int64_t>, VFConfigType>(
            config, tilingStrategy, BackInferStrategy::TILING_STRATEGY, opStrategyMap);
}

template <typename VFConfigType>
mlir::FailureOr<SmallVector<vpux::Dim>> vpux::VPU::backInferVFTilingDim(
        VFConfigType& config, vpux::Dim outputDim, std::unordered_map<mlir::Operation*, vpux::Dim>& opDimMap) {
    return backInferVFTiling<vpux::Dim, vpux::Dim, VFConfigType>(config, outputDim, BackInferStrategy::TILING_DIM,
                                                                 opDimMap);
}

VPU::VerticalFusionOp vpux::VPU::fuseOpsInBlock(mlir::PatternRewriter& rewriter, VPU::VerticalFusionOp vfOp,
                                                mlir::Operation* prevOp, mlir::ArrayAttr tilingInfo /*nullptr*/) {
    SmallVector<mlir::Operation*> prevOperations;
    auto prevOperands = prevOp->getOperands();
    SmallVector<mlir::Value> prevBlockArgs = prevOp->getOperands();
    mlir::Operation* lastOp = prevOp;
    const auto getOpPointer = [](auto& op) -> mlir::Operation* {
        return &op;
    };
    if (auto prevBlock = mlir::dyn_cast<VPU::VerticalFusionOp>(prevOp)) {
        prevBlockArgs.clear();
        llvm::copy(prevBlock.getBody()->getOperations() | transformed(getOpPointer),
                   std::back_inserter(prevOperations));
        llvm::copy(prevBlock.getBody()->getArguments(), std::back_inserter(prevBlockArgs));
        lastOp = prevBlock.getBody()->getTerminator()->getOperands().back().getDefiningOp();
    } else {
        prevOperations.push_back(prevOp);
    }

    SmallVector<size_t> argNumLastOp;
    SmallVector<size_t> argNumCurrentOp;
    mlir::DenseMap<size_t, size_t> opArgMapper;
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange blockArgs) {
        mlir::IRMapping mapper;

        const auto curBlockArgs = vfOp.getBody()->getArguments();

        // map new operands with previous ones for both blocks
        for (size_t i = 0; i < blockArgs.size(); ++i) {
            if (i < prevBlockArgs.size()) {
                // map operands of first block with current ones
                mapper.map(prevBlockArgs[i], blockArgs[i]);

                // in case there is operand in second block which also
                // can be mapped with this operands - map them too
                if (opArgMapper.count(i) != 0) {
                    mapper.map(curBlockArgs[opArgMapper[i]], blockArgs[i]);
                }
            } else {
                // map other operands
                if (argNumCurrentOp.size() > i - prevBlockArgs.size() &&
                    curBlockArgs.size() > argNumCurrentOp[i - prevBlockArgs.size()]) {
                    mapper.map(curBlockArgs[argNumCurrentOp[i - prevBlockArgs.size()]], blockArgs[i]);
                }
                if (opArgMapper.count(i) != 0) {
                    mapper.map(curBlockArgs[opArgMapper[i]], blockArgs[i]);
                }
            }
        }

        SmallVector<mlir::Value> newResults;

        const auto copyOps = [&](auto operations) {
            for (auto* op : operations) {
                if (!mlir::isa<VPU::YieldOp>(op)) {
                    auto* clonedOp = builder.clone(*op, mapper);
                    if (op == lastOp && !argNumLastOp.empty()) {
                        for (auto index : argNumLastOp) {
                            mapper.map(curBlockArgs[index], clonedOp->getResult(0));
                        }
                    }
                } else {
                    for (auto operand : op->getOperands()) {
                        if (operand.getDefiningOp() != lastOp) {
                            newResults.push_back(mapper.lookupOrDefault(operand));
                        }
                    }
                }
            }
        };

        copyOps(prevOperations);
        copyOps(vfOp.getBody()->getOperations() | transformed(getOpPointer));

        builder.create<VPU::YieldOp>(loc, newResults.back());
    };

    SmallVector<mlir::Value> newOperands(prevOperands.begin(), prevOperands.end());

    VPUX_THROW_WHEN(lastOp == nullptr, "Couldn't find last operation in region {0}", prevOp);

    // for all operands in current region
    // sort them in following baskets
    // argNumLastOp - if operand is previous region
    // argNumCurrentOp - arguments of current region
    // opArgMapper - in case operand is already in the list,
    // map this operand and argument of current block in order to
    // create right correlation
    for (auto arg : vfOp.getBody()->getArguments()) {
        auto operand = vfOp.getOperand(arg.getArgNumber());
        if (operand.getDefiningOp() == prevOp) {
            argNumLastOp.push_back(arg.getArgNumber());
        } else {
            const auto value = llvm::find(newOperands, operand);
            if (value == newOperands.end()) {
                newOperands.push_back(operand);
                argNumCurrentOp.push_back(arg.getArgNumber());
            } else {
                opArgMapper[std::distance(newOperands.begin(), value)] = arg.getArgNumber();
            }
        }
    }

    if (tilingInfo == nullptr) {
        tilingInfo = vfOp.getTilingStrategy();
    }

    return rewriter.create<VPU::VerticalFusionOp>(vfOp.getLoc(), vfOp->getResultTypes(), newOperands, bodyBuilder,
                                                  tilingInfo);
}

bool vpux::VPU::isSpatialTiling(ArrayRef<int64_t> strategy) {
    if (strategy.size() <= Dims4D::Act::numSpatialDims) {
        return false;
    }

    for (auto index : irange(Dims4D::Act::numSpatialDims)) {
        if (strategy[Dims4D::Act::getSpatialDim(index).ind()] > 1) {
            return true;
        }
    }
    return false;
};

mlir::Operation* vpux::VPU::findParent(mlir::Value operand) {
    auto parent = operand.getDefiningOp();

    while (parent != nullptr && isPureViewOp(parent)) {
        parent = parent->getOperand(0).getDefiningOp();
    }

    return parent;
}

SmallVector<mlir::OpOperand*> vpux::VPU::findUses(mlir::Operation* operation) {
    SmallVector<mlir::OpOperand*> uses;

    for (auto& use : operation->getUses()) {
        if (!isPureViewOp(use.getOwner())) {
            uses.emplace_back(&use);
            continue;
        }

        auto usersBelow = findUses(use.getOwner());
        if (!usersBelow.empty()) {
            llvm::copy(usersBelow, std::back_inserter(uses));
        }
    }

    return uses;
}

// Previous operation will be scheduled early when:
// 1. Previous operation inputs are all from network inputs
// 2. Next operation inputs are not all from network inputs
// This function detect below case:
// VF1 input is from network input, it will be scheduled very early
// VF3 has input from VF2, it has scheduling dependency with VF2
// Therefore, VF1 will be scheduled much earlier than VF3
// There will be a spilling between VF1 and VF3 even though VF1 is a CMX operation
/*
    BlockArg    ...
        |        |
       VF1      VF2
        \       /
           VF3
            |
*/
bool vpux::VPU::isPrevOperationEarlyScheduled(mlir::Operation* prevOp, mlir::Operation* nextOp) {
    auto areOperandsFromNetworkInputs = [&](mlir::Operation* operation) {
        return llvm::all_of(operation->getOperands(), [&](mlir::Value operand) {
            auto parentOp = findParent(operand);
            return parentOp == nullptr;
        });
    };

    // Check if previous operation inputs are all from network inputs
    if (!areOperandsFromNetworkInputs(prevOp)) {
        return false;
    }

    // Check if any other input of the next operation is not from network input
    for (auto vfOperand : nextOp->getOperands()) {
        auto parentOp = findParent(vfOperand);
        if (parentOp == prevOp) {
            continue;
        }

        if (parentOp != nullptr) {
            return true;
        }
    }

    return false;
}

bool vpux::VPU::spillingCopyOpsCanBeOverlapped(config::ArchKind arch) {
    return getDMAChannelsWithIndependentLinkAgents(arch) !=
           SmallVector<VPUIP::DmaChannelType>{VPUIP::DmaChannelType::NOT_SPECIFIED};
}

bool vpux::VPU::isOpTiled(mlir::Operation* op) {
    if (op == nullptr) {
        return false;
    }
    if (!op->hasAttr(vpux::tilingStrategy)) {
        return false;
    }
    const auto tilingDims = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(op->getAttr(vpux::tilingStrategy)));
    return llvm::any_of(tilingDims, [](auto value) {
        return value != 1;
    });
}

bool vpux::VPU::onlySupportPartialTilingDims(vpux::VPU::TilingViewLikeOpInterface viewOp) {
    auto dims = DimsOrder::fromValue(viewOp->getResult(0)).toPermutation();
    return llvm::any_of(dims, [&](auto dim) {
        return !viewOp.isSupportedTilingDim({dim});
    });
};

// Explicit instantiation of the template function for V1/V2 VFConfig
template mlir::FailureOr<SmallVector<SmallVector<int64_t>>> vpux::VPU::backInferVFTilingStrategy<
        vpux::VPU::VF::v1::VFConfig>(vpux::VPU::VF::v1::VFConfig& config, ArrayRef<int64_t> tilingStrategy,
                                     std::unordered_map<mlir::Operation*, SmallVector<int64_t>>& opStrategyMap);

template mlir::FailureOr<SmallVector<SmallVector<int64_t>>> vpux::VPU::backInferVFTilingStrategy<
        vpux::VPU::VF::v2::VFConfig>(vpux::VPU::VF::v2::VFConfig& config, ArrayRef<int64_t> tilingStrategy,
                                     std::unordered_map<mlir::Operation*, SmallVector<int64_t>>& opStrategyMap);

template mlir::FailureOr<SmallVector<vpux::Dim>> vpux::VPU::backInferVFTilingDim<vpux::VPU::VF::v1::VFConfig>(
        vpux::VPU::VF::v1::VFConfig& config, vpux::Dim outputDim,
        std::unordered_map<mlir::Operation*, vpux::Dim>& opDimMap);

template mlir::FailureOr<SmallVector<vpux::Dim>> vpux::VPU::backInferVFTilingDim<vpux::VPU::VF::v2::VFConfig>(
        vpux::VPU::VF::v2::VFConfig& config, vpux::Dim outputDim,
        std::unordered_map<mlir::Operation*, vpux::Dim>& opDimMap);
