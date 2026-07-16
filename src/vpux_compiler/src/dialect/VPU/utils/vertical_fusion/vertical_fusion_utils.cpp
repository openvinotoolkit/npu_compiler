//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vf_axis_increment.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/profiling/reports/api.hpp"

#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/TypeSwitch.h>

#include <queue>
#include <variant>

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
                            if (auto channelAlignOp = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(currentOp)) {
                                const auto channelDim = tile.shape.size() == 5 ? DimsGroups5D::Act::C : Dims4D::Act::C;
                                if (tile.shape[channelDim] % channelAlignOp.getOutputChannelAlignment() != 0) {
                                    return mlir::failure();
                                }
                            }

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
            log.trace("TileInfo inserted for operation at {0} {1} tile {2}, {3}", currentOp->getName(),
                      currentOp->getLoc(), tileNumber, tile);
        }

        // Process each operand of the current operation
        for (const auto& op : currentOp->getOperands() | indexed) {
            const auto operand = op.value();
            const auto indexOp = op.index();

            if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                // Store block argument info
                storage.insert(std::make_pair(arg.getArgNumber(), currentOp), tileNumber, inputTiling.tiles[indexOp]);
                log.trace("TileInfo inserted for argument {0} tile {1}, {2}", arg.getArgNumber(), tileNumber,
                          inputTiling.tiles[indexOp]);
                continue;
            }

            if (!fusedOps.empty() && !fusedOps.contains(operand.getDefiningOp())) {
                continue;
            }

            // Create the tile for the operand and add it to the work queue
            auto& oneTile = inputTiling.tiles[indexOp];
            auto inputTile = TileInfo(oneTile.shape, oneTile.offsets, oneTile.axis, tile.isCompletedTile);
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
                limit = std::max(limit / MINIMUM_LENGTH_TILING, int64_t(1));
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
                if (!tilingViewLikeOp.isSupportedTilingDim({curAxis})) {
                    isAllowed = false;
                    break;
                }
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
            if (!isLegalTilingDim(tilingViewLikeOp, curTiling)) {
                return mlir::failure();
            }
            curTiling = VPU::backInfer<ArgType, ResultType>(tilingViewLikeOp, curTiling, strategy);
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
mlir::FailureOr<SmallVector<vpux::Dim>> vpux::VPU::backInferVFTilingDim(
        VFConfigType& config, vpux::Dim outputDim, std::unordered_map<mlir::Operation*, vpux::Dim>& opDimMap) {
    return backInferVFTiling<vpux::Dim, vpux::Dim, VFConfigType>(config, outputDim, BackInferStrategy::TILING_DIM,
                                                                 opDimMap);
}

VPU::VerticalFusionOp vpux::VPU::fuseOpsInBlock(mlir::OpBuilder& rewriter, VPU::VerticalFusionOp vfOp,
                                                mlir::Operation* prevOp, mlir::ArrayAttr tilingInfo /*nullptr*/,
                                                bool isManualConfigured /*false*/) {
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
    mlir::UnitAttr isManualConfiguredAttr = isManualConfigured ? mlir::UnitAttr::get(vfOp.getContext()) : nullptr;
    return rewriter.create<VPU::VerticalFusionOp>(vfOp.getLoc(), vfOp->getResultTypes(), newOperands, bodyBuilder,
                                                  tilingInfo, isManualConfiguredAttr);
}

VPU::VerticalFusionOp vpux::VPU::fuseOpsInBlock(mlir::RewriterBase& rewriter, ArrayRef<VPU::VerticalFusionOp> ops,
                                                mlir::ArrayAttr tilingInfo /*nullptr*/,
                                                bool isManualConfigured /*false*/) {
    VPUX_THROW_WHEN(ops.size() <= 1, "Cannot fuse VF op list with size {0}", ops.size());

    auto mergedOp = ops.back();
    bool isMergedOpTemporary = false;
    for (size_t i = ops.size() - 1; i > 0; --i) {
        auto oldMergedOp = mergedOp;
        auto prevVFOp = ops[i - 1];
        rewriter.setInsertionPoint(mergedOp);
        mergedOp =
                vpux::VPU::fuseOpsInBlock(rewriter, mergedOp, prevVFOp.getOperation(), tilingInfo, isManualConfigured);
        if (mergedOp == nullptr) {
            if (oldMergedOp != ops.back()) {
                rewriter.eraseOp(oldMergedOp);
            }
            return nullptr;
        }

        if (isMergedOpTemporary && oldMergedOp != ops.back()) {
            rewriter.eraseOp(oldMergedOp);
        }
        isMergedOpTemporary = true;
    }

    return mergedOp;
}

// fuseSingleViewOpsChainInBlock is needed when the predecessor is a linear chain of pure
// view-like ops (e.g. Reshape -> MemPermute -> QuantizeCast) rather than a single op or an
// existing VPU::VerticalFusionOp. Avoid calling fuseOpsInBlock in a loop; it causes spurious
// intermediate VF blocks that corrupt the IR.
//
// Procedure:
//   1. Validates adjacency so every pair is a true producer-consumer link.
//   2. Collects external inputs of the whole chain as new VF block arguments in one pass,
//      avoiding duplicate operand entries.
//   3. Clones the chain ops in topological order (farthest first) before cloning vfOp's body,
//      so SSA dominance is maintained in the new block.
//   4. Replaces old VF block arguments that were fed by chain outputs with the cloned results,
//      while external-operand arguments are re-wired to the corresponding new block arguments.
VPU::VerticalFusionOp vpux::VPU::fuseSingleViewOpsChainInBlock(mlir::OpBuilder& rewriter, VPU::VerticalFusionOp vfOp,
                                                               ArrayRef<mlir::Operation*> prevOpChain,
                                                               mlir::ArrayAttr tilingInfo /*nullptr*/,
                                                               bool isManualConfigured /*false*/) {
    VPUX_THROW_WHEN(prevOpChain.empty(), "Cannot fuse an empty producer chain into {0}", vfOp);

    // Validate that each adjacent pair forms one producer-consumer link in the chain.
    for (size_t i = 0; i + 1 < prevOpChain.size(); ++i) {
        auto* childOp = prevOpChain[i];
        auto* parentOp = prevOpChain[i + 1];
        const auto isLinked = llvm::any_of(childOp->getOperands(), [&](mlir::Value operand) {
            return operand.getDefiningOp() == parentOp;
        });
        VPUX_THROW_WHEN(!isLinked, "Invalid producer chain: op {0} is not fed by op {1}", childOp->getName(),
                        parentOp->getName());
    }

    const auto isOpInChain = [&](mlir::Operation* op) {
        return llvm::is_contained(prevOpChain, op);
    };

    // Clone from farthest producer to the one closest to vfOp.
    SmallVector<mlir::Operation*> chainOps(prevOpChain.begin(), prevOpChain.end());
    std::reverse(chainOps.begin(), chainOps.end());

    SmallVector<mlir::Value> newOperands;
    mlir::DenseMap<mlir::Value, size_t> operandToIdx;
    const auto appendOperand = [&](mlir::Value operand) {
        if (operandToIdx.count(operand) == 0) {
            operandToIdx[operand] = newOperands.size();
            newOperands.push_back(operand);
        }
    };

    // 1) Collect chain external inputs.
    for (auto* op : chainOps) {
        for (auto operand : op->getOperands()) {
            if (!isOpInChain(operand.getDefiningOp())) {
                appendOperand(operand);
            }
        }
    }

    // 2) Collect current VF external inputs and remember which block args are fed by chain values.
    SmallVector<size_t> vfArgNumsFromChain;
    vfArgNumsFromChain.reserve(vfOp.getBody()->getNumArguments());
    mlir::DenseMap<size_t, mlir::Value> vfArgToChainValue;
    mlir::DenseMap<size_t, size_t> vfArgToExternalOperandIdx;
    for (auto arg : vfOp.getBody()->getArguments()) {
        const auto argNum = arg.getArgNumber();
        const auto operand = vfOp.getOperand(argNum);
        if (isOpInChain(operand.getDefiningOp())) {
            vfArgNumsFromChain.push_back(argNum);
            vfArgToChainValue[argNum] = operand;
            continue;
        }
        appendOperand(operand);
        vfArgToExternalOperandIdx[argNum] = operandToIdx[operand];
    }

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange blockArgs) {
        mlir::IRMapping mapper;
        const auto curBlockArgs = vfOp.getBody()->getArguments();

        // Map newly created block arguments back to original external values.
        for (size_t i = 0; i < newOperands.size(); ++i) {
            mapper.map(newOperands[i], blockArgs[i]);
        }

        // Clone the whole producer chain first.
        for (auto* op : chainOps) {
            if (!mlir::isa<VPU::YieldOp>(op)) {
                builder.clone(*op, mapper);
            }
        }

        // Map old VF block args that were connected to the chain to cloned chain results.
        for (auto argNum : vfArgNumsFromChain) {
            mapper.map(curBlockArgs[argNum], mapper.lookupOrDefault(vfArgToChainValue[argNum]));
        }

        // Map old VF block args that were external operands.
        for (const auto& item : vfArgToExternalOperandIdx) {
            mapper.map(curBlockArgs[item.first], blockArgs[item.second]);
        }

        SmallVector<mlir::Value> newResults;
        const auto getOpPointer = [](auto& op) -> mlir::Operation* {
            return &op;
        };
        for (auto* op : vfOp.getBody()->getOperations() | transformed(getOpPointer)) {
            if (!mlir::isa<VPU::YieldOp>(op)) {
                builder.clone(*op, mapper);
                continue;
            }
            for (auto operand : op->getOperands()) {
                newResults.push_back(mapper.lookupOrDefault(operand));
            }
        }

        builder.create<VPU::YieldOp>(loc, newResults.back());
    };

    if (tilingInfo == nullptr) {
        tilingInfo = vfOp.getTilingStrategy();
    }
    mlir::UnitAttr isManualConfiguredAttr = isManualConfigured ? mlir::UnitAttr::get(vfOp.getContext()) : nullptr;
    return rewriter.create<VPU::VerticalFusionOp>(vfOp.getLoc(), vfOp->getResultTypes(), newOperands, bodyBuilder,
                                                  tilingInfo, isManualConfiguredAttr);
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
    auto producerOut = findProducerValue(operand);
    return producerOut != nullptr ? producerOut.getDefiningOp() : nullptr;
}

mlir::OpResult vpux::VPU::findProducerValue(mlir::Value operand) {
    auto producerValue = mlir::dyn_cast_if_present<mlir::OpResult>(operand);
    while (producerValue != nullptr && isPureViewOp(producerValue.getOwner())) {
        auto parent = producerValue.getOwner();
        producerValue = mlir::dyn_cast_if_present<mlir::OpResult>(parent->getOperand(0));
    }

    return producerValue;
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
    auto inputRank = mlir::cast<vpux::NDTypeInterface>(viewOp->getOperand(0).getType()).getRank();
    auto outputRank = mlir::cast<vpux::NDTypeInterface>(viewOp->getResult(0).getType()).getRank();
    if (inputRank != outputRank) {
        return true;
    }

    auto dims = DimsOrder::fromValue(viewOp->getResult(0)).toPermutation();
    return llvm::any_of(dims, [&](auto dim) {
        return !viewOp.isSupportedTilingDim({dim});
    });
};

SmallVector<mlir::Operation*> vpux::VPU::getParentViewLikeOpsInVF(mlir::Operation* operation) {
    SmallVector<mlir::Operation*> parents;
    for (auto operand : operation->getOperands()) {
        auto parentOp = operand.getDefiningOp();
        if (parentOp != nullptr && VPU::isPureViewOp(parentOp)) {
            parents.push_back(parentOp);
            parents.append(getParentViewLikeOpsInVF(parentOp));
        }
    }
    llvm::sort(parents, [](auto op1, auto op2) {
        return op1->isBeforeInBlock(op2);
    });
    return parents;
}

VPU::DistributedTensorType vpux::VPU::inferDistributedTypeThroughViewOps(VPU::DistributedTensorType srcType,
                                                                         ArrayRef<mlir::Operation*> viewOps) {
    vpux::NDTypeInterface type =
            vpux::getTensorType(srcType.getShape(), srcType.getElementType(), srcType.getDimsOrder(),
                                srcType.getMemSpace(), getBounds(srcType), getDynamicDimsMask(srcType));
    auto getDataShape = [](vpux::NDTypeInterface type) {
        if (auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(type)) {
            auto ndType = mlir::cast<vpux::NDTypeInterface>(sparseType.getData());
            return ndType.getShape();
        }
        return type.getShape();
    };

    auto isViewOpSizeChanged = [&](mlir::Operation* op) {
        auto inType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
        auto outType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());

        return getDataShape(inType) != getDataShape(outType) && inType.getElemTypeSize() == outType.getElemTypeSize() &&
               inType.getDimsOrder() == outType.getDimsOrder();
    };

    auto distribution = VPU::DistributionInfo::getClassFromAttr(srcType.getDistribution());
    for (auto viewOp : viewOps) {
        if (auto distCastOp = mlir::dyn_cast<VPU::DistributedCastOpInterface>(viewOp)) {
            auto castedTypeWithDistribution = distCastOp.inferCastedTypeAndDistribution(type, distribution);
            if (mlir::failed(castedTypeWithDistribution)) {
                return nullptr;
            }
            type = mlir::cast<vpux::NDTypeInterface>(castedTypeWithDistribution.value().first);
            distribution = castedTypeWithDistribution.value().second;
            continue;
        } else if (isViewOpSizeChanged(viewOp) && VPU::isDistributionWithExplicitShapesAndOffsets(distribution)) {
            // handle slice-like ops with size change, e.g. Slice
            auto sliceInShape = getDataShape(viewOp->getOperand(0).getType());
            auto sliceOutShape = getDataShape(viewOp->getResult(0).getType());

            const auto sliceDims = IE::getDiffInOutSizeDims(sliceInShape, sliceOutShape);
            if (VPU::isSegmentedLikeDistributionMode(type, distribution)) {
                // Skip for segmented-like distribution and the size change is on the distributed dim,
                auto tilingDim = Dim(VPU::getDistributedTilingAxis(distribution.getNumTiles()));
                if (llvm::is_contained(sliceDims, tilingDim)) {
                    return nullptr;
                }
            }

            auto possibleDistribution = VPU::DistributionInfo::getAttrFromClass(viewOp->getContext(), distribution);
            auto newDistributionAttr = VPU::updateSliceLikeOpsAlignment(viewOp->getContext(), type.getShape(),
                                                                        sliceOutShape, possibleDistribution);
            auto explicitDistribution =
                    VPU::getExplicitDistrAttrForSliceLikeOps(newDistributionAttr, sliceOutShape, type.getShape().raw(),
                                                             type.getElementType(), viewOp->getContext());
            distribution = VPU::DistributionInfo::getClassFromAttr(explicitDistribution);
            type = mlir::cast<vpux::NDTypeInterface>(viewOp->getResult(0).getType());
            continue;
        }
    }

    TensorDistributionMap distributionMap;
    distributionMap.insert(std::make_pair(type, distribution));
    return mlir::cast<VPU::DistributedTensorType>(getDistributedTypeFromDistributionMap(type, distributionMap));
};

mlir::BlockArgument vpux::VPU::getLinkedArgumentBetweenVFOps(VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) {
    for (auto blockArg : currentOp.getBody()->getArguments()) {
        auto operand = currentOp.getOperand(blockArg.getArgNumber());
        if (operand.getDefiningOp() == prevOp.getOperation()) {
            return blockArg;
        }
    }
    return nullptr;
}

// Explicit instantiation of the template function for V1/V2 VFConfig
template mlir::FailureOr<SmallVector<vpux::Dim>> vpux::VPU::backInferVFTilingDim<vpux::VPU::VF::v1::VFConfig>(
        vpux::VPU::VF::v1::VFConfig& config, vpux::Dim outputDim,
        std::unordered_map<mlir::Operation*, vpux::Dim>& opDimMap);

template mlir::FailureOr<SmallVector<vpux::Dim>> vpux::VPU::backInferVFTilingDim<vpux::VPU::VF::v2::VFConfig>(
        vpux::VPU::VF::v2::VFConfig& config, vpux::Dim outputDim,
        std::unordered_map<mlir::Operation*, vpux::Dim>& opDimMap);
