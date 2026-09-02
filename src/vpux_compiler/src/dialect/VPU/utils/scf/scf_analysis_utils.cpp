//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/dialect_processors.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/LoopLikeInterface.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>
#include <iomanip>

namespace vpux::VPU {

bool isBlockArgument(mlir::Value val) {
    return mlir::isa<mlir::BlockArgument>(val);
}

bool isConstant(mlir::Value value) {
    // Check for constant ops (e.g., arith::ConstantOp, etc.)
    return llvm::isa<mlir::arith::ConstantOp>(value.getDefiningOp());
}

bool isTensorDim(mlir::Value value) {
    return llvm::isa<mlir::tensor::DimOp>(value.getDefiningOp());
}

// OpChainAnalysis class implementation
OpChainAnalysis::OpChainAnalysis(const Logger& log)
        : _registry(DialectProcessorRegistry::createDefault(getInterpolateScalesBoundResolver())), _log(log) {
}

OpChainAnalysis::OpChainAnalysis(const OpChainAnalysis& other)
        : _registry(DialectProcessorRegistry::createDefault(getInterpolateScalesBoundResolver())), _log(other._log) {
}

OpChainAnalysis& OpChainAnalysis::operator=(const OpChainAnalysis& other) {
    if (this != &other) {
        _registry = DialectProcessorRegistry::createDefault(getInterpolateScalesBoundResolver());
        _log = other._log;
        _chainCache.clear();
    }
    return *this;
}

// Coverity requires an explicit destructor and does not treat '= default' as user-defined.
OpChainAnalysis::~OpChainAnalysis() {
}

std::optional<int64_t> OpChainAnalysis::getIntegerFromValue(mlir::Value value, bool processOpChain) {
    if (auto dimOp = mlir::dyn_cast_or_null<mlir::tensor::DimOp>(value.getDefiningOp())) {
        return vpux::VPU::getIntValueFromDimOp(dimOp, _log);
    }

    auto integerValue = mlir::getConstantIntValue(value);
    if (integerValue.has_value()) {
        return integerValue;
    }

    if (processOpChain) {
        llvm::DenseMap<mlir::Value, int64_t> localOperandMap;
        auto parentOpsChain = collectParentOpsChain(value);

        if (evaluateOpChain(parentOpsChain, localOperandMap)) {
            if (localOperandMap.contains(value)) {
                return localOperandMap[value];
            }
        }
    }

    return std::nullopt;
}

void OpChainAnalysis::updateChainCache(mlir::Value val,
                                       const llvm::SmallSetVector<mlir::Operation*, DEFAULT_ARG_SET_SIZE>& chain) {
    _chainCache[val] = chain;
}

llvm::SmallSetVector<mlir::Operation*, DEFAULT_ARG_SET_SIZE> OpChainAnalysis::collectParentOpsChain(mlir::Value val) {
    if (auto cacheIterator = _chainCache.find(val); cacheIterator != _chainCache.end()) {
        return cacheIterator->second;
    }

    if (val == nullptr || val.getDefiningOp() == nullptr) {
        return {};
    }

    SmallVector<mlir::Operation*, DEFAULT_ARG_SET_SIZE> startNodes = {val.getDefiningOp()};

    auto stopSearch = [](mlir::Operation* op) {
        return mlir::isa<mlir::tensor::DimOp>(op);
    };

    auto getNeighbors = [&](mlir::Operation* op) -> llvm::SmallSetVector<mlir::Operation*, DEFAULT_ARG_SET_SIZE> {
        llvm::SmallSetVector<mlir::Operation*, DEFAULT_ARG_SET_SIZE> neighbors;
        for (auto operand : op->getOperands()) {
            if (auto definingOp = operand.getDefiningOp()) {
                if (!stopSearch(definingOp)) {
                    neighbors.insert(definingOp);
                }
            }
        }
        return neighbors;
    };

    auto results = vpux::VPU::collectOpsInTopologicalOrder(startNodes, getNeighbors, stopSearch);
    llvm::SmallSetVector<mlir::Operation*, DEFAULT_ARG_SET_SIZE> opChain(results.begin(), results.end());

    updateChainCache(val, opChain);
    return opChain;
}

bool OpChainAnalysis::processOperations(llvm::ArrayRef<mlir::Operation*> operations,
                                        llvm::DenseMap<mlir::Value, int64_t>& valueMap) {
    // Create a self-referential block processor that dialect processors can use
    // to recursively process nested blocks (e.g., in scf.if, scf.for)
    BlockProcessor blockProcessor = [this](mlir::Block* block,
                                           llvm::DenseMap<mlir::Value, int64_t>& blockValueMap) -> bool {
        SmallVector<mlir::Operation*> blockOps;
        for (auto& op : *block) {
            blockOps.push_back(&op);
        }
        return processOperations(blockOps, blockValueMap);
    };

    for (auto* op : operations) {
        if (op == nullptr) {
            _log.trace("Encountered null operation pointer");
            return false;
        }

        auto* processor = _registry->getProcessor(op);
        if (processor == nullptr) {
            auto* dialect = op->getDialect();
            _log.trace("Unsupported dialect ({0}) encountered", dialect ? dialect->getNamespace() : "unknown");
            return false;
        }

        if (!processor->processOperation(op, valueMap, blockProcessor)) {
            _log.trace("Failed to process {0} operation: {1}", processor->getDialectName(), op->getName());
            return false;
        }
    }

    return true;
}

bool OpChainAnalysis::evaluateOpChain(llvm::SmallSetVector<mlir::Operation*, DEFAULT_ARG_SET_SIZE>& opChain,
                                      llvm::DenseMap<mlir::Value, int64_t>& localOperandMap) {
    SmallVector<mlir::Operation*> operations(opChain.begin(), opChain.end());
    return processOperations(operations, localOperandMap);
}

void getBlockArgsRecursive(mlir::Block& block, llvm::SmallSetVector<mlir::Value, DEFAULT_ARG_SET_SIZE>& blockArgs,
                           const vpux::Logger& log) {
    for (auto& op : block) {
        for (auto operand : op.getOperands()) {
            if (isBlockArgument(operand)) {
                blockArgs.insert(operand);
            }
        }

        for (auto& region : op.getRegions()) {
            for (auto& nestedBlock : region) {
                getBlockArgsRecursive(nestedBlock, blockArgs, log);
            }
        }
    }
}

void OpChainAnalysis::traverseAndGetBlockArgs(mlir::Value val,
                                              llvm::SmallSetVector<mlir::Value, DEFAULT_ARG_SET_SIZE>& blockArgs) {
    if (isBlockArgument(val)) {
        blockArgs.insert(val);
        return;
    }

    if (isConstant(val) || isTensorDim(val)) {
        return;
    }

    llvm::SmallPtrSet<mlir::Operation*, DEFAULT_ARG_SET_SIZE> visitedOps;
    SmallVector<mlir::Operation*> worklist;

    auto defOp = val.getDefiningOp();
    if (defOp != nullptr) {
        worklist.push_back(defOp);
    }

    while (!worklist.empty()) {
        mlir::Operation* currentOp = worklist.pop_back_val();
        if (visitedOps.contains(currentOp)) {
            continue;
        }

        visitedOps.insert(currentOp);

        for (auto operand : currentOp->getOperands()) {
            if (isBlockArgument(operand)) {
                blockArgs.insert(operand);
            } else if (!isConstant(operand) && !isTensorDim(operand)) {
                if (auto* definingOp = operand.getDefiningOp()) {
                    worklist.push_back(definingOp);
                }
            }
        }

        for (auto& region : currentOp->getRegions()) {
            for (auto& block : region.getBlocks()) {
                getBlockArgsRecursive(block, blockArgs, _log);
            }
        }
    }

    // Keep only loop IVs; function arguments (e.g. Interpolate scales) are not
    // valid inputs for generateValueMap().
    llvm::SmallSetVector<mlir::Value, DEFAULT_ARG_SET_SIZE> loopBlockArgs;
    for (auto blockArg : blockArgs) {
        auto arg = mlir::cast<mlir::BlockArgument>(blockArg);
        auto parentOp = arg.getOwner()->getParentOp();
        if (mlir::isa<mlir::LoopLikeOpInterface>(parentOp)) {
            loopBlockArgs.insert(blockArg);
        }
    }
    blockArgs = std::move(loopBlockArgs);
}

SmallVector<mlir::Value> reorderBlocksArgs(const llvm::SmallSetVector<mlir::Value, DEFAULT_ARG_SET_SIZE>& blockArgs) {
    if (blockArgs.size() == 1) {
        return llvm::to_vector(blockArgs);
    }

    // Collect blocks and create a mapping from blocks to their corresponding block arguments
    llvm::DenseMap<mlir::Block*, mlir::Value> blockToArg;
    SmallVector<mlir::Block*> blocks;

    for (auto blockArg : blockArgs) {
        auto arg = mlir::dyn_cast<mlir::BlockArgument>(blockArg);
        auto parentOp = arg.getOwner()->getParentOp();
        auto loopOp = mlir::dyn_cast<mlir::LoopLikeOpInterface>(parentOp);
        if (loopOp == nullptr) {
            return {};
        }
        auto& region = loopOp->getRegion(0);
        if (!region.hasOneBlock()) {
            return {};
        }
        auto* block = &region.front();
        blocks.push_back(block);
        blockToArg[block] = blockArg;
    }

    // Verify all blocks are in a proper nested loop structure (not siblings)
    // Check if any pair of blocks neither dominates the other (sibling loops)
    mlir::DominanceInfo dom;
    for (size_t i = 0; i < blocks.size(); ++i) {
        for (size_t j = i + 1; j < blocks.size(); ++j) {
            // For nested loops, one must dominate the other
            if (!dom.dominates(blocks[i], blocks[j]) && !dom.dominates(blocks[j], blocks[i])) {
                // Sibling loops detected - not supported
                return {};
            }
        }
    }

    // Sort blocks by dominance order
    llvm::sort(blocks, [&](mlir::Block* a, mlir::Block* b) {
        // If a dominates b, a should come first (outer loop before inner loop)
        if (dom.dominates(a, b)) {
            return true;
        }
        // Otherwise b dominates a (already verified no siblings exist)
        return false;
    });

    // Map sorted blocks back to their arguments
    SmallVector<mlir::Value> orderedBlockOperands;
    orderedBlockOperands.reserve(blocks.size());
    for (auto* block : blocks) {
        orderedBlockOperands.push_back(blockToArg[block]);
    }

    return orderedBlockOperands;
}

void iterativeDfs(
        llvm::ArrayRef<mlir::Operation*> startNodes,
        llvm::function_ref<llvm::SmallSetVector<mlir::Operation*, DEFAULT_ARG_SET_SIZE>(mlir::Operation*)> getNeighbors,
        llvm::function_ref<void(mlir::Operation*)> visitPostOrder,
        llvm::function_ref<bool(mlir::Operation*)> stopCheckFn) {
    llvm::SmallPtrSet<mlir::Operation*, DEFAULT_ARG_SET_SIZE> visited;
    struct Node {
        mlir::Operation* operation;
        bool visited;
    };

    SmallVector<Node> stack;
    for (auto node : startNodes) {
        if (node == nullptr || visited.count(node) != 0) {
            continue;
        }

        stack.push_back({node, false});

        while (!stack.empty()) {
            auto& currentNode = stack.back();
            mlir::Operation* currentOp = currentNode.operation;

            if (!currentNode.visited) {
                if (!visited.insert(currentOp).second) {
                    stack.pop_back();
                    continue;
                }

                if (stopCheckFn && stopCheckFn(currentOp)) {
                    visitPostOrder(currentOp);
                    stack.pop_back();
                    continue;
                }

                currentNode.visited = true;
                auto neighbors = getNeighbors(currentOp);
                for (auto neighbor : neighbors) {
                    if (neighbor != nullptr && !visited.count(neighbor)) {
                        stack.push_back({neighbor, false});
                    }
                }
            } else {
                visitPostOrder(currentOp);
                stack.pop_back();
            }
        }
    }
}

std::tuple<int64_t, int64_t, int64_t> OpChainAnalysis::getLoopBoundsAndStep(mlir::scf::ForOp forOp,
                                                                            ValueRangeMap* valueHints) {
    auto lowerBoundOpt = getIntegerFromValue(forOp.getLowerBound(), true);
    auto upperBoundOpt = getIntegerFromValue(forOp.getUpperBound(), true);
    auto stepOpt = getIntegerFromValue(forOp.getStep(), true);

    // When the upper bound is dynamic try to evaluate it using the caller-supplied hints map, which may contain
    // pre-seeded single-value entries such as Interpolate scale bounds.
    if (!upperBoundOpt.has_value() && valueHints) {
        auto result = getOpFoldResultValue(forOp.getUpperBound(), *valueHints, OpChainAnalysis::MODE::MAX_VALUE);
        if (result.has_value() && !result->empty()) {
            upperBoundOpt = result->front();
        }
    }

    if (!lowerBoundOpt.has_value() || !upperBoundOpt.has_value() || !stepOpt.has_value()) {
        VPUX_THROW("Failed to get integer values for scf.for operation bounds and step");
    }

    if (forOp->hasAttr(UPPERBOUND_ATTR)) {
        upperBoundOpt = forOp->getAttrOfType<mlir::IntegerAttr>(UPPERBOUND_ATTR).getInt();
    }

    auto low = lowerBoundOpt.value();
    auto high = upperBoundOpt.value();
    auto st = stepOpt.value();
    if (st <= 0 || low > high || low < 0 || high < 0) {
        VPUX_THROW("Received values for scf.for operation are invalid: low={0}, high={1}, step={2}", low, high, st);
    }

    return {low, high, st};
}

SmallVector<int64_t> OpChainAnalysis::getForallInductionDimRange(mlir::scf::ForallOp forallOp,
                                                                 mlir::BlockArgument& dimInductionArg,
                                                                 ValueRangeMap& valueMap) {
    auto dimInductionIdx = dimInductionArg.getArgNumber();
    auto lowerMixed = forallOp.getMixedLowerBound();
    auto upperMixed = forallOp.getMixedUpperBound();
    auto stepMixed = forallOp.getMixedStep();
    if (dimInductionIdx >= lowerMixed.size() || dimInductionIdx >= upperMixed.size() ||
        dimInductionIdx >= stepMixed.size()) {
        _log.warning("Skipping as specified dimension with idx {0} outside of available range {1}", dimInductionIdx,
                     upperMixed.size());
        return {};
    }

    auto lowerFolded = getOpFoldResultValue(lowerMixed[dimInductionIdx], valueMap, OpChainAnalysis::MODE::ALL_VALUES);
    auto upperFolded = getOpFoldResultValue(upperMixed[dimInductionIdx], valueMap, OpChainAnalysis::MODE::ALL_VALUES);
    auto stepFolded = getOpFoldResultValue(stepMixed[dimInductionIdx], valueMap, OpChainAnalysis::MODE::ALL_VALUES);
    if (!lowerFolded || !upperFolded || !stepFolded || lowerFolded->empty() || upperFolded->empty() ||
        stepFolded->empty()) {
        _log.warning("Failed to fold forall bounds/steps for dim {0}", dimInductionIdx);
        return {};
    }

    const auto stepsSize = stepFolded->size();
    const auto lowerSize = lowerFolded->size();
    const auto upperSize = upperFolded->size();
    const auto maxSize = std::max({lowerSize, upperSize, stepsSize});
    const auto canBroadcast = [&](size_t sz) {
        return sz == 1 || sz == maxSize;
    };
    if (!(canBroadcast(lowerSize) && canBroadcast(upperSize) && canBroadcast(stepsSize))) {
        _log.warning("Mismatched forall bounds/steps sizes for dim {0}: lb={1}, ub={2}, st={3}", dimInductionIdx,
                     lowerSize, upperSize, stepsSize);
        return {};
    }

    const auto broadcastAt = [&](const SmallVector<int64_t>& v, size_t idx) {
        return v.size() == 1 ? v.front() : v[idx];
    };
    llvm::SmallSetVector<int64_t, 16> rangesSet;
    for (size_t idx = 0; idx < maxSize; ++idx) {
        const auto lb = broadcastAt(*lowerFolded, idx);
        const auto ub = broadcastAt(*upperFolded, idx);
        const auto st = broadcastAt(*stepFolded, idx);
        if (st <= 0) {
            _log.warning("Non-positive step {0} for forall dim {1}", st, dimInductionIdx);
            continue;
        }
        for (int64_t i = lb; i < ub; i += st) {
            rangesSet.insert(i);
        }
    }

    return rangesSet.takeVector();
}

// E#197702 Add support for other loop operations like scf.forall
bool OpChainAnalysis::generateValueMap(llvm::ArrayRef<mlir::Value> blockOperands, ValueRangeMap& valueMap) {
    auto getLoopIVRange = [&](mlir::Value blockOperand) -> SmallVector<int64_t> {
        auto dimInductionArg = mlir::dyn_cast<mlir::BlockArgument>(blockOperand);
        if (dimInductionArg == nullptr) {
            return {};
        }
        if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(dimInductionArg.getOwner()->getParentOp())) {
            SmallVector<int64_t> vals;
            auto [low, high, step] = getLoopBoundsAndStep(forOp, &valueMap);
            for (auto i = low; i < high; i += step) {
                vals.push_back(i);
            }
            return vals;
        }
        if (auto forallOp = mlir::dyn_cast<mlir::scf::ForallOp>(dimInductionArg.getOwner()->getParentOp())) {
            // A potentially more performant way is to pass all blockOperands at least in the forall case, to avoid
            // repeated processing and bounds calculation
            return getForallInductionDimRange(forallOp, dimInductionArg, valueMap);
        }
        return {};
    };

    for (auto blockOperand : blockOperands) {
        if (valueMap.find(blockOperand) != valueMap.end()) {
            continue;
        }

        auto initValues = getLoopIVRange(blockOperand);
        if (initValues.empty()) {
            _log.trace("Failed to initialize values for block operand from ForOp");
            return false;
        }

        valueMap[blockOperand] = std::move(initValues);
    }

    return true;
}

std::optional<SmallVector<int64_t>> OpChainAnalysis::processCallChain(mlir::Value val, ValueRangeMap& valueMap,
                                                                      OpChainAnalysis::MODE mode) {
    // Propagate single-value entries from ValueRangeMap to the scalar operand map
    auto seedScalarMap = [&](llvm::DenseMap<mlir::Value, int64_t>& localOperandMap) {
        for (const auto& [seedValue, seedRange] : valueMap) {
            if (seedRange.size() == 1) {
                localOperandMap.try_emplace(seedValue, seedRange.front());
            }
        }
    };

    llvm::SmallSetVector<mlir::Value, DEFAULT_ARG_SET_SIZE> blockOperands;
    traverseAndGetBlockArgs(val, blockOperands);

    if (blockOperands.empty()) {
        _log.trace("Block operands empty, try evaluating op chain directly");
        auto callChain = collectParentOpsChain(val);
        llvm::DenseMap<mlir::Value, int64_t> localOperandMap;
        seedScalarMap(localOperandMap);
        if (!evaluateOpChain(callChain, localOperandMap)) {
            _log.trace("Failed to evaluate op chain without block operands");
            return std::nullopt;
        }
        auto it = localOperandMap.find(val);
        if (it == localOperandMap.end()) {
            _log.trace("Value not found after evaluating op chain");
            return std::nullopt;
        }
        SmallVector<int64_t> res{it->second};
        return res;
    }

    auto orderedBlockOperands = reorderBlocksArgs(blockOperands);
    bool allBlockOperandsHaveValues =
            std::all_of(orderedBlockOperands.begin(), orderedBlockOperands.end(), [&](mlir::Value operand) {
                return valueMap.find(operand) != valueMap.end();
            });
    if (!allBlockOperandsHaveValues && !generateValueMap(orderedBlockOperands, valueMap)) {
        _log.trace("Failed to generate value map for block operands");
        return std::nullopt;
    }

    SmallVector<const SmallVector<int64_t>*, DEFAULT_ARG_SET_SIZE> ranges;
    std::vector<int64_t> sizes;
    size_t totalCombinations = 1;
    for (mlir::Value operand : orderedBlockOperands) {
        auto valueIterator = valueMap.find(operand);
        if (valueIterator == valueMap.end() || valueIterator->second.empty()) {
            return std::nullopt;
        }
        sizes.push_back(valueIterator->second.size());
        ranges.push_back(&valueIterator->second);
        totalCombinations *= valueIterator->second.size();
    }

    auto decodeIndices = [&](int combo, mlir::ArrayRef<int64_t> sizes, SmallVector<int64_t>& indices) {
        VPUX_THROW_UNLESS(indices.size() == sizes.size(), "Mismatch between indices and sizes in decodeIndices");
        int divisor = 1;
        for (size_t i = 0; i < sizes.size(); ++i) {
            VPUX_THROW_WHEN(sizes[i] == 0, "Size cannot be zero for index {0}", i);
            indices[i] = (combo / divisor) % sizes[i];
            divisor *= sizes[i];
        }
    };

    auto parentOpsChain = collectParentOpsChain(val);

    SmallVector<int64_t> resultRange;
    SmallVector<int64_t> indices(ranges.size());
    for (size_t i = 0; i < totalCombinations; ++i) {
        decodeIndices(static_cast<int>(i), sizes, indices);

        llvm::DenseMap<mlir::Value, int64_t> localOperandMap;
        seedScalarMap(localOperandMap);
        for (size_t j = 0; j < orderedBlockOperands.size(); ++j) {
            localOperandMap[orderedBlockOperands[j]] = (*ranges[j])[indices[j]];
        }

        if (!evaluateOpChain(parentOpsChain, localOperandMap)) {
            return std::nullopt;
        }

        auto resultIterator = localOperandMap.find(val);
        if (resultIterator != localOperandMap.end()) {
            resultRange.push_back(resultIterator->second);
        } else {
            return std::nullopt;
        }
    }

    if (resultRange.empty()) {
        return std::nullopt;
    }

    if (mode == MODE::MAX_VALUE) {
        return SmallVector<int64_t>({*llvm::max_element(resultRange)});
    }

    return resultRange;
}

std::optional<SmallVector<int64_t>> OpChainAnalysis::getOpFoldResultValue(mlir::OpFoldResult val,
                                                                          ValueRangeMap& valueMap,
                                                                          OpChainAnalysis::MODE mode) {
    auto integerValue = mlir::getConstantIntValue(val);
    if (integerValue.has_value()) {
        return SmallVector<int64_t>{integerValue.value()};
    }

    if (auto value = mlir::dyn_cast<mlir::Value>(val)) {
        if (auto dimOp = mlir::dyn_cast_or_null<mlir::tensor::DimOp>(value.getDefiningOp())) {
            auto dimValue = vpux::VPU::getIntValueFromDimOp(dimOp, _log);
            if (dimValue.has_value()) {
                return SmallVector<int64_t>{dimValue.value()};
            } else {
                return std::nullopt;
            }
        } else {
            return processCallChain(value, valueMap, mode);
        }
    }

    return std::nullopt;
}

SmallVector<mlir::Operation*> collectOpsInTopologicalOrder(
        llvm::ArrayRef<mlir::Operation*> startNodes,
        llvm::function_ref<llvm::SmallSetVector<mlir::Operation*, DEFAULT_ARG_SET_SIZE>(mlir::Operation*)> getNeighbors,
        llvm::function_ref<bool(mlir::Operation*)> stopCheckFn) {
    SmallVector<mlir::Operation*> sortedOps;
    iterativeDfs(
            startNodes, getNeighbors,
            [&](mlir::Operation* op) {
                sortedOps.push_back(op);
            },
            stopCheckFn);

    return sortedOps;
}

void AnalysisContext::generateIterationPoints() {
    if (_inputRanges.empty()) {
        return;
    }

    // Calculate total number of combinations
    size_t totalCombinations =
            std::accumulate(_inputRanges.begin(), _inputRanges.end(), size_t{1}, [](size_t product, const auto& range) {
                return product * range.values.size();
            });

    _points.reserve(totalCombinations);

    // Generate Cartesian product
    // _inputRanges is ordered from slowest (outermost) to fastest (innermost)
    // We iterate in reverse order so the last dimension changes fastest
    for (size_t combo = 0; combo < totalCombinations; ++combo) {
        SmallVector<int64_t> point;
        point.resize(_inputRanges.size());

        size_t divisor = 1;
        // Iterate backwards: last dimension (fastest) changes most frequently
        for (size_t dimIdx = _inputRanges.size(); dimIdx > 0; --dimIdx) {
            size_t actualDimIdx = dimIdx - 1;
            size_t valueIdx = (combo / divisor) % _inputRanges[actualDimIdx].values.size();
            point[actualDimIdx] = _inputRanges[actualDimIdx].values[valueIdx];
            divisor *= _inputRanges[actualDimIdx].values.size();
        }

        _points.push_back(std::move(point));
    }
}

bool AnalysisContext::arePointsContiguous(ArrayRef<int64_t> point1, ArrayRef<int64_t> point2) {
    if (point1.size() != point2.size()) {
        return false;
    }

    if (point1.size() == 0) {
        return false;
    }

    if (point1.size() != _strides.size()) {
        return false;
    }

    bool seenDiffer = false;
    for (size_t i = 0; i < point1.size(); ++i) {
        int64_t diff = std::abs(point1[i] - point2[i]);
        if (diff > _strides[i]) {
            return false;
        }

        if (diff == _strides[i]) {
            if (seenDiffer) {
                return false;
            }
            seenDiffer = true;
        }
    }

    return seenDiffer;
}

}  // namespace vpux::VPU
