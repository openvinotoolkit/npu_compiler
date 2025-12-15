//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/function_outlining_splitter.hpp"
#include "vpux/utils/core/dense_map.hpp"

#include <queue>

using namespace vpux;

namespace {

class NaiveSplitter {
public:
    NaiveSplitter(size_t numSplits, Logger log): _numSplits(numSplits), _log(log) {
    }
    SmallVector<OutliningInstance> getOutliningInstances(mlir::func::FuncOp mainFunction);

private:
    SmallVector<size_t> findSplittingPoints(mlir::func::FuncOp mainFunction);
    DenseMap<mlir::Operation*, size_t> getOpDistances(mlir::func::FuncOp mainFunction, size_t& maxDistance);
    std::optional<size_t> findSplitIdx(mlir::Operation* op);
    void addParentOpsToSplit(mlir::Operation* op, IRSlice& currentSplit,
                             std::set<mlir::Operation*>& currentSplitCoveredOps, size_t currentSplitIdx);
    void addOperandToSplitIfInput(mlir::Value operand, SmallVector<mlir::Value>& currentSplitInputs,
                                  size_t currentSplitIdx);
    void addResultToSplitIfOutput(mlir::Value result, SmallVector<mlir::Value>& currentSplitOutputs,
                                  size_t currentSplitIdx);

private:
    size_t _numSplits;
    Logger _log;

    DenseMap<mlir::Operation*, size_t> _distances;
    SmallVector<size_t> _splittingPoints;
};

bool isPreprocessingOpAndHasBlockArgParent(mlir::Operation* op) {
    if (!mlir::isa_and_nonnull<IE::ConvertOp, IE::TransposeOp, IE::FakeQuantizeOp, IE::FakeConvertOp>(op)) {
        return false;
    }
    if (mlir::isa<mlir::BlockArgument>(op->getOperand(0))) {
        return true;
    }
    return isPreprocessingOpAndHasBlockArgParent(op->getOperand(0).getDefiningOp());
}

/**
 * Iterate over operations and identify which IR split they belong to.
 * To achieve this, every operation is iterated over and placed in a split depending on the distance of the operation
 * from the start of the model. The distance is computed as a prerequisite, only on operations found on the activation
 * path (e.g. not constants). The operations are visited in the order found in the IR, while skipping operations that do
 * not have a distance set. As such, it is possible to reach an operation on the activation path which does not have all
 * of its operands visited; e.g.:
 *    Op1    <- distance 0
 *    |
 *    |  cst
 *    |  /
 *    Op2    <- distance 1
 * For this example, when `Op2` is visited, the constant should be first placed in the current split before the actual
 * operation. It is also possible to reach an operation which has dependencies in another split:
 *    Op1
 *    |  \
 *    |   Op2
 *    |   /
 *  -----/---> splitting point
 *    | /
 *    Op3
 * In this example, when `Op3` is visited, it depends on both `Op1` and `Op2` which are found in a previous split. The
 * values produced by these two operations should be set as input arguments to the split in which `Op3` is found.
 * In summary, to satisfy such dependencies, the following behavior takes place:
 *   - if the parent operation is found on the activation path, there are two possibilities:
 *      - if it is placed in a previous split, the operand is marked as an input argument to the current split
 *      - if it is placed in the current split, skip it as the operation has been already handled
 *   - if the parent operation is not found on the activation path (e.g. constant), place it and any intermediate
 *   operation in the current split as well
 */
SmallVector<OutliningInstance> NaiveSplitter::getOutliningInstances(mlir::func::FuncOp mainFunction) {
    _splittingPoints = findSplittingPoints(mainFunction);
    if (_splittingPoints.empty() || _splittingPoints.size() + 1 != _numSplits) {
        return {};
    }

    SmallVector<OutliningInstance> splits(_numSplits, OutliningInstance(1));
    SmallVector<std::set<mlir::Operation*>> coveredOps(_numSplits);

    mainFunction.walk([&](mlir::Operation* op) {
        const auto maybeSplitIdx = findSplitIdx(op);
        if (!maybeSplitIdx.has_value()) {
            return;
        }
        const auto splitIdx = maybeSplitIdx.value();

        auto& currentSplit = splits[splitIdx].front();
        auto& currentSplitCoveredOps = coveredOps[splitIdx];
        if (currentSplitCoveredOps.find(op) == currentSplitCoveredOps.end()) {
            for (auto operand : op->getOperands()) {
                addOperandToSplitIfInput(operand, currentSplit.inputs, splitIdx);
                if (auto parentOp = operand.getDefiningOp()) {
                    if (currentSplitCoveredOps.find(parentOp) != currentSplitCoveredOps.end()) {
                        continue;
                    }
                    addParentOpsToSplit(parentOp, currentSplit, currentSplitCoveredOps, splitIdx);
                }
            }
            currentSplit.operations.push_back(op);
            currentSplitCoveredOps.insert(op);
        }

        for (auto result : op->getResults()) {
            addResultToSplitIfOutput(result, currentSplit.outputs, splitIdx);
        }
    });

    return splits;
}

// Finds the splitting points of the given IR
// This is achieved by computing the distance of each operation from the start of the network, followed by finding the
// spitting points based on the maximum distance and previously configured number of splits
SmallVector<size_t> NaiveSplitter::findSplittingPoints(mlir::func::FuncOp mainFunction) {
    size_t maxDistance = 0;
    _distances = getOpDistances(mainFunction, maxDistance);
    if (maxDistance < _numSplits) {
        _log.trace("Furthest operation is found at distance {0} from the start. Should be at least {1} for outlining",
                   maxDistance, _numSplits);
        return {};
    }

    SmallVector<size_t> splittingPoints;
    size_t start = 0;
    const auto splitSize = maxDistance / _numSplits;
    for (size_t split = 0; split < _numSplits - 1; ++split) {
        splittingPoints.push_back(start + splitSize);
        start += splitSize;
    }
    return splittingPoints;
}

// Find the distance of each operation from the start of the network, as well as the maximum distance
// Only operations on the activations path will have distances set (e.g. constants and intermediate operations until
// consumer will not have a distance set)
DenseMap<mlir::Operation*, size_t> NaiveSplitter::getOpDistances(mlir::func::FuncOp mainFunction, size_t& maxDistance) {
    DenseMap<mlir::Operation*, size_t> distances;
    maxDistance = 0;

    mlir::DenseSet<mlir::Operation*> queuedOps;
    std::queue<mlir::Operation*> ops;
    const auto pushOp = [&](mlir::Operation* op) {
        if (!queuedOps.contains(op)) {
            ops.push(op);
            queuedOps.insert(op);
        }
    };

    const auto popOp = [&]() {
        auto op = ops.front();
        ops.pop();
        queuedOps.erase(op);
        return op;
    };

    for (auto blockArg : mainFunction.getArguments()) {
        for (auto userOp : blockArg.getUsers()) {
            if (mlir::isa<mlir::func::ReturnOp>(userOp)) {
                continue;
            }
            pushOp(userOp);
            distances[userOp] = 0;
        }
    }

    while (!ops.empty()) {
        auto op = popOp();

        for (auto result : op->getResults()) {
            for (auto userOp : result.getUsers()) {
                if (mlir::isa<mlir::func::ReturnOp>(userOp)) {
                    continue;
                }

                distances[userOp] = std::max(distances[userOp], distances[op] + 1);
                maxDistance = std::max(maxDistance, distances[userOp]);

                pushOp(userOp);
            }
        }
    }

    return distances;
}

// Finds which IR split the given operation belongs to, depending on the distance from the start of the model
std::optional<size_t> NaiveSplitter::findSplitIdx(mlir::Operation* op) {
    auto distanceIt = _distances.find(op);
    if (distanceIt == _distances.end()) {
        return std::nullopt;
    }

    for (size_t i = 0; i < _splittingPoints.size(); ++i) {
        if (distanceIt->second <= _splittingPoints[i]) {
            return i;
        }
    }
    return _splittingPoints.size();
}

// Traverses recursively through the parent operations until an operation previously covered or found in another IR
// split is found. The intermediate operations are added to the current split
void NaiveSplitter::addParentOpsToSplit(mlir::Operation* op, IRSlice& currentSplit,
                                        std::set<mlir::Operation*>& currentSplitCoveredOps, size_t currentSplitIdx) {
    if (op == nullptr) {
        return;
    }
    if (currentSplitCoveredOps.find(op) != currentSplitCoveredOps.end()) {
        return;
    }
    const auto splitIdx = findSplitIdx(op);
    if (!isPreprocessingOpAndHasBlockArgParent(op) && splitIdx.has_value() && splitIdx.value() != currentSplitIdx) {
        return;
    }
    for (auto operand : op->getOperands()) {
        addOperandToSplitIfInput(operand, currentSplit.inputs, currentSplitIdx);
        addParentOpsToSplit(operand.getDefiningOp(), currentSplit, currentSplitCoveredOps, currentSplitIdx);
    }

    currentSplit.operations.push_back(op);
    currentSplitCoveredOps.insert(op);
};

// Add the operand to the list of the input values for the current split, in case it is a block argument or the producer
// operation is found in another split
void NaiveSplitter::addOperandToSplitIfInput(mlir::Value operand, SmallVector<mlir::Value>& currentSplitInputs,
                                             size_t currentSplitIdx) {
    VPUX_THROW_WHEN(operand == nullptr, "Invalid operand");
    if (llvm::find(currentSplitInputs, operand) != currentSplitInputs.end()) {
        return;
    }
    if (mlir::isa<mlir::BlockArgument>(operand)) {
        currentSplitInputs.push_back(operand);
        return;
    }
    auto parentOp = operand.getDefiningOp();
    if (parentOp == nullptr) {
        return;
    }
    const auto parentSplitIdx = findSplitIdx(parentOp);
    if (!parentSplitIdx.has_value() || parentSplitIdx.value() == currentSplitIdx) {
        return;
    }
    if (isPreprocessingOpAndHasBlockArgParent(parentOp)) {
        return;
    }
    currentSplitInputs.push_back(operand);
};

// Add the result to the list of the output values for the current split, in case one of its users is a return
// operation or found in another split
void NaiveSplitter::addResultToSplitIfOutput(mlir::Value result, SmallVector<mlir::Value>& currentSplitOutputs,
                                             size_t currentSplitIdx) {
    VPUX_THROW_WHEN(result == nullptr, "Invalid result");
    if (llvm::find(currentSplitOutputs, result) != currentSplitOutputs.end()) {
        return;
    }
    bool isOutput = llvm::any_of(result.getUsers(), [&](mlir::Operation* userOp) {
        if (mlir::isa<mlir::func::ReturnOp>(userOp)) {
            return true;
        }
        const auto userSplitIdx = findSplitIdx(userOp);
        if (userSplitIdx.has_value() && userSplitIdx.value() != currentSplitIdx) {
            return true;
        }
        return false;
    });
    if (isOutput && !isPreprocessingOpAndHasBlockArgParent(result.getDefiningOp())) {
        currentSplitOutputs.push_back(result);
    }
};

}  // namespace

namespace vpux {
namespace IE {

FunctionOutlinerNaive::FunctionOutlinerNaive(size_t numSplits, Logger log): _numSplits(numSplits), _log(log) {
    _log.setName("function-outliner-naive");
}

SmallVector<OutliningInstance> FunctionOutlinerNaive::getOutliningTargets(mlir::func::FuncOp mainFunction) {
    _log.debug("Searching for outlining targets with a naive split strategy");

    if (_numSplits <= 1) {
        _log.debug("Number of splits {0} should be larger than 1", _numSplits);
        return {};
    }

    NaiveSplitter naiveSplitter(_numSplits, _log);
    const auto outliningInstances = naiveSplitter.getOutliningInstances(mainFunction);

    printOutliningInstances(outliningInstances, _log);

    const auto hasReadValueOp = [](std::vector<mlir::Operation*> operations) -> bool {
        for (const auto& op : operations) {
            if (mlir::isa_and_nonnull<IE::ReadValueOp>(op)) {
                return true;
            }
        }
        return false;
    };
    const auto hasAssignOp = [](std::vector<mlir::Operation*> operations) -> bool {
        for (const auto& op : operations) {
            if (mlir::isa_and_nonnull<IE::AssignOp>(op)) {
                return true;
            }
        }
        return false;
    };

    // Avoid Splitting function when there is a slice with no outputs
    auto hasEmptySlice = llvm::any_of(outliningInstances, [&](const OutliningInstance& outlineInstance) -> bool {
        for (auto& slice : outlineInstance) {
            if (slice.operations.empty() || (slice.inputs.empty() && !hasReadValueOp(slice.operations)) ||
                (slice.outputs.empty() && !hasAssignOp(slice.operations))) {
                return true;
            }
        }
        return false;
    });

    if (hasEmptySlice) {
        _log.debug("At least one instance has no outputs values, which results in an empty function");
        return {};
    }

    return outliningInstances;
}

}  // namespace IE
}  // namespace vpux
