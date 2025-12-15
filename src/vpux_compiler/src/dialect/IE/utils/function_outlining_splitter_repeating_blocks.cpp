//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/hash.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/dense_map.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <deque>

using namespace vpux;

namespace {

//
// RepeatingBlocksIdentifier
//

class RepeatingBlocksIdentifier {
public:
    RepeatingBlocksIdentifier(size_t minOpsInBlock, size_t maxNumIterations, bool separateFunctions,
                              bool weightsAsInputs, const Logger& log)
            : _minOpsInBlock(minOpsInBlock),
              _maxNumIterations(maxNumIterations),
              _separateFunctions(separateFunctions),
              _weightsAsInputs(weightsAsInputs),
              _log(log) {
        // The CLI argument parser already ensures that these are mutually exclusive. Just to be safe, we prohibit the
        // construction of an invalid instance.
        VPUX_THROW_WHEN(separateFunctions && weightsAsInputs,
                        "'Separate functions' and 'weights as inputs' cannot be enabled at the same time");
    }
    SmallVector<OutliningInstance> getOutliningInstances(mlir::func::FuncOp mainFunction);

private:
    using BlockId = size_t;
    using InstanceId = size_t;
    using BlockPair = std::pair<BlockId, BlockId>;
    struct InstancePair {
        InstanceId parentId;
        InstanceId childId;
        llvm::hash_code parentOpHash;
        llvm::hash_code childOpHash;
        size_t parentResultIdx;
        size_t childOperandIdx;

        struct HashFunction {
            size_t operator()(const InstancePair& i) const {
                const auto h1 = llvm::hash_value(i.parentId);
                const auto h2 = llvm::hash_value(i.childId);
                const auto h3 = llvm::hash_value(i.parentResultIdx);
                const auto h4 = llvm::hash_value(i.childOperandIdx);
                return llvm::hash_combine(h1, h2, h3, h4, i.parentOpHash, i.childOpHash);
            }
        };
        bool operator==(const InstancePair& other) const {
            return parentId == other.parentId && childId == other.childId && parentOpHash == other.parentOpHash &&
                   childOpHash == other.childOpHash && parentResultIdx == other.parentResultIdx &&
                   childOperandIdx == other.childOperandIdx;
        }
    };

    struct MergeCandidateInfo {
        std::vector<InstancePair> instances;
        std::unordered_set<InstancePair, InstancePair::HashFunction> conflicts;
    };

    struct OpInstance {
        InstanceId id;
        std::set<mlir::Operation*> operations;

        OpInstance(InstanceId id, const std::set<mlir::Operation*>& operations): id(id), operations(operations) {
        }
    };
    using OpInstances = std::vector<OpInstance>;

    using OpValuePair = std::pair<llvm::hash_code, size_t>;

private:
    void identifyUniqueOperations(mlir::func::FuncOp mainFunction);
    bool tryMergeAdjacentBlocks();
    SmallVector<size_t> findMergeCandidateIdx(ArrayRef<MergeCandidateInfo> mergeCandidates,
                                              const InstancePair& currentInstancePair);
    std::optional<MergeCandidateInfo> chooseMergeCandidate(
            const std::unordered_map<BlockPair, std::vector<MergeCandidateInfo>>& mergeCandidates);
    bool performMerge(const MergeCandidateInfo& candidateInfo);
    void removeLeftoverBlocks();

    std::optional<InstanceId> getInstanceId(mlir::Operation* op);
    void addInputsOutputsForSeparateFunctions(IRSlice& instance, InstanceId instanceId, mlir::Operation* op);
    void addInputsOutputsForSharedFunction(IRSlice& instance, InstanceId instanceId, mlir::Operation* op,
                                           SmallVector<OpValuePair>& instanceInputs,
                                           SmallVector<OpValuePair>& instanceOutputs);
    SmallVector<OutliningInstance> prepareOutliningInstances(mlir::func::FuncOp mainFunction);
    mlir::LogicalResult validateOutliningInstances(ArrayRef<OutliningInstance> outliningInstances);

    void printBlocks(StringLiteral note);

private:
    size_t _minOpsInBlock;
    size_t _maxNumIterations;
    bool _separateFunctions;
    bool _weightsAsInputs;
    Logger _log;

    std::unordered_map<mlir::Operation*, llvm::hash_code> _opHash{};

    // Op -> ID of the instance it belongs to
    std::unordered_map<mlir::Operation*, InstanceId> _opInstance{};

    // ID of instance -> ID of the block it belongs to
    std::unordered_map<InstanceId, BlockId> _instanceBlock{};

    // The blocks of repeating operations, where the key represents the ID of the block and the value contains all of
    // the instances of operations that belong to the block. For example, if the IR contains the following operations:
    //   A -> B -> C -> A -> B
    // the map could contain: block_id -> {{A, B}, {A, B}}.
    std::unordered_map<BlockId, OpInstances> _blocks{};

    InstanceId _lastInstanceId = 0;
    BlockId _lastBlockId = 0;
};

/**
 * @brief Identify operations that repeat in the IR and place them in a unique block. The identification is done based
 * on the operation type, attributes, operand and result types. After this step, each block should contain a single
 * operation that appears multiple times in the IR (i.e. each block should have multiple instances in the IR).
 */
void RepeatingBlocksIdentifier::identifyUniqueOperations(mlir::func::FuncOp mainFunction) {
    _log.trace("Identifying unique operations");

    // The mapping from each operation hash to the id of the repeating block it belongs to
    DenseMap<llvm::hash_code, BlockId> uniqueOpBlock;

    mainFunction.walk([&](mlir::Operation* op) {
        // Constants are skipped as they should not represent operations which could differentiate between repeating
        // blocks
        if (mlir::isa<Const::DeclareOp>(op)) {
            return;
        }

        const auto hash = hashOperation(op);
        _opHash[op] = hash;
        if (!uniqueOpBlock.contains(hash)) {
            uniqueOpBlock[hash] = _lastBlockId++;
        }

        auto instanceId = _lastInstanceId++;
        _opInstance[op] = instanceId;
        _instanceBlock[instanceId] = uniqueOpBlock[hash];
        _blocks[uniqueOpBlock[hash]].push_back({instanceId, {op}});
    });

    // Remove blocks that have non-repeating operations
    for (auto& block : llvm::make_early_inc_range(_blocks)) {
        auto& instances = block.second;
        if (instances.size() >= 2) {
            continue;
        }
        for (auto& instance : instances) {
            for (auto op : instance.operations) {
                _opInstance.erase(op);
            }
            _instanceBlock.erase(instance.id);
        }
        _blocks.erase(block.first);
    }

    printBlocks("after identifyUniqueOperations");
}

/**
 * @brief Try to merge adjacent blocks of operations
 * @details Implementation overview:
 *  - Identify an instance of a block X which is adjacent with an instance of block Y (this can be of the same block or
 *    a different block). The adjacency is determined by the direct users of the operations from the current instance.
 *  - Add the pair of instances as a merge candidate for blocks X-Y
 *  - Check if the merge candidate has a conflict with any other candidates and update the conflicts for these
 *    candidates
 *  - After all merge candidates have been found, select the candidate which would result in the largest number of
 *    instances. If there are multiple such candidates, select the one which has the smallest amount of conflicts
 *  - If a candidate is found, perform the merge
 * @return true if a merge has been done
 */
bool RepeatingBlocksIdentifier::tryMergeAdjacentBlocks() {
    // For each pair of blocks, stores the lists of instances which can be merged
    // Note: there are multiple lists of instances per pair of blocks to cover cases where one instance conflicts
    // another instance in the same pair of blocks
    std::unordered_map<BlockPair, std::vector<MergeCandidateInfo>> blockPairsAndMergeCandidates;

    for (auto& block : _blocks) {
        const auto blockId = block.first;
        const auto& opInstances = block.second;
        for (auto& instance : opInstances) {
            const auto instanceId = instance.id;
            for (auto op : instance.operations) {
                const auto opHash = _opHash[op];
                for (auto result : op->getResults()) {
                    const auto resultIdx = result.getResultNumber();
                    for (auto& use : result.getUses()) {
                        auto userOp = use.getOwner();
                        const auto userOperandIdx = use.getOperandNumber();
                        const auto userInstanceIt = _opInstance.find(userOp);
                        if (userInstanceIt == _opInstance.end()) {
                            continue;
                        }
                        const auto userInstanceId = userInstanceIt->second;
                        // Skip operations which are part of the same instance
                        if (instanceId == userInstanceId) {
                            continue;
                        }

                        const auto userBlockIt = _instanceBlock.find(userInstanceId);
                        VPUX_THROW_WHEN(userBlockIt == _instanceBlock.end(), "Missing block id for instance {0}",
                                        userInstanceId);
                        const auto userBlockId = userBlockIt->second;

                        // Prevent instances from the same block from being merged. This is done to prevent LLMs from
                        // growing the identified blocks too large, as they often have consecutive identical instances
                        // in their topology
                        if (blockId == userBlockId) {
                            continue;
                        }

                        const auto blockPair = std::make_pair(blockId, userBlockId);

                        const auto userOpHash = _opHash[userOp];
                        const auto currentInstancePair =
                                InstancePair{instanceId, userInstanceId, opHash, userOpHash, resultIdx, userOperandIdx};

                        auto& mergeCandidates = blockPairsAndMergeCandidates[blockPair];

                        const auto candidateIndices = findMergeCandidateIdx(mergeCandidates, currentInstancePair);

                        const auto addCandidate = [&](size_t candidateIdx) {
                            auto candidateIt = mergeCandidates.begin() + candidateIdx;

                            // Update the list of conflicts for each relevant pair of instances
                            for (auto& otherCandidate : blockPairsAndMergeCandidates) {
                                auto& candidateBlockPair = otherCandidate.first;
                                if (candidateBlockPair.first != blockId && candidateBlockPair.first != userBlockId &&
                                    candidateBlockPair.second != blockId && candidateBlockPair.second != userBlockId) {
                                    continue;
                                }
                                for (auto& otherCandidateInfo : otherCandidate.second) {
                                    for (auto& instancePair : otherCandidateInfo.instances) {
                                        if (instancePair.parentId != instanceId &&
                                            instancePair.parentId != userInstanceId &&
                                            instancePair.childId != instanceId &&
                                            instancePair.childId != userInstanceId) {
                                            continue;
                                        }
                                        otherCandidateInfo.conflicts.insert(currentInstancePair);
                                        candidateIt->conflicts.insert(instancePair);
                                    }
                                }
                            }
                            // Add instance pair to the list of candidates
                            candidateIt->instances.push_back(currentInstancePair);
                        };

                        if (candidateIndices.empty()) {
                            mergeCandidates.emplace_back();
                            addCandidate(mergeCandidates.size() - 1);
                        } else {
                            for (auto idx : candidateIndices) {
                                addCandidate(idx);
                            }
                        }
                    }
                }
            }
        }
    }

    if (blockPairsAndMergeCandidates.empty()) {
        return false;
    }

    const auto candidateInfo = chooseMergeCandidate(blockPairsAndMergeCandidates);
    if (!candidateInfo.has_value()) {
        return false;
    }
    auto status = performMerge(candidateInfo.value());
    printBlocks("after tryMergeAdjacentBlocks");
    return status;
}

/**
 * @brief Find which list of instances the current candidate should be added to. In case no list is found, nothing is
 * returned
 */
SmallVector<size_t> RepeatingBlocksIdentifier::findMergeCandidateIdx(ArrayRef<MergeCandidateInfo> mergeCandidates,
                                                                     const InstancePair& currentInstancePair) {
    const auto parentId = currentInstancePair.parentId;
    const auto childId = currentInstancePair.childId;

    SmallVector<size_t> candidateIdx;
    for (size_t idx = 0; idx < mergeCandidates.size(); ++idx) {
        const auto& firstInstance = mergeCandidates[idx].instances.front();
        // If either the hashes or result/operand indices are different, skip the current list of
        // instances as it is not the same as the current candidate
        if (currentInstancePair.parentOpHash != firstInstance.parentOpHash ||
            currentInstancePair.childOpHash != firstInstance.childOpHash ||
            currentInstancePair.parentResultIdx != firstInstance.parentResultIdx ||
            currentInstancePair.childOperandIdx != firstInstance.childOperandIdx) {
            continue;
        }

        // Find the first list of instances that has no conflicts with the current candidate
        auto hasConflict = llvm::any_of(mergeCandidates[idx].instances, [&](const InstancePair& i) {
            return i.parentId == parentId || i.parentId == childId || i.childId == parentId || i.childId == childId;
        });
        if (!hasConflict) {
            candidateIdx.push_back(idx);
        }
    }
    return candidateIdx;
}

/**
 * @brief Select the candidate which would result in the largest number of instances. If there are multiple such
 * candidates, select the one which has the smallest amount of conflicts
 * TODO E#111873: Consider returning multiple candidates, in case there are no conflicts between the selections
 * @return the merge candidate if one is found, nothing otherwise
 */
std::optional<RepeatingBlocksIdentifier::MergeCandidateInfo> RepeatingBlocksIdentifier::chooseMergeCandidate(
        const std::unordered_map<BlockPair, std::vector<MergeCandidateInfo>>& blockPairsAndMergeCandidates) {
    // Find the merge candidate that has the larges number of instances
    std::vector<std::vector<MergeCandidateInfo>::const_iterator> largestCandidates;
    size_t largestNumInstances = 0;
    for (auto& candidateInfos : blockPairsAndMergeCandidates) {
        for (auto it = candidateInfos.second.begin(); it != candidateInfos.second.end(); ++it) {
            const auto candidateNumInstances = it->instances.size();
            if (candidateNumInstances < largestNumInstances) {
                continue;
            } else if (candidateNumInstances > largestNumInstances) {
                largestCandidates.clear();
                largestNumInstances = candidateNumInstances;
            }

            largestCandidates.push_back(it);
        }
    }

    if (largestNumInstances == 1) {
        return std::nullopt;
    }

    // In case there are more than one such candidates, choose the one that has the smallest number of conflicts
    // If all of the largest candidates have the same number of conflicts, the first one is chosen
    auto candidateIt = largestCandidates.front();
    size_t smallestNumConflicts = candidateIt->conflicts.size();
    for (auto it : largestCandidates) {
        const auto candidateNumConflicts = it->conflicts.size();
        if (candidateNumConflicts < smallestNumConflicts) {
            candidateIt = it;
            smallestNumConflicts = candidateNumConflicts;
        }
    }

    return *candidateIt;
}

/**
 * @brief Perform the merge for the given candidate by creating a new block which contains the merged instances
 * @return true if the merge has been done successfully
 */
bool RepeatingBlocksIdentifier::performMerge(const MergeCandidateInfo& candidateInfo) {
    if (candidateInfo.instances.empty()) {
        return false;
    }

    const auto parentBlockId = _instanceBlock[candidateInfo.instances.front().parentId];
    const auto childBlockId = _instanceBlock[candidateInfo.instances.front().childId];
    const auto newBlockId = _lastBlockId++;
    for (auto& instance : candidateInfo.instances) {
        const auto newInstanceId = _lastInstanceId++;

        // Find the instances that are part of the merge from the parent and child blocks, respectively
        const auto findOrigInstance = [&](const size_t blockId, const size_t instanceId) {
            auto oldInstanceIt = llvm::find_if(_blocks[blockId], [&](const OpInstance& blockInstance) {
                return blockInstance.id == instanceId;
            });
            VPUX_THROW_WHEN(oldInstanceIt == _blocks[blockId].end(), "Could not find instance {0} in block {1}",
                            instanceId, blockId);
            return oldInstanceIt;
        };
        auto origInstanceParentIt = findOrigInstance(parentBlockId, instance.parentId);
        auto origInstanceChildIt = findOrigInstance(childBlockId, instance.childId);
        auto& origInstanceParent = *origInstanceParentIt;
        auto& origInstanceChild = *origInstanceChildIt;

        // Update the hashes of the operations that are being merged, so that the hash of an operation depends on its
        // merge history. This is important to be able to differentiate between identical operations that later end up
        // in the same instance, so that merge candidates from the two operations are not considered the same.
        // A unique seed is also used to distinguish between the operations that come from the parent and child
        // instances, for the case where two instances from the same block are merged. This is currently unnecessary
        // since such merges are not allowed, but they might be allowed in the future
        for (auto op : origInstanceParent.operations) {
            _opHash[op] = llvm::hash_combine(_opHash[op], newBlockId, /*parentSeed=*/1);
        }
        for (auto op : origInstanceChild.operations) {
            _opHash[op] = llvm::hash_combine(_opHash[op], newBlockId, /*childSeed=*/2);
        }

        // Move all operations into a new instance and add the instance to a new block
        origInstanceParent.operations.merge(origInstanceChild.operations);
        OpInstance newInstance(newInstanceId, std::move(origInstanceParent.operations));
        _blocks[newBlockId].push_back(newInstance);

        // Mark the instance as being part of the new block and erase the blocks ids for the old instances
        _instanceBlock[newInstanceId] = newBlockId;
        _instanceBlock.erase(instance.parentId);
        _instanceBlock.erase(instance.childId);

        // Mark the operations as being part of the new instance
        for (auto op : newInstance.operations) {
            _opInstance[op] = newInstanceId;
        }

        // Remove the old instances from the previous blocks. In case the block is empty, remove it as well
        _blocks[parentBlockId].erase(origInstanceParentIt);
        if (parentBlockId == childBlockId) {
            // The erase method can invalidate old iterators, so the second iterator has to be retrieved again in case
            // both belong to the same block
            origInstanceChildIt = findOrigInstance(childBlockId, instance.childId);
        }
        _blocks[childBlockId].erase(origInstanceChildIt);
        if (_blocks[childBlockId].empty()) {
            _blocks.erase(childBlockId);
        }
        if (_blocks[parentBlockId].empty()) {
            _blocks.erase(parentBlockId);
        }
    }

    return true;
}

/**
 * @brief Remove blocks which have only one instance in the IR or which have too few operations
 */
void RepeatingBlocksIdentifier::removeLeftoverBlocks() {
    for (auto& block : llvm::make_early_inc_range(_blocks)) {
        const auto& instances = block.second;
        if (instances.size() >= 2 && instances.front().operations.size() >= _minOpsInBlock) {
            continue;
        }
        for (auto& opInstance : block.second) {
            for (auto op : opInstance.operations) {
                _opInstance.erase(op);
            }
            _instanceBlock.erase(opInstance.id);
        }
        _blocks.erase(block.first);
    }
    printBlocks("after removeLeftoverBlocks");
}

std::optional<RepeatingBlocksIdentifier::InstanceId> RepeatingBlocksIdentifier::getInstanceId(mlir::Operation* op) {
    const auto opInstanceIt = _opInstance.find(op);
    if (opInstanceIt == _opInstance.end()) {
        return std::nullopt;
    }
    return opInstanceIt->second;
}

SmallVector<mlir::Operation*> getConstantParents(mlir::Operation* op, SmallVector<mlir::Value>& blockArgs) {
    if (op == nullptr) {
        return {};
    }
    if (mlir::isa<Const::DeclareOp>(op)) {
        return SmallVector<mlir::Operation*>{op};
    }

    // Quantized weights could be represented as subgraphs, such as:
    //              Cst         Cst
    //               |           |
    // Weights -> Subtract -> Multiply -> [user]
    // These subgraphs are included into the outlined function, so that the low-precision pipeline can correctly
    // quantize the user operation
    if (mlir::isa<IE::SubtractOp, IE::MultiplyOp, IE::ConvertOp, IE::FakeQuantizeOp, IE::ReshapeOp,
                  IE::AffineReshapeOp>(op)) {
        SmallVector<mlir::Operation*> parentConstOps;
        for (auto operand : op->getOperands()) {
            const auto parentOp = operand.getDefiningOp();
            if (parentOp == nullptr) {
                blockArgs.push_back(operand);
                continue;
            }
            const auto ops = getConstantParents(parentOp, blockArgs);
            if (ops.empty()) {
                return {};
            }
            parentConstOps.append(ops.begin(), ops.end());
        }
        parentConstOps.push_back(op);
        return parentConstOps;
    }

    return {};
}

mlir::LogicalResult duplicateNeededParentOps(IRSlice& instance, mlir::Operation* parentOp) {
    // In case the parent operation is a constant (or constant subgraph), it should be placed in the current instance
    // regardless of where it was placed initially
    SmallVector<mlir::Value> blockArgs;
    const auto constParents = getConstantParents(parentOp, blockArgs);
    if (!constParents.empty()) {
        for (auto constParent : constParents) {
            if (llvm::find(instance.operations, constParent) == instance.operations.end()) {
                instance.operations.push_back(constParent);
            }
        }
        for (auto blockArg : blockArgs) {
            if (llvm::find(instance.inputs, blockArg) == instance.inputs.end()) {
                instance.inputs.push_back(blockArg);
            }
        }
        return mlir::success();
    }

    return mlir::failure();
}

void RepeatingBlocksIdentifier::addInputsOutputsForSeparateFunctions(IRSlice& instance, InstanceId instanceId,
                                                                     mlir::Operation* op) {
    // Add the dependencies of the operation to the current instance or mark them input values
    for (auto operand : op->getOperands()) {
        const bool operandAlreadyCovered = llvm::find(instance.inputs, operand) != instance.inputs.end();
        if (operandAlreadyCovered) {
            continue;
        }
        auto parentOp = operand.getDefiningOp();
        // Operand is a block argument in the original function
        if (parentOp == nullptr) {
            instance.inputs.push_back(operand);
            continue;
        }
        const auto parentInstanceId = getInstanceId(parentOp);
        bool parentOutsideInstance = !parentInstanceId.has_value() || parentInstanceId.value() != instanceId;
        if (parentOutsideInstance) {
            if (mlir::succeeded(duplicateNeededParentOps(instance, parentOp))) {
                continue;
            }
            instance.inputs.push_back(operand);
        }
    }

    instance.operations.push_back(op);

    // Mark the results of the operation as outputs if they have users outside the current instance
    for (auto result : op->getResults()) {
        // The result is already in the list of output values
        const bool resultAlreadyCovered = llvm::find(instance.outputs, result) != instance.outputs.end();
        if (resultAlreadyCovered) {
            continue;
        }
        const auto userOutsideInstance = llvm::any_of(result.getUsers(), [&](mlir::Operation* userOp) {
            const auto userInstanceId = getInstanceId(userOp);
            return !userInstanceId.has_value() || userInstanceId.value() != instanceId;
        });
        if (userOutsideInstance) {
            instance.outputs.push_back(result);
        }
    }
}  // namespace

void RepeatingBlocksIdentifier::addInputsOutputsForSharedFunction(IRSlice& instance, InstanceId instanceId,
                                                                  mlir::Operation* op,
                                                                  SmallVector<OpValuePair>& instanceInputs,
                                                                  SmallVector<OpValuePair>& instanceOutputs) {
    // Add the dependencies of the operation to the current instance or mark them input values
    // All instances of a repeating block must have the same input values, even if not all of them are connected
    // to the operations inside an instance. This is required since all instances will be represented as calls
    // to the same function. For this reason, as an initial step, all the outside input connections to the
    // instances are collected and only later the input values deduced
    for (auto& operand : op->getOpOperands()) {
        const bool operandAlreadyCovered =
                llvm::find_if(instanceInputs, [&](auto& input) {
                    return input.first == _opHash[op] && input.second == operand.getOperandNumber();
                }) != instanceInputs.end();
        if (operandAlreadyCovered) {
            continue;
        }

        auto parentOp = operand.get().getDefiningOp();
        // Operand is a block argument in the original function
        if (parentOp == nullptr) {
            instanceInputs.emplace_back(_opHash[op], operand.getOperandNumber());
            continue;
        }

        const auto maybeParentInstanceId = getInstanceId(parentOp);
        // Parent operation is not placed in the current instance
        if (!maybeParentInstanceId.has_value() || maybeParentInstanceId.value() != instanceId) {
            // An exception is the case where the parent operation is a constant; this operations should be
            // placed in the current instance regardless of where it was placed initially
            if (mlir::isa<Const::DeclareOp>(parentOp) && !_weightsAsInputs) {
                auto constAlreadyAdded = llvm::find(instance.operations, parentOp) != instance.operations.end();
                if (!constAlreadyAdded) {
                    instance.operations.push_back(parentOp);
                }
                continue;
            }
            instanceInputs.emplace_back(_opHash[op], operand.getOperandNumber());
        }
    }

    instance.operations.push_back(op);

    // Similar to the inputs, collect all the output connection types across all instances, since each instance
    // must have the same number of output values. After all connections are collected, the output values will
    // be deduced for every instance
    for (auto result : op->getResults()) {
        bool resultAlreadyCovered = llvm::find_if(instanceOutputs, [&](auto& output) {
                                        return output.first == _opHash[op] && output.second == result.getResultNumber();
                                    }) != instanceOutputs.end();
        if (resultAlreadyCovered) {
            continue;
        }

        auto anyUserOutside = llvm::any_of(result.getUsers(), [&](mlir::Operation* userOp) {
            const auto maybeUserInstanceId = getInstanceId(userOp);
            return !maybeUserInstanceId.has_value() || maybeUserInstanceId.value() != instanceId;
        });
        if (anyUserOutside) {
            instanceOutputs.emplace_back(_opHash[op], result.getResultNumber());
        }
    }
}

/**
 * @brief Based on the blocks identified ahead of time, sort the operations in each instance topologically and mark the
 * input and output values of the instance
 */
SmallVector<OutliningInstance> RepeatingBlocksIdentifier::prepareOutliningInstances(mlir::func::FuncOp mainFunction) {
    SmallVector<OutliningInstance> outliningInstances(_blocks.size());
    SmallVector<SmallVector<OpValuePair>> outliningInstancesInputs(_blocks.size());
    SmallVector<SmallVector<OpValuePair>> outliningInstancesOutputs(_blocks.size());

    std::unordered_map<BlockId, size_t> blockOutliningIdx;
    size_t lastBlock = 0;

    const auto getBlockId = [&](InstanceId instanceId) -> BlockId {
        const auto instanceBlockIt = _instanceBlock.find(instanceId);
        VPUX_THROW_WHEN(instanceBlockIt == _instanceBlock.end(), "Missing block for instance {0}", instanceId);
        return instanceBlockIt->second;
    };
    const auto getInstanceIdx = [&](BlockId blockId, InstanceId instanceId) -> size_t {
        const auto blockIt = _blocks.find(blockId);
        VPUX_THROW_WHEN(blockIt == _blocks.end(), "Missing block with id {0}", blockId);
        const auto& opInstances = blockIt->second;
        for (size_t idx = 0; idx < opInstances.size(); ++idx) {
            if (opInstances[idx].id == instanceId) {
                return idx;
            }
        }
        VPUX_THROW("Could not find instance {0} in block {1}", instanceId, blockId);
    };

    mainFunction.walk([&](mlir::Operation* op) {
        const auto maybeInstanceId = getInstanceId(op);
        if (!maybeInstanceId.has_value()) {
            return;
        }
        const auto instanceId = maybeInstanceId.value();
        const auto blockId = getBlockId(instanceId);

        if (blockOutliningIdx.find(blockId) == blockOutliningIdx.end()) {
            const auto index = lastBlock++;
            blockOutliningIdx[blockId] = index;
            outliningInstances[index].resize(_blocks[blockId].size());
        }

        const auto outliningInstanceIdx = blockOutliningIdx[blockId];
        auto& instances = outliningInstances[outliningInstanceIdx];

        const auto idx = getInstanceIdx(blockId, instanceId);
        auto& instance = instances[idx];

        if (_separateFunctions) {
            addInputsOutputsForSeparateFunctions(instance, instanceId, op);
        } else {
            auto& instanceInputs = outliningInstancesInputs[outliningInstanceIdx];
            auto& instanceOutputs = outliningInstancesOutputs[outliningInstanceIdx];
            addInputsOutputsForSharedFunction(instance, instanceId, op, instanceInputs, instanceOutputs);
        }
    });

    if (!_separateFunctions) {
        // Extract the inputs and outputs of all instances for each block type. If a block has N instances, it is
        // possible that some of the instances (e.g. the last one) do not have the same number of users as the other
        // instances. However, since all instances call the same function, they must return all values even if some are
        // unused. Similarly, some instances could use one value multiple times, in which case the value has to be
        // passed multiple times for that instance's call to cover the other instances that receive different values
        // instead
        for (const auto& [instances, inputs, outputs] :
             zip(outliningInstances, outliningInstancesInputs, outliningInstancesOutputs)) {
            for (const auto& [instanceIdx, instance] : instances | indexed) {
                for (const auto& input : inputs) {
                    const auto inputHash = input.first;
                    const auto operandNumber = input.second;
                    auto opIt = llvm::find_if(instance.operations, [&](mlir::Operation* op) {
                        return inputHash == _opHash[op];
                    });
                    VPUX_THROW_WHEN(opIt == instance.operations.end(),
                                    "Missing operation with hash {0} in instance {1}", input.first, instanceIdx);
                    instance.inputs.push_back((*opIt)->getOperand(operandNumber));

                    // The first instance is expected to be used for outlining the function. Each argument is used only
                    // once, therefore we explicitly map each argument to its user, so that the outliner can correctly
                    // connect the arguments to the operations. The first instance might pass the same value multiple
                    // times, in which case the outliner is not aware to which user the value should be connected (other
                    // instances may pass different values, after all)
                    if (instanceIdx == 0) {
                        instance.inputUserMapping.emplace_back(*opIt, operandNumber);
                    }
                }

                for (auto& output : outputs) {
                    auto opIt = llvm::find_if(instance.operations, [&](mlir::Operation* op) {
                        return output.first == _opHash[op];
                    });
                    VPUX_THROW_WHEN(opIt == instance.operations.end(),
                                    "Missing operation with hash {0} in instance {1}", output.first, instanceIdx);
                    instance.outputs.push_back((*opIt)->getResult(output.second));
                }
            }
        }
    }

    return outliningInstances;
}

/**
 * @brief Check whether the instances are valid in terms of the IR order
 * @details Outlining some patterns could lead to cyclical dependencies, which cannot be correctly represented in the
 * form of an IR. For example, for the following IR:
 *   %0 = op(...)
 *   %1 = op(%0)
 *   %2 = op(%0, %1)
 * If operations %0 and %2 are part of a repeating block and get outlined into a function, the resulting IR would be:
 *   %1 = op(%call)
 *   %call = call(..., %1)
 */
mlir::LogicalResult RepeatingBlocksIdentifier::validateOutliningInstances(
        ArrayRef<OutliningInstance> outliningInstances) {
    const auto validateParentOperation = [](const IRSlice& slice, mlir::Operation* firstOutputOpInIR,
                                            mlir::Value operand) -> mlir::LogicalResult {
        if (operand.getDefiningOp() == nullptr) {
            return mlir::success();
        }

        mlir::DenseSet<mlir::Operation*> sliceOps(slice.operations.begin(), slice.operations.end());

        mlir::DenseSet<mlir::Operation*> visitedOps;
        std::deque<mlir::Operation*> parentOps;
        parentOps.push_back(operand.getDefiningOp());

        while (!parentOps.empty()) {
            const auto parentOp = parentOps.front();
            parentOps.pop_front();
            visitedOps.insert(parentOp);

            if (parentOp == nullptr) {
                continue;
            }
            if (parentOp->isBeforeInBlock(firstOutputOpInIR)) {
                continue;
            }
            if (sliceOps.contains(parentOp)) {
                continue;
            }
            for (auto operand : parentOp->getOperands()) {
                if (llvm::find(slice.outputs, operand) != slice.outputs.end()) {
                    return mlir::failure();
                }
                if (auto op = operand.getDefiningOp(); op != nullptr && !visitedOps.contains(op)) {
                    parentOps.push_back(operand.getDefiningOp());
                }
            }
        }
        return mlir::success();
    };

    for (auto& instance : outliningInstances) {
        for (auto slice : instance) {
            auto firstOutputOpInIR = slice.operations.front();
            for (auto output : slice.outputs) {
                auto op = output.getDefiningOp();
                if (op != nullptr && op->isBeforeInBlock(firstOutputOpInIR)) {
                    firstOutputOpInIR = op;
                }
            }
            for (auto input : slice.inputs) {
                if (mlir::failed(validateParentOperation(slice, firstOutputOpInIR, input))) {
                    return mlir::failure();
                }
            }
        }
    }
    return mlir::success();
}

void RepeatingBlocksIdentifier::printBlocks(StringLiteral note) {
    if (!_log.isActive(LogLevel::Trace)) {
        return;
    }
    _log.trace("Printing blocks: {0}", note);
    for (auto& block : _blocks) {
        _log.nest().trace("Block {0} with {1} instances:", block.first, block.second.size());
        for (auto& instance : block.second) {
            _log.nest(2).trace("- instance id {0}", instance.id);
            for (auto& op : instance.operations) {
                _log.nest(3).trace("- {0} at {1} ({2}):", op->getName(), op->getLoc(), _opHash[op]);
            }
        }
    }
}

/**
 * @brief Identify the repeating blocks in the given function and collect information about all of their instances
 * @details For the function operation given as a parameter, this method will try to identify all of the slices of the
 * IR which repeat multiple times. For example, let's assume that the IR contains the following operations:
 *    /> B \                   /> B \
 *  A       > D -> E -> F -> A       > G -> E -> F -> E -> F
 *    \> C /                   \> C /
 * The method would identify two repeating blocks:
 *   - one containing {A, B, C} with two instances
 *   - one containing {E, F} with three instances
 * The return value of this method contains information for each instance of each block: the operations sorted
 * topologically, the input values whose producers are not part of the instance and the output values which have users
 * outside of the instance. The dependencies for the instance's operations (i.e. constants) are also included into the
 * list of operations. This is done in order for the outliner to easily identify which constant should map to which
 * operation.
 */
SmallVector<OutliningInstance> RepeatingBlocksIdentifier::getOutliningInstances(mlir::func::FuncOp mainFunction) {
    // Step 1. Identify operations that repeat in the IR and place them in a unique block.
    identifyUniqueOperations(mainFunction);

    // Step 2. Try to merge adjacent blocks of operations. This is repeated until no more merges are done or until the
    // maximum number of iterations is reached
    _log.trace("Trying to merge adjacent blocks");
    for (size_t i = 0; i < _maxNumIterations; ++i) {
        _log.trace("Iteration {0}", i);
        if (!tryMergeAdjacentBlocks()) {
            _log.trace("No merge could be performed. Stopping attempts");
            break;
        }
    }

    // Step 3. Remove blocks which have only one instance or fewer operations than the configured minimum
    removeLeftoverBlocks();

    // Step 4. Sort the instances in each repeating block topologically and include all dependencies
    const auto outliningInstances = prepareOutliningInstances(mainFunction);

    // Step5 5. Check whether the identified instances are valid
    if (mlir::failed(validateOutliningInstances(outliningInstances))) {
        _log.debug("The identified instances are invalid");
        return {};
    }

    return outliningInstances;
}
};  // namespace

namespace vpux {
namespace IE {

FunctionOutlinerRepeatingBlocks::FunctionOutlinerRepeatingBlocks(size_t minOpsInBlock, size_t maxNumIterations,
                                                                 bool separateFunctions, bool weightsAsInputs,
                                                                 Logger log)
        : _minOpsInBlock(minOpsInBlock),
          _maxNumIterations(maxNumIterations),
          _separateFunctions(separateFunctions),
          _weightsAsInputs(weightsAsInputs),
          _log(log) {
    _log.setName("function-outliner-repeating-blocks");
}

SmallVector<OutliningInstance> FunctionOutlinerRepeatingBlocks::getOutliningTargets(mlir::func::FuncOp mainFunction) {
    _log.debug("Searching for outlining targets that are repeating in the IR");

    if (_minOpsInBlock == 0) {
        _log.debug("Minimum number of operations in block {0} should be larger than 0", _minOpsInBlock);
        return {};
    }

    RepeatingBlocksIdentifier repeatingBlocksIdentifier(_minOpsInBlock, _maxNumIterations, _separateFunctions,
                                                        _weightsAsInputs, _log);
    const auto outliningInstances = repeatingBlocksIdentifier.getOutliningInstances(mainFunction);

    printOutliningInstances(outliningInstances, _log);

    return outliningInstances;
}

}  // namespace IE
}  // namespace vpux
