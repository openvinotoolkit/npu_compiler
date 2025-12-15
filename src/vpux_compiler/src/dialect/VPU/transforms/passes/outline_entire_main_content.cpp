//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_OUTLINEENTIREMAINCONTENT
#define GEN_PASS_DEF_OUTLINEENTIREMAINCONTENT
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

struct FuncInfo {
    std::string funcName;
    SmallVector<mlir::Value> inputs;
    SmallVector<mlir::Value> outputs;
    SmallVector<mlir::Type> inputTypes;
    SmallVector<mlir::Type> outputTypes;
    mlir::func::FuncOp funcOp;
};

struct Instance {
    mlir::Operation* nextIROp;
    std::vector<mlir::Operation*> ops;
};

class OutlineEntireMainContentPass final :
        public VPU::impl::OutlineEntireMainContentBase<OutlineEntireMainContentPass> {
public:
    explicit OutlineEntireMainContentPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final {
        net::NetworkInfoOp netInfo;
        mlir::func::FuncOp mainFuncOp;
        auto moduleOp = getOperation();
        net::NetworkInfoOp::getFromModule(moduleOp, netInfo, mainFuncOp);

        bool containsCallOps = false;
        bool containsNonCallOps = false;
        for (auto& op : mainFuncOp.getOps()) {
            if (mlir::isa<mlir::func::ReturnOp>(&op)) {
                continue;
            }
            const auto isCallOp = mlir::isa<mlir::func::CallOp>(&op);
            containsCallOps |= isCallOp;
            containsNonCallOps |= !isCallOp;
            if (containsCallOps && containsNonCallOps) {
                break;
            }
        }
        if (!(containsCallOps && containsNonCallOps)) {
            _log.trace("The main function does not contain a mix of call and non-call operations. Nothing to do");
            return;
        }

        std::vector<Instance> instances;
        {
            std::vector<mlir::Operation*> ops;
            const auto mainOps = mainFuncOp.getOps();
            for (auto& op : mainOps) {
                if (mlir::isa<mlir::func::CallOp, mlir::func::ReturnOp>(&op)) {
                    if (!ops.empty()) {
                        instances.push_back(Instance{&op, ops});
                        ops.clear();
                    }
                    continue;
                }
                ops.push_back(&op);
            }
        }
        if (instances.empty()) {
            _log.trace("No instances for outlining were found");
            return;
        }

        addConstantsToInstances(mainFuncOp, instances);

        std::vector<FuncInfo> funcsInfo(instances.size());
        for (const auto& [instanceIdx, instance] : instances | indexed) {
            auto funcName = printToString("{0}_outline{1}", mainFuncOp.getName(), instanceIdx + 1);
            SmallVector<mlir::Value> inputs, outputs;
            gatherInputsOutputs(instance.ops, inputs, outputs);
            SmallVector<mlir::Type> inputTypes, outputTypes;
            for (auto input : inputs) {
                const auto types = prepareType(input.getType());
                inputTypes.append(types.begin(), types.end());
            }
            for (auto output : outputs) {
                const auto types = prepareType(output.getType());
                outputTypes.append(types.begin(), types.end());
            }
            funcsInfo[instanceIdx] = FuncInfo{std::move(funcName),   std::move(inputs),      std::move(outputs),
                                              std::move(inputTypes), std::move(outputTypes),
                                              /*funcOp=*/nullptr};
        }

        // In case an instance only contains constants and these get duplicated in the user instances, the original
        // instance would be left without users, which eventually leads to an empty function being created
        removeEmptyInstances(instances, funcsInfo);

        _log.info("Creating {0} functions", instances.size());

        buildFuncOps(mainFuncOp, funcsInfo, instances);
        buildCallOps(mainFuncOp, funcsInfo, instances);
        removeOrigOps(instances);
    }

    // Collect the parent operations that are not found in the instance, starting with the operation given as a
    // parameter. The function will iterate through subgraphs of copies and group-view-like operations, and collect all
    // the operations up to the constants (if the constants exist)
    SmallVector<mlir::Operation*> getConstantParents(mlir::Operation* op, ArrayRef<mlir::Operation*> instanceOps) {
        if (op == nullptr) {
            return {};
        }
        if (llvm::find(instanceOps, op) != instanceOps.end()) {
            return {};
        }

        if (mlir::isa<Const::DeclareOp>(op)) {
            return {op};
        } else if (mlir::isa<VPU::CopyOp>(op)) {
            if (const auto parentOp = op->getOperand(0).getDefiningOp()) {
                auto parentConstOps = getConstantParents(parentOp, instanceOps);
                if (parentConstOps.empty()) {
                    return {};
                }
                parentConstOps.push_back(op);
                return parentConstOps;
            }
        } else if (mlir::isa<VPU::GroupedViewLikeOpInterface>(op)) {
            SmallVector<mlir::Operation*> parentConstOps;
            for (auto operand : op->getOperands()) {
                if (const auto parentOp = operand.getDefiningOp()) {
                    parentConstOps.append(getConstantParents(parentOp, instanceOps));
                }
            }
            parentConstOps.push_back(op);
            return parentConstOps;
        }

        return {};
    }

    // Include constant operations in all instances that use them, to ensure all functions that get outlined contain the
    // used constants. If instead of a constant, the dependency is a subgraph (e.g. `constant -> copy` or sparsity
    // subgraph), the entire subgraph is included in the instance. This inclusion is done by cloning the dependencies
    // into the IR slice associated with each user instance
    void addConstantsToInstances(mlir::func::FuncOp mainFuncOp, std::vector<Instance>& instances) {
        auto builder = mlir::OpBuilder(mainFuncOp);

        size_t instanceIdx = 0;
        for (auto& instance : instances) {
            std::vector<mlir::Operation*> duplicatedOps;
            for (auto op : instance.ops) {
                for (auto& operand : op->getOpOperands()) {
                    const auto constantParentOps = getConstantParents(operand.get().getDefiningOp(), instance.ops);
                    if (constantParentOps.empty()) {
                        continue;
                    }

                    std::vector<mlir::Operation*> newOperations(constantParentOps.size());
                    DenseMap<mlir::Value, mlir::Value> oldToNewMap;
                    builder.setInsertionPoint(instance.ops.front());
                    for (const auto& parentOpIt : constantParentOps | indexed) {
                        const auto parentOp = parentOpIt.value();
                        mlir::IRMapping mapper;
                        for (auto operand : parentOp->getOperands()) {
                            if (oldToNewMap.contains(operand)) {
                                mapper.map(operand, oldToNewMap[operand]);
                            }
                        }
                        _log.trace("Cloning parent {0} at {1} to instance {2}", parentOp->getName(), parentOp->getLoc(),
                                   instanceIdx);
                        auto clonedOp = builder.clone(*parentOp, mapper);
                        newOperations[parentOpIt.index()] = clonedOp;
                        for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
                            oldToNewMap[parentOp->getResult(i)] = clonedOp->getResult(i);
                        }
                    }

                    VPUX_THROW_WHEN(!oldToNewMap.contains(operand.get()), "Expected parent to be cloned");
                    operand.set(oldToNewMap[operand.get()]);

                    duplicatedOps.insert(duplicatedOps.end(), newOperations.begin(), newOperations.end());
                }
            }
            instance.ops.insert(instance.ops.begin(), duplicatedOps.begin(), duplicatedOps.end());
            instanceIdx++;
        }
    }

    void gatherInputsOutputs(ArrayRef<mlir::Operation*> ops, SmallVector<mlir::Value>& inputs,
                             SmallVector<mlir::Value>& outputs) {
        for (auto op : ops) {
            for (auto operand : op->getOperands()) {
                const bool operandAlreadyCovered = llvm::find(inputs, operand) != inputs.end();
                if (operandAlreadyCovered) {
                    continue;
                }
                if (mlir::isa_and_nonnull<mlir::BlockArgument>(operand)) {
                    inputs.push_back(operand);
                    continue;
                }
                if (auto parentOp = operand.getDefiningOp()) {
                    const auto isOpExternal = llvm::find(ops, parentOp) == ops.end();
                    if (isOpExternal) {
                        inputs.push_back(operand);
                    }
                }
            }
            for (auto result : op->getResults()) {
                const bool resultAlreadyCovered = llvm::find(outputs, result) != outputs.end();
                if (resultAlreadyCovered) {
                    continue;
                }
                const auto hasExternalUser = llvm::any_of(result.getUsers(), [&](mlir::Operation* userOp) {
                    const auto isOpExternal = llvm::find(ops, userOp) == ops.end();
                    return isOpExternal;
                });
                if (hasExternalUser) {
                    outputs.push_back(result);
                }
            }
        }
    }

    // Prepare the type for being a function argument / result:
    // - in case the type is sparse, ungroup all types so that they can be represented as individual values
    // - in case the type is not found in DDR, change the memory space to DDR; this is done to accommodate a
    // scheduler limitation where each argument / result of a function must be in DDR
    SmallVector<mlir::Type> prepareType(mlir::Type origType) {
        SmallVector<mlir::Type> origTypes = {origType};
        if (auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(origType)) {
            origTypes = {sparseType.getData(), sparseType.getSparsityMap(), sparseType.getStorageElementTable()};
        }
        SmallVector<mlir::Type> preparedTypes;
        for (auto type : origTypes) {
            if (type == nullptr) {
                continue;
            }
            if (mlir::cast<NDTypeInterface>(type).getMemoryKind() == VPU::MemoryKind::DDR) {
                preparedTypes.push_back(type);
                continue;
            }
            auto newType = type;
            if (auto distributedType = mlir::dyn_cast<VPU::DistributedTypeInterface>(type);
                distributedType != nullptr && distributedType.containsDistributedTypes()) {
                newType = VPU::getCompactTypeFromDistributed(type);
            }
            preparedTypes.push_back(mlir::cast<NDTypeInterface>(newType).changeMemSpace(VPU::MemoryKind::DDR));
        }
        return preparedTypes;
    }

    mlir::Value groupSparseValue(mlir::OpBuilder& builder, mlir::ValueRange sparseValues,
                                 VPU::SparseTensorType sparseType) {
        VPUX_THROW_WHEN(sparseValues.empty(), "Empty sparse values for type {0}", sparseType);
        auto producerOp = sparseValues.front().getDefiningOp();
        const auto loc = (producerOp != nullptr) ? appendLoc(producerOp->getLoc(), "_group")
                                                 : mlir::UnknownLoc::get(&getContext());
        const auto data = sparseValues.front();
        mlir::Value sparsityMap = nullptr;
        mlir::Value seTable = nullptr;
        {
            size_t metadataIdx = 1;
            if (sparseType.getSparsityMap() != nullptr) {
                VPUX_THROW_WHEN(sparseValues.size() < metadataIdx + 1, "Missing sparsity map value for type {0}",
                                sparseType);
                sparsityMap = sparseValues[metadataIdx++];
            }
            if (sparseType.getStorageElementTable() != nullptr) {
                VPUX_THROW_WHEN(sparseValues.size() < metadataIdx + 1,
                                "Missing storage element table value for type {0}", sparseType);
                seTable = sparseValues[metadataIdx++];
            }
        }
        auto groupOp =
                builder.create<VPU::GroupSparseTensorOp>(loc, data, sparsityMap, seTable, sparseType.getIsWeights(),
                                                         sparseType.getSparsityCompression(), sparseType.getSeAttr());
        return groupOp->getResult(0);
    }

    mlir::ResultRange ungroupSparseValue(mlir::OpBuilder& builder, mlir::Value sparseValue) {
        auto producerOp = sparseValue.getDefiningOp();
        const auto loc = (producerOp != nullptr) ? appendLoc(producerOp->getLoc(), "_ungroup")
                                                 : mlir::UnknownLoc::get(&getContext());
        auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(sparseValue.getType());
        VPUX_THROW_WHEN(sparseType == nullptr, "Expected value to have sparse type, but got type {0}",
                        sparseValue.getType());
        auto ungroupOp =
                builder.create<VPU::UngroupSparseTensorOp>(loc, sparseType.getData(), sparseType.getSparsityMap(),
                                                           sparseType.getStorageElementTable(), sparseValue);
        return ungroupOp->getResults();
    }

    /**
     * @brief Create a copy operation that generates the given output type
     * @details This function is intended to be used to move the input / output values of the new functions to /
     * from DDR, if the original value from main was not found in DDR. This is done to accommodate a scheduler
     * limitation, where each input / output value of a function must be in DDR
     * @param builder - the builder used to create operations
     * @param origValue - the original value from the main function that corresponds to the input of the copy
     * operation
     * @param copyInput - the value that should be used as the input of the new copy operation
     * @param newType - the type that should be produced by the new copy operation
     */
    mlir::Operation* createCopy(mlir::OpBuilder& builder, mlir::Value origValue, mlir::Value copyInput,
                                mlir::Type newType) {
        auto producerOp = origValue.getDefiningOp();
        const auto loc = (producerOp != nullptr) ? producerOp->getLoc() : mlir::UnknownLoc::get(&getContext());
        const auto newMemSpace = mlir::cast<NDTypeInterface>(newType).getMemSpace();

        return builder.create<VPU::CopyOp>(loc, newType, copyInput, newMemSpace);
    }

    void buildFuncOps(mlir::func::FuncOp mainFuncOp, std::vector<FuncInfo>& funcsInfo, ArrayRef<Instance> instances) {
        for (const auto& instance : instances | indexed) {
            const auto& ops = instance.value().ops;

            const auto& funcName = funcsInfo[instance.index()].funcName;
            _log.nest().trace("Creating function '{0}'", funcName);

            const auto& inputs = funcsInfo[instance.index()].inputs;
            const auto& outputs = funcsInfo[instance.index()].outputs;
            const auto& inputTypes = funcsInfo[instance.index()].inputTypes;
            const auto& outputTypes = funcsInfo[instance.index()].outputTypes;

            auto builder = mlir::OpBuilder(mainFuncOp);
            const auto funcLoc = appendLoc(mainFuncOp.getLoc(), "_outline{0}", instance.index() + 1);
            const auto funcType = mlir::FunctionType::get(&getContext(), inputTypes, outputTypes);
            auto funcOp = builder.create<mlir::func::FuncOp>(funcLoc, funcName, funcType);
            funcOp.setPrivate();
            funcsInfo[instance.index()].funcOp = funcOp;

            auto funcOpBlock = funcOp.addEntryBlock();
            builder.setInsertionPointToStart(funcOpBlock);

            DenseMap<mlir::Value, mlir::Value> oldToNewMap;
            for (size_t origIdx = 0, newIdx = 0; origIdx < inputs.size(); ++origIdx) {
                const auto origInput = inputs[origIdx];
                if (auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(origInput.getType())) {
                    size_t numTypes = (sparseType.getData() != nullptr) ? 1 : 0;
                    numTypes += (sparseType.getSparsityMap() != nullptr) ? 1 : 0;
                    numTypes += (sparseType.getStorageElementTable() != nullptr) ? 1 : 0;
                    SmallVector<mlir::Value> funcArgs(funcOp.getArguments().begin() + newIdx,
                                                      funcOp.getArguments().begin() + newIdx + numTypes);
                    auto groupedValue = groupSparseValue(builder, funcArgs, sparseType);
                    if (origInput.getType() != groupedValue.getType()) {
                        auto copyOp = createCopy(builder, origInput, groupedValue, origInput.getType());
                        groupedValue = copyOp->getResult(0);
                    }
                    oldToNewMap[origInput] = groupedValue;
                    newIdx += numTypes;
                    continue;
                }
                mlir::Value newInput = funcOp.getArgument(newIdx);
                if (origInput.getType() != funcOp.getArgument(newIdx).getType()) {
                    auto copyOp = createCopy(builder, origInput, funcOp.getArgument(newIdx), origInput.getType());
                    newInput = copyOp->getResult(0);
                }
                oldToNewMap[origInput] = newInput;
                ++newIdx;
            }

            for (const auto op : ops) {
                mlir::IRMapping mapper;
                for (auto operand : op->getOperands()) {
                    mapper.map(operand, oldToNewMap[operand]);
                }
                auto clonedOp = builder.clone(*op, mapper);
                clonedOp->setLoc(appendLoc(clonedOp->getLoc(), formatv("_outline{0}", instance.index() + 1).str()));
                for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
                    oldToNewMap[op->getResult(i)] = clonedOp->getResult(i);
                }
            }

            SmallVector<mlir::Value> funcOutputFromSlices;
            for (size_t origIdx = 0, newIdx = 0; origIdx < outputs.size(); ++origIdx) {
                const auto origOutput = outputs[origIdx];
                if (mlir::isa<VPU::SparseTensorType>(origOutput.getType())) {
                    const auto ungroupedValues = ungroupSparseValue(builder, oldToNewMap[origOutput]);
                    for (const auto& p : ungroupedValues | indexed) {
                        auto ungroupedValue = p.value();
                        const auto& ungroupedValueIdx = p.index();
                        if (ungroupedValue.getType() != outputTypes[newIdx + ungroupedValueIdx]) {
                            auto copyOp = createCopy(builder, origOutput, ungroupedValue,
                                                     outputTypes[newIdx + ungroupedValueIdx]);
                            ungroupedValue = copyOp->getResult(0);
                        }
                        funcOutputFromSlices.push_back(ungroupedValue);
                    }
                    newIdx += ungroupedValues.size();
                    continue;
                }
                auto newOutput = oldToNewMap[origOutput];
                if (origOutput.getType() != outputTypes[newIdx]) {
                    auto copyOp = createCopy(builder, origOutput, oldToNewMap[origOutput], outputTypes[newIdx]);
                    newOutput = copyOp->getResult(0);
                }
                funcOutputFromSlices.push_back(newOutput);
                ++newIdx;
            }
            const auto returnLoc = appendLoc(mainFuncOp.getLoc(), "_outline{0}_return", instance.index() + 1);
            builder.create<mlir::func::ReturnOp>(returnLoc, funcOutputFromSlices);
        }
    }

    void buildCallOps(mlir::func::FuncOp mainFuncOp, ArrayRef<FuncInfo> funcsInfo, ArrayRef<Instance> instances) {
        DenseMap<mlir::Value, SmallVector<mlir::Value>> sparseValueMapping;
        DenseMap<mlir::Value, mlir::Value> oldToNewArgMap;
        for (const auto& arg : mainFuncOp.getArguments()) {
            oldToNewArgMap[arg] = arg;
        }

        for (const auto& instance : instances | indexed) {
            const auto& funcInfo = funcsInfo[instance.index()];
            VPUX_THROW_WHEN(funcInfo.funcOp == nullptr, "Missing func op for instance '{0}'", instance.index());

            _log.nest().trace("Creating call op to function '{0}'", funcInfo.funcName);

            auto builder = mlir::OpBuilder(instance.value().nextIROp);

            SmallVector<mlir::Value> newInputs;
            for (const auto input : funcInfo.inputs) {
                if (auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(input.getType())) {
                    VPUX_THROW_UNLESS(sparseValueMapping.contains(input), "Missing mapping for sparse value {0}",
                                      input);
                    const auto& inputs = sparseValueMapping.at(input);
                    newInputs.append(inputs.begin(), inputs.end());
                    continue;
                }
                if (oldToNewArgMap.contains(input)) {
                    newInputs.push_back(oldToNewArgMap[input]);
                } else {
                    newInputs.push_back(input);
                }
            }

            const auto callLoc = appendLoc(funcInfo.funcOp->getLoc(), "_call");
            auto newCall = builder.create<mlir::func::CallOp>(callLoc, funcInfo.funcOp, newInputs);

            for (size_t origIdx = 0, newIdx = 0; origIdx < funcInfo.outputs.size();) {
                if (auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(funcInfo.outputs[origIdx].getType())) {
                    size_t numTypes = (sparseType.getData() != nullptr) ? 1 : 0;
                    numTypes += (sparseType.getSparsityMap() != nullptr) ? 1 : 0;
                    numTypes += (sparseType.getStorageElementTable() != nullptr) ? 1 : 0;
                    sparseValueMapping[funcInfo.outputs[origIdx]] = newCall->getResults().slice(newIdx, numTypes);
                    ++origIdx;
                    newIdx += numTypes;
                    continue;
                }
                oldToNewArgMap[funcInfo.outputs[origIdx]] = newCall->getResult(newIdx);
                ++origIdx;
                ++newIdx;
            }
        }

        mainFuncOp.walk([&](mlir::Operation* op) {
            for (auto i : irange(op->getNumOperands())) {
                if (oldToNewArgMap.find(op->getOperand(i)) != oldToNewArgMap.end()) {
                    op->setOperand(i, oldToNewArgMap[op->getOperand(i)]);
                }
            }
        });
    }

    void removeEmptyInstances(std::vector<Instance>& instances, std::vector<FuncInfo>& funcsInfo) {
        SmallVector<size_t> emptyInstances;
        for (size_t instanceIdx = 0; instanceIdx < instances.size(); ++instanceIdx) {
            if (funcsInfo[instanceIdx].outputs.empty()) {
                emptyInstances.push_back(instanceIdx);
            }
        }
        for (auto instanceIdx : emptyInstances | reversed) {
            instances.erase(instances.begin() + instanceIdx);
            funcsInfo.erase(funcsInfo.begin() + instanceIdx);
        }
    }

    void removeOrigOps(ArrayRef<Instance> instances) {
        _log.trace("Removing original ops");
        for (const auto& instance : instances | reversed) {
            for (auto op : instance.ops | reversed) {
                op->erase();
            }
        }
    }
};

}  // namespace

//
// createOutlineEntireMainContentPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createOutlineEntireMainContentPass(const Logger& log) {
    return std::make_unique<OutlineEntireMainContentPass>(log);
}
