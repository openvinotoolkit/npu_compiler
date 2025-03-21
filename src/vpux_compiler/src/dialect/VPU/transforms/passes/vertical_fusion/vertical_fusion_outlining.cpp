//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_config.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Visitors.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_VERTICALFUSIONOUTLINING
#define GEN_PASS_DEF_VERTICALFUSIONOUTLINING
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

struct FuncInfo {
    SmallVector<mlir::Type> inputTypes;
    SmallVector<mlir::Type> outputTypes;
    std::string funcNames;
};

namespace {

//
// VerticalFusionOutliningPass
//
class VerticalFusionOutliningPass final : public VPU::impl::VerticalFusionOutliningBase<VerticalFusionOutliningPass> {
public:
    VerticalFusionOutliningPass() = default;
    VerticalFusionOutliningPass(const VPU::TilingOptions& TilingOptions, Logger log);

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    void safeRunOnModule() final;

private:
    // Initialize fields from pass options
    void initializeFromOptions();

private:
    // checks for supported patterns which will be added with root op
    bool isProcessInputOp(mlir::Operation* op);
    bool isConcatWithOutlinedInputs(mlir::Operation* op);
    bool isProcessOutputOp(mlir::Operation* op);

    // recursively process operations on the input, add supported ops to the outlining instance
    void processInputsRecursively(mlir::Operation& op, OpOrderedSet& storedOperations);
    // add operation with inputs and outputs to outlining instance
    void processOperation(mlir::Operation& op, OpOrderedSet& storedOperations);
    // recursively process operations on the output, add supported ops to the outlining instance
    void processOutputsRecursively(mlir::Operation& op, OpOrderedSet& storedOperations);

    // Move storage op to instance
    void moveOpsToInstance(OpOrderedSet& instanceOps, SmallVector<OpOrderedSet>& instances, bool target = false);
    // Get outlining instances separated by VF regions
    SmallVector<OutliningInstance> getOutliningInstances(mlir::func::FuncOp netFunc, const int64_t tileThreshold);
    // Convert stored operations to outlining instance
    void createOutliningInstanceFromStorage(ValueOrderedSet& storedInputs, ValueOrderedSet& storedOutputs,
                                            OpOrderedSet& storedOperations, SmallVector<OutliningInstance>& instances);

    void buildFuncOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets);
    void buildCallOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets);

private:
    mlir::DenseSet<mlir::Operation*> _outlinedOperations;
    // Thresholds for outlining, avoid creating very small functions
    int64_t _numInstanceThreshold = 1;
    int64_t _verticalFusionTileThreshold = 1;
};

bool isSupportedSkipOp(mlir::Operation* op) {
    // TODO: E#140556 include more view ops
    return mlir::isa<VPU::ShapeCastOp, VPU::AffineReshapeOp, VPU::PermuteCastOp, VPU::ExpandOp>(op);
}

mlir::Operation* trySearchForRootOp(mlir::Operation* op) {
    if (!isSupportedSkipOp(op)) {
        return nullptr;
    }
    if (!op->getResult(0).hasOneUse()) {
        return nullptr;
    }
    return *op->getResult(0).getUsers().begin();
}

bool isConstOperandOp(mlir::Operation* op) {
    if (mlir::isa<VPU::StorageElementTableOp, Const::DeclareOp>(op)) {
        return true;
    }

    if (mlir::isa<VPU::GroupedViewLikeOpInterface>(op)) {
        return llvm::all_of(op->getOperands(), [&](mlir::Value v) {
            if (mlir::isa<mlir::BlockArgument>(v)) {
                return true;
            }
            auto parentOp = v.getDefiningOp();
            return isConstOperandOp(parentOp);
        });
    }

    return false;
}

bool VerticalFusionOutliningPass::isProcessInputOp(mlir::Operation* op) {
    if (isConstOperandOp(op)) {
        return true;
    }

    if (_outlinedOperations.find(op) != _outlinedOperations.end()) {
        return false;
    }

    if (auto tryOp = trySearchForRootOp(op)) {
        // Do no add 'ViewOp -> Return' to current outlining instance
        if (mlir::isa<mlir::func::ReturnOp>(tryOp)) {
            return false;
        }
        return true;
    }

    // TODO: E#140556 generic view ops
    if (mlir::isa<VPU::SliceOp>(op)) {
        // Only add 'BlockArg -> Slice' to current outlining instance
        if (mlir::isa<mlir::BlockArgument>(op->getOperand(0))) {
            return true;
        }
        return false;
    }

    return false;
}

bool VerticalFusionOutliningPass::isConcatWithOutlinedInputs(mlir::Operation* op) {
    if (!mlir::isa<ConcatOp>(op)) {
        return false;
    }
    for (auto operand : op->getOperands()) {
        auto parentOp = operand.getDefiningOp();
        if (parentOp == nullptr) {
            continue;
        }
        if (_outlinedOperations.contains(parentOp)) {
            continue;
        }
        return false;
    }
    return true;
}

bool VerticalFusionOutliningPass::isProcessOutputOp(mlir::Operation* op) {
    // skip supported ops, search for root op
    while (auto tryOp = trySearchForRootOp(op)) {
        if (mlir::isa<mlir::func::ReturnOp>(tryOp)) {
            return true;
        }
        op = tryOp;
    }

    // TODO: E#140556 generic view ops
    if (mlir::isa<VPU::SliceOp>(op)) {
        // 'BlockArg -> Slice' case handled in 'isProcessInputOp'
        if (mlir::isa<mlir::BlockArgument>(op->getOperand(0))) {
            return false;
        }
        if (op->getOperand(0).getDefiningOp<VPU::ConcatOp>() != nullptr) {
            // 'Concat -> Slice' or 'Concat -> Concat' patterns will
            // be processed recursively
            for (auto userOp : op->getOperand(0).getUsers()) {
                if (!mlir::isa<VPU::SliceOp, VPU::ConcatOp>(userOp)) {
                    // slice input has other uses, do not process
                    return false;
                }
            }
        }
        return true;
    }
    return false;
}

void VerticalFusionOutliningPass::processOperation(mlir::Operation& op, OpOrderedSet& storedOperations) {
    if (storedOperations.find(&op) != storedOperations.end()) {
        return;
    }
    storedOperations.insert(&op);
    _outlinedOperations.insert(&op);
}

void VerticalFusionOutliningPass::processInputsRecursively(mlir::Operation& op, OpOrderedSet& storedOperations) {
    // recursively add supported ops to the outlining instance
    for (auto operand : op.getOperands()) {
        auto parentOp = operand.getDefiningOp();
        if (parentOp == nullptr) {
            continue;
        }
        if (isProcessInputOp(parentOp)) {
            processInputsRecursively(*parentOp, storedOperations);
            processOperation(*parentOp, storedOperations);
        }
    }
}

void VerticalFusionOutliningPass::processOutputsRecursively(mlir::Operation& op, OpOrderedSet& storedOperations) {
    // recursively add supported ops to the outlining instance
    mlir::SmallVector<mlir::Operation*> stepOps;
    for (auto result : op.getResults()) {
        for (auto userOp : result.getUsers()) {
            if (_outlinedOperations.find(userOp) != _outlinedOperations.end()) {
                continue;
            }
            if (isProcessOutputOp(userOp)) {
                stepOps.push_back(userOp);
            } else if (isConcatWithOutlinedInputs(userOp)) {
                stepOps.push_back(userOp);
            }
        }
    }
    // BFS to preserve order of operations
    for (auto stepOp : stepOps) {
        processOperation(*stepOp, storedOperations);
        processOutputsRecursively(*stepOp, storedOperations);
    }
}

void VerticalFusionOutliningPass::createOutliningInstanceFromStorage(ValueOrderedSet& storedInputs,
                                                                     ValueOrderedSet& storedOutputs,
                                                                     OpOrderedSet& storedOperations,
                                                                     SmallVector<OutliningInstance>& instances) {
    auto currentSlice = IRSlice();
    for (auto op : storedOperations) {
        currentSlice.operations.push_back(op);
    }

    for (auto operand : llvm::make_early_inc_range(storedInputs)) {
        if (storedOutputs.find(operand) != storedOutputs.end()) {
            storedInputs.erase(operand);
        }
    }

    for (auto operand : llvm::make_early_inc_range(storedOutputs)) {
        const auto hasOutsideUser = llvm::any_of(operand.getUsers(), [&](mlir::Operation* op) {
            return storedOperations.find(op) == storedOperations.end();
        });
        if (!hasOutsideUser) {
            storedOutputs.erase(operand);
        } else if (operand.getType().isa<VPU::SparseTensorType>()) {
            // TODO: E#140551 support GroupSparseTensorOp as function arg
            return;
        }
    }

    for (auto inputs : storedInputs) {
        currentSlice.inputs.push_back(inputs);
    }
    for (auto outputs : storedOutputs) {
        currentSlice.outputs.push_back(outputs);
    }

    if (currentSlice.operations.empty() || currentSlice.inputs.empty() || currentSlice.outputs.empty()) {
        _log.error("At least one instance has no outputs values, which results in an empty function");
        return;
    }

    // stored ops moved into outlined instance
    storedInputs.clear();
    storedOutputs.clear();
    storedOperations.clear();
    instances.push_back(OutliningInstance{std::move(currentSlice)});
}

void VerticalFusionOutliningPass::moveOpsToInstance(OpOrderedSet& instanceOps, SmallVector<OpOrderedSet>& instances,
                                                    bool target) {
    if (!target) {
        // check if only skip ops in instance
        const auto onlySkipOps = llvm::all_of(instanceOps, [&](mlir::Operation* op) {
            return isSupportedSkipOp(op);
        });

        // try to add ops to parent instances
        if (onlySkipOps) {
            auto firstOp = *instanceOps.begin();
            if (auto parentOp = firstOp->getOperand(0).getDefiningOp()) {
                for (auto& instance : instances) {
                    if (instance.find(parentOp) != instance.end()) {
                        // found parent instance
                        instance.insert(instanceOps.begin(), instanceOps.end());
                        instanceOps.clear();
                        return;
                    }
                }
            }
        }
    }

    // move to new instance
    instances.push_back(instanceOps);
    instanceOps.clear();
}

SmallVector<OutliningInstance> VerticalFusionOutliningPass::getOutliningInstances(mlir::func::FuncOp netFunc,
                                                                                  const int64_t tileThreshold) {
    // High level overview:
    // Walk though operations in IR and add them to stored ops with inputs and outputs
    // Recursively add supported inputs and outputs for the operations to storage.
    // If outlining instance was found (vertical fusion operation satisfying tile threshold),
    // outlining instance need to be created, creating outlining instance can have 3 cases:

    // CASE1. There are some operation stored and current operation is vertical fusion outlining instance
    //          1. Create outlining instance for the stored operations
    //          2. Add VF operation to storage, recursively add supported input and output ops
    //          3. Create outlining instance for the VF operation

    // CASE3. Current operation is vertical fusion outlining instance
    //          1. Add VF operation to storage, recursively add supported input and output ops
    //          2. Create outlining instance for the VF operation

    // CASE2. ReturnOp is reached in the main function:
    //          1. Recursively add supported input operations to the ReturnOp
    //          2. If there are any operations stored - create outlining instance for the last operations.

    SmallVector<OpOrderedSet> opInstances;
    OpOrderedSet instanceOps;

    const auto isOpInCurrentOutliningInstance = [&](mlir::Operation* op) {
        return instanceOps.find(op) != instanceOps.end();
    };

    const auto isParallelConcatInput = [&](mlir::Operation* op) {
        /*                          ... ...  Op
                                      \  |  /  \
        Check for pattern:      ...    Concat  Concat
                                  \    /
                                   VFOp
        */
        // parallel concat sequences can be well optimized, but require:
        // "producers and consumers in the same function"
        // TODO: E#140555 optimize concat across functions
        const bool vfOpInStorage = isOpInCurrentOutliningInstance(op);
        for (auto input : op->getOperands()) {
            if (auto concatOp = input.getDefiningOp<VPU::ConcatOp>()) {
                const bool concatOpInStorage = isOpInCurrentOutliningInstance(concatOp);
                for (auto concatInput : concatOp.getInputs()) {
                    for (auto user : concatInput.getUsers()) {
                        if (user == concatOp || !mlir::isa_and_nonnull<VPU::ConcatOp>(user)) {
                            continue;
                        }
                        if (vfOpInStorage == concatOpInStorage &&
                            concatOpInStorage == isOpInCurrentOutliningInstance(user)) {
                            continue;
                        }
                        // VFOp is a consumer of parallel concat, can not outline since
                        // would result in parallel concat consumers in different function
                        return true;
                    }
                }
            }
        }
        return false;
    };

    const auto isVfOutliningInstanceCandidate = [&](mlir::Operation* op) {
        if (auto vfOp = mlir::dyn_cast_or_null<VPU::VerticalFusionOp>(op)) {
            const auto tilingStrategy = parseIntArrayAttr<int64_t>(vfOp.getTilingStrategy());
            const auto numTiles = std::accumulate(tilingStrategy.begin(), tilingStrategy.end(), int64_t(1),
                                                  std::multiplies<int64_t>());
            if (numTiles < tileThreshold) {
                return false;
            }
            if (isParallelConcatInput(op)) {
                return false;
            }
            return true;
        }
        return false;
    };

    for (auto& op : netFunc.getOps()) {
        if (_outlinedOperations.find(&op) != _outlinedOperations.end()) {
            // skip already outlined operations
            continue;
        }
        if (isProcessInputOp(&op) || isProcessOutputOp(&op)) {
            // process input and output operations will be added
            // with their root operation to outlining instance
            continue;
        }

        // Building outlining instance CASE 1
        if (!instanceOps.empty() && isVfOutliningInstanceCandidate(&op)) {
            // operation exist in outlining instance and current op is target for outlining
            // need to add new instance from the previous operations in storage
            // and instance for current op will be build in CASE 3
            moveOpsToInstance(instanceOps, opInstances);
        }

        // Building outlining instance CASE 2
        if (mlir::isa<mlir::func::ReturnOp>(op)) {
            // recurse and add input ops
            processInputsRecursively(op, instanceOps);
            if (!instanceOps.empty()) {
                // add final outlining instance for operations linked to results of function
                moveOpsToInstance(instanceOps, opInstances);
            }
            // result op does not need to be added, new ops will be created during outlining
            continue;
        }

        // add operation(s) to storage of current outlining instance
        processInputsRecursively(op, instanceOps);
        processOperation(op, instanceOps);
        processOutputsRecursively(op, instanceOps);

        // Building outlining instance CASE 3
        if (isVfOutliningInstanceCandidate(&op)) {
            // current op is target for outlining, add new instance
            moveOpsToInstance(instanceOps, opInstances, true);
        }
    }

    // convert to outlining instances
    SmallVector<OutliningInstance> instances;
    OpOrderedSet storedOperations;
    ValueOrderedSet storedInputs;
    ValueOrderedSet storedOutputs;

    for (auto& opInstance : opInstances) {
        storedOperations.insert(opInstance.begin(), opInstance.end());
        for (auto& op : storedOperations) {
            for (auto operand : op->getOperands()) {
                storedInputs.insert(operand);
            }
            for (auto result : op->getResults()) {
                storedOutputs.insert(result);
            }
        }
        createOutliningInstanceFromStorage(storedInputs, storedOutputs, storedOperations, instances);
    }

    return instances;
}

void VerticalFusionOutliningPass::buildFuncOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                                               ArrayRef<OutliningInstance> outlinedTargets) {
    IE::CNNNetworkOp netInfo;
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

    auto builder = mlir::OpBuilder(moduleOp.getBodyRegion());
    builder.setInsertionPoint(netFunc);

    auto* ctx = moduleOp.getContext();
    for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
        const auto& slice = slices.front();
        const size_t sliceIdx = 0;
        const auto funcType = mlir::FunctionType::get(ctx, ArrayRef(funcsInfo[targetIdx][sliceIdx].inputTypes),
                                                      ArrayRef(funcsInfo[targetIdx][sliceIdx].outputTypes));
        const auto funcLoc = appendLoc(netFunc.getLoc(), "_part{0}", targetIdx + 1);
        auto func = builder.create<mlir::func::FuncOp>(funcLoc, funcsInfo[targetIdx][sliceIdx].funcNames, funcType);
        func.setPrivate();
        OpBuilderLogger builderLog(_log.nest());
        auto builder = mlir::OpBuilder::atBlockEnd(func.addEntryBlock(), &builderLog);

        mlir::DenseMap<mlir::Value, mlir::Value> oldToNewMap;
        for (size_t i = 0; i < slice.inputs.size(); i++) {
            oldToNewMap[slice.inputs[i]] = func.getArgument(i);
        }
        for (const auto op : slice.operations) {
            mlir::IRMapping mapper;
            for (auto operand : op->getOperands()) {
                mapper.map(operand, oldToNewMap[operand]);
            }
            auto clonedOp = builder.clone(*op, mapper);

            for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
                oldToNewMap[op->getResult(i)] = clonedOp->getResult(i);
            }
        }
        SmallVector<mlir::Value> funcOutputFromSlices;
        for (const auto output : slice.outputs) {
            funcOutputFromSlices.push_back(oldToNewMap[output]);
        }
        const auto returnLoc = appendLoc(netFunc.getLoc(), "_part{0}_return", targetIdx + 1);
        builder.create<mlir::func::ReturnOp>(returnLoc, funcOutputFromSlices);
    }
}

void VerticalFusionOutliningPass::buildCallOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                                               ArrayRef<OutliningInstance> outlinedTargets) {
    IE::CNNNetworkOp netInfo;
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(&netFunc.getBody().front(), &builderLog);
    DenseMap<mlir::Value, mlir::Value> oldToNewArgMap;
    for (const auto& arg : netFunc.getArguments()) {
        oldToNewArgMap[arg] = arg;
    }

    for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
        const auto& slice = slices.front();
        const size_t sliceIdx = 0;

        SmallVector<mlir::Value> newInputs;
        for (const auto input : slice.inputs) {
            newInputs.push_back(oldToNewArgMap[input]);
        }

        const auto callLoc = appendLoc(netFunc.getLoc(), "_part{0}_call", targetIdx + 1);
        auto newCall = builder.create<mlir::func::CallOp>(callLoc, funcsInfo[targetIdx][sliceIdx].funcNames,
                                                          funcsInfo[targetIdx][sliceIdx].outputTypes, newInputs);
        for (const auto& res : newCall.getResults()) {
            size_t idx = res.getResultNumber();
            oldToNewArgMap[slice.outputs[idx]] = res;
        }
    }
    netFunc.walk([&](mlir::func::ReturnOp ret) {
        for (auto i : irange(ret.getNumOperands())) {
            ret.setOperand(i, oldToNewArgMap[ret.getOperand(i)]);
        }
    });
}

VerticalFusionOutliningPass::VerticalFusionOutliningPass(const VPU::TilingOptions& TilingOptions, Logger log) {
    Base::initLogger(log, Base::getArgumentName());
    Base::copyOptionValuesFrom(TilingOptions);

    initializeFromOptions();
}

mlir::LogicalResult VerticalFusionOutliningPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    initializeFromOptions();

    return mlir::success();
}

void VerticalFusionOutliningPass::initializeFromOptions() {
    if (verticalFusionTileThreshold.hasValue()) {
        _verticalFusionTileThreshold = verticalFusionTileThreshold.getValue();
    }

    if (numInstanceThreshold.hasValue()) {
        _numInstanceThreshold = numInstanceThreshold.getValue();
    }
}

//
// safeRunOnModule
//

void VerticalFusionOutliningPass::safeRunOnModule() {
    IE::CNNNetworkOp netInfo;
    mlir::func::FuncOp netFunc;
    auto moduleOp = getOperation();
    IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

    // TODO E#150569: remove this condition when this pass is compatible with pre-outlined functions
    bool containsCallOps = false;
    netFunc.walk([&](mlir::Operation* op) {
        if (mlir::isa<mlir::func::CallOp>(op)) {
            containsCallOps = true;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    if (containsCallOps) {
        _log.info("The main function already contains call operations. Skipping pass");
        return;
    }

    _log.trace("Searching for outlining instances separated by vertical fusion regions");
    const auto outliningInstances = getOutliningInstances(netFunc, _verticalFusionTileThreshold);
    if (static_cast<int64_t>(outliningInstances.size()) < _numInstanceThreshold) {
        _log.trace("Can not perform vertical fusion outlining");
        return;
    }

    _log.info("Creating {0} functions", outliningInstances.size());

    SmallVector<SmallVector<FuncInfo>> funcsInfo(outliningInstances.size());
    for (const auto& [targetIdx, slices] : outliningInstances | indexed) {
        const auto slice = slices.front();
        SmallVector<mlir::Type> inputTypes;
        SmallVector<mlir::Type> outputTypes;
        for (const auto input : slice.inputs) {
            inputTypes.push_back(input.getType());
        }
        for (const auto output : slice.outputs) {
            outputTypes.push_back(output.getType());
        }
        auto funcName = printToString("{0}_{1}{2}", netFunc.getName(), "vf", targetIdx + 1);
        _log.trace("Build function with name {0}", funcName);
        funcsInfo[targetIdx].push_back({std::move(inputTypes), std::move(outputTypes), std::move(funcName)});
    }

    buildFuncOps(moduleOp, funcsInfo, outliningInstances);
    buildCallOps(moduleOp, funcsInfo, outliningInstances);
}

}  // namespace

//
// createVerticalFusionOutliningPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createVerticalFusionOutliningPass() {
    return std::make_unique<VerticalFusionOutliningPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPU::createVerticalFusionOutliningPass(const TilingOptions& TilingOptions,
                                                                         Logger log) {
    return std::make_unique<VerticalFusionOutliningPass>(TilingOptions, log);
}
