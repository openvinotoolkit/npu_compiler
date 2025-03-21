//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/hash.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/format.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/RegionUtils.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_CONCATREPEATINGBLOCKSOUTLINING
#define GEN_PASS_DEF_CONCATREPEATINGBLOCKSOUTLINING
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

struct FuncInfo {
    SmallVector<mlir::Type> inputTypes;
    SmallVector<mlir::Type> outputTypes;
    std::string funcNames;
};

using FunctionCalls = std::map<mlir::func::FuncOp, std::vector<mlir::func::CallOp>>;
using CallFunction = std::map<mlir::func::CallOp, mlir::func::FuncOp>;

using InputBranch = std::unordered_map<size_t, SmallVector<mlir::Operation*>>;

//
// ConcatOutliner
//

class ConcatOutliner final {
private:
    int64_t _minSeqLength;
    bool _singleFunctionPerConcat;
    bool _wrapMainIntoCallOps = true;
    Logger _log;

public:
    ConcatOutliner(int64_t minSeqLength, bool singleFunctionPerConcat, const Logger& log)
            : _minSeqLength(minSeqLength), _singleFunctionPerConcat(singleFunctionPerConcat), _log(log) {
    }

    SmallVector<OutliningInstance> getOutliningInstances(mlir::func::FuncOp funcOp) {
        SmallVector<OutliningInstance> outliningInstances;

        const auto& concatOps = funcOp.getOps<VPU::ConcatOp>();
        for (const auto& concatOp : concatOps) {
            _log.trace("Got {0} at '{1}", concatOp->getName(), concatOp->getLoc());

            auto inputBranches = findParallelRepeatingBranches(concatOp);
            if (!areInputBranchesValid(inputBranches)) {
                _log.nest().trace("The input branches are not valid for outlining. Skipping operation");
                continue;
            }
            _log.nest().trace("Creating outlining instance");
            auto outliningInstance = createOutliningInstance(inputBranches);
            outliningInstances.push_back(outliningInstance);
        }

        _log.trace("Found {0} outlining instances", outliningInstances.size());

        if (_wrapMainIntoCallOps) {
            _log.trace("Wrapping remaining operations into call ops");
            return wrapRemainingOpsIntoCallOps(funcOp, outliningInstances);
        }

        return outliningInstances;
    }

private:
    /**
     * @brief Find the input branches of the concat operation if they have the same structure
     * @details Iterates over every input branch of the concat and collects the parts that are identical across all
     * branches. In case the branches start to diverge at some point, only the identical parts will be taken into
     * consideration. The constants will be excluded from the checks and the final branches will not contain them.
     */
    SmallVector<InputBranch> findParallelRepeatingBranches(VPU::ConcatOp concatOp) {
        const auto numConcatInputs = concatOp.getNumOperands();
        SmallVector<InputBranch> inputBranches(numConcatInputs);

        const auto isParentOpValid = [](mlir::Operation* parentOp) {
            if (parentOp == nullptr) {
                return false;
            }
            // Constants are not included into the branch search, as they are included separately later
            if (parentOp->hasTrait<mlir::OpTrait::ConstantLike>()) {
                return false;
            }
            // Note: it is possible for an operation to have all results / uses in the branch, so this condition could
            // be relaxed
            if (parentOp->getNumResults() > 1 || !parentOp->getResult(0).hasOneUse()) {
                return false;
            }
            return true;
        };
        const auto tryToAddLevel = [&](size_t level) {
            for (auto& inputBranch : inputBranches) {
                for (auto op : inputBranch[level - 1]) {
                    for (auto input : op->getOperands()) {
                        auto parentOp = input.getDefiningOp();
                        if (!isParentOpValid(parentOp)) {
                            continue;
                        }
                        inputBranch[level].push_back(parentOp);
                    }
                }
            }
        };
        const auto eraseLevel = [&](size_t level) {
            for (auto& inputBranch : inputBranches) {
                if (inputBranch.find(level) != inputBranch.end()) {
                    inputBranch.erase(level);
                }
            }
        };

        size_t level = 0;
        for (size_t inputIdx = 0; inputIdx < numConcatInputs; ++inputIdx) {
            if (auto parentOp = concatOp.getOperand(inputIdx).getDefiningOp()) {
                if (!isParentOpValid(parentOp)) {
                    continue;
                }
                // A concat can receive the same value multiple times. For such cases, the input is inserted only
                // once as the input branch is the same
                const auto isParentOpReused = llvm::any_of(inputBranches, [&](InputBranch& branch) {
                    if (branch[level].empty()) {
                        return false;
                    }
                    return parentOp == branch[level].front();
                });
                if (isParentOpReused) {
                    continue;
                }
                inputBranches[inputIdx][level].push_back(parentOp);
            }
        }

        inputBranches.erase(std::remove_if(inputBranches.begin(), inputBranches.end(),
                                           [&](const InputBranch& branch) {
                                               return branch.empty();
                                           }),
                            inputBranches.end());
        if (inputBranches.empty()) {
            return {};
        }

        // Arbitrary limit, to prevent the loop from running infinitely in case something goes wrong
        constexpr size_t maxNumLevels = 100;

        ++level;
        while (level < maxNumLevels) {
            tryToAddLevel(level);

            auto& firstBranch = inputBranches.front();
            if (firstBranch[level].empty()) {
                break;
            }

            const auto anyBranchDiffers =
                    std::any_of(inputBranches.begin() + 1, inputBranches.end(), [&](InputBranch& branch) {
                        if (branch[level].size() != firstBranch[level].size()) {
                            return true;
                        }
                        for (size_t opIdx = 0; opIdx < branch[level].size(); ++opIdx) {
                            if (hashOperation(firstBranch[level][opIdx]) != hashOperation(branch[level][opIdx])) {
                                return true;
                            }
                        }
                        return false;
                    });
            if (anyBranchDiffers) {
                eraseLevel(level);
                break;
            }

            ++level;
        }

        if (level >= maxNumLevels) {
            _log.nest().debug("Reached the maximum level {0}. It is possible that more operations could be outlined",
                              maxNumLevels);
        }

        return inputBranches;
    }

    bool areInputBranchesValid(ArrayRef<InputBranch> inputBranches) {
        if (inputBranches.size() < 2) {
            _log.nest().trace("Less than two branches were found");
            return false;
        }
        for (const auto& inputBranch : inputBranches | indexed) {
            int64_t numComputeOps = 0;
            for (const auto& [level, operations] : inputBranch.value()) {
                for (auto op : operations) {
                    if (!op->hasTrait<mlir::OpTrait::ConstantLike>() && !VPU::isPureViewOp(op)) {
                        ++numComputeOps;
                    }
                }
            }
            if (numComputeOps < _minSeqLength) {
                _log.nest().trace("Branch {0} has {1} compute operations. Minimum is {2}", inputBranch.index(),
                                  numComputeOps, _minSeqLength);
                return false;
            }
        }
        _log.nest().trace("Found {0} valid branches", inputBranches.size());
        return true;
    }

    void addInstanceInputsOutputs(IRSlice& instance) {
        SmallVector<mlir::Operation*> extraConstants;
        for (auto op : instance.operations) {
            if (op->hasTrait<mlir::OpTrait::ConstantLike>()) {
                continue;
            }

            // Add external operands as inputs to the instance
            for (auto operand : op->getOperands()) {
                const bool operandAlreadyCovered = llvm::find(instance.inputs, operand) != instance.inputs.end();
                if (operandAlreadyCovered) {
                    continue;
                }

                auto parentOp = operand.getDefiningOp();
                bool operandIsBlockArg = parentOp == nullptr;
                if (operandIsBlockArg) {
                    instance.inputs.push_back(operand);
                    continue;
                }
                bool parentIsOutsideInstance = llvm::find(instance.operations, parentOp) == instance.operations.end();
                if (parentIsOutsideInstance) {
                    // Include the constants that are used by the operations into the instance as well, so that function
                    // passes are able to read their value when necessary, after outlining
                    if (parentOp->hasTrait<mlir::OpTrait::ConstantLike>()) {
                        if (llvm::find(extraConstants, parentOp) == extraConstants.end()) {
                            extraConstants.push_back(parentOp);
                        }
                        continue;
                    }
                    instance.inputs.push_back(operand);
                }
            }

            // Add external results as outputs to the instance
            for (auto result : op->getResults()) {
                const bool resultAlreadyCovered = llvm::find(instance.outputs, result) != instance.outputs.end();
                if (resultAlreadyCovered) {
                    continue;
                }

                auto anyUserOutsideInstance = llvm::any_of(result.getUsers(), [&](mlir::Operation* userOp) {
                    return llvm::find(instance.operations, userOp) == instance.operations.end();
                });
                if (anyUserOutsideInstance) {
                    instance.outputs.push_back(result);
                }
            }
        }
        instance.operations.insert(instance.operations.begin(), extraConstants.begin(), extraConstants.end());
    }

    OutliningInstance createOutliningInstance(SmallVector<InputBranch>& inputBranches) {
        if (_singleFunctionPerConcat) {
            // Create a single slice that contains all of the input branches and the concat operation
            const auto getConcatOp = [&]() {
                auto& firstBranch = inputBranches.front();
                const auto firstOp = firstBranch[0].front();
                const auto concatOp = mlir::dyn_cast<VPU::ConcatOp>(*firstOp->getUsers().begin());
                VPUX_THROW_WHEN(concatOp == nullptr, "Missing Concat user");
                return concatOp;
            };

            IRSlice instance;
            for (auto& inputBranch : inputBranches) {
                for (int64_t level = static_cast<int64_t>(inputBranch.size()) - 1; level >= 0; --level) {
                    for (auto op : inputBranch[level]) {
                        instance.operations.push_back(op);
                    }
                }
            }
            instance.operations.push_back(getConcatOp());
            llvm::sort(instance.operations, [](auto* lhs, auto* rhs) {
                return lhs->isBeforeInBlock(rhs);
            });
            addInstanceInputsOutputs(instance);

            OutliningInstance newInstance;
            newInstance.push_back(std::move(instance));
            return newInstance;
        }

        // Create separate slices for each input branches
        OutliningInstance newInstance;
        for (auto& inputBranch : inputBranches) {
            IRSlice instance;
            for (int64_t level = static_cast<int64_t>(inputBranch.size()) - 1; level >= 0; --level) {
                for (auto op : inputBranch[level]) {
                    instance.operations.push_back(op);
                }
            }
            llvm::sort(instance.operations, [](auto* lhs, auto* rhs) {
                return lhs->isBeforeInBlock(rhs);
            });
            addInstanceInputsOutputs(instance);
            newInstance.push_back(std::move(instance));
        }
        return newInstance;
    }

    void collectNonOutlinedParents(mlir::Value operand, std::vector<mlir::Operation*>& parentOps,
                                   mlir::DenseSet<mlir::Operation*>& outlinedOps) {
        auto parentOp = operand.getDefiningOp();
        if (parentOp == nullptr) {
            return;
        }
        if (parentOp->hasTrait<mlir::OpTrait::ConstantLike>()) {
            return;
        }
        if (outlinedOps.find(parentOp) != outlinedOps.end()) {
            return;
        }
        parentOps.push_back(parentOp);
        outlinedOps.insert(parentOp);
        for (auto parentOperand : parentOp->getOperands()) {
            collectNonOutlinedParents(parentOperand, parentOps, outlinedOps);
        }
    }

    // There is a limitation in the scheduler, as it does not support scenarios where call operations are mixed with
    // non-call operations. As a workaround, the remaining non-call operations are moved from main into individual
    // functions and replaced with call operations. The end-result will have main only contain call operations
    SmallVector<OutliningInstance> wrapRemainingOpsIntoCallOps(mlir::func::FuncOp funcOp,
                                                               SmallVector<OutliningInstance>& outliningInstances) {
        if (outliningInstances.empty()) {
            return {};
        }

        mlir::DenseSet<mlir::Operation*> outlinedOpStorage;
        for (auto& instance : outliningInstances) {
            for (auto slice : instance) {
                outlinedOpStorage.insert(slice.operations.begin(), slice.operations.end());
            }
        }

        // For all instances, collect the non-outlined parent operations and store them into an outlining instance
        SmallVector<OutliningInstance> allOutliningInstances;
        for (auto& instance : outliningInstances) {
            std::vector<std::vector<mlir::Operation*>> perSliceParentOps(instance.size());
            for (const auto sliceIt : instance | indexed) {
                const auto& slice = sliceIt.value();
                for (auto input : slice.inputs) {
                    collectNonOutlinedParents(input, perSliceParentOps[sliceIt.index()], outlinedOpStorage);
                }
            }

            if (perSliceParentOps.empty()) {
                allOutliningInstances.push_back(instance);
                continue;
            }

            for (auto& sliceParentOps : llvm::make_early_inc_range(perSliceParentOps)) {
                if (sliceParentOps.empty()) {
                    continue;
                }
                llvm::sort(sliceParentOps, [](auto* lhs, auto* rhs) {
                    return lhs->isBeforeInBlock(rhs);
                });

                IRSlice slice;
                slice.operations = std::move(sliceParentOps);
                addInstanceInputsOutputs(slice);
                allOutliningInstances.push_back(OutliningInstance{std::move(slice)});
            }

            allOutliningInstances.push_back(instance);
        }

        // Also collect the parent operations of the return op and store them into an outlining instance
        const auto& returnOps = funcOp.getOps<mlir::func::ReturnOp>();
        std::vector<mlir::Operation*> resultParentOps;
        for (auto returnOp : returnOps) {
            for (auto operand : returnOp->getOperands()) {
                collectNonOutlinedParents(operand, resultParentOps, outlinedOpStorage);
            }
        }
        if (!resultParentOps.empty()) {
            llvm::sort(resultParentOps, [](auto* lhs, auto* rhs) {
                return lhs->isBeforeInBlock(rhs);
            });

            IRSlice slice;
            slice.operations = std::move(resultParentOps);
            addInstanceInputsOutputs(slice);
            allOutliningInstances.push_back(OutliningInstance{std::move(slice)});
        }

        _log.nest().trace("Found a total of {0} outlining instances", allOutliningInstances.size());

        if (allOutliningInstances.size() == 1 && allOutliningInstances.front().size() == 1) {
            _log.trace("Found a single slice to outline which would contain the entire main function. Skipping",
                       allOutliningInstances.size());
            return {};
        }

        return allOutliningInstances;
    }
};  // namespace

//
// ConcatRepeatingBlocksOutliningPass
//

class ConcatRepeatingBlocksOutliningPass final :
        public VPU::impl::ConcatRepeatingBlocksOutliningBase<ConcatRepeatingBlocksOutliningPass> {
private:
    int64_t _minSeqLength = 1;
    bool _singleFunctionPerConcat = true;

public:
    explicit ConcatRepeatingBlocksOutliningPass(int64_t minSeqLength, const Logger& log): _minSeqLength(minSeqLength) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final {
        if (mlir::failed(Base::initialize(ctx))) {
            return mlir::failure();
        }
        if (minSeqLength.hasValue()) {
            _minSeqLength = minSeqLength.getValue();
        }
        if (singleFunctionPerConcat.hasValue()) {
            _singleFunctionPerConcat = singleFunctionPerConcat.getValue();
        }
        return mlir::success();
    }

private:
    void safeRunOnModule() final {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp mainFuncOp;
        auto moduleOp = getOperation();
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, mainFuncOp);

        _log.debug("Searching for outlining instances around concat operations");
        ConcatOutliner outliner(_minSeqLength, _singleFunctionPerConcat, _log);
        const auto outliningInstances = outliner.getOutliningInstances(mainFuncOp);
        if (outliningInstances.empty()) {
            _log.debug("Found no candidate instances");
            return;
        }
        printOutliningInstances(outliningInstances);

        if (mlir::failed(validateOutliningInstances(outliningInstances))) {
            _log.debug("The outlined instances failed validation");
            return;
        }

        outlineTargets(moduleOp, mainFuncOp, outliningInstances);
    }

    mlir::LogicalResult validateOutliningInstances(ArrayRef<OutliningInstance> outliningInstances) {
        mlir::DenseSet<mlir::Operation*> visitedOutlinedOps;

        for (const auto& instanceIt : outliningInstances | indexed) {
            auto& instance = instanceIt.value();
            if (instance.empty()) {
                _log.debug("Instance {0} is empty", instanceIt.index());
                return mlir::failure();
            }

            for (const auto& sliceIt : instance | indexed) {
                const auto& slice = sliceIt.value();

                for (const auto& inputIt : slice.inputs | indexed) {
                    const auto& input = inputIt.value();

                    const auto parentOp = input.getDefiningOp();
                    if (parentOp == nullptr && !mlir::isa_and_nonnull<mlir::BlockArgument>(input)) {
                        _log.debug("Instance {0} -> slice {1}: input {2} is null", instanceIt.index(), sliceIt.index(),
                                   inputIt.index());
                        return mlir::failure();
                    }
                    if (parentOp != nullptr) {
                        const auto parentInsideSlice =
                                llvm::find(slice.operations, input.getDefiningOp()) != slice.operations.end();
                        if (parentInsideSlice) {
                            _log.debug("Instance {0} -> slice {1}: input {2} has parent inside slice",
                                       instanceIt.index(), sliceIt.index(), inputIt.index());
                            return mlir::failure();
                        }
                    }

                    const auto anyUserInSlice = llvm::any_of(input.getUsers(), [&](mlir::Operation* user) {
                        return llvm::find(slice.operations, user) != slice.operations.end();
                    });
                    if (!anyUserInSlice) {
                        _log.debug("Instance {0} -> slice {1}: input {2} has no users inside the slice",
                                   instanceIt.index(), sliceIt.index(), inputIt.index());
                        return mlir::failure();
                    }
                }

                for (const auto& outputIt : slice.outputs | indexed) {
                    const auto& output = outputIt.value();

                    const auto parentOp = output.getDefiningOp();
                    if (parentOp == nullptr && !mlir::isa_and_nonnull<mlir::BlockArgument>(output)) {
                        _log.debug("Instance {0} -> slice {1}: output {2} is null", instanceIt.index(), sliceIt.index(),
                                   outputIt.index());
                        return mlir::failure();
                    }
                    if (parentOp != nullptr) {
                        const auto parentOutsideSlice =
                                llvm::find(slice.operations, parentOp) == slice.operations.end();
                        if (parentOutsideSlice) {
                            _log.debug("Instance {0} -> slice {1}: output {2} has parent outside slice",
                                       instanceIt.index(), sliceIt.index(), outputIt.index());
                            return mlir::failure();
                        }
                    }

                    const auto anyUserOutsideSlice = llvm::any_of(output.getUsers(), [&](mlir::Operation* user) {
                        return llvm::find(slice.operations, user) == slice.operations.end();
                    });
                    if (!anyUserOutsideSlice) {
                        _log.debug("Instance {0} -> slice {1}: output {2} has no users outside the slice",
                                   instanceIt.index(), sliceIt.index(), outputIt.index());
                        return mlir::failure();
                    }
                }

                mlir::DenseSet<mlir::Operation*> sliceOps;
                for (auto op : slice.operations) {
                    if (op == nullptr) {
                        _log.debug("Instance {0} -> slice {1}: at least one operation is null", instanceIt.index(),
                                   sliceIt.index());
                        return mlir::failure();
                    }
                    const auto [it, wasInserted] = sliceOps.insert(op);
                    if (!wasInserted) {
                        _log.debug("Instance {0} -> slice {1}: operation {2} at {3} is duplicated within the slice",
                                   instanceIt.index(), sliceIt.index(), op->getName(), op->getLoc());
                        return mlir::failure();
                    }
                    const auto isOpDuplicated =
                            !op->hasTrait<mlir::OpTrait::ConstantLike>() && visitedOutlinedOps.contains(op);
                    if (isOpDuplicated) {
                        _log.debug(
                                "Instance {0} -> slice {1}: operation {2} at {3} is also duplicated in another slice",
                                instanceIt.index(), sliceIt.index(), op->getName(), op->getLoc());
                        return mlir::failure();
                    }
                    visitedOutlinedOps.insert(op);
                }
            }
        }
        return mlir::success();
    }

    void printOutliningInstances(ArrayRef<OutliningInstance> outliningInstances) {
        if (!_log.isActive(LogLevel::Debug)) {
            return;
        }
        _log.debug("Functions to outline: {0}", outliningInstances.size());
        for (auto& outliningInstance : outliningInstances) {
            _log.nest().debug("Number of instances in IR: {0}", outliningInstance.size());
            for (const auto& p : outliningInstance | indexed) {
                const auto& slice = p.value();
                _log.nest().debug("Instance {0}", p.index());
                _log.nest(2).debug("Input values: {0}", slice.inputs.size());
                for (auto input : slice.inputs) {
                    auto producerOp = input.getDefiningOp();
                    if (producerOp != nullptr) {
                        _log.nest(3).debug("{0} at {1}", producerOp->getName(), producerOp->getLoc());
                        continue;
                    }
                    _log.nest(3).debug("{0}", input);
                }
                _log.nest(2).debug("Output values: {0}", slice.outputs.size());
                for (auto output : slice.outputs) {
                    auto producerOp = output.getDefiningOp();
                    if (producerOp != nullptr) {
                        _log.nest(3).debug("{0} at {1}", producerOp->getName(), producerOp->getLoc());
                        continue;
                    }
                    _log.nest(3).debug("{0}", output);
                }
                _log.nest(2).debug("Number of operations in slice: {0}", slice.operations.size());
                for (auto op : slice.operations) {
                    _log.nest(3).debug("Operation {0} at {1}", op->getName(), op->getLoc());
                }
                if (!slice.inputUserMapping.empty()) {
                    _log.nest(2).debug("Input user mapping");
                    for (const auto& [argIdx, user] : slice.inputUserMapping | indexed) {
                        _log.nest(3).debug("Argument {0}, user operation {1}, operand {2}", argIdx,
                                           user.first->getName(), user.second);
                    }
                }
            }
        }
    }

    void outlineTargets(mlir::ModuleOp moduleOp, mlir::func::FuncOp mainFuncOp,
                        ArrayRef<OutliningInstance> outliningInstances) {
        size_t numFunctions = 0;
        for (const auto& instance : outliningInstances) {
            numFunctions += instance.size();
        }
        _log.info("Creating {0} functions", numFunctions);

        SmallVector<SmallVector<FuncInfo>> funcsInfo(outliningInstances.size());
        for (const auto& [targetIdx, slices] : outliningInstances | indexed) {
            for (const auto& [sliceIdx, slice] : slices | indexed) {
                SmallVector<mlir::Type> inputTypes;
                SmallVector<mlir::Type> outputTypes;
                for (const auto input : slice.inputs) {
                    inputTypes.push_back(input.getType());
                }
                for (const auto output : slice.outputs) {
                    outputTypes.push_back(output.getType());
                }
                auto funcName = _singleFunctionPerConcat
                                        ? printToString("{0}_concat{1}", mainFuncOp.getName(), targetIdx + 1)
                                        : printToString("{0}_concat{1}_input{2}", mainFuncOp.getName(), targetIdx + 1,
                                                        sliceIdx + 1);
                funcsInfo[targetIdx].push_back({std::move(inputTypes), std::move(outputTypes), std::move(funcName)});
            }
        }

        buildFuncOps(moduleOp, funcsInfo, outliningInstances);
        buildCallOps(moduleOp, funcsInfo, outliningInstances);
    }

    void buildFuncOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp mainFuncOp;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, mainFuncOp);

        auto builder = mlir::OpBuilder(moduleOp.getBodyRegion());
        builder.setInsertionPoint(mainFuncOp);

        auto* ctx = moduleOp.getContext();
        for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
            for (const auto& [sliceIdx, slice] : slices | indexed) {
                _log.trace("Creating func target {0}, slice {1}", targetIdx, sliceIdx);
                const auto funcLoc = appendLoc(mainFuncOp.getLoc(), "_concat{0}_input{1}", targetIdx + 1, sliceIdx + 1);
                const auto funcType = mlir::FunctionType::get(ctx, ArrayRef(funcsInfo[targetIdx][sliceIdx].inputTypes),
                                                              ArrayRef(funcsInfo[targetIdx][sliceIdx].outputTypes));
                auto func =
                        builder.create<mlir::func::FuncOp>(funcLoc, funcsInfo[targetIdx][sliceIdx].funcNames, funcType);
                func.setPrivate();

                auto builder = mlir::OpBuilder::atBlockEnd(func.addEntryBlock());

                DenseMap<mlir::Value, mlir::Value> oldToNewMap;
                for (size_t i = 0; i < slice.inputs.size(); i++) {
                    oldToNewMap[slice.inputs[i]] = func.getArgument(i);
                }
                for (const auto op : slice.operations) {
                    mlir::IRMapping mapper;
                    for (auto operand : op->getOperands()) {
                        mapper.map(operand, oldToNewMap[operand]);
                    }
                    auto clonedOp = builder.clone(*op, mapper);
                    clonedOp->setLoc(appendLoc(clonedOp->getLoc(),
                                               formatv("_concat{0}_input{1}", targetIdx + 1, sliceIdx + 1).str()));
                    for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
                        oldToNewMap[op->getResult(i)] = clonedOp->getResult(i);
                    }
                }

                SmallVector<mlir::Value> funcOutputFromSlices;
                for (const auto output : slice.outputs) {
                    funcOutputFromSlices.push_back(oldToNewMap[output]);
                }
                const auto returnLoc =
                        appendLoc(mainFuncOp.getLoc(), "_concat{0}_input{1}_return", targetIdx + 1, sliceIdx + 1);
                builder.create<mlir::func::ReturnOp>(returnLoc, funcOutputFromSlices);
            }
        }
    }

    void buildCallOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp mainFuncOp;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, mainFuncOp);

        auto builder = mlir::OpBuilder::atBlockBegin(&mainFuncOp.getBody().front());
        DenseMap<mlir::Value, mlir::Value> oldToNewArgMap;

        SmallVector<mlir::Value> prevOutput;
        for (const auto& arg : mainFuncOp.getArguments()) {
            oldToNewArgMap[arg] = arg;
        }

        for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
            for (const auto& [sliceIdx, slice] : slices | indexed) {
                SmallVector<mlir::Value> newInputs;
                for (const auto input : slice.inputs) {
                    if (oldToNewArgMap.contains(input)) {
                        newInputs.push_back(oldToNewArgMap[input]);
                    } else {
                        newInputs.push_back(input);
                    }
                    if (auto producerOp = newInputs.back().getDefiningOp()) {
                        if (!producerOp->isBeforeInBlock(&(*builder.getInsertionPoint()))) {
                            builder.setInsertionPointAfter(producerOp);
                        }
                    }
                }

                const auto callLoc = appendLoc(mainFuncOp.getLoc(), "_fn{0}_call{1}", targetIdx + 1, sliceIdx);
                auto newCall =
                        builder.create<mlir::func::CallOp>(callLoc, funcsInfo[targetIdx][sliceIdx].funcNames,
                                                           funcsInfo[targetIdx][sliceIdx].outputTypes, newInputs);
                for (const auto& res : newCall.getResults()) {
                    size_t idx = res.getResultNumber();
                    oldToNewArgMap[slice.outputs[idx]] = res;
                }
            }
        }
        mainFuncOp.walk([&](mlir::Operation* op) {
            for (auto i : irange(op->getNumOperands())) {
                if (oldToNewArgMap.find(op->getOperand(i)) != oldToNewArgMap.end()) {
                    op->setOperand(i, oldToNewArgMap[op->getOperand(i)]);
                }
            }
        });
        mainFuncOp.walk([&](mlir::Operation* op) {
            for (auto operand : op->getOperands()) {
                auto producerOp = operand.getDefiningOp();
                if (producerOp == nullptr) {
                    continue;
                }
                if (op->isBeforeInBlock(producerOp)) {
                    op->moveAfter(producerOp);
                }
            }
        });
    }
};

}  // namespace

//
// createConcatRepeatingBlocksOutliningPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createConcatRepeatingBlocksOutliningPass(int64_t minSeqLength,
                                                                                const Logger& log) {
    return std::make_unique<ConcatRepeatingBlocksOutliningPass>(minSeqLength, log);
}
