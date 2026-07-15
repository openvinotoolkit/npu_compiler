//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/core/format.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/RegionUtils.h>

namespace vpux::IE {
#define GEN_PASS_DECL_OUTLINER
#define GEN_PASS_DEF_OUTLINER
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {
// It is possible for an instance of a repeating block to only contain a subset of the
// operations found in a slice of the IR. When that happens, operations interleaved with
// those from the instance may need to be moved after the newly-inserted call operation.
// Walk in program order: if an op needs to move past its producer, any of its users that
// also need reordering appear later in opsToReorder and will be handled in this same loop.
void reorderOperations(SmallVector<mlir::Operation*> opsToReorder) {
    for (auto* op : opsToReorder) {
        for (auto operand : op->getOperands()) {
            auto* producerOp = operand.getDefiningOp();
            if (producerOp == nullptr) {
                continue;
            }
            if (op->isBeforeInBlock(producerOp)) {
                op->moveAfter(producerOp);
            }
        }
    }
}
}  // namespace

namespace outliner {

//
// DefaultOutliner
//

class DefaultOutliner final : public OutlinerBase {
public:
    DefaultOutliner(std::unique_ptr<IFunctionOutliner> splitter, const Logger& log, std::string prefix)
            : OutlinerBase(std::move(splitter), log), _prefix(std::move(prefix)) {
    }

    static constexpr StringRef name() {
        return "default";
    }

private:
    std::string _prefix;

    void buildFuncOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override {
        auto netFunc = net::getMainFunc(moduleOp);

        auto builder = mlir::OpBuilder(moduleOp.getBodyRegion());
        builder.setInsertionPoint(netFunc);

        auto* ctx = moduleOp.getContext();
        for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
            const auto& slice = slices.front();
            const size_t sliceIdx = 0;
            const auto funcType = mlir::FunctionType::get(ctx, ArrayRef(funcsInfo[targetIdx][sliceIdx].inputTypes),
                                                          ArrayRef(funcsInfo[targetIdx][sliceIdx].outputTypes));
            const auto funcLoc = appendLoc(netFunc.getLoc(), "{0}{1}", _prefix, targetIdx + 1);
            auto func = builder.create<mlir::func::FuncOp>(funcLoc, funcsInfo[targetIdx][sliceIdx].funcName, funcType);
            func.setNested();

            OpBuilderLogger builderLog(getLogger().nest());
            auto builder = mlir::OpBuilder::atBlockEnd(func.addEntryBlock(), &builderLog);

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

                // Override the connection from the block arguments of the function to the user operation if it was
                // explicitly mapped by the analysis. This helps cover the case where the first instance (the one being
                // cloned) has some inputs reused, while the other instances may use separate arguments
                // E.g.: %0 = ...
                //       %1 = Add(%0, %0)  // instance 1
                //       %2 = Add(%0, %1)  // instance 2
                // The outlined function containing Add should have two operands, each connected to one operand. The
                // first call will pass the same value twice, while the second instance will pass different values
                if (!slice.inputUserMapping.empty()) {
                    for (auto& operand : clonedOp->getOpOperands()) {
                        if (!mlir::isa<mlir::BlockArgument>(operand.get())) {
                            continue;
                        }
                        const auto inputMappingIt = llvm::find_if(
                                slice.inputUserMapping, [&](const std::pair<mlir::Operation*, size_t>& user) {
                                    return user.first == op && user.second == operand.getOperandNumber();
                                });
                        if (inputMappingIt == slice.inputUserMapping.end()) {
                            continue;
                        }
                        const auto argIdx = std::distance(slice.inputUserMapping.begin(), inputMappingIt);
                        clonedOp->setOperand(operand.getOperandNumber(), func.getArgument(argIdx));
                    }
                }

                // The input pre-processing operations might be duplicated in multiple functions depending on how
                // their results are used, so their location is customized during cloning
                extendOpLoc(clonedOp, "{0}_{1}", _prefix, targetIdx + 1);
                for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
                    oldToNewMap[op->getResult(i)] = clonedOp->getResult(i);
                }
            }

            SmallVector<mlir::Value> funcOutputFromSlices;
            for (const auto output : slice.outputs) {
                funcOutputFromSlices.push_back(oldToNewMap[output]);
            }
            const auto returnLoc = appendLoc(netFunc.getLoc(), "{0}{1}_return", _prefix, targetIdx + 1);
            builder.create<mlir::func::ReturnOp>(returnLoc, funcOutputFromSlices);
        }
    }

    void buildCallOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override {
        auto netFunc = net::getMainFunc(moduleOp);

        OpBuilderLogger builderLog(getLogger().nest());
        auto builder = mlir::OpBuilder::atBlockBegin(&netFunc.getBody().front(), &builderLog);
        DenseMap<mlir::Value, mlir::Value> oldToNewArgMap;

        SmallVector<mlir::Value> prevOutput;
        for (const auto& arg : netFunc.getArguments()) {
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

                const auto callLoc = appendLoc(netFunc.getLoc(), "{0}_{1}_call_{2}", _prefix, targetIdx + 1, sliceIdx);
                auto newCall = builder.create<mlir::func::CallOp>(callLoc, funcsInfo[targetIdx].front().funcName,
                                                                  funcsInfo[targetIdx].front().outputTypes, newInputs);
                for (const auto& res : newCall.getResults()) {
                    size_t idx = res.getResultNumber();
                    oldToNewArgMap[slice.outputs[idx]] = res;
                }
            }
        }

        SmallVector<mlir::Operation*> opsToReorder;
        netFunc.walk([&](mlir::Operation* op) {
            bool changedOperands = false;
            for (auto i : irange(op->getNumOperands())) {
                if (oldToNewArgMap.find(op->getOperand(i)) != oldToNewArgMap.end()) {
                    op->setOperand(i, oldToNewArgMap[op->getOperand(i)]);
                    changedOperands = true;
                }
            }
            if (changedOperands) {
                opsToReorder.push_back(op);
            }
        });
        reorderOperations(std::move(opsToReorder));
    }

    void updateMainFuncOp(mlir::ModuleOp moduleOp, ArrayRef<OutliningInstance> outlinedTargets) override {
        VPUX_UNUSED(outlinedTargets);
        auto netFunc = net::getMainFunc(moduleOp);
        vpux::runLocalDCE(netFunc);
    }
};

//
// Batching
//

class Batching final : public OutlinerBase {
public:
    Batching(const Logger& log): OutlinerBase(std::make_unique<vpux::IE::FunctionOutlinerBatching>(log), log) {
    }

    static constexpr StringRef name() {
        return "batching";
    }

private:
    void buildFuncOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override {
        auto netFunc = net::getMainFunc(moduleOp);

        auto builder = mlir::OpBuilder(moduleOp.getBodyRegion());
        builder.setInsertionPoint(netFunc);

        auto* ctx = moduleOp.getContext();
        for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
            const auto& slice = slices.front();
            const size_t sliceIdx = 0;
            const auto funcLoc = appendLoc(netFunc.getLoc(), "fn{0}_block{1}", targetIdx + 1, sliceIdx + 1);
            const auto funcType = mlir::FunctionType::get(ctx, ArrayRef(funcsInfo[targetIdx][sliceIdx].inputTypes),
                                                          ArrayRef(funcsInfo[targetIdx][sliceIdx].outputTypes));
            auto func = builder.create<mlir::func::FuncOp>(funcLoc, funcsInfo[targetIdx][sliceIdx].funcName, funcType);
            func.setNested();

            OpBuilderLogger builderLog(getLogger().nest());
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
                // The input pre-processing operations might be duplicated in multiple functions depending on how
                // their results are used, so their location is customized during cloning
                extendOpLoc(clonedOp, "fn_{0}", targetIdx + 1);
                for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
                    oldToNewMap[op->getResult(i)] = clonedOp->getResult(i);
                }
            }
            SmallVector<mlir::Value> funcOutputFromSlices;
            for (const auto output : slice.outputs) {
                funcOutputFromSlices.push_back(oldToNewMap[output]);
            }
            const auto returnLoc = appendLoc(netFunc.getLoc(), "fn_{0}_block_{1}_return", targetIdx + 1, sliceIdx + 1);
            builder.create<mlir::func::ReturnOp>(returnLoc, funcOutputFromSlices);
        }
    }

    void buildCallOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override {
        auto netFunc = net::getMainFunc(moduleOp);

        OpBuilderLogger builderLog(getLogger().nest());
        auto builder = mlir::OpBuilder::atBlockBegin(&netFunc.getBody().front(), &builderLog);
        DenseMap<mlir::Value, mlir::Value> oldToNewArgMap;

        SmallVector<mlir::Value> prevOutput;
        for (const auto& arg : netFunc.getArguments()) {
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

                const auto callLoc = appendLoc(netFunc.getLoc(), "fn_{0}_call_{1}", targetIdx + 1, sliceIdx);
                auto newCall =
                        builder.create<mlir::func::CallOp>(callLoc, funcsInfo[targetIdx][sliceIdx].funcName,
                                                           funcsInfo[targetIdx][sliceIdx].outputTypes, newInputs);
                for (const auto& res : newCall.getResults()) {
                    size_t idx = res.getResultNumber();
                    oldToNewArgMap[slice.outputs[idx]] = res;
                }
            }
        }
        netFunc.walk([&](mlir::Operation* op) {
            for (auto i : irange(op->getNumOperands())) {
                if (oldToNewArgMap.find(op->getOperand(i)) != oldToNewArgMap.end()) {
                    op->setOperand(i, oldToNewArgMap[op->getOperand(i)]);
                }
            }
        });
    }
};

}  // namespace outliner

namespace {

//
// OutlinerPass
//

class OutlinerPass final : public IE::impl::OutlinerBase<OutlinerPass> {
public:
    explicit OutlinerPass(const DefaultHWOptionsBase& outlingOptions, const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(outlingOptions);

        _options = vpux::IE::OutlinerPassOptions::createFromString(functionOutlining);
    }

    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;

private:
    void safeRunOnModule() final;

private:
    std::string _mode;
    vpux::IE::OutlinerPassOptions _options;
};

mlir::LogicalResult OutlinerPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    _log.info("initializeOptions");
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }

    // Throws an exception if functionOutlining is not a valid string.
    _options = vpux::IE::OutlinerPassOptions::createFromString(functionOutlining);

    return mlir::success();
}

//
// safeRunOnModule
//

void OutlinerPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    for (size_t i = 0; i < _options.count(); ++i) {
        if (i >= 1) {
            _log.warning("Execution of fallback outliner solutions is not yet implemented!");
            break;
        }

        if (const auto* opt = _options.getIf<vpux::IE::NaiveOptions>(i)) {
            outliner::DefaultOutliner outliner(std::make_unique<vpux::IE::FunctionOutlinerNaive>(opt->numParts, _log),
                                               _log, "part");
            outliner.outline(moduleOp, "part");
        } else if (const auto* opt = _options.getIf<vpux::IE::RepeatingBlocksOptions>(i)) {
            outliner::DefaultOutliner outliner(
                    std::make_unique<vpux::IE::FunctionOutlinerRepeatingBlocks>(
                            opt->minOpsInBlock, opt->maxNumIterations, opt->weightsAsInputs, _log),
                    _log, "fn");
            if (outliner.outline(moduleOp, "fn")) {
                outliner::DefaultOutliner exhaustive(std::make_unique<vpux::IE::FunctionOutlinerExhaustive>(_log), _log,
                                                     "rest");
                exhaustive.outline(moduleOp, "rest");
            }
        } else if (const auto* opt = _options.getIf<vpux::IE::BatchingOptions>(i)) {
            std::ignore = opt;
            if (_options.count() != 1) {
                mlir::emitError(moduleOp->getLoc(),
                                printToString("Outliner must not use \"batching\" outlining together with other "
                                              "options, total options count: {0}",
                                              _options.count()));
                signalPassFailure();
                return;
            }

            if (!config::hasCompileMethodDebatch(moduleOp)) {
                _log.info("{0} ignores \"batching\" outlining as \"config.debatch\" wasn't found", getName());
                return;
            }
            outliner::Batching outliner(_log);
            outliner.outline(moduleOp, "batching");
        } else {
            llvm_unreachable("Validity of mode should have been checked in pass initialization!");
        }
    }
}

}  // namespace

//
// createOutlinerPass
//
std::unique_ptr<mlir::Pass> vpux::IE::createOutlinerPass(const DefaultHWOptionsBase& outlingOptions, Logger log) {
    return std::make_unique<OutlinerPass>(outlingOptions, log);
}

std::unique_ptr<mlir::Pass> vpux::IE::createOutlinerPass(Logger log) {
    return createOutlinerPass(DefaultHWOptionsBase{}, log);
}
