//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_LOOPOUTLINER
#define GEN_PASS_DEF_LOOPOUTLINER
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// LoopOutliner
//

class LoopOutliner final : public IE::impl::LoopOutlinerBase<LoopOutliner> {
public:
    explicit LoopOutliner(Logger log): _log(std::move(log)) {
        _log.setName("loop-outliner");
    }

protected:
    Logger getLogger() const {
        return _log;
    }

private:
    void safeRunOnModule() final;
    void collectLoopInfo(mlir::func::FuncOp function, SmallVector<IE::LoopOp>& loopList,
                         SmallVector<SmallVector<mlir::Type>>& inputTypesList,
                         SmallVector<SmallVector<mlir::Type>>& outputTypesList);
    void createOutlinedFunction(mlir::MLIRContext* ctx, mlir::func::FuncOp netFunc, mlir::OpBuilder& builder,
                                OpBuilderLogger& builderLog, size_t targetIdx, IE::LoopOp origLoop,
                                ArrayRef<mlir::Type> inputTypes, ArrayRef<mlir::Type> outputTypes);
    void cloneLoopBody(mlir::OpBuilder& funcBuilder, mlir::Region* bodyModule,
                       mlir::DenseMap<mlir::Value, mlir::Value>& oldToNewMap);
    void createReturnOp(mlir::OpBuilder& funcBuilder, mlir::func::FuncOp netFunc, size_t targetIdx,
                        IE::LoopTerminatorOp loopTerminator, mlir::DenseMap<mlir::Value, mlir::Value>& oldToNewMap,
                        int64_t execCondIndex, bool skipConstCond);
    void createCallOp(mlir::func::FuncOp netFunc, OpBuilderLogger& builderLog, mlir::Block* bodyBlock, size_t targetIdx,
                      const std::string& funcName, ArrayRef<mlir::Type> outputTypes,
                      ArrayRef<mlir::BlockArgument> bodyInputs, IE::LoopTerminatorOp loopTerminator,
                      int64_t execCondIndex, bool skipConstCond);
    bool isLoopBodyInternalExecCondConstAndOneUse(IE::LoopOp& origLoop);
    Logger _log;
};

//
// safeRunOnModule
//

void LoopOutliner::safeRunOnModule() {
    auto moduleOp = getOperation();
    IE::CNNNetworkOp netInfo;
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);
    auto* ctx = moduleOp.getContext();

    SmallVector<IE::LoopOp> loopList;
    SmallVector<SmallVector<mlir::Type>> inputTypesList;
    SmallVector<SmallVector<mlir::Type>> outputTypesList;

    collectLoopInfo(netFunc, loopList, inputTypesList, outputTypesList);

    auto builder = mlir::OpBuilder(moduleOp.getBodyRegion());
    builder.setInsertionPoint(netFunc);
    OpBuilderLogger builderLog(getLogger().nest());

    for (const auto& [targetIdx, origLoop] : loopList | indexed) {
        createOutlinedFunction(ctx, netFunc, builder, builderLog, targetIdx, origLoop, inputTypesList[targetIdx],
                               outputTypesList[targetIdx]);
    }
}

void LoopOutliner::collectLoopInfo(mlir::func::FuncOp function, SmallVector<IE::LoopOp>& loopList,
                                   SmallVector<SmallVector<mlir::Type>>& inputTypesList,
                                   SmallVector<SmallVector<mlir::Type>>& outputTypesList) {
    function->walk([&](IE::LoopOp loopOp) {
        loopList.push_back(loopOp);
        auto bodyBlock = &loopOp.getBodyModule().getBlocks().front();
        auto bodyInputs = bodyBlock->getArguments();
        SmallVector<mlir::Type> inputTypes;
        SmallVector<mlir::Type> outputTypes;
        for (const auto input : bodyInputs) {
            inputTypes.push_back(input.getType());
        }
        auto blockOutput = mlir::cast<IE::LoopTerminatorOp>(bodyBlock->getTerminator()).getOperands();
        const bool skipConstCond = isLoopBodyInternalExecCondConstAndOneUse(loopOp);
        const auto execCondIndex = loopOp.getExecCondIndex();
        for (const auto& [idx, output] : blockOutput | indexed) {
            if (skipConstCond && execCondIndex == static_cast<int64_t>(idx)) {
                continue;
            }
            outputTypes.push_back(output.getType());
        }
        inputTypesList.push_back(std::move(inputTypes));
        outputTypesList.push_back(std::move(outputTypes));
    });
}

void LoopOutliner::createOutlinedFunction(mlir::MLIRContext* ctx, mlir::func::FuncOp netFunc, mlir::OpBuilder& builder,
                                          OpBuilderLogger& builderLog, size_t targetIdx, IE::LoopOp origLoop,
                                          ArrayRef<mlir::Type> inputTypes, ArrayRef<mlir::Type> outputTypes) {
    const auto funcType = mlir::FunctionType::get(ctx, inputTypes, outputTypes);
    const auto funcLoc = appendLoc(netFunc.getLoc(), "loop_body{0}", targetIdx + 1);
    const auto funcName = printToString("{0}_{1}{2}", netFunc.getName(), "loop_body", targetIdx + 1);
    auto func = builder.create<mlir::func::FuncOp>(funcLoc, funcName, funcType);
    func.setPrivate();
    auto funcBuilder = mlir::OpBuilder::atBlockEnd(func.addEntryBlock(), &builderLog);

    auto bodyModule = &origLoop.getBodyModule();
    auto bodyBlock = &bodyModule->getBlocks().front();
    auto bodyInputs = bodyBlock->getArguments();
    const auto execCondIndex = origLoop.getExecCondIndex();
    const bool skipConstCond = isLoopBodyInternalExecCondConstAndOneUse(origLoop);
    const auto loopTerminator = mlir::cast<IE::LoopTerminatorOp>(bodyBlock->getTerminator());
    mlir::DenseMap<mlir::Value, mlir::Value> oldToNewMap;
    for (size_t i = 0; i < bodyInputs.size(); i++) {
        oldToNewMap[bodyInputs[i]] = func.getArgument(i);
    }

    cloneLoopBody(funcBuilder, bodyModule, oldToNewMap);

    createReturnOp(funcBuilder, netFunc, targetIdx, loopTerminator, oldToNewMap, execCondIndex, skipConstCond);

    createCallOp(netFunc, builderLog, bodyBlock, targetIdx, funcName, outputTypes, bodyInputs, loopTerminator,
                 execCondIndex, skipConstCond);
}

void LoopOutliner::cloneLoopBody(mlir::OpBuilder& funcBuilder, mlir::Region* bodyModule,
                                 mlir::DenseMap<mlir::Value, mlir::Value>& oldToNewMap) {
    for (auto& op : bodyModule->getOps()) {
        if (mlir::isa<IE::LoopTerminatorOp>(op)) {
            continue;
        }
        mlir::IRMapping mapper;
        for (auto operand : op.getOperands()) {
            mapper.map(operand, oldToNewMap[operand]);
        }
        auto newOp = funcBuilder.clone(op, mapper);

        for (size_t i = 0; i < newOp->getResults().size(); i++) {
            oldToNewMap[op.getResult(i)] = newOp->getResult(i);
        }
    }
}

void LoopOutliner::createReturnOp(mlir::OpBuilder& funcBuilder, mlir::func::FuncOp netFunc, size_t targetIdx,
                                  IE::LoopTerminatorOp loopTerminator,
                                  mlir::DenseMap<mlir::Value, mlir::Value>& oldToNewMap, int64_t execCondIndex,
                                  bool skipConstCond) {
    SmallVector<mlir::Value> funcOutputFromBody;
    for (const auto& [idx, output] : loopTerminator.getOperands() | indexed) {
        if (execCondIndex == static_cast<int64_t>(idx) && skipConstCond) {
            continue;
        }
        funcOutputFromBody.push_back(oldToNewMap[output]);
    }
    const auto returnLoc = appendLoc(netFunc.getLoc(), "_loop_body{0}_return", targetIdx + 1);
    funcBuilder.create<mlir::func::ReturnOp>(returnLoc, funcOutputFromBody);
}

void LoopOutliner::createCallOp(mlir::func::FuncOp netFunc, OpBuilderLogger& builderLog, mlir::Block* bodyBlock,
                                size_t targetIdx, const std::string& funcName, ArrayRef<mlir::Type> outputTypes,
                                ArrayRef<mlir::BlockArgument> bodyInputs, IE::LoopTerminatorOp loopTerminator,
                                int64_t execCondIndex, bool skipConstCond) {
    auto callbuilder = mlir::OpBuilder::atBlockBegin(bodyBlock, &builderLog);

    SmallVector<mlir::Value> newInputs;
    for (const auto input : bodyInputs) {
        newInputs.push_back(input);
    }

    const auto callLoc = appendLoc(netFunc.getLoc(), "_loop_body{0}_call", targetIdx + 1);
    auto newCall = callbuilder.create<mlir::func::CallOp>(callLoc, funcName, outputTypes, newInputs);
    for (const auto& [idx, res] : newCall.getResults() | indexed) {
        auto index = static_cast<int64_t>(idx);
        index = (skipConstCond && index >= execCondIndex) ? index + 1 : index;
        loopTerminator.setOperand(index, res);
    }
}

bool LoopOutliner::isLoopBodyInternalExecCondConstAndOneUse(IE::LoopOp& loopOp) {
    const auto execCondIndex = loopOp.getExecCondIndex();
    auto bodyBlock = &loopOp.getBodyModule().getBlocks().front();
    auto blockOutput = mlir::cast<IE::LoopTerminatorOp>(bodyBlock->getTerminator()).getOperands();
    auto condOutput = blockOutput[execCondIndex];
    if (auto constExecCond = condOutput.getDefiningOp<Const::DeclareOp>()) {
        return constExecCond.getOutput().hasOneUse();
    }
    return false;
}

}  // namespace

//
// createLoopOutlinerPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createLoopOutlinerPass(Logger log) {
    return std::make_unique<LoopOutliner>(log);
}
