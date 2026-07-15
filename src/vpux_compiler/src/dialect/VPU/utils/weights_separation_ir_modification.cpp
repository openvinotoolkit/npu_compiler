//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/weights_separation_ir_modification.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"

using namespace vpux;

namespace vpux::VPU {

namespace {
std::vector<VPU::CallChainData> collectCallChains(mlir::func::FuncOp funcOp) {
    std::vector<VPU::CallChainData> functions;
    funcOp.walk([&](mlir::func::CallOp callOp) {
        functions.push_back({callOp, getCalledFunction(callOp)});
    });
    return functions;
}

std::vector<VPU::CallChainData> findChildren(const VPU::CallChainTree::Node& node) {
    auto funcOp = node.data().second;
    auto chains = collectCallChains(funcOp);
    // Note: sort call-chains lexicographically (using function names) to ensure
    // outlining-independent processing. while this disregards the call
    // sequence, this allows to avoid differences in schedule generation when
    // independent calls get reordered in IR:
    // ```cpp
    //  %call1 = call @foo1(...)
    //  %call2 = call @foo2(...)
    //  // vs:
    //  %call2 = call @foo2(...)
    //  %call1 = call @foo1(...)
    //
    //  // independent usage of calls:
    //  %op1 = VPU.Convolution(%call1)
    //  %ops2 = VPU.Convolution(%call2)
    // ```
    std::sort(chains.begin(), chains.end(), [](const VPU::CallChainData& x, const VPU::CallChainData& y) {
        auto xFunc = x.second;
        auto yFunc = y.second;
        // lexicographical comparison
        return xFunc.getSymName() < yFunc.getSymName();
    });

    return chains;
}

// Casts the resulting value "back" to its original quantized type. This is
// normally done in main and thus in VPU dialect.
mlir::Value castToQuantizedType(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                const VPU::IoBoundaryAdapter::TypeInfo& info) {
    if (!info.valid()) {
        return input;
    }
    return builder.create<VPU::QuantizeCastOp>(appendLoc(loc, "quant_cast"), input, info.quantizedType);
}
}  // namespace

CallChainTree getOutliningRepresentation(mlir::func::FuncOp startFunc) {
    VPU::CallChainTree tree({VPU::CallChainTree::Node(VPU::CallChainData{nullptr, startFunc}, {})}, findChildren);
    return tree;
}

void MainFunctionUpdater::hoistCalleeArgsToCaller(mlir::func::FuncOp callerOp, mlir::func::FuncOp calleeOp) {
    auto& callerArgDeduplicator = getNonConstArgCache(callerOp);
    const auto& calleeArgDeduplicator = _argCaches.at(calleeOp);
    for (auto it : calleeArgDeduplicator.getSortedArgs()) {
        const auto& [entry, blockArg] = *it;
        const auto uniqueLoc = appendLoc(blockArg.getLoc(), "from_{0}", calleeOp.getSymName());
        // propagate the argument from child to parent - there's nothing
        // else that has to be done just yet
        std::ignore = callerArgDeduplicator.addArgument(uniqueLoc, entry, blockArg.getType());
    }
}

void MainFunctionUpdater::fixCallSite(mlir::OpBuilder& callerBuilder, mlir::func::FuncOp callerOp,
                                      mlir::func::FuncOp calleeOp, mlir::func::CallOp oldCall) {
    const auto& callerDeduplicator = _argCaches.at(callerOp);
    const auto& calleeDeduplicator = _argCaches.at(calleeOp);

    auto newCallArguments = to_std_vector(oldCall.getOperands());
    const auto& afterInitCalleeArgs = calleeDeduplicator.getSortedArgs();
    // old arguments remain "as is", new arguments are appended
    newCallArguments.resize(newCallArguments.size() + afterInitCalleeArgs.size());
    for (auto it : afterInitCalleeArgs) {
        const auto& [entry, calleeArg] = *it;
        auto callerArg = callerDeduplicator.findArgument(entry);
        assert(calleeArg.getArgNumber() >= oldCall.getOperands().size() &&
               "Call-site is invalidated: added arguments must always be present after the original ones");
        newCallArguments[calleeArg.getArgNumber()] = callerArg;
    }

    callerBuilder.setInsertionPoint(oldCall);
    auto newCall = callerBuilder.create<mlir::func::CallOp>(oldCall.getLoc(), calleeOp, newCallArguments);
    oldCall.replaceAllUsesWith(newCall.getResults());
    oldCall->erase();
}

WsArgumentCache& MainFunctionUpdater::getNonConstArgCache(mlir::func::FuncOp funcOp) {
    assert(_argCaches.contains(funcOp) && "Argument caches must already be set up and be functional");
    return _argCaches.find(funcOp)->second;
}

MainFunctionUpdater::MainFunctionUpdater(const Logger& log, mlir::ModuleOp moduleOp, VPU::IsWorthyToCollect isWorthy)
        : _log(log), _isWorthy(std::move(isWorthy)) {
    moduleOp.walk([&](mlir::func::FuncOp funcOp) {
        _argCaches.insert({funcOp, WsArgumentCache(funcOp)});
    });
}

bool MainFunctionUpdater::visit(const Node& node) {
    auto currOp = node.data().second;
    if (_hasSeenThisFunction(currOp)) {
        return false;
    }

    // when visiting the function, update IR inside the current function
    // according to the main schedule transformations.
    _log.trace("Visiting {0} to update main schedule", currOp.getSymName());
    const auto constants = VPU::collectMoveWorthyConstants(_log, currOp, _isWorthy);

    // in main we only care about input boundary - "quantization" has to be
    // done on input arguments to restore real types.
    vpux::VPU::IoBoundaryAdapter mainIoAdaptor{/*wrapInput=*/&castToQuantizedType,
                                               /*wrapOutput=*/&vpux::VPU::IoBoundaryAdapter::identity};
    auto& mainArgDeduplicator = getNonConstArgCache(currOp);

    // Note: created externally once to ensure operation builder has correct
    // insertion point.
    VPU::ConstOpConverter funcConverter(currOp, _log);

    for (auto it : llvm::enumerate(constants)) {
        auto declareOp = it.value();
        auto idx = it.index();
        VPU::TransformationsSplit split(declareOp);
        auto projection = split.take(VPU::WeightsSeparationSchedule::Main);
        const auto uniqueLoc = appendLoc(split.getLoc(), "main_cst{0}", idx);
        auto mainArg = mainArgDeduplicator.addArgument(uniqueLoc, VPU::ConstArg(projection), projection.argType);
        auto valueInMain = funcConverter.convertToIrForm(uniqueLoc, projection, mainArg, mainIoAdaptor,
                                                         VPU::WeightsSeparationSchedule::Main);

        _log.trace("Replacing '{0}' with '{1}'", declareOp, valueInMain);
        declareOp.replaceAllUsesWith(valueInMain);
    }

    // Note: removal of operations is done separately, after construction of
    // new IR, to ensure that operation builder is not invalidated.
    for (auto op : constants) {
        op.erase();
    }

    return true;
}

void MainFunctionUpdater::endVisit(const Node& node) {
    auto currOp = node.data().second;
    auto bodyBlock = &currOp.getFunctionBody().front();
    OpBuilderLogger builderLog(_log.nest());
    auto currBuilder = mlir::OpBuilder::atBlockBegin(bodyBlock, &builderLog);

    // At the end of the visitation, it is certain that the children are
    // already processed (by definition of the procedure). Thus, we can
    // forward children's arguments up the call-chain and fix the calls to
    // children accordingly:
    // ```
    // func.func foo() {
    //   call bar()
    // }
    // ```
    // becomes:
    // ```
    // func.func foo(%bar_cst: ...) {
    //   call bar(%bar_cst)
    // }
    // ```
    for (const auto& child : node.children()) {
        auto [callOp, childOp] = child.data();
        hoistCalleeArgsToCaller(currOp, childOp);
        fixCallSite(currBuilder, currOp, childOp, callOp);
    }

    // Since argument propagation is done, function signature could be
    // updated.
    const auto mainFuncResults = currOp.getFunctionType().getResults();
    // in "main", only inputs change
    currOp.setFunctionType(
            mlir::FunctionType::get(currOp.getContext(), bodyBlock->getArgumentTypes(), mainFuncResults));

    _log.trace("Finished visiting {0}", currOp.getSymName());
}

WsArgumentCache MainFunctionUpdater::takeArgCache(mlir::func::FuncOp funcOp) {
    return std::move(getNonConstArgCache(funcOp));
}
}  // namespace vpux::VPU
