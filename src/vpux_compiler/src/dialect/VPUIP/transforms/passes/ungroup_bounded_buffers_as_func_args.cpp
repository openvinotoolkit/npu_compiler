//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/bounded_buffer.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes/bounded_buffer_pass_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"

#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LogicalResult.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNGROUPBOUNDEDBUFFERSASFUNCARGS
#define GEN_PASS_DEF_UNGROUPBOUNDEDBUFFERSASFUNCARGS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {
void adaptFuncInputArgs(mlir::func::FuncOp funcOp, net::NetworkInfoOp netInfo, Logger log) {
    auto& entryBlock = funcOp.front();
    auto builder = mlir::OpBuilder::atBlockBegin(&entryBlock);

    auto mainFuncOp = net::getMainFunc(funcOp);

    log.trace("Old block inputs: {0}", entryBlock.getArgumentTypes());

    log = log.nest(2);
    auto funcLoc = funcOp.getLoc();
    const auto originalInputSize = funcOp.getFunctionType().getInputs().size();
    for (const auto& index : irange(originalInputSize)) {
        auto origInArg = entryBlock.getArgument(index);
        const auto inArgType = origInArg.getType();
        const auto boundedType = mlir::dyn_cast_or_null<VPUIP::BoundedBufferType>(inArgType);

        if (boundedType == nullptr) {
            continue;
        }

        log.trace("Found dynamic input {0}", inArgType);
        const auto& [boundedMemRef, dynamicShapeMemRef] = VPUIP::unpackBoundedBufferType(boundedType);

        if (funcOp == mainFuncOp) {
            VPUIP::addShapeTensorDataInfo(funcOp, dynamicShapeMemRef, netInfo.getInputsInfo().front(),
                                          netInfo.getInputsDataInfo()[index].getName(),
                                          /*current dataBufferCount*/ netInfo.getInputsDataInfo().size());
        }

        auto arg0 = entryBlock.insertArgument(index + 1, boundedMemRef, funcLoc);
        auto arg1 = entryBlock.insertArgument(entryBlock.getNumArguments(), dynamicShapeMemRef, funcLoc);

        builder.setInsertionPointToStart(&entryBlock);

        auto groupOp = builder.create<VPUIP::GroupBoundedBufferOp>(appendLoc(funcLoc, "arg_to_call_op_{0}", index),
                                                                   arg0, arg1);

        log.nest(2).trace("Wrapped newly added network arguments into GroupBoundedBufferOp.");
        origInArg.replaceAllUsesWith(groupOp->getResult(0));
        log.nest(2).trace("CallOp uses of old arg replaced with value {0}", groupOp->getResult(0));

        entryBlock.eraseArgument(index);
    }

    log = log.unnest(2);
    log.trace("New function inputs: {0}", entryBlock.getArgumentTypes());
    const auto newInFuncTypes =
            mlir::FunctionType::get(funcOp.getContext(), entryBlock.getArgumentTypes(), funcOp.getResultTypes());
    funcOp.setType(newInFuncTypes);
}

void adaptFuncOutputArgs(mlir::func::FuncOp funcOp, net::NetworkInfoOp netInfo, Logger log) {
    auto& entryBlock = funcOp.front();
    auto builder = mlir::OpBuilder::atBlockBegin(&entryBlock);

    auto mainFuncOp = net::getMainFunc(funcOp);

    auto returnOps = funcOp.getOps<mlir::func::ReturnOp>();
    if (const auto returnOpsCount = std::distance(returnOps.begin(), returnOps.end()); returnOpsCount != 1) {
        VPUX_THROW("Expected to find one ReturnOp, but got {0}", returnOpsCount);
    }

    auto returnOp = *returnOps.begin();
    log.trace("Old function outputs: {0}", returnOp->getOperandTypes());

    log = log.nest(2);
    const auto originalOutputSize = funcOp.getFunctionType().getResults().size();
    for (const auto& index : irange(originalOutputSize)) {
        const auto output = funcOp.getFunctionType().getResult(index);
        const auto boundedType = mlir::dyn_cast_or_null<VPUIP::BoundedBufferType>(output);
        if (boundedType == nullptr) {
            continue;
        }

        log.trace("Found dynamic output {0}", output);
        const auto boundedBufferTypes = VPUIP::unpackBoundedBufferType(boundedType);

        if (funcOp == mainFuncOp) {
            VPUIP::addShapeTensorDataInfo(funcOp, boundedBufferTypes.dynamicShapeType, netInfo.getOutputsInfo().front(),
                                          netInfo.getOutputsDataInfo()[index].getName(),
                                          /*current dataBufferCount*/ netInfo.getOutputsDataInfo().size());
        }

        builder.setInsertionPoint(returnOp);
        const auto operand = returnOp.getOperand(index);
        auto ungroupOp =
                builder.create<VPUIP::UngroupBoundedBufferOp>(appendLoc(funcOp.getLoc(), "output_{0}", index), operand);
        log.nest(2).trace("Created UngroupBoundedBufferOp for newly added network results");

        returnOp->eraseOperand(index);
        returnOp->insertOperands(index, ungroupOp.getData());
        returnOp->insertOperands(returnOp->getOpOperands().size(), ungroupOp.getDynamicShape());
    }

    log = log.unnest(2);
    log.trace("New function outputs: {0}", returnOp->getOperandTypes());
    const auto newOutFuncTypes =
            mlir::FunctionType::get(funcOp.getContext(), entryBlock.getArgumentTypes(), returnOp->getOperandTypes());
    funcOp.setType(newOutFuncTypes);
}

//
// UngroupBoundedBuffersAsFuncArgs
//

class UngroupBoundedBuffersAsFuncArgs final :
        public VPUIP::impl::UngroupBoundedBuffersAsFuncArgsBase<UngroupBoundedBuffersAsFuncArgs> {
public:
    explicit UngroupBoundedBuffersAsFuncArgs(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void UngroupBoundedBuffersAsFuncArgs::safeRunOnModule() {
    auto module = getOperation();

    auto funcOps = module.getOps<mlir::func::FuncOp>();
    net::NetworkInfoOp netInfo = *module.getOps<net::NetworkInfoOp>().begin();

    const auto isDynamicOperand = [&](auto type) {
        return mlir::isa<vpux::VPUIP::BoundedBufferType>(type);
    };

    for (auto funcOp : funcOps) {
        if (funcOp.isExternal()) {
            /*
            Example of external functions:
            module @VPU.SW {
                func.func private @builtin_softmax(%input : memref<*xf16>, %output :
            memref<*xf16>, %axis : i64) attributes {VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"}
            }
            */
            _log.trace("Can't convert external Function '@{0}' at {1}", funcOp.getSymName(), funcOp.getLoc());
            continue;
        }

        const auto hasDynamicInputs = llvm::any_of(funcOp.getFunctionType().getInputs(), isDynamicOperand);
        if (hasDynamicInputs) {
            _log.trace("Adapt inputs for func '@{0}' at {1}.", funcOp.getSymName(), funcOp.getLoc());
            adaptFuncInputArgs(funcOp, netInfo, _log.nest(2));
        }

        const auto hasDynamicOutputs = llvm::any_of(funcOp.getFunctionType().getResults(), isDynamicOperand);
        if (hasDynamicOutputs) {
            _log.trace("Adapt outputs for func '@{0}' at {1}.", funcOp.getSymName(), funcOp.getLoc());
            adaptFuncOutputArgs(funcOp, netInfo, _log.nest(2));
        }

        _log.trace("Handle CallOp connections in func '@{0}' at {1}.", funcOp.getSymName(), funcOp.getLoc());

        _log = _log.nest(2);

        mlir::OpBuilder builder(funcOp.getContext());
        auto callOps = funcOp.getOps<mlir::func::CallOp>();
        for (auto callOp : llvm::make_early_inc_range(callOps)) {
            _log.trace("Checking call op at loc {0}", callOp.getLoc());
            builder.setInsertionPoint(callOp);

            SmallVector<mlir::Value> newOperands;
            SmallVector<mlir::Value> dynShapeOperands;

            for (auto [index, operand] : callOp->getOperands() | indexed) {
                auto boundedType = mlir::dyn_cast_or_null<VPUIP::BoundedBufferType>(operand.getType());
                if (boundedType == nullptr) {
                    newOperands.push_back(operand);
                    continue;
                }

                auto ungroupOp = builder.create<VPUIP::UngroupBoundedBufferOp>(
                        appendLoc(callOp.getLoc(), "operand_{0}", index), operand);
                newOperands.push_back(ungroupOp.getData());
                dynShapeOperands.push_back(ungroupOp.getDynamicShape());
                _log.nest(2).trace("Found Bounded operand of Call Op. Replaced by Ungroup op: {0}", ungroupOp);
            }
            newOperands.append(dynShapeOperands);

            _log.nest().trace("New operands for CallOp {0}", newOperands);

            SmallVector<mlir::Type> newResultsType;
            SmallVector<mlir::Type> dynShapeResultTypes;
            SmallVector<std::pair<int64_t, int64_t>> dataShapeResultPairs;
            int64_t numBoundedBuffs = 0;
            for (const auto& [index, result] : callOp.getResults() | indexed) {
                auto boundedType = mlir::dyn_cast_or_null<VPUIP::BoundedBufferType>(result.getType());
                if (boundedType == nullptr) {
                    newResultsType.push_back(result.getType());
                    continue;
                }

                const auto& [boundedMemRef, dynamicShapeMemRef] = VPUIP::unpackBoundedBufferType(boundedType);
                newResultsType.push_back(boundedMemRef);
                dynShapeResultTypes.push_back(dynamicShapeMemRef);

                const auto dynamicShapeResultIndex = numBoundedBuffs + callOp->getNumResults();
                dataShapeResultPairs.push_back({index, dynamicShapeResultIndex});
                numBoundedBuffs++;
                _log.nest(2).trace("Found Bounded result of CallOp. Data - Dynamic Shape result indices: [{0}, {1}]",
                                   index, dynamicShapeResultIndex);
            }
            newResultsType.append(dynShapeResultTypes);

            _log.nest().trace("New Result Types for CallOp {0}", newResultsType);

            if (dynShapeOperands.empty() && newResultsType.empty()) {
                _log.nest().trace("CallOp has no dynamic inputs/outputs.");
                continue;
            }

            auto newCallOp = builder.create<mlir::func::CallOp>(callOp.getLoc(), callOp.getCalleeAttr(), newResultsType,
                                                                newOperands);
            newCallOp->setAttrs(callOp->getAttrs());
            _log.nest().trace("Created new CallOp at loc {0}", newCallOp->getLoc());

            SmallVector<mlir::Value> results(newCallOp.getResults().begin(),
                                             newCallOp.getResults().begin() + callOp->getNumResults());

            for (const auto& [dataIdx, dynShapeIdx] : dataShapeResultPairs) {
                _log.nest(2).trace("Create GroupBoundedBufferOp for result pair [{0}, {1}].", dataIdx, dynShapeIdx);
                auto groupOp = builder.create<VPUIP::GroupBoundedBufferOp>(
                        appendLoc(callOp.getLoc(), "results_{0}_{1}", dataIdx, dynShapeIdx),
                        newCallOp.getResult(dataIdx), newCallOp.getResult(dynShapeIdx));
                results[dataIdx] = groupOp.getResult();
            }
            _log.nest().trace("Replace uses of old CallOp.");

            callOp.replaceAllUsesWith(results);
            callOp.erase();
        }

        _log.unnest(2);
    }
}

}  // namespace

//
// UngroupBoundedBuffersAsFuncArgs
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUngroupBoundedBuffersAsFuncArgsPass(Logger log) {
    return std::make_unique<UngroupBoundedBuffersAsFuncArgs>(log);
}
