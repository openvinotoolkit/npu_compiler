//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes/bounded_buffer_pass_utils.hpp"

#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
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
// Returns whether the given function has any dynamic memrefs in its signature.
bool functionHasDynamicSignature(mlir::func::FuncOp funcOp) {
    const auto funcType = funcOp.getFunctionType();
    const auto isDynamicMemref = [](mlir::Type type) {
        const auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
        return memrefType != nullptr && !memrefType.hasStaticShape();
    };
    return llvm::any_of(funcType.getInputs(), isDynamicMemref) || llvm::any_of(funcType.getResults(), isDynamicMemref);
}

// Collects all NPU modules that have dynamic inputs or outputs and thus require
// adjustments by this pass. Supports both default and host compilations.
SmallVector<mlir::ModuleOp> findDynamicNpuModules(Logger log, mlir::ModuleOp topModule) {
    log.trace("Finding all NPU modules that have dynamism");

    log = log.nest();
    SmallVector<mlir::ModuleOp> suitableModules;
    if (!config::isHostCompileMode(topModule)) {
        log.trace("This is NOT host compilation, expecting top-level module to be the only NPU module");
        // not host compile means we should have a single module which
        // represents the NPU schedule.
        suitableModules.push_back(topModule);
        return suitableModules;
    }

    log.trace("This IS host compilation, expecting nested modules to be NPU kernels");
    const auto innerModules = topModule.getOps<mlir::ModuleOp>();
    for (auto innerModule : innerModules) {
        const auto networkInfos = innerModule.getOps<net::NetworkInfoOp>();
        if (networkInfos.empty()) {
            // not an NPU module
            log.trace("Module '{0}' is not an NPU kernel", innerModule.getSymName().value_or("UNKNOWN"));
            continue;
        }

        log.trace("Module '{0}' is an NPU kernel", innerModule.getSymName().value());
        // Note: assume that if an entry point has no dynamic inputs nor
        // outputs, this model is not dynamic and doesn't require this pass to
        // be run.
        auto mainFunc = net::getMainFunc(innerModule);
        if (functionHasDynamicSignature(mainFunc)) {
            log.trace("Module '{0}' HAS dynamism", innerModule.getSymName().value());
            suitableModules.push_back(innerModule);
        } else {
            log.trace("Module '{0}' does NOT have dynamism", innerModule.getSymName().value());
        }
    }

    return suitableModules;
}

// Discovers the bounds for a given input value, including the value that
// "provides" this bounds. For example,
// ```mlir
// func.func @foo(%input: memref<1x2x?x?xf32>) ...
//  %cast = Core.ReinterpretCast(%input) : memref<1x2x?x?xf32>
//      -> memref<1x2x?x?xf32, {bounds = [1, 2, 3, 4]>
// ```
// In the case of IR above, this function returns '%cast' value and bounds equal
// to [1, 2, 3, 4].
std::pair<mlir::Value, BoundsRef> findBoundedInput(Logger log, mlir::Value input) {
    auto bounds = getBounds(input.getType());
    if (!bounds.empty()) {
        log.trace("Found direct bounds for input {0}: {1}", input, bounds);
        return {input, bounds};
    }

    // expect the input to have a reinterpret cast chain as a single user that
    // has the bounds information.
    auto users = input.getUsers();
    while (std::distance(users.begin(), users.end()) == 1) {
        auto castOp = mlir::dyn_cast<Core::ReinterpretCastOp>(*users.begin());
        if (castOp == nullptr) {
            break;
        }

        auto castResult = castOp.getResult();
        auto resultBounds = getBounds(castResult.getType());
        if (!resultBounds.empty()) {
            log.trace("Found indirect bounds for input {0} via {1}: {2}", input, castResult, resultBounds);
            return {castResult, resultBounds};
        }

        users = castResult.getUsers();
    }
    return {nullptr, {}};
}

// Discovers the bounds of a given result value, including the value that
// "provides" this bounds. For example,
// ```mlir
//  %out = ...
//  %cast = Core.ReinterpretCast(%out) : memref<1x2x?x?xf32, {bounds = [1, 2, 3, 4]>
//      -> memref<1x2x?x?xf32>
//  return %cast
// ```
// In the case of IR above, this function returns '%out' value and bounds equal
// to [1, 2, 3, 4].
std::pair<mlir::Value, BoundsRef> findBoundedOutput(Logger log, mlir::Value output) {
    auto bounds = getBounds(output.getType());
    if (!bounds.empty()) {
        log.trace("Found direct bounds for output {0}: {1}", output, bounds);
        return {output, bounds};
    }

    // expect the output to be a result of a reinterpret cast operation that has
    // the bounds information.
    auto castOp = output.getDefiningOp<Core::ReinterpretCastOp>();
    while (castOp != nullptr) {
        auto castInput = castOp.getInput();
        auto inputBounds = getBounds(castInput.getType());
        if (!inputBounds.empty()) {
            log.trace("Found indirect bounds for output {0} via {1}: {2}", output, castInput, inputBounds);
            return {castInput, inputBounds};
        }

        castOp = castInput.getDefiningOp<Core::ReinterpretCastOp>();
    }
    return {nullptr, {}};
}

bool isDynamicOperand(mlir::Type type) {
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
    return memrefType && !memrefType.hasStaticShape();
}

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
        if (!isDynamicOperand(origInArg.getType())) {
            continue;
        }

        auto [value, bounds] = findBoundedInput(log, origInArg);
        if (bounds.empty()) {
            continue;
        }

        log.trace("Found dynamic input {0}", value);
        const auto dynamicMemrefType = mlir::cast<mlir::MemRefType>(value.getType());
        assert(!dynamicMemrefType.hasStaticShape() &&
               "Expecting a dynamic shape for a bounded memref in function inputs");
        const auto& [boundedMemRef, dynamicShapeMemRef] = VPUIP::unpackDynamicMemRefType(dynamicMemrefType, bounds);

        if (funcOp == mainFuncOp) {
            VPUIP::addShapeTensorDataInfo(funcOp, dynamicShapeMemRef, netInfo.getInputsInfo().front(),
                                          netInfo.getInputsDataInfo()[index].getName(),
                                          /*current dataBufferCount*/ netInfo.getInputsDataInfo().size());
        }

        value.setType(boundedMemRef);
        auto newShapeArgument = entryBlock.insertArgument(entryBlock.getNumArguments(), dynamicShapeMemRef, funcLoc);

        builder.setInsertionPointToStart(&entryBlock);
        if (!mlir::isa<mlir::BlockArgument>(value)) {
            builder.setInsertionPointAfter(value.getDefiningOp());
        }
        auto groupOp = VPUIP::GroupBoundedBufferOp::create(builder, appendLoc(funcLoc, "arg_to_call_op_{0}", index),
                                                           dynamicMemrefType, value, newShapeArgument);

        log.nest(2).trace("Wrapped newly added network arguments into GroupBoundedBufferOp.");
        value.replaceAllUsesExcept(groupOp->getResult(0), groupOp);
        log.nest(2).trace("CallOp uses of old arg replaced with value {0}", groupOp->getResult(0));
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
        const auto operand = returnOp.getOperand(index);
        if (!isDynamicOperand(operand.getType())) {
            continue;
        }

        auto [value, bounds] = findBoundedOutput(log, operand);
        if (bounds.empty()) {
            continue;
        }

        log.trace("Found dynamic output {0}", value);
        const auto dynamicMemrefType = mlir::cast<mlir::MemRefType>(value.getType());
        assert(!dynamicMemrefType.hasStaticShape() &&
               "Expecting a dynamic shape for a bounded memref in function results");
        const auto& [boundedMemRef, dynamicShapeMemRef] = VPUIP::unpackDynamicMemRefType(dynamicMemrefType, bounds);

        if (funcOp == mainFuncOp) {
            VPUIP::addShapeTensorDataInfo(funcOp, dynamicShapeMemRef, netInfo.getOutputsInfo().front(),
                                          netInfo.getOutputsDataInfo()[index].getName(),
                                          /*current dataBufferCount*/ netInfo.getOutputsDataInfo().size());
        }

        builder.setInsertionPoint(returnOp);
        if (!mlir::isa<mlir::BlockArgument>(value)) {
            builder.setInsertionPointAfter(value.getDefiningOp());
        }
        auto ungroupOp = VPUIP::UngroupBoundedBufferOp::create(builder, appendLoc(funcOp.getLoc(), "output_{0}", index),
                                                               {boundedMemRef, dynamicShapeMemRef}, value);
        value.replaceUsesWithIf(ungroupOp.getData(), [](mlir::OpOperand& use) {
            // Note: new ungroup operation can only be used by either a direct
            // return, or an indirect return via a reinterpret cast; all other
            // uses must remain unchanged as there is no guarantee that the IR
            // would be valid otherwise. in theory, not all reinterpret casts
            // are applicable - only those that have a return op at the end and
            // no other uses but, given the limited scope of this algorithm, and
            // the expected IR prior to this pass, any other more complex
            // patterns are deemed non-existent.
            auto owner = use.getOwner();
            return mlir::isa<mlir::func::ReturnOp, Core::ReinterpretCastOp>(owner);
        });
        log.nest(2).trace("Created UngroupBoundedBufferOp for newly added network results");

        returnOp->insertOperands(returnOp->getOpOperands().size(), ungroupOp.getDynamicShape());
    }

    log = log.unnest(2);
    log.trace("New function outputs: {0}", returnOp->getOperandTypes());
    const auto newOutFuncTypes =
            mlir::FunctionType::get(funcOp.getContext(), entryBlock.getArgumentTypes(), returnOp->getOperandTypes());
    funcOp.setType(newOutFuncTypes);
}

void rewriteNestedCallForHiddenBoundaries(Logger log, mlir::OpBuilder& builder, Core::NestedCallOp nestedCallOp,
                                          mlir::func::FuncOp callee) {
    // Rebuild nested calls with extra shape operands/results when callee boundaries were expanded
    const auto funcType = callee.getFunctionType();
    assert(funcType.getNumInputs() >= nestedCallOp.getNumOperands() && "This pass adds new shape arguments");
    const size_t extraShapeAllocationsCount = funcType.getNumInputs() - nestedCallOp.getNumOperands();
    assert(funcType.getNumResults() >= nestedCallOp.getNumResults() && "This pass adds new shape results");
    const size_t extraShapeResultsCount = funcType.getNumResults() - nestedCallOp.getNumResults();
    log.trace("Rewrite nested call at {0} for callee '@{1}': add {2} shape operands, {3} shape results",
              nestedCallOp.getLoc(), callee.getSymName(), extraShapeAllocationsCount, extraShapeResultsCount);
    builder.setInsertionPoint(nestedCallOp);

    SmallVector<mlir::Value> newOperands;
    newOperands.reserve(nestedCallOp.getNumOperands() + extraShapeAllocationsCount);
    auto oldOperands = nestedCallOp.getOperands();
    newOperands.assign(oldOperands.begin(), oldOperands.end());
    for (size_t i = 0; i < extraShapeAllocationsCount; ++i) {
        auto alloc = mlir::memref::AllocOp::create(
                builder, appendLoc(nestedCallOp.getLoc(), "dyn_shape_alloc{0}", i),
                mlir::cast<mlir::MemRefType>(funcType.getInput(nestedCallOp.getNumOperands() + i)));
        newOperands.push_back(alloc.getResult());
    }

    SmallVector<mlir::Type> newResultTypes;
    newResultTypes.reserve(nestedCallOp.getNumResults() + extraShapeResultsCount);
    auto oldResultTypes = nestedCallOp.getResultTypes();
    newResultTypes.assign(oldResultTypes.begin(), oldResultTypes.end());
    for (size_t i = 0; i < extraShapeResultsCount; ++i) {
        newResultTypes.push_back(funcType.getResult(nestedCallOp.getNumResults() + i));
    }

    auto newNestedCallOp = Core::NestedCallOp::create(builder, nestedCallOp.getLoc(), nestedCallOp.getCalleeAttr(),
                                                      newResultTypes, newOperands);
    newNestedCallOp->setAttrs(nestedCallOp->getAttrs());

    // Note: at this stage, new results (with newly added shape arguments) are
    // still unused anywhere in the IR - can only replace old call with a new
    // call - that has more results - when shapes are unused.
    SmallVector<mlir::Value> newResults(newNestedCallOp.getResults().begin(),
                                        newNestedCallOp.getResults().begin() + nestedCallOp.getNumResults());
    nestedCallOp.replaceAllUsesWith(newResults);
    nestedCallOp.erase();
}

class UngroupBoundedBuffersAsFuncArgs final :
        public VPUIP::impl::UngroupBoundedBuffersAsFuncArgsBase<UngroupBoundedBuffersAsFuncArgs> {
public:
    explicit UngroupBoundedBuffersAsFuncArgs(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    void ungroupBoundedBuffers(mlir::ModuleOp module);
};

void UngroupBoundedBuffersAsFuncArgs::safeRunOnModule() {
    auto topModule = getOperation();

    const auto npuModules = findDynamicNpuModules(_log, topModule);
    for (auto npuModule : npuModules) {
        ungroupBoundedBuffers(npuModule);
    }

    if (!config::isHostCompileMode(topModule)) {
        return;
    }
    // in host-compile, there is a need to additionally adjust the nested call
    // operations of the host functions calling into NPU kernel functions that
    // have been changed signature-wise.
    _log.trace("Ungrouping done, adjusting call-sites of host module '{0}'",
               topModule.getSymName().value_or("UNKNOWN"));
    const auto immediateFunctions = topModule.getOps<mlir::func::FuncOp>();
    for (auto funcOp : immediateFunctions) {
        auto builder = mlir::OpBuilder::atBlockBegin(&funcOp.getBody().front());
        SmallVector<std::pair<Core::NestedCallOp, mlir::func::FuncOp>> nestedCalls;
        funcOp.walk([&](Core::NestedCallOp nestedCallOp) {
            auto callee = getCalledFunction(nestedCallOp);
            const auto funcType = callee.getFunctionType();
            // if there is a mismatch in function signature and the call
            // signature, it means that the function has dynamism and thus the
            // calls to it need adjustment.
            const bool functionIsDynamic = (nestedCallOp.getNumOperands() != funcType.getNumInputs()) ||
                                           (nestedCallOp.getNumResults() != funcType.getNumResults());
            if (functionIsDynamic) {
                nestedCalls.emplace_back(nestedCallOp, callee);
            }
        });
        for (auto [nestedCallOp, funcOp] : nestedCalls) {
            builder.setInsertionPoint(nestedCallOp);
            rewriteNestedCallForHiddenBoundaries(_log, builder, nestedCallOp, funcOp);
        }
    }
}

void UngroupBoundedBuffersAsFuncArgs::ungroupBoundedBuffers(mlir::ModuleOp module) {
    auto funcOps = module.getOps<mlir::func::FuncOp>();
    net::NetworkInfoOp netInfo = *module.getOps<net::NetworkInfoOp>().begin();

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
                if (!isDynamicOperand(operand.getType())) {
                    newOperands.push_back(operand);
                    continue;
                }

                const auto& [boundedMemRef, dynamicShapeMemRef] =
                        VPUIP::unpackDynamicMemRefType(mlir::cast<mlir::MemRefType>(operand.getType()));
                auto ungroupOp =
                        VPUIP::UngroupBoundedBufferOp::create(builder, appendLoc(callOp.getLoc(), "operand_{0}", index),
                                                              {boundedMemRef, dynamicShapeMemRef}, operand);
                newOperands.push_back(ungroupOp.getData());
                dynShapeOperands.push_back(ungroupOp.getDynamicShape());
                _log.nest(2).trace("Found dynamic operand of Call Op. Replaced by Ungroup op: {0}", ungroupOp);
            }
            newOperands.append(dynShapeOperands);

            _log.nest().trace("New operands for CallOp {0}", newOperands);

            SmallVector<mlir::Type> newResultsType;
            SmallVector<mlir::Type> dynShapeResultTypes;
            SmallVector<std::tuple<int64_t, int64_t, mlir::Type>> dataShapeResults;
            int64_t numBoundedBuffs = 0;
            for (const auto& [index, result] : callOp.getResults() | indexed) {
                if (!isDynamicOperand(result.getType())) {
                    newResultsType.push_back(result.getType());
                    continue;
                }

                const auto& [boundedMemRef, dynamicShapeMemRef] =
                        VPUIP::unpackDynamicMemRefType(mlir::cast<mlir::MemRefType>(result.getType()));
                newResultsType.push_back(boundedMemRef);
                dynShapeResultTypes.push_back(dynamicShapeMemRef);

                const auto dynamicShapeResultIndex = numBoundedBuffs + callOp->getNumResults();
                dataShapeResults.push_back({index, dynamicShapeResultIndex, result.getType()});
                numBoundedBuffs++;
                _log.nest(2).trace("Found dynamic result of CallOp. Data - Dynamic Shape result indices: [{0}, {1}]",
                                   index, dynamicShapeResultIndex);
            }
            newResultsType.append(dynShapeResultTypes);

            _log.nest().trace("New Result Types for CallOp {0}", newResultsType);

            if (dynShapeOperands.empty() && newResultsType.empty()) {
                _log.nest().trace("CallOp has no dynamic inputs/outputs.");
                continue;
            }

            auto newCallOp = mlir::func::CallOp::create(builder, callOp.getLoc(), callOp.getCalleeAttr(),
                                                        newResultsType, newOperands);
            newCallOp->setAttrs(callOp->getAttrs());
            _log.nest().trace("Created new CallOp at loc {0}", newCallOp->getLoc());

            SmallVector<mlir::Value> results(newCallOp.getResults().begin(),
                                             newCallOp.getResults().begin() + callOp->getNumResults());

            for (const auto& [dataIdx, dynShapeIdx, origResultType] : dataShapeResults) {
                _log.nest(2).trace("Create GroupBoundedBufferOp for result pair [{0}, {1}] with return type '{2}'.",
                                   dataIdx, dynShapeIdx, origResultType);
                auto groupOp = VPUIP::GroupBoundedBufferOp::create(
                        builder, appendLoc(callOp.getLoc(), "results_{0}_{1}", dataIdx, dynShapeIdx), origResultType,
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

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUngroupBoundedBuffersAsFuncArgsPass(Logger log) {
    return std::make_unique<UngroupBoundedBuffersAsFuncArgs>(log);
}
