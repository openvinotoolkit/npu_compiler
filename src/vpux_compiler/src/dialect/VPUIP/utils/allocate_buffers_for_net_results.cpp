//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/allocate_buffers_for_net_results.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/allocate_buffers.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dynamic_memref_bounds.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <functional>
#include <type_traits>

using namespace vpux;

namespace {

// Updates the func op and entry block.
// Any args appended to the entry block are added to `appendedEntryArgs`.
void updateFuncOp(mlir::func::FuncOp func, SmallVectorImpl<mlir::BlockArgument>& appendedEntryArgs) {
    auto functionType = func.getFunctionType();

    // Add the new arguments to the function type.
    auto newArgTypes =
            to_small_vector(llvm::concat<const mlir::Type>(functionType.getInputs(), functionType.getResults()));
    auto newFunctionType = mlir::FunctionType::get(func.getContext(), newArgTypes, functionType.getResults());
    func.setType(newFunctionType);

    const auto numInputs = functionType.getNumInputs();
    for (auto resultType : functionType.getResults() | indexed) {
        // Transfer the result attributes to arg attributes.
        const auto idx = checked_cast<unsigned>(resultType.index());
        func.setArgAttrs(numInputs + idx, func.getResultAttrs(idx));

        // Add the new arguments to the function type.
        auto newArg = func.front().addArgument(resultType.value(), func.getLoc());
        appendedEntryArgs.push_back(newArg);
    }
}

// Function to create callback, which provides location for result. It tries to get access to location from
// net::NetworkInfoOp, but in tests this information may be unavailable, so empty callback will be returned
std::function<std::optional<mlir::Location>(mlir::OpOperand&)> getResultLocationProvider(mlir::func::FuncOp func,
                                                                                         vpux::Logger& log) {
    auto moduleOp = getModuleOp(func);

    while (auto parentModule = moduleOp->getParentOfType<mlir::ModuleOp>()) {
        moduleOp = parentModule;
    }
    auto netInfoOps = to_small_vector(moduleOp.getOps<net::NetworkInfoOp>());
    if (netInfoOps.size() != 1) {
        log.warning("Can't get location for output. If it isn't a test, please, debug this.");
        return [](mlir::OpOperand&) -> const std::optional<mlir::Location> {
            return std::nullopt;
        };
    }

    net::NetworkInfoOp netInfo = netInfoOps.front();
    auto entryPointFuncOp = net::getMainFunc(moduleOp);

    if (func == entryPointFuncOp) {
        auto outputsInfo = to_small_vector(netInfo.getOutputsInfo().getOps<net::DataInfoOp>());
        return [outputsInfo = std::move(outputsInfo)](mlir::OpOperand& operand) -> const std::optional<mlir::Location> {
            const auto loc = outputsInfo[operand.getOperandNumber()]->getLoc();
            VPUX_THROW_WHEN(mlir::isa<mlir::UnknownLoc>(loc), "Network output {0} must have location",
                            operand.getOperandNumber());
            return loc;
        };
    }

    // This is outlined function.
    auto baseName = printToString("{0}_outputBuff", func.getName());
    return [=, baseName = std::move(baseName)](mlir::OpOperand& operand) -> const std::optional<mlir::Location> {
        if (mlir::isa<mlir::BlockArgument>(operand.get())) {
            auto retOp = operand.getOwner();
            auto funcOp = retOp->getParentOfType<mlir::func::FuncOp>();
            return appendLoc(funcOp->getLoc(), "{0}{1}", baseName.c_str(), operand.getOperandNumber());
        }

        auto producerOp = operand.get().getDefiningOp();
        return appendLoc(producerOp->getLoc(), "{0}{1}", baseName.c_str(), operand.getOperandNumber());
    };
}

inline mlir::Value getCopyOpOutput(VPUIP::CopyOp copyOp) {
    return copyOp.getOutput();
}

inline mlir::Value getCopyOpOutput(mlir::memref::CopyOp copyOp) {
    return copyOp.getTarget();
}

// Updates all ReturnOps in the scope of the given FuncOp by copying the associated buffer contents into the given
// out-params.
template <typename T>
void updateReturnOps(mlir::func::FuncOp func, ArrayRef<mlir::BlockArgument> appendedEntryArgs, vpux::Logger& log) {
    const auto locProvider = getResultLocationProvider(func, log);

    func.walk([&](mlir::func::ReturnOp op) {
        mlir::OpBuilder builder(op);
        for (auto& opOperand : op->getOpOperands()) {
            auto opLoc = op->getLoc();
            if (auto realLoc = locProvider(opOperand)) {
                opLoc = realLoc.value();
            }
            auto idx = opOperand.getOperandNumber();
            auto copyOp = builder.create<T>(opLoc, op.getOperand(idx), appendedEntryArgs[idx]);
            opOperand.set(getCopyOpOutput(copyOp));
        }
    });
}

// Peels view-like ops (memref.subview, memref.cast, ...) and returns the underlying buffer.
// Two values share memory when their roots are the same value.
mlir::Value getViewRoot(mlir::Value value) {
    while (auto viewOp = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(value.getDefiningOp())) {
        value = viewOp.getViewSource();
    }
    return value;
}

// Returns true when `value` is a block argument, or a view of one (for example memref.subview of an
// scf.for iter_arg). Such a target denotes memory owned by the caller, so letting the call write into it
// directly preserves the semantics of the copy being removed.
bool isRootedAtBlockArg(mlir::Value value) {
    return mlir::isa<mlir::BlockArgument>(getViewRoot(value));
}

// Collects the view-like ops and their operands that define `target` and are located after `callOp`, in
// definition order. Those ops must be moved above the call for `target` to be usable as a call operand.
// Returns false when an op cannot be moved, i.e. it has side effects or lives in another block.
bool collectOpsToHoistAboveCall(mlir::Value target, mlir::Operation* callOp,
                                SmallVector<mlir::Operation*>& opsToHoist) {
    auto* defOp = target.getDefiningOp();
    if (defOp == nullptr || defOp->getBlock() != callOp->getBlock() || defOp->isBeforeInBlock(callOp)) {
        return true;
    }
    if (!mlir::isMemoryEffectFree(defOp)) {
        return false;
    }
    for (auto operand : defOp->getOperands()) {
        if (!collectOpsToHoistAboveCall(operand, callOp, opsToHoist)) {
            return false;
        }
    }
    if (!llvm::is_contained(opsToHoist, defOp)) {
        opsToHoist.push_back(defOp);
    }
    return true;
}

// Returns true when the copies can be replaced by letting the call write into `target` directly.
// The write to `target` moves from the last copy up to the call, so every operation in between must leave
// `target` alone, otherwise it would observe the new contents early or have its own write overwritten.
// The copies being erased are skipped, and memory effect free ops cannot interfere.
// Values that alias `target` through view ops are compared by their root buffer.
bool canMoveWriteToCall(mlir::Value target, mlir::Operation* callOp, ArrayRef<mlir::memref::CopyOp> copies) {
    auto lastCopy = copies.back();
    // The linear scan below only makes sense when all the copies sit in the call's block
    if (llvm::any_of(copies, [&](mlir::memref::CopyOp copyOp) {
            return copyOp->getBlock() != callOp->getBlock();
        })) {
        return false;
    }

    const auto targetRoot = getViewRoot(target);
    for (auto* op = callOp->getNextNode(); op != lastCopy.getOperation(); op = op->getNextNode()) {
        const auto isErasedCopy = llvm::any_of(copies, [&](mlir::memref::CopyOp copyOp) {
            return copyOp.getOperation() == op;
        });
        if (mlir::isMemoryEffectFree(op) || isErasedCopy) {
            continue;
        }
        if (llvm::any_of(op->getOperands(), [&](mlir::Value operand) {
                return getViewRoot(operand) == targetRoot;
            })) {
            return false;
        }
    }
    return true;
}

// Follows the chain of memref.copy ops from a call result and returns the first target that is owned by the
// caller, either a block argument (the function's own appended output argument, or an enclosing loop's
// iter_arg) or a view of one, so it can be reused as the call's output argument directly.
// Copies traversed along  the way are recorded in `copiesToErase`, as they become redundant once the call
// writes into that target.  memref.dim users are ignored as shape queries. Returns null if the chain branches,
// has another consumer, or never reaches such a target.
mlir::Value findCopyChainBlockArgTarget(mlir::Value result, SmallVector<mlir::memref::CopyOp>& copiesToErase) {
    mlir::Value current = result;
    while (true) {
        mlir::memref::CopyOp onwardCopy = nullptr;
        for (auto* user : current.getUsers()) {
            if (mlir::isa<mlir::memref::DimOp>(user)) {
                continue;
            }
            auto copyUser = mlir::dyn_cast<mlir::memref::CopyOp>(user);
            if (copyUser == nullptr) {
                // Some other consumer reads the buffer - fusing it away would drop that use.
                return nullptr;
            }
            if (copyUser.getTarget() == current) {
                // The copy that produced `current`, not a consumer of it.
                continue;
            }
            if (copyUser.getSource() != current || onwardCopy != nullptr) {
                // `current` is overwritten by another copy, or there is more than one onward copy: ambiguous.
                return nullptr;
            }
            onwardCopy = copyUser;
        }
        if (onwardCopy == nullptr) {
            return nullptr;
        }
        copiesToErase.push_back(onwardCopy);
        current = onwardCopy.getTarget();
        if (isRootedAtBlockArg(current)) {
            return current;
        }
    }
}

// Tries to eliminate the memref.copy(s) after a call result by reusing the terminal caller-owned copy
// target as the call's output argument, casting it if needed and erasing the redundant copies. The view ops
// defining the target are hoisted above the call when needed, so that the target dominates its new use.
// Returns nullptr if the chain never reaches such a target, or if reusing it would change behaviour.
mlir::Value tryEliminateResultCopy(mlir::CallOpInterface callOp, mlir::Value result, size_t index,
                                   mlir::OpBuilder& builder) {
    SmallVector<mlir::memref::CopyOp> copiesToErase;
    auto copyTarget = findCopyChainBlockArgTarget(result, copiesToErase);
    if (copyTarget == nullptr) {
        return nullptr;
    }

    if (!canMoveWriteToCall(copyTarget, callOp, copiesToErase)) {
        return nullptr;
    }

    SmallVector<mlir::Operation*> opsToHoist;
    if (!collectOpsToHoistAboveCall(copyTarget, callOp, opsToHoist)) {
        return nullptr;
    }
    for (auto* op : opsToHoist) {
        op->moveBefore(callOp);
    }

    auto funcOp = getCalledFunction(callOp);
    auto funcType = funcOp.getFunctionType();
    size_t numInputs = funcType.getNumInputs() - funcType.getNumResults();
    auto memRefType = mlir::cast<mlir::MemRefType>(funcType.getInput(numInputs + index));

    mlir::Value outParam = copyTarget;
    if (memRefType != outParam.getType()) {
        auto castBufferOp = builder.create<mlir::memref::CastOp>(callOp.getLoc(), memRefType, outParam);
        outParam = castBufferOp.getResult();
    }
    for (auto copyOp : copiesToErase) {
        copyOp.erase();
    }
    return outParam;
}

// Updates call op
void updateCallOp(const mlir::DenseSet<mlir::CallOpInterface>& callOps, vpux::Logger& log) {
    for (auto callOp : llvm::make_early_inc_range(callOps)) {
        mlir::OpBuilder builder(callOp);

        SmallVector<mlir::Value> outParams;
        SmallVector<mlir::Value> currentResults;
        SmallVector<mlir::Type> resultTypes;

        // Only Core.NestedCall-style callees (@Module::@func) have a submodule-scoped NetworkInfo
        // that can supply upper bounds for dynamic memref allocation. Flat calls (func.call @main,
        // func.call @output_shape) must not go through the bounds-based allocation path
        auto callableRef = callOp.getCallableForCallee();
        auto symbRef = mlir::dyn_cast<mlir::SymbolRefAttr>(callableRef);
        bool isOutlinedFunction = symbRef && !symbRef.getNestedReferences().empty();

        for (auto [index, result] : llvm::enumerate(callOp->getResults())) {
            mlir::Type resType = result.getType();
            // E-140551: add support for VPUIP.SparseBuffer, allocateBuffersOfType has the allocation logic for
            // VPUIP.SparseBuffer. Need real use cases. Remove the following VPUX_THROW_WHEN to check if it works.
            VPUX_THROW_WHEN(
                    !mlir::isa<mlir::MemRefType>(resType) && !mlir::isa<vpux::VPUIP::DistributedBufferType>(resType),
                    "Only MemRefType and DistributedBufferType are supported for now, got {0}", result.getType());

            mlir::Value outParam = tryEliminateResultCopy(callOp, result, index, builder);
            if (outParam == nullptr) {
                // Dynamic memrefs in outlined kernel results have no dominating destination to reuse, so
                // allocate a buffer using upper bounds from the callee's NetworkInfo, large enough for the
                // worst-case runtime shape. Static memrefs and non-outlined calls use the standard path.
                if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(resType);
                    memrefType && memrefType.getNumDynamicDims() > 0 && isOutlinedFunction) {
                    outParam = VPUIP::allocateCallBoundaryMemref(callOp, index, /*isInput=*/false, memrefType, builder,
                                                                 log, "add-buffers-for-net-results");
                } else {
                    outParam = VPUIP::allocateBuffersOfType(log, callOp.getLoc(), builder, resType).front();
                }
            }
            outParams.push_back(outParam);

            currentResults.push_back(result);
            resultTypes.push_back(resType);
        }

        auto newOperands = to_vector(callOp->getOperands());
        newOperands.append(outParams.begin(), outParams.end());

        auto funcOp = getCalledFunction(callOp);
        for (auto& arg : funcOp.getArguments()) {
            auto argType = arg.getType();
            auto argIndex = arg.getArgNumber();

            if (argType != newOperands[argIndex].getType()) {
                auto castBufferOp =
                        builder.create<mlir::memref::CastOp>(callOp.getLoc(), argType, newOperands[argIndex]);
                newOperands[argIndex] = castBufferOp.getResult();
            }
        }

        auto newCallOp = callOp->clone();
        newCallOp->setOperands(newOperands);
        builder.insert(newCallOp);

        callOp->replaceAllUsesWith(newCallOp->getResults());

        newCallOp->setAttrs(callOp->getAttrs());
        callOp.erase();
    }
}

}  // namespace

namespace vpux::VPUIP {
template <typename CopyOp>
void updateFuncBoundariesForNetResults(const mlir::DenseSet<mlir::func::FuncOp>& funcOps, vpux::Logger& log) {
    for (auto func : funcOps) {
        SmallVector<mlir::BlockArgument> appendedEntryArgs;
        updateFuncOp(func, appendedEntryArgs);
        updateReturnOps<CopyOp>(func, appendedEntryArgs, log);
    }
}

void updateCallsForNetResults(const mlir::DenseSet<mlir::CallOpInterface>& callOps, vpux::Logger& log) {
    updateCallOp(callOps, log);
}

void allocateBuffersForNetResults(const mlir::DenseSet<mlir::CallOpInterface>& callOps,
                                  const mlir::DenseSet<mlir::func::FuncOp>& funcOps, vpux::Logger& log) {
    updateFuncBoundariesForNetResults<VPUIP::CopyOp>(funcOps, log);
    updateCallsForNetResults(callOps, log);
}

template void updateFuncBoundariesForNetResults<VPUIP::CopyOp>(const mlir::DenseSet<mlir::func::FuncOp>& funcOps,
                                                               Logger& log);
template void updateFuncBoundariesForNetResults<mlir::memref::CopyOp>(const mlir::DenseSet<mlir::func::FuncOp>& funcOps,
                                                                      Logger& log);
}  // namespace vpux::VPUIP
