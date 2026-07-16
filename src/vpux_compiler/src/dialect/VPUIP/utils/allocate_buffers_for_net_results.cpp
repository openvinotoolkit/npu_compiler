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

bool isUsedOnlyByMemrefCopy(mlir::Value value) {
    auto users = value.getUsers();
    return std::distance(users.begin(), users.end()) == 1 && mlir::isa<mlir::memref::CopyOp>(*users.begin());
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

            mlir::Value outParam = nullptr;
            if (isUsedOnlyByMemrefCopy(result)) {
                auto funcOp = getCalledFunction(callOp);
                auto funcType = funcOp.getFunctionType();
                size_t numInputs = funcType.getNumInputs() - funcType.getNumResults();
                auto memRefType = mlir::cast<mlir::MemRefType>(funcType.getInput(numInputs + index));
                mlir::memref::CopyOp copyOp = mlir::cast<mlir::memref::CopyOp>(*result.getUsers().begin());
                outParam = copyOp.getTarget();
                builder.setInsertionPointAfter(copyOp);
                if (memRefType != outParam.getType()) {
                    auto castBufferOp = builder.create<mlir::memref::CastOp>(callOp.getLoc(), memRefType, outParam);
                    outParam = castBufferOp.getResult();
                }
                copyOp.erase();
            }
            if (outParam == nullptr) {
                // Dynamic memrefs in outlined kernel results must be allocated using upper bounds
                // from the callee's NetworkInfo so the buffer is large enough for the worst-case
                // runtime shape. Static memrefs and non-outlined calls use the standard path
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
void allocateBuffersForNetResults(const mlir::DenseSet<mlir::CallOpInterface>& callOps,
                                  const mlir::DenseSet<mlir::func::FuncOp>& funcOps, vpux::Logger& log) {
    for (auto func : funcOps) {
        SmallVector<mlir::BlockArgument> appendedEntryArgs;
        updateFuncOp(func, appendedEntryArgs);
        updateReturnOps<CopyOp>(func, appendedEntryArgs, log);
    }

    updateCallOp(callOps, log);
}

template void allocateBuffersForNetResults<VPUIP::CopyOp>(const mlir::DenseSet<mlir::CallOpInterface>& callOps,
                                                          const mlir::DenseSet<mlir::func::FuncOp>& funcOps,
                                                          Logger& log);
template void allocateBuffersForNetResults<mlir::memref::CopyOp>(const mlir::DenseSet<mlir::CallOpInterface>& callOps,
                                                                 const mlir::DenseSet<mlir::func::FuncOp>& funcOps,
                                                                 Logger& log);
}  // namespace vpux::VPUIP
