//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/allocate_buffers_for_net_results.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <functional>

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
    auto netInfoOps = to_small_vector(moduleOp.getOps<net::NetworkInfoOp>());
    if (netInfoOps.size() != 1) {
        log.warning("Can't get location for output. If it isn't a test, please, debug this.");
        return [](mlir::OpOperand&) -> const std::optional<mlir::Location> {
            return std::nullopt;
        };
    }

    net::NetworkInfoOp netInfo = netInfoOps.front();
    mlir::func::FuncOp entryPointFuncOp;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, entryPointFuncOp);

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

// Updates call op
void updateCallOp(ArrayRef<mlir::CallOpInterface> callOps, vpux::Logger& log) {
    for (auto callOp : llvm::make_early_inc_range(callOps)) {
        mlir::OpBuilder builder(callOp);

        SmallVector<mlir::Value> outParams;
        SmallVector<mlir::Value> currentResults;
        SmallVector<mlir::Type> resultTypes;
        for (auto result : callOp->getResults()) {
            mlir::Type resType = result.getType();
            // E-140551: add support for VPUIP.SparseBuffer, allocateBuffersOfType has the allocation logic for
            // VPUIP.SparseBuffer. Need real use cases. Remove the following VPUX_THROW_WHEN to check if it works.
            VPUX_THROW_WHEN(
                    !mlir::isa<mlir::MemRefType>(resType) && !mlir::isa<vpux::VPUIP::DistributedBufferType>(resType),
                    "Only MemRefType and DistributedBufferType are supported for now, got {0}", result.getType());

            auto outParam = allocateBuffersOfType(log, callOp.getLoc(), builder, resType).front();
            outParams.push_back(outParam);

            currentResults.push_back(result);
            resultTypes.push_back(resType);
        }

        auto newOperands = to_vector(callOp->getOperands());
        newOperands.append(outParams.begin(), outParams.end());

        auto newCallOp = callOp->clone();
        newCallOp->setOperands(newOperands);
        builder.insert(newCallOp);

        callOp->replaceAllUsesWith(newCallOp->getResults());

        newCallOp->setAttrs(callOp->getAttrs());
        callOp.erase();
    }
}

}  // namespace

template <typename CopyOp>
void vpux::allocateBuffersForNetResults(ArrayRef<mlir::CallOpInterface> callOps, ArrayRef<mlir::func::FuncOp> funcOps,
                                        vpux::Logger& log) {
    for (auto func : funcOps) {
        SmallVector<mlir::BlockArgument> appendedEntryArgs;
        updateFuncOp(func, appendedEntryArgs);
        updateReturnOps<CopyOp>(func, appendedEntryArgs, log);
    }

    updateCallOp(callOps, log);
}

template void vpux::allocateBuffersForNetResults<VPUIP::CopyOp>(ArrayRef<mlir::CallOpInterface> callOps,
                                                                ArrayRef<mlir::func::FuncOp> funcOps, Logger& log);
template void vpux::allocateBuffersForNetResults<mlir::memref::CopyOp>(ArrayRef<mlir::CallOpInterface> callOps,
                                                                       ArrayRef<mlir::func::FuncOp> funcOps,
                                                                       Logger& log);
