//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <functional>

namespace vpux {
#define GEN_PASS_DECL_ADDBUFFERSFORNETRESULTS
#define GEN_PASS_DEF_ADDBUFFERSFORNETRESULTS
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

// Updates the func op and entry block.
//
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
// IE::CNNNetworkOp, but in tests this information may be unavailable, so empty callback will be returned
std::function<const std::optional<mlir::Location>(mlir::OpOperand&)> getResultLocationProvider(mlir::func::FuncOp func,
                                                                                               vpux::Logger log) {
    auto moduleOp = getModuleOp(func);
    auto netOps = to_small_vector(moduleOp.getOps<IE::CNNNetworkOp>());
    if (netOps.size() != 1) {
        log.warning("Can't get location for output. If it isn't a test, please, debug this.");
        return [](mlir::OpOperand&) -> const std::optional<mlir::Location> {
            return std::nullopt;
        };
    }

    IE::CNNNetworkOp netOp = netOps.front();
    mlir::func::FuncOp entryPointFuncOp;
    IE::CNNNetworkOp::getFromModule(moduleOp, netOp, entryPointFuncOp);

    if (func == entryPointFuncOp) {
        const auto outputsInfo = to_small_vector(netOp.getOutputsInfo().getOps<IE::DataInfoOp>());
        return [=](mlir::OpOperand& operand) -> const std::optional<mlir::Location> {
            const auto loc = outputsInfo[operand.getOperandNumber()]->getLoc();
            VPUX_THROW_WHEN(loc.isa<mlir::UnknownLoc>(), "Network output {0} must have location",
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

// Updates all ReturnOps in the scope of the given FuncOp by copying the associated buffer contents into the given
// out-params.
void updateReturnOps(mlir::func::FuncOp func, ArrayRef<mlir::BlockArgument> appendedEntryArgs, vpux::Logger log) {
    const auto locProvider = getResultLocationProvider(func, log);
    func.walk([&](mlir::func::ReturnOp op) {
        mlir::OpBuilder builder(op);
        for (auto& opOperand : op->getOpOperands()) {
            auto opLoc = op->getLoc();
            if (auto realLoc = locProvider(opOperand)) {
                opLoc = realLoc.value();
            }
            auto idx = opOperand.getOperandNumber();
            auto copyOp = builder.create<VPUIP::CopyOp>(opLoc, op.getOperand(idx), appendedEntryArgs[idx]);
            opOperand.set(copyOp.getOutput());
        }
    });
}

// Updates call op
void updateCallOp(mlir::ModuleOp module) {
    module.walk([&](mlir::func::CallOp callOp) {
        mlir::OpBuilder builder(callOp);

        SmallVector<mlir::Value> outParams;
        SmallVector<mlir::Value> currentResults;
        SmallVector<mlir::Type> resultTypes;
        for (auto result : callOp.getResults()) {
            auto resType = result.getType().dyn_cast<mlir::MemRefType>();
            // TODO: E#140551 add support for VPUIP.SparseBuffer, possibly use allocateBuffers()
            VPUX_THROW_WHEN(resType == nullptr, "Only MemRefType is supported for now, got {0}", result.getType());

            auto outParam = builder.create<mlir::memref::AllocOp>(callOp.getLoc(), resType);
            outParams.push_back(outParam);

            currentResults.push_back(result);
            resultTypes.push_back(resType);
        }

        auto newOperands = to_vector(callOp.getOperands());
        newOperands.append(outParams.begin(), outParams.end());

        auto newCallOp =
                builder.create<mlir::func::CallOp>(callOp.getLoc(), callOp.getCalleeAttr(), resultTypes, newOperands);

        for (const auto& [result, newResult] : zip(currentResults, newCallOp.getResults())) {
            result.replaceAllUsesWith(newResult);
        }

        newCallOp->setAttrs(callOp->getAttrs());
        callOp.erase();
    });
}

//
// AddBuffersForNetResults
//

class AddBuffersForNetResults final : public impl::AddBuffersForNetResultsBase<AddBuffersForNetResults> {
public:
    explicit AddBuffersForNetResults(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

//
// safeRunOnFunc
//

void AddBuffersForNetResults::safeRunOnModule() {
    auto module = getOperation();

    for (auto func : module.getOps<mlir::func::FuncOp>()) {
        if (func.isExternal()) {
            _log.trace("Can't convert external Function '@{0}'", func.getSymName());
            signalPassFailure();
        }

        SmallVector<mlir::BlockArgument> appendedEntryArgs;
        updateFuncOp(func, appendedEntryArgs);
        updateReturnOps(func, appendedEntryArgs, _log);
    }

    updateCallOp(module);
}

}  // namespace

//
// createAddBuffersForNetResults
//

std::unique_ptr<mlir::Pass> vpux::createAddBuffersForNetResults(Logger log) {
    return std::make_unique<AddBuffersForNetResults>(log);
}
