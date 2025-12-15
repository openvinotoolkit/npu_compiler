//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/dialect/core/IR/ops.hpp>
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux::Core {
#define GEN_PASS_DECL_ADDNETINFOTOMODULE
#define GEN_PASS_DEF_ADDNETINFOTOMODULE
#include "vpux/compiler/dialect/core/passes.hpp.inc"
}  // namespace vpux::Core

using namespace vpux;

namespace {
//
// AddNetInfoToModule
//

class AddNetInfoToModule final : public Core::impl::AddNetInfoToModuleBase<AddNetInfoToModule> {
public:
    explicit AddNetInfoToModule(Logger log, bool hasTensorSemantics) {
        Base::initLogger(log, Base::getArgumentName());
        this->hasTensorSemantics = hasTensorSemantics;
    }

private:
    void safeRunOnModule() final;
};

net::NetworkInfoOp createNetInfoForFuncOp(mlir::func::FuncOp funcOp, bool hasTensorSemantics) {
    auto ctx = funcOp.getContext();
    mlir::OpBuilder builder(ctx);
    auto netInfo = builder.create<net::NetworkInfoOp>(appendLoc(funcOp.getLoc(), "nested_network_info"),
                                                      mlir::FlatSymbolRefAttr::get(ctx, funcOp.getName()), false);
    net::setupSections(netInfo);

    auto funcType = funcOp.getFunctionType();

    // Handle inputs
    auto& inputRegion = netInfo.getInputsInfo();
    builder.setInsertionPointToStart(&inputRegion.front());

    auto numOfInputs =
            hasTensorSemantics ? funcType.getNumInputs() : funcType.getNumInputs() - funcType.getNumResults();

    for (unsigned i = 0; i < numOfInputs; ++i) {
        auto argType = mlir::cast<vpux::NDTypeInterface>(funcType.getInput(i));
        const auto newType = mlir::RankedTensorType::get(argType.getShape(), argType.getElementType(), nullptr);
        auto name = formatv("in_{0}", i).str();
        builder.create<net::DataInfoOp>(appendLoc(funcOp.getLoc(), name), name, newType);
    }

    // Handle outputs
    auto& outputsRegion = netInfo.getOutputsInfo();
    builder.setInsertionPointToStart(&outputsRegion.front());

    for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
        auto resType = mlir::cast<vpux::NDTypeInterface>(funcType.getResult(i));
        const auto newType = mlir::RankedTensorType::get(resType.getShape(), resType.getElementType(), nullptr);
        auto name = formatv("out_{0}", i).str();
        builder.create<net::DataInfoOp>(appendLoc(funcOp.getLoc(), name), name, newType);
    }

    return netInfo;
}

void AddNetInfoToModule::safeRunOnModule() {
    auto module = getOperation();
    auto ctx = &getContext();
    mlir::OpBuilder builder(ctx);

    auto funcOps = module.getOps<mlir::func::FuncOp>();
    auto it = funcOps.begin();

    // Module without funcOp indicates reserved memory module. Skip this pass
    // for such modules.
    if (std::distance(it, funcOps.end()) == 0) {
        return;
    }

    if (std::distance(it, funcOps.end()) != 1) {
        module->emitError("Module must contain exactly one function to add NetworkInfoOp");
        return signalPassFailure();
    }

    if (!module.getOps<net::NetworkInfoOp>().empty()) {
        module->emitError("Module already contains a NetworkInfoOp, cannot add another one");
        return signalPassFailure();
    }

    auto netInfo = createNetInfoForFuncOp(*it, hasTensorSemantics);
    builder.setInsertionPointToStart(module.getBody());
    builder.insert(netInfo);
    _log.trace("Added NetworkInfoOp to module '{0}'", module.getSymName());
}

}  // namespace

//
// createAddNetInfoToModulePass
//

std::unique_ptr<mlir::Pass> vpux::Core::createAddNetInfoToModulePass(Logger log, bool hasTensorSemantics) {
    return std::make_unique<AddNetInfoToModule>(log, hasTensorSemantics);
}
