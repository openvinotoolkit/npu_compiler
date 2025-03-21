//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Pass/Pass.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_OUTLINELINALGSWLAYERS
#define GEN_PASS_DEF_OUTLINELINALGSWLAYERS
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {
mlir::func::FuncOp outlineSwLayer(mlir::MLIRContext* ctx, mlir::ModuleOp module, mlir::linalg::GenericOp layerOp,
                                  size_t counter) {
    auto dpsInputs = vpux::to_small_vector(layerOp.getInputs().getTypes());
    auto outputTypes = vpux::to_small_vector(layerOp.getOutputs().getTypes());

    // Preserve DPS for the function arguments
    dpsInputs.append(vpux::to_small_vector(outputTypes));
    auto funcType = mlir::FunctionType::get(ctx, dpsInputs, outputTypes);

    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody());

    auto funcName = printToString("generated_{0}", counter);
    auto funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), funcName, funcType);
    auto funcOpBody = funcOp.addEntryBlock();
    builder.setInsertionPointToEnd(funcOpBody);

    auto newLayerOp = builder.clone(*layerOp.getOperation());
    newLayerOp->setOperands(funcOpBody->getArguments());

    // Need to return the value produced by the computation ops, otherwise they will get removed at canonicalization
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), newLayerOp->getResults());
    return funcOp;
}

//
// OutlineLinalgSwLayersPass
//

class OutlineLinalgSwLayersPass final :
        public ShaveCodeGen::impl::OutlineLinalgSwLayersBase<OutlineLinalgSwLayersPass> {
public:
    explicit OutlineLinalgSwLayersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    };

private:
    void safeRunOnModule() final;
};

void OutlineLinalgSwLayersPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto moduleOp = getOperation();
    IE::CNNNetworkOp netInfo;
    mlir::func::FuncOp func;
    IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, func);

    auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);
    size_t counter = 0;

    func.walk([&](mlir::linalg::GenericOp op) {
        auto outlinedFunc = outlineSwLayer(&ctx, swModule, op, counter);
        mlir::OpBuilder builder(&ctx);
        builder.setInsertionPointAfter(op);
        auto fullSymRef = mlir::SymbolRefAttr::get(
                swModule.getSymNameAttr(),
                llvm::ArrayRef<mlir::FlatSymbolRefAttr>(mlir::FlatSymbolRefAttr::get(outlinedFunc)));

        auto genericSwLayerOp =
                builder.create<VPU::GenericSwLayerOp>(op.getLoc(), op.getResultTypes(), fullSymRef, op.getInputs());
        op.replaceAllUsesWith(genericSwLayerOp.getResults());
        op.erase();
        counter++;
    });
}

}  // namespace

//
// createOutlineLinalgSwLayersPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createOutlineLinalgSwLayersPass(Logger log) {
    return std::make_unique<OutlineLinalgSwLayersPass>(log);
}
