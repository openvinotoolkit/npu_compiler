//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/analysis.hpp"
#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Pass/Pass.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_ENCAPSULATECODEGENOPS
#define GEN_PASS_DEF_ENCAPSULATECODEGENOPS
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

//
// EncapsulateCodeGenOpsPass
//

class EncapsulateCodeGenOpsPass final :
        public ShaveCodeGen::impl::EncapsulateCodeGenOpsBase<EncapsulateCodeGenOpsPass> {
public:
    explicit EncapsulateCodeGenOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    mlir::Operation* wrapInCodeGenCapsule(mlir::Operation* op, mlir::OpBuilder& opBuilder);
};

mlir::Operation* EncapsulateCodeGenOpsPass::wrapInCodeGenCapsule(mlir::Operation* op, mlir::OpBuilder& opBuilder) {
    opBuilder.setInsertionPoint(op);
    auto cgCapsule = IE::CodeGenCapsuleOp::create(opBuilder, op->getLoc(), op->getResultTypes(), op->getOperands());
    auto& cgBlock = cgCapsule.getContent().emplaceBlock();

    opBuilder.setInsertionPointToEnd(&cgBlock);
    auto yieldOp = IE::CGCYieldOp::create(opBuilder, opBuilder.getUnknownLoc(), mlir::ValueRange());

    op->replaceAllUsesWith(cgCapsule->getResults());
    op->moveBefore(yieldOp);
    yieldOp->setOperands(op->getResults());

    for (auto operand : op->getOperands()) {
        auto arg = cgBlock.addArgument(operand.getType(), operand.getLoc());
        op->replaceUsesOfWith(operand, arg);
    }
    return cgCapsule;
}

void EncapsulateCodeGenOpsPass::safeRunOnFunc() {
    auto func = getOperation();

    auto builder = mlir::OpBuilder::atBlockBegin(func->getBlock());

    auto& fusionChainAnalysis = getAnalysis<ShaveCodeGen::FusionChainAnalysis>();

    auto computeOpChains = fusionChainAnalysis.getComputeOpChains();

    for (const auto& chain : computeOpChains) {
        assert(chain.size() > 0);
        std::vector<mlir::Operation*> newCgcChain = {};
        for (auto computeOp : chain) {
            auto cgcOp = wrapInCodeGenCapsule(computeOp, builder);
            newCgcChain.push_back(cgcOp);
        }
        fusionChainAnalysis.appendCodeGenCapsuleChain(newCgcChain);
    }
    fusionChainAnalysis.setState(ShaveCodeGen::FusionChainAnalysis::State::CodeGenCapsuleChains);
    markAllAnalysesPreserved();
}

}  // namespace

//
// createEncapsulateCodeGenOpsPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createEncapsulateCodeGenOpsPass(Logger log) {
    return std::make_unique<EncapsulateCodeGenOpsPass>(log);
}
