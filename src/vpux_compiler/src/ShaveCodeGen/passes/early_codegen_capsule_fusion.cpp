//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/analysis.hpp"
#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/iterator_range.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_EARLYCODEGENCAPSULEFUSION
#define GEN_PASS_DEF_EARLYCODEGENCAPSULEFUSION
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

//
// EarlyCodeGenCapsuleFusionPass
//

class EarlyCodeGenCapsuleFusionPass final :
        public ShaveCodeGen::impl::EarlyCodeGenCapsuleFusionBase<EarlyCodeGenCapsuleFusionPass> {
public:
    explicit EarlyCodeGenCapsuleFusionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

IE::CodeGenCapsuleOp fuseCapsules(IE::CodeGenCapsuleOp producer, IE::CodeGenCapsuleOp consumer) {
    auto producerBlock = producer.getBody();
    auto consumerBlock = consumer.getBody();

    // Simplistic approach as at this point, consumer capsule always contains only 1 compute op + yield
    auto consumerComputeOps = vpux::to_small_vector(consumerBlock->getOps<IE::ShaveCodeGenSupportedOpInterface>());
    VPUX_THROW_UNLESS(consumerComputeOps.size() == 1, "Only 1 compute op is expected in consumer capsule");
    auto producerYield = mlir::cast<IE::CGCYieldOp>(producerBlock->getTerminator());
    auto opToMove = consumerComputeOps[0];

    auto producerResults = producer->getResults();
    auto producerOperands = producer->getOperands();

    // Track producer operands as vector, such that additional operands can be appended
    // A concrete case for this is when the cosumer has a BlockArg dependency which is not present in the producer
    // capsule
    auto possiblyUpdatedProducerOperands = to_small_vector(producerOperands);

    // Map the "opToMove" to it's new operands
    mlir::IRMapping mapper;
    for (auto operandIt : consumer->getOperands() | indexed) {
        // First check whether the consumer operand is a direct result of the producer capsule
        auto opResultCorrespondent = llvm::find(producerResults, operandIt.value());
        if (opResultCorrespondent != producerResults.end()) {
            mapper.map(opToMove->getOperand(operandIt.index()),
                       producerYield->getOperand(opResultCorrespondent.getIndex()));
            continue;
        }
        // Then check whether the consumer operand is an already-existant operand of the producer capsule
        auto producerOperandCorrespondent = llvm::find(producerOperands, operandIt.value());
        if (producerOperandCorrespondent != producerOperands.end()) {
            mapper.map(opToMove->getOperand(operandIt.index()),
                       producerBlock->getArgument(producerOperandCorrespondent.getIndex()));
            continue;
        }
        // Final possibility is that the operand is a BlockArg which is not yet used by the producer capsule
        // If this is the case, add it to the producer operands vector & append it to the producer block arguments
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operandIt.value());
        if (blockArg) {
            auto newArg = producerBlock->addArgument(blockArg.getType(), producer->getLoc());
            possiblyUpdatedProducerOperands.push_back(blockArg);
            mapper.map(opToMove->getOperand(operandIt.index()), newArg);
        } else {
            VPUX_THROW(
                    "Unsupported and/or unexpected external dependency detected in the CodeGenCapsule fusion process.");
        }
    }

    // Move consumer compute op in producer block, remap it's operands & update accordingly the yield
    opToMove->moveBefore(producerYield);
    for (auto valueMapping : mapper.getValueMap()) {
        opToMove->replaceUsesOfWith(valueMapping.getFirst(), valueMapping.getSecond());
    }
    // This assumes that final fused capsule results are fixed to be the consumer capsule results
    producerYield->setOperands(opToMove->getResults());

    // Need to create new CodeGenCapsule, as op results are immutable
    mlir::OpBuilder builder(producer->getContext());
    builder.setInsertionPoint(producer);
    auto fusedOp = IE::CodeGenCapsuleOp::create(builder, producer->getLoc(), consumer->getResultTypes(),
                                                possiblyUpdatedProducerOperands);
    producer.replaceAllUsesWith(fusedOp->getResults());
    consumer.replaceAllUsesWith(fusedOp->getResults());

    // Move fused block to the newly created fused CodeGenCapsule
    auto dummyBlock = &(fusedOp.getContent().emplaceBlock());
    producerBlock->moveBefore(dummyBlock);
    dummyBlock->erase();

    // Erase the ops that were fused, as the new CodeGenCapsule fully replaces them
    producer.erase();
    consumer.erase();
    return fusedOp;
}

void EarlyCodeGenCapsuleFusionPass::safeRunOnFunc() {
    auto fusionChainAnalysisOpt = getCachedAnalysis<ShaveCodeGen::FusionChainAnalysis>();
    if (!fusionChainAnalysisOpt.has_value()) {
        mlir::emitError(getOperation()->getLoc(), "FusionChainAnalysis is expected to be cached");
        signalPassFailure();
        return;
    }
    auto& fusionChainAnalysis = fusionChainAnalysisOpt.value().get();

    auto chains = fusionChainAnalysis.getCodeGenCapsulesChains();
    for (auto chain : chains) {
        if (chain.size() > 1) {
            auto concreteChainHead = mlir::cast<IE::CodeGenCapsuleOp>(chain[0]);
            auto chainIt = chain.begin();
            chainIt++;
            auto toFuseRange = llvm::make_range(chainIt, chain.end());
            for (auto toBeFusedOp : llvm::make_early_inc_range(toFuseRange)) {
                concreteChainHead = fuseCapsules(mlir::cast<IE::CodeGenCapsuleOp>(concreteChainHead),
                                                 mlir::cast<IE::CodeGenCapsuleOp>(toBeFusedOp));
            }
        }
    }
    fusionChainAnalysis.invalidate();
}

}  // namespace

//
// createEarlyCodeGenCapsuleFusionPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createEarlyCodeGenCapsuleFusionPass(Logger log) {
    return std::make_unique<EarlyCodeGenCapsuleFusionPass>(log);
}
