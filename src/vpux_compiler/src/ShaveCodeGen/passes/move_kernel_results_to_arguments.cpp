//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Bufferization/Transforms/Transforms.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/Utils/MemRefUtils.h>
#include <mlir/Pass/Pass.h>

#include <iterator>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_MOVEKERNELRESULTSTOARGUMENTS
#define GEN_PASS_DEF_MOVEKERNELRESULTSTOARGUMENTS
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

//
// MoveKernelResultsToArgumentsPass
//

class MoveKernelResultsToArgumentsPass final :
        public ShaveCodeGen::impl::MoveKernelResultsToArgumentsBase<MoveKernelResultsToArgumentsPass> {
public:
    explicit MoveKernelResultsToArgumentsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    };

private:
    void safeRunOnModule() final;
};

void MoveKernelResultsToArgumentsPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    auto& ctx = getContext();
    auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);
    auto funcOps = swModule.getOps<mlir::func::FuncOp>();

    for (auto func : funcOps) {
        if (func.isExternal()) {
            continue;
        }

        auto block = &func.getBody().front();
        auto terminator = mlir::cast<mlir::func::ReturnOp>(block->getTerminator());
        auto loc = terminator->getLoc();

        auto inputs = vpux::to_small_vector(block->getArgumentTypes());
        auto outputs = vpux::to_small_vector(terminator->getOperandTypes());

        const auto isTensor = [](mlir::Type type) {
            return mlir::isa<mlir::TensorType>(type);
        };
        if (llvm::none_of(outputs, isTensor)) {
            continue;
        }
        if (!llvm::all_of(outputs, isTensor)) {
            func.emitError("expected kernel function '")
                    << func.getSymName() << "' to return only tensors, but got a mix of tensor and non-tensor results";
            return signalPassFailure();
        }

        // Find insertion point: before the first scalar argument
        auto firstScalarIt = llvm::find_if(inputs, [](auto type) {
            return !mlir::isa<mlir::ShapedType>(type);
        });
        size_t insertionPoint = std::distance(inputs.begin(), firstScalarIt);

        // Insert bufferized arguments before the first scalar argument
        size_t currentInsertPos = insertionPoint;
        for (auto outputType : outputs) {
            auto memrefTy = mlir::bufferization::getMemRefTypeWithStaticIdentityLayout(
                    mlir::cast<mlir::TensorType>(outputType));
            inputs.insert(inputs.begin() + currentInsertPos, memrefTy);
            block->insertArgument(currentInsertPos, memrefTy, loc);
            currentInsertPos++;
        }

        auto builder = mlir::OpBuilder::atBlockEnd(block);
        for (auto it : terminator->getOpOperands() | indexed) {
            auto correspondentInput = block->getArgument(insertionPoint + it.index());
            auto opOperand = &it.value();
            auto mat = builder.create<mlir::bufferization::MaterializeInDestinationOp>(loc, opOperand->get(),
                                                                                       correspondentInput);
            // Set writable and restrict attributes to enable empty tensor elimination and possibly other
            // optimizations. The result buffer is clearly writable and restrict is correct because result
            // buffers should not alias other arguments.
            mat.setWritable(true);
            mat.setRestrict(true);
        }

        builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{});
        terminator->erase();

        func.setType(mlir::FunctionType::get(&ctx, inputs, {}));
    }

    mlir::IRRewriter rewriter(&ctx);
    if (failed(mlir::bufferization::eliminateEmptyTensors(rewriter, swModule))) {
        return signalPassFailure();
    }
}

}  // namespace

//
// createMoveKernelResultsToArgumentsPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createMoveKernelResultsToArgumentsPass(Logger log) {
    return std::make_unique<MoveKernelResultsToArgumentsPass>(log);
}
