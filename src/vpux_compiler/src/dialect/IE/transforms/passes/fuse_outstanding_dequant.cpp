//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEOUTSTANDINGDEQUANT
#define GEN_PASS_DEF_FUSEOUTSTANDINGDEQUANT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseOutstandingDequantPass
//

class FuseOutstandingDequantPass final : public IE::impl::FuseOutstandingDequantBase<FuseOutstandingDequantPass> {
public:
    explicit FuseOutstandingDequantPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
};

mlir::LogicalResult FuseOutstandingDequantPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    return mlir::success();
}

class DequantizeWithNCERewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeWithNCERewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DequantizeWithNCERewriter::matchAndRewrite(IE::DequantizeOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    // Get the Dequantize Op input and output types
    const auto dequantInputType = origOp.getInput().getType();
    const auto elemType = mlir::cast<mlir::ShapedType>(dequantInputType).getElementType();
    const auto dequantUniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType);
    const bool isPerChannel = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(elemType);

    const auto dequantOutputType = origOp.getOutput().getType();

    auto maybeNCETask = origOp.getInput().getDefiningOp();
    if (maybeNCETask == nullptr) {
        return matchFailed(rewriter, origOp, "Producer is a block argument");
    }
    if (!maybeNCETask->getResult(0).hasOneUse()) {
        return matchFailed(rewriter, origOp, "NCE task has more than one consumer");
    }

    if (!mlir::isa<IE::LayerWithPostOpInterface>(maybeNCETask)) {
        SmallVector<mlir::Operation*> targetOps;
        mlir::Operation* operation = origOp;
        _log.trace("[{0}] Search quantized NCE task for {1} at {2}", this->getDebugName(), origOp->getName(),
                   origOp->getLoc());
        while (operation) {
            auto input = (*operation->getOperands().begin()).getDefiningOp();

            if (!mlir::isa<IE::ElemTypeInfoOpInterface, IE::LayerWithPostOpInterface>(input)) {
                return matchFailed(rewriter, origOp,
                                   "Ancestor {0} at {1} is neither FakeQuantize agnostic operation nor NCE operation",
                                   input->getName(), input->getLoc());
            }

            if (!input->hasOneUse()) {
                return matchFailed(rewriter, origOp, "Ancestor {0} at {1} has more than one consumer", input->getName(),
                                   input->getLoc());
            }

            if (mlir::isa<IE::ElemTypeInfoOpInterface>(input)) {
                if (input->getNumOperands() > 1) {
                    return matchFailed(rewriter, origOp, "Ancestor {0} at {1} has more than one ancestors",
                                       input->getName(), input->getLoc());
                }
                _log.trace("[{0}] Push ElemTypeInfoOpInterface {1} at {2}", this->getDebugName(), input->getName(),
                           input->getLoc());
                targetOps.push_back(input);
                operation = input;
                continue;
            }

            if (mlir::isa<IE::LayerWithPostOpInterface>(input)) {
                _log.trace("[{0}] Found NCE task {1} at {2}, stop pattern searching", this->getDebugName(),
                           input->getName(), input->getLoc());
                maybeNCETask = input;
                break;
            }
        }

        _log.trace("[{0}] Capture the pattern for {1} at {2}", this->getDebugName(), origOp->getName(),
                   origOp->getLoc());

        if (!IE::arch37xx::isMixPrecisionSupported(maybeNCETask, !isPerChannel, _log)) {
            return matchFailed(rewriter, origOp, "Producer {0} is not supported", maybeNCETask->getName());
        }

        auto* newNCETask = rewriter.clone(*maybeNCETask);
        vpux::NDTypeInterface newType = newNCETask->getResult(0).getType();
        newType = newType.changeElemType(dequantOutputType.getElementType());
        newNCETask->getResult(0).setType(newType);
        newNCETask->moveBefore(targetOps.back());

        _log.trace("[{0}] Replace {1} {2} at {3} with {4} {5} at {6}", this->getDebugName(), maybeNCETask->getName(),
                   maybeNCETask->getResult(0).getType(), maybeNCETask->getLoc(), newNCETask->getName(),
                   newNCETask->getResult(0).getType(), newNCETask->getLoc());
        rewriter.replaceOp(maybeNCETask, newNCETask->getResult(0));

        // [NCE with quantized output]->[ElemTypeInfoOpInterface] ... ->[Dequantize] pattern is captured
        // Rewrite the sub-graph.
        for (auto iterator = targetOps.rbegin(); iterator != targetOps.rend(); ++iterator) {
            _log.trace("[{0}] Change {1} at {2} to {3}", this->getDebugName(), (*iterator)->getName(),
                       (*iterator)->getLoc(), (*iterator)->getResult(0).getType());
            inferReturnTypes(*iterator, InferShapedTypeMode::ELEM_TYPE);
        }

        // Remove old Dequantize ops.
        _log.trace("[{0}] Replace {1} at {2} with {3} at {4}", this->getDebugName(), origOp->getName(),
                   origOp->getLoc(), targetOps.front()->getName(), targetOps.front()->getLoc());
        rewriter.replaceOp(origOp, targetOps.front()->getResult(0));
    } else {
        if (!IE::arch37xx::isMixPrecisionSupported(maybeNCETask, !isPerChannel, _log)) {
            return matchFailed(rewriter, origOp, "Producer {0} is not supported", maybeNCETask->getName());
        }

        auto* newNCETask = rewriter.clone(*maybeNCETask);
        mlir::Value result = newNCETask->getResult(0);
        result.setType(dequantOutputType);
        if (dequantUniformType && !isPerChannel && !dequantUniformType.isSigned() &&
            dequantUniformType.getZeroPoint() == 0) {
            // Replace implicit ReLU in the Dequantize Op with an actual ReLU Op
            result = rewriter.create<IE::ReLUOp>(takeOpLoc(origOp, "as_relu"), dequantOutputType, result);
        }
        rewriter.replaceOp(origOp, result);
        rewriter.eraseOp(maybeNCETask);
    }

    return mlir::success();
}

void FuseOutstandingDequantPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DequantizeWithNCERewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFuseOutstandingDequant
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseOutstandingDequant(Logger log) {
    return std::make_unique<FuseOutstandingDequantPass>(log);
}
