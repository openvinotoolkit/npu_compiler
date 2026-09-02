//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZEQUANTIZATIONFORATTENTION
#define GEN_PASS_DEF_OPTIMIZEQUANTIZATIONFORATTENTION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

struct DequantChain {
    IE::ConvertOp convert;
    IE::MultiplyOp multiply;
    mlir::Value dqScale;
};

bool matchDequantChain(IE::ScatterUpdateOp scatterOp, DequantChain& result) {
    auto mulOp = scatterOp.getInput().getDefiningOp<IE::MultiplyOp>();
    if (mulOp == nullptr) {
        return false;
    }
    for (auto [convertCandidate, scaleCandidate] :
         {std::pair{mulOp.getInput1(), mulOp.getInput2()}, std::pair{mulOp.getInput2(), mulOp.getInput1()}}) {
        auto convertOp = convertCandidate.getDefiningOp<IE::ConvertOp>();
        if (convertOp == nullptr) {
            continue;
        }
        auto inElemType = mlir::cast<mlir::RankedTensorType>(convertOp.getInput().getType()).getElementType();
        if (inElemType.isSignedInteger(8)) {
            result = {convertOp, mulOp, scaleCandidate};
            return true;
        }
    }
    return false;
}

struct QuantChain {
    IE::MultiplyOp multiply;
    IE::RoundOp round;
    IE::ClampOp clamp;
    IE::ConvertOp convert;
    mlir::Value qScale;
};

bool matchQuantChain(IE::ScatterUpdateOp scatterOp, QuantChain& result) {
    // Among the users of the ScatterUpdate output, find one that starts a quant chain.
    for (auto* user : scatterOp.getOutput().getUsers()) {
        auto mulOp = mlir::dyn_cast_if_present<IE::MultiplyOp>(user);
        if (mulOp == nullptr) {
            continue;
        }

        // Identify which operand of the Multiply is the scatter output and which is the scale.
        mlir::Value qScale;
        if (mulOp.getInput1() == scatterOp.getOutput()) {
            qScale = mulOp.getInput2();
        } else if (mulOp.getInput2() == scatterOp.getOutput()) {
            qScale = mulOp.getInput1();
        } else {
            continue;
        }

        // Multiply -> Round
        if (!mulOp.getOutput().hasOneUse()) {
            continue;
        }
        auto roundOp = mlir::dyn_cast_if_present<IE::RoundOp>(*mulOp.getOutput().getUsers().begin());
        if (roundOp == nullptr) {
            continue;
        }

        // Round -> Clamp
        if (!roundOp.getOutput().hasOneUse()) {
            continue;
        }
        auto clampOp = mlir::dyn_cast_if_present<IE::ClampOp>(*roundOp.getOutput().getUsers().begin());
        if (clampOp == nullptr) {
            continue;
        }

        // Verify clamp range matches si8: [-128, 127]
        constexpr double si8Min = -128.0;
        constexpr double si8Max = 127.0;
        const auto isValidSi8Range =
                (clampOp.getMin().convertToDouble() == si8Min && clampOp.getMax().convertToDouble() == si8Max);
        if (!isValidSi8Range) {
            continue;
        }

        // Clamp -> Convert(fp -> si8)
        if (!clampOp.getOutput().hasOneUse()) {
            continue;
        }
        auto convertOp = mlir::dyn_cast_if_present<IE::ConvertOp>(*clampOp.getOutput().getUsers().begin());
        if (convertOp == nullptr) {
            continue;
        }
        if (!convertOp.getDstElemType().isSignedInteger(8)) {
            continue;
        }

        result.multiply = mulOp;
        result.round = roundOp;
        result.clamp = clampOp;
        result.convert = convertOp;
        result.qScale = qScale;
        return true;
    }

    return false;
}

bool hasAttentionUser(mlir::Operation* op) {
    if (op == nullptr) {
        return false;
    }
    if (mlir::isa_and_present<IE::SDPAOp, IE::AttentionOp, IE::FlashSDPAOp>(op)) {
        return true;
    }
    if (!mlir::isa_and_present<IE::ViewLikeOpInterface, IE::BroadcastOp>(op)) {
        return false;
    }
    if (!op->getResult(0).hasOneUse()) {
        return false;
    }
    return hasAttentionUser(*op->getResult(0).getUsers().begin());
}

class OptimizeQuantizationForScatter final : public mlir::OpRewritePattern<IE::ScatterUpdateOp> {
public:
    OptimizeQuantizationForScatter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterUpdateOp>(ctx), _log(log) {
        setDebugName("OptimizeQuantizationForScatter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterUpdateOp scatterOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult OptimizeQuantizationForScatter::matchAndRewrite(IE::ScatterUpdateOp scatterOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), scatterOp->getName(), scatterOp->getLoc());

    // 1. Match dequantization chain on the data input side.
    DequantChain dequantChain;
    if (!matchDequantChain(scatterOp, dequantChain)) {
        return mlir::failure();
    }

    // 2. Match quantization chain on the output side.
    QuantChain quantChain;
    if (!matchQuantChain(scatterOp, quantChain)) {
        return mlir::failure();
    }

    // 3. Verify scatter output feeds an attention layer (skip the quant chain user)
    auto scatterFeedsAttention = llvm::any_of(scatterOp.getOutput().getUsers(), [&](mlir::Operation* user) {
        return user != quantChain.multiply.getOperation() && hasAttentionUser(user);
    });
    if (!scatterFeedsAttention) {
        return mlir::failure();
    }

    _log.trace("Matched KV cache ScatterUpdate pattern at '{0}'", scatterOp->getLoc());

    mlir::Value curToken = scatterOp.getUpdates();
    auto qMul = rewriter.create<IE::MultiplyOp>(takeOpLoc(scatterOp, "quant_mul"), curToken, quantChain.qScale,
                                                quantChain.multiply.getAutoBroadcastAttr());

    auto qRound = rewriter.create<IE::RoundOp>(takeOpLoc(scatterOp, "quant_round"), qMul.getOutput(),
                                               quantChain.round.getModeAttr());

    auto qClamp = rewriter.create<IE::ClampOp>(takeOpLoc(scatterOp, "quant_clamp"), qRound.getOutput(),
                                               quantChain.clamp.getMinAttr(), quantChain.clamp.getMaxAttr());
    auto qConvert = rewriter.create<IE::ConvertOp>(takeOpLoc(scatterOp, "quant_convert"), qClamp.getOutput(),
                                                   quantChain.convert.getDstElemTypeAttr());

    // ScatterUpdate in si8
    auto si8Scatter = rewriter.create<IE::ScatterUpdateOp>(
            takeOpLoc(scatterOp, "scatter_si8"), dequantChain.convert.getInput(), scatterOp.getIndices(),
            qConvert.getOutput(), scatterOp.getAxis(), scatterOp.getAxisValueAttr());

    // Dequantize for SDPA consumers: Convert + Multiply(dq_scale)
    auto dqConvert = rewriter.create<IE::ConvertOp>(takeOpLoc(scatterOp, "dequant_convert"), si8Scatter.getOutput(),
                                                    dequantChain.convert.getDstElemTypeAttr());
    auto dqMul = rewriter.create<IE::MultiplyOp>(takeOpLoc(scatterOp, "dequant_mul"), dqConvert.getOutput(),
                                                 dequantChain.dqScale, dequantChain.multiply.getAutoBroadcastAttr());

    rewriter.replaceAllUsesWith(quantChain.convert.getOutput(), si8Scatter.getOutput());
    rewriter.replaceAllUsesWith(scatterOp.getOutput(), dqMul.getOutput());

    rewriter.eraseOp(quantChain.convert);
    rewriter.eraseOp(quantChain.clamp);
    rewriter.eraseOp(quantChain.round);
    rewriter.eraseOp(quantChain.multiply);
    rewriter.eraseOp(scatterOp);

    if (dequantChain.multiply.getOutput().use_empty()) {
        rewriter.eraseOp(dequantChain.multiply);
    }
    if (dequantChain.convert.getOutput().use_empty()) {
        rewriter.eraseOp(dequantChain.convert);
    }
    return mlir::success();
}

//
// OptimizeQuantizationForAttention
//

class OptimizeQuantizationForAttention final :
        public IE::impl::OptimizeQuantizationForAttentionBase<OptimizeQuantizationForAttention> {
public:
    explicit OptimizeQuantizationForAttention(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void OptimizeQuantizationForAttention::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<OptimizeQuantizationForScatter>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createOptimizeQuantizationForAttentionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeQuantizationForAttentionPass(Logger log) {
    return std::make_unique<OptimizeQuantizationForAttention>(log);
}
