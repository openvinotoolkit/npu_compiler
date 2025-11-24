//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPTRANSPOSEWITHFQ
#define GEN_PASS_DEF_SWAPTRANSPOSEWITHFQ
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// SwapTransposeWithFQ
//

class SwapTransposeWithFQ final : public IE::impl::SwapTransposeWithFQBase<SwapTransposeWithFQ> {
public:
    explicit SwapTransposeWithFQ(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class TransposeOpConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// TransposeOpConverter
//

class SwapTransposeWithFQ::TransposeOpConverter final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    TransposeOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TransposeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapTransposeWithFQ::TransposeOpConverter::matchAndRewrite(IE::TransposeOp origOp,
                                                                               mlir::PatternRewriter& rewriter) const {
    const auto transposeIn = origOp.getInput();
    if (auto origQuantOp = transposeIn.getDefiningOp<IE::QuantizeOp>()) {
        auto transposeOp = rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_in"), origQuantOp.getInput(),
                                                            nullptr, origOp.getOrderValueAttr());

        auto newOp = rewriter.replaceOpWithNewOp<IE::QuantizeOp>(origOp, transposeOp.getOutput(),
                                                                 origQuantOp.getDstElemType());
        extendOpLoc(newOp, "as_quant");
    } else if (auto origFqOp = transposeIn.getDefiningOp<IE::FakeQuantizeOp>()) {
        auto transposeOp = rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_in"), origFqOp.getInput(),
                                                            nullptr, origOp.getOrderValueAttr());

        auto newOp = rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(
                origOp, transposeOp.getOutput(), origFqOp.getInputLow(), origFqOp.getInputHigh(),
                origFqOp.getOutputLow(), origFqOp.getOutputHigh(), origFqOp.getLevelsAttr(),
                origFqOp.getLowFpTypeAttr(), origFqOp.getAutoBroadcast());
        extendOpLoc(newOp, "as_fq");
    }

    return mlir::success();
}

void SwapTransposeWithFQ::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalOp = [&](IE::TransposeOp op) -> bool {
        const auto transposeIn = op.getInput();
        if (auto maybeQuantOp = transposeIn.getDefiningOp<IE::QuantizeOp>()) {
            // Check that Quantize has per-tensor quantization.
            const auto axis = IE::getQuantAxisIndex(maybeQuantOp, _log);
            if (axis.has_value()) {
                return true;
            }

            // It turned out that this approach gives performance gain mostly in this case:
            // NetworkInput (NCHW) -> Quantize -> Transpose
            // Quantize will eventually become an NCE task, which requires NHWC layout.
            // If Quantize and Transpose is swapped, transpose and NHWC repack can be fused together.
            // Also, sometimes such fusion results in PermuteCast, which does nothing in runtime.
            return !mlir::isa<mlir::BlockArgument>(maybeQuantOp.getInput());
        } else if (auto maybeFqOp = transposeIn.getDefiningOp<IE::FakeQuantizeOp>()) {
            // Check that FQ has per-tensor quantization.
            if (!IE::isPerTensorFQ({maybeFqOp})) {
                return true;
            }

            // For OV 2.0 API U8 we can have:
            // NetworkInput (NCHW) -> Convert -> FQ -> Transpose. Because of this will remain a
            // dequantize layer, this dequant layer will introduce 2 mem permutes because of the layout.
            // This Transpose will be done as PermuteCast lately.
            if (mlir::isa_and_nonnull<IE::ConvertOp>(maybeFqOp.getInput().getDefiningOp()) &&
                mlir::isa<mlir::BlockArgument>(maybeFqOp.getInput().getDefiningOp()->getOperand(0)) &&
                mlir::isa_and_nonnull<IE::FakeQuantizeOp>(*op.getResult().getUsers().begin())) {
                return false;
            }

            return !mlir::isa<mlir::BlockArgument>(maybeFqOp.getInput());
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::TransposeOp>(isLegalOp);
    target.addLegalOp<IE::QuantizeOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapTransposeWithFQ::TransposeOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapTransposeWithFQPass(Logger log) {
    return std::make_unique<SwapTransposeWithFQ>(log);
}
