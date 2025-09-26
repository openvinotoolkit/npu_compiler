//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTSUBTRACTTOADD
#define GEN_PASS_DEF_CONVERTSUBTRACTTOADD
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

Const::DeclareOp getInputConstantOp(mlir::Value input) {
    if (input == nullptr) {
        return nullptr;
    }
    if (auto inputFq = input.getDefiningOp<IE::FakeQuantizeOp>()) {
        if (auto fqConstInput = inputFq.getInput().getDefiningOp<Const::DeclareOp>()) {
            return fqConstInput;
        }
    } else if (auto inputConst = input.getDefiningOp<Const::DeclareOp>()) {
        return inputConst;
    }
    return nullptr;
}

mlir::Value createNegativeFqVal(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value fqVal,
                                mlir::RankedTensorType storageType) {
    auto valConst = fqVal.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(valConst.getContentAttr().isSplat(), "Second input's FQ constant is not splat");
    auto valConstContent = valConst.getContent();
    auto inValue = valConstContent.getSplatValue<float>();
    auto negativeValue = (inValue == 0) ? 0 : -1 * inValue;
    return Const::createFloatConst(rewriter, loc, storageType, negativeValue);
}

IE::FakeQuantizeOp createNewFq(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value fqInput,
                               IE::FakeQuantizeOp initialFqOp) {
    auto fqValType = mlir::cast<vpux::NDTypeInterface>(initialFqOp.getInputHigh().getType()).getElementType();
    const auto storageType = mlir::RankedTensorType::get({1, 1, 1, 1}, fqValType);

    mlir::Value inLow =
            createNegativeFqVal(rewriter, appendLoc(loc, "in_low"), initialFqOp.getInputHigh(), storageType);
    mlir::Value inHigh =
            createNegativeFqVal(rewriter, appendLoc(loc, "in_high"), initialFqOp.getInputLow(), storageType);
    mlir::Value outLow =
            createNegativeFqVal(rewriter, appendLoc(loc, "out_low"), initialFqOp.getOutputHigh(), storageType);
    mlir::Value outHigh =
            createNegativeFqVal(rewriter, appendLoc(loc, "out_high"), initialFqOp.getOutputLow(), storageType);

    return rewriter.create<IE::FakeQuantizeOp>(appendLoc(loc, "new_fq"), fqInput, inLow, inHigh, outLow, outHigh,
                                               initialFqOp.getLevelsAttr(), initialFqOp.getLowFpTypeAttr(),
                                               initialFqOp.getAutoBroadcastAttr());
}

//
// ConvertSubtractToDWConvAdd
//

class ConvertSubtractToDWConvAdd final : public mlir::OpRewritePattern<IE::SubtractOp> {
public:
    ConvertSubtractToDWConvAdd(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SubtractOp>(ctx), _log(log) {
        setDebugName("ConvertSubtractToDWConvAdd");
    }

    mlir::LogicalResult matchAndRewrite(IE::SubtractOp subOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Three typical pattern conversions:
//
// Scenario 1: FakeQuantize is optional
// - The first input is either a tensor or a constant (but cannot be used as a bias - Scenario 2)
// - The second input is a tensor
//                                                                Input_2         Filter(-1)
// Input_1 / Constant       Input_2                                  |                |
//         |                   |                               FakeQuantize_1   FakeQuantize_3
//   FakeQuantize_0      FakeQuantize_1       =>                            \   /
//                 \    /                           Input_1 / Constant    GroupConv
//                Subtract                                  |                  |
//                    |                              FakeQuantize_0     FakeQuantize_4
//              FakeQuantize_2                                     \   /
//                                                                  Add
//                                                                   |
//                                                             FakeQuantize_2
//
// Scenario 2:
// - The first input is a constant, either a splat or with one dimension larger than 1, matching the output channel size
// - The second input is a tensor
//
//   Constant(splat / inC == outC)      Input_2             Input_2    Filter(-1)   Input_0(Bias)
//                |________________________|        =>         |___________|______________|
//                             |                                           |
//                         Subtract                                     GroupConv
//
//
// Scenario 3: FakeQuantize is optional
// - The first input is a tensor
// - The second input is a constant
//
//      Input_1             Constant                        Input_1          Constant(rescale -1)
//         |                   |                               |                       |
//   FakeQuantize_0      FakeQuantize_1       =>        FakeQuantize_0           FakeQuantize_3
//                 \    /                                            \            /
//                Subtract                                                 Add
//                    |                                                     |
//              FakeQuantize_2                                        FakeQuantize_2
//
// Scenario 4:
// - Constant input of Multiply can be either the first and the second input
// - Multiply has to be the second input of Subtract, so that the Subtract can be
//   converted to Add without additional changes
//
//          MulInput1    Constant (MulInput2)                 MulInput1    Constant (rescale -1)
//                \       /                                        \       /
//    SubInput1   Multiply (SubInput2)       =>         SubInput1   Multiply
//        \       /                                             \   /
//         Subtract                                              Add
//

mlir::LogicalResult ConvertSubtractToDWConvAdd::matchAndRewrite(IE::SubtractOp subOp,
                                                                mlir::PatternRewriter& rewriter) const {
    auto subOpLoc = subOp.getLoc();
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), subOp->getName(), subOp->getLoc());

    auto input1 = subOp.getInput1();
    auto input2 = subOp.getInput2();

    const auto input1Shape = getShape(input1);
    const auto input2Shape = getShape(input2);
    if (input1Shape.size() != 4 || input2Shape.size() != 4) {
        return mlir::failure();
    }

    if (auto mulOp = input2.getDefiningOp<IE::MultiplyOp>()) {
        auto handleConstantInput = [&](auto constInputOp, mlir::Value otherInput, mlir::Value maybeFqInput) {
            const auto& constInputContent = constInputOp.getContentAttr();
            auto negativeContent = constInputContent.transform().rescale(-1.0).get();
            mlir::Value negativeInput = rewriter.create<Const::DeclareOp>(constInputOp.getLoc(), constInputOp.getType(),
                                                                          std::move(negativeContent));
            auto fqInput = maybeFqInput.getDefiningOp<IE::FakeQuantizeOp>();
            if (fqInput != nullptr) {
                negativeInput = createNewFq(rewriter, fqInput.getLoc(), negativeInput, fqInput).getOutput();
            }
            auto negativeMulOp = rewriter.create<IE::MultiplyOp>(
                    takeOpLoc(mulOp, "neg"), negativeInput, otherInput, mulOp.getAutoBroadcastAttr(),
                    /*post_op=*/nullptr, /*clamp=*/nullptr, /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);
            auto addOp = rewriter.replaceOpWithNewOp<IE::AddOp>(
                    subOp, input1, negativeMulOp.getOutput(), subOp.getAutoBroadcastAttr(),
                    /*post_op=*/nullptr, /*clamp=*/nullptr, /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);
            extendOpLoc(addOp, "as_add");
            return mlir::success();
        };

        if (auto constInput1Op = getInputConstantOp(mulOp.getInput1())) {
            return handleConstantInput(constInput1Op, mulOp.getInput2(), mulOp.getInput1());
        } else if (auto constInput2Op = getInputConstantOp(mulOp.getInput2())) {
            return handleConstantInput(constInput2Op, mulOp.getInput1(), mulOp.getInput2());
        }
    }

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(subOp.getOutput().getType());
    auto outputChannel = outputType.getShape()[Dims4D::Act::C];
    auto constInput1Op = getInputConstantOp(input1);
    auto fqInput1 = input1.getDefiningOp<IE::FakeQuantizeOp>();
    const auto isLegalToConvertConstantInput1ToBias =
            (constInput1Op != nullptr) && (fqInput1 == nullptr) &&
            (input1Shape.totalSize() == 1 || input1Shape.totalSize() == outputChannel);

    mlir::Value biasInput = nullptr;
    if (isLegalToConvertConstantInput1ToBias) {
        auto broadcastContentSetup = constInput1Op.transformContentAttr();
        auto newInput1Shape = to_small_vector(input1Shape);
        newInput1Shape[Dims4D::Act::C.ind()] = outputChannel;

        if (input1Shape.totalSize() == 1) {
            broadcastContentSetup = broadcastContentSetup.broadcast(Dims4D::Act::C, outputChannel);
        }
        auto constInput1Type = mlir::cast<vpux::NDTypeInterface>(constInput1Op.getType());
        auto newInput1Type = constInput1Type.changeShape(ShapeRef(newInput1Shape));
        biasInput = rewriter.create<Const::DeclareOp>(subOpLoc, newInput1Type, broadcastContentSetup.get());
    }

    auto fqInput2 = input2.getDefiningOp<IE::FakeQuantizeOp>();

    mlir::Value negativeInput = input2;
    if (auto constInput2 = getInputConstantOp(input2)) {
        const auto& constInput2Content = constInput2.getContentAttr();
        auto negativeContent = constInput2Content.transform().rescale(-1.0).get();
        negativeInput = rewriter.create<Const::DeclareOp>(appendLoc(subOpLoc, "neg_cst"), constInput2.getType(),
                                                          std::move(negativeContent));
    } else {
        const auto elemType = outputType.getElementType();
        const auto inputC = input2Shape[Dims4D::Act::C];
        const auto filterStorageType = mlir::RankedTensorType::get(SmallVector<int64_t>{inputC, 1, 1, 1}, elemType);
        auto filter = Const::createFloatConst(rewriter, appendLoc(subOpLoc, "conv_filter"), filterStorageType, -1.0f);

        if (fqInput2 != nullptr) {
            const auto fqArgType = mlir::RankedTensorType::get({1, 1, 1, 1}, elemType);
            // TODO(E#145633): FQ(-1,-1) will cause GroupConv output 3x larger than reference for unknown reason.
            auto low = Const::createFloatConst(rewriter, subOpLoc, fqArgType, -1.0f);
            auto high = Const::createFloatConst(rewriter, subOpLoc, fqArgType, 0.0f);
            auto filterFQ = rewriter.create<IE::FakeQuantizeOp>(subOpLoc, filter, low, high, low, high,
                                                                fqInput2.getLevelsAttr(), fqInput2.getLowFpTypeAttr(),
                                                                fqInput2.getAutoBroadcastAttr());
            filter = filterFQ.getOutput();
        }

        auto dilationsAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{1, 1});
        auto stridesAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{1, 1});
        auto padBeginAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
        auto padEndAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
        auto groupAttr = getIntAttr(rewriter, inputC);

        auto dwConv =
                rewriter.create<IE::GroupConvolutionOp>(appendLoc(subOpLoc, "as_gconv"), input2, filter, biasInput,
                                                        stridesAttr, padBeginAttr, padEndAttr, dilationsAttr, groupAttr,
                                                        /*post_opAttr=*/nullptr, /*clampAttr=*/nullptr,
                                                        /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);
        negativeInput = dwConv.getOutput();
        if (biasInput) {
            rewriter.replaceOp(subOp, negativeInput);
            return mlir::success();
        }
    }

    if (fqInput2 != nullptr) {
        negativeInput = createNewFq(rewriter, subOpLoc, negativeInput, fqInput2).getOutput();
    }

    auto addOp =
            rewriter.replaceOpWithNewOp<IE::AddOp>(subOp, input1, negativeInput, subOp.getAutoBroadcastAttr(),
                                                   /*post_op=*/nullptr, /*clamp=*/nullptr, /*outputPadding=*/nullptr,
                                                   /*inputPadding=*/nullptr);
    extendOpLoc(addOp, "add_out");
    return mlir::success();
}

//
// ConvertSubtractToAddPass
//

class ConvertSubtractToAddPass final : public IE::impl::ConvertSubtractToAddBase<ConvertSubtractToAddPass> {
public:
    explicit ConvertSubtractToAddPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertSubtractToAddPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto isLegalSubtractOp = [&](IE::SubtractOp op) {
        const auto input1Type = mlir::cast<vpux::NDTypeInterface>(op.getInput1().getType());
        const auto input2Type = mlir::cast<vpux::NDTypeInterface>(op.getInput2().getType());
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getResult().getType());

        // tracking number: E#171091
        // When input1 is constant and input2 is dynamic, we need a GroupConv operation that supports dynamic shapes for
        // input2 (Scenario 1 and Scenario 2). Currently, dynamic shape support is not available.
        if (outputType.getShape().isDynamic()) {
            return true;
        }

        // SubTract with INTEGER input/ouput can not be executed as NCE op, and should be kept like Subtract and be
        // executed on shave.
        const auto hasIntInput = mlir::isa<mlir::IntegerType>(input1Type.getElementType()) ||
                                 mlir::isa<mlir::IntegerType>(input2Type.getElementType());

        // SubTract with FLOAT32 input/ouput can not be executed as NCE op, and should be kept like Subtract and be
        // executed on shave.
        const auto hasFP32Input = mlir::isa<mlir::Float32Type>(input1Type.getElementType()) ||
                                  mlir::isa<mlir::Float32Type>(input2Type.getElementType());
        const auto convertGroupConv = getInputConstantOp(op.getInput2()) == nullptr;

        // Check if a broadcast operation is needed. If the input and output types do not match, and it cannot be
        // optimized through constant folding, then broadcasting is required
        const auto needsBroadcast = (input1Type != input2Type) &&
                                    ((input1Type != outputType && getInputConstantOp(op.getInput1()) == nullptr) ||
                                     (input2Type != outputType && getInputConstantOp(op.getInput2()) == nullptr));

        // Check if an expansion operation is needed. If the size of the output shape is not a multiple of the channel
        // alignment, then expansion is required
        const auto alignment = VPU::NCEInvariant::getAlignment(outputType.getElementType());
        const auto needsExpansion = (outputType.getShape().totalSize() % alignment) != 0;

        // if the second input is a multiply with constant, marked as illegal to enable the optimization
        if (auto mulOp = op.getInput2().getDefiningOp<IE::MultiplyOp>()) {
            if (getInputConstantOp(mulOp.getInput1()) || getInputConstantOp(mulOp.getInput2())) {
                return false;
            }
        }

        return hasIntInput || needsBroadcast || needsExpansion || (hasFP32Input && convertGroupConv);
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::GroupConvolutionOp>();
    target.addLegalOp<IE::AddOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<IE::NegativeOp>();
    target.addLegalOp<IE::MultiplyOp>();
    target.addDynamicallyLegalOp<IE::SubtractOp>(isLegalSubtractOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertSubtractToDWConvAdd>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertSubtractToAddPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertSubtractToAddPass(Logger log) {
    return std::make_unique<ConvertSubtractToAddPass>(log);
}
