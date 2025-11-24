//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/factories/convert_divide_to_multiply_benefit_strategy.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTDIVIDETOMULTIPLY
#define GEN_PASS_DEF_CONVERTDIVIDETOMULTIPLY
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class ConvertDivideToMultiplyPass final : public IE::impl::ConvertDivideToMultiplyBase<ConvertDivideToMultiplyPass> {
public:
    explicit ConvertDivideToMultiplyPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

// Checks if IE.Divide op operates on floating point type
bool isFloatDivision(IE::DivideOp divideOp) {
    const auto elementType = divideOp.getOutput().getType().getElementType();
    return mlir::isa<mlir::FloatType>(elementType);
}

// Checks if the current user is IE.Divide op that operates on floating point type and its second input is origOp
bool isDivideUser(mlir::Operation* origOp, mlir::Operation* user) {
    if (auto divideOp = mlir::dyn_cast<IE::DivideOp>(user); divideOp != nullptr) {
        return divideOp.getInput2().getDefiningOp() == origOp && isFloatDivision(divideOp);
    }
    return false;
}

mlir::Value createNewFQ(mlir::PatternRewriter& rewriter, IE::FakeQuantizeOp origFqOp, mlir::Value input, float inLow,
                        float inHigh, float outLow, float outHigh) {
    auto inLowConstType =
            mlir::cast<mlir::RankedTensorType>(origFqOp.getInputLow().getDefiningOp<Const::DeclareOp>().getType());
    auto outLowConstType =
            mlir::cast<mlir::RankedTensorType>(origFqOp.getOutputLow().getDefiningOp<Const::DeclareOp>().getType());

    rewriter.setInsertionPoint(origFqOp);
    auto newInLowConst = Const::createConst(rewriter, origFqOp->getLoc(), inLowConstType, ArrayRef(inLow));
    auto newInHighConst = Const::createConst(rewriter, origFqOp->getLoc(), outLowConstType, ArrayRef(inHigh));
    auto newOutLowConst = Const::createConst(rewriter, origFqOp->getLoc(), inLowConstType, ArrayRef(outLow));
    auto newOutHighConst = Const::createConst(rewriter, origFqOp->getLoc(), outLowConstType, ArrayRef(outHigh));

    auto newFakeQuantizeOp = rewriter.create<IE::FakeQuantizeOp>(
            origFqOp->getLoc(), origFqOp.getType(), input, newInLowConst, newInHighConst, newOutLowConst,
            newOutHighConst, origFqOp.getLevelsAttr(), origFqOp.getLowFpTypeAttr(), origFqOp.getAutoBroadcastAttr());

    // We replace the old FakeQuantize op with the new one only for Divide users
    // if their second input is the old FakeQuantize
    rewriter.replaceUsesWithIf(origFqOp, newFakeQuantizeOp, [&](mlir::OpOperand& opOperand) {
        return isDivideUser(origFqOp, opOperand.getOwner());
    });

    return newFakeQuantizeOp.getOutput();
}

mlir::Value createNewDQ(mlir::PatternRewriter& rewriter, IE::DequantizeOp origDqOp, mlir::Value input) {
    rewriter.setInsertionPoint(origDqOp);

    auto newDequantizeOp = rewriter.create<IE::DequantizeOp>(origDqOp->getLoc(), input, origDqOp.getDstElemTypeAttr());

    // We replace the old Dequantize op with the new one only for Divide users
    // if their second input is the old Dequantize
    rewriter.replaceUsesWithIf(origDqOp, newDequantizeOp, [&](mlir::OpOperand& opOperand) {
        return isDivideUser(origDqOp, opOperand.getOwner());
    });

    return newDequantizeOp.getOutput();
}

mlir::FailureOr<mlir::Value> replaceWithNewFakeQuantizeOp(mlir::PatternRewriter& rewriter, Const::DeclareOp constOp,
                                                          IE::FakeQuantizeOp fakeQuantize) {
    const auto inLowSplat = vpux::Const::template getSplatValue<float>(fakeQuantize.getInputLow());
    const auto inHighSplat = vpux::Const::template getSplatValue<float>(fakeQuantize.getInputHigh());
    const auto outLowSplat = vpux::Const::template getSplatValue<float>(fakeQuantize.getOutputLow());
    const auto outHighSplat = vpux::Const::template getSplatValue<float>(fakeQuantize.getOutputHigh());

    if (mlir::failed(inLowSplat) || mlir::failed(inHighSplat) || mlir::failed(outLowSplat) ||
        mlir::failed(outHighSplat)) {
        return mlir::failure();
    }

    const auto inLowVal = inLowSplat.value();
    const auto inHighVal = inHighSplat.value();
    const auto outLowVal = outLowSplat.value();
    const auto outHighVal = outHighSplat.value();

    auto newCstAttr = constOp.transformContentAttr().scalarMultInverse().get();
    rewriter.setInsertionPoint(constOp);
    auto newCstOp = rewriter.create<Const::DeclareOp>(constOp->getLoc(), newCstAttr.getType(), std::move(newCstAttr));

    // Get inversed values for new input low/high params for FQ
    const auto inversedConstContent = newCstOp.getContent();
    const auto inversedConstContentVals = inversedConstContent.getValues<float>();

    // New FQ range should contain 0 so new params are calculated as:
    // newInputLow = min(0, min(inversedConstContentVals))
    // newInputHigh = max(0, max(inversedConstContentVals))
    // e.g. origInput = [1, 2, 3] inLow = 0, inHigh = 3
    // inversedInput = [1; 0.5; 0.33] newInLow = 0, newInHigh = 1
    const auto minInversedVal = std::min_element(inversedConstContentVals.begin(), inversedConstContentVals.end());
    const auto maxInversedVal = std::max_element(inversedConstContentVals.begin(), inversedConstContentVals.end());
    const auto newInLowVal = std::min(0.f, *minInversedVal);
    const auto newInHighVal = std::max(0.f, *maxInversedVal);

    // To get new output range we have to quantize original values, inverse them and find min/max:
    // quantizedConstContentVal[Max|Min] = (val[Max|Min] - in_low) / (in_high - in_low) * (out_high - out_low) +
    // out_low
    // inversedQuantizedConstContentVal[Max|Min] = 1 / quantizedConstContentVal[Min|Max]
    // newInputLow = min(0, min(inversedQuantizedConstContentValMin))
    // newInputHigh = max(0, max(inversedQuantizedConstContentValMax))
    auto getQuantizedMinMaxVal = [&](float inversedVal) {
        const float origVal = 1.f / inversedVal;
        const float quantizedVal = (origVal - inLowVal) / (inHighVal - inLowVal) * (outHighVal - outLowVal) + outLowVal;
        VPUX_THROW_WHEN(quantizedVal == 0.f, "Cannot divide by zero");
        return 1.f / quantizedVal;
    };

    const auto newOutLowVal = std::min(0.f, getQuantizedMinMaxVal(*minInversedVal));
    const auto newOutHighVal = std::max(0.f, getQuantizedMinMaxVal(*maxInversedVal));

    return createNewFQ(rewriter, fakeQuantize, newCstOp.getOutput(), newInLowVal, newInHighVal, newOutLowVal,
                       newOutHighVal);
}

mlir::FailureOr<mlir::Value> replaceWithNewDequantizeOp(mlir::PatternRewriter& rewriter, Const::DeclareOp constOp,
                                                        IE::DequantizeOp dequantize, Logger log) {
    // Per axis quantization or other complex cases are not yet supported
    auto oldType = constOp.getContentAttr().getType();
    auto oldElemType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(oldType.getElementType());
    if (oldElemType == nullptr) {
        log.trace("Only supporting UniformQuantizedType");
        return mlir::failure();
    }

    auto storageType = oldElemType.getStorageType();
    auto oldScale = oldElemType.getScale();
    auto oldZP = oldElemType.getZeroPoint();

    if (!storageType.isInteger(8)) {
        log.trace("Only supporting INT8 cases, but got {0}", storageType);
        return mlir::failure();
    }

    float rescaleFactor = oldElemType.getStorageTypeMax() - oldElemType.getStorageTypeMin() + 1;
    auto FP32ElemType = mlir::Float32Type::get(rewriter.getContext());
    auto newCstAttr = constOp.transformContentAttr()
                              .add(-oldZP)
                              .castElemType(FP32ElemType)
                              .scalarMultInverse()
                              .rescale(rescaleFactor)
                              .get();
    auto newCstOp = rewriter.create<Const::DeclareOp>(constOp->getLoc(), newCstAttr.getType(), newCstAttr);
    auto constContent = newCstOp.getContent();
    auto constContentVals = constContent.getValues<float>();
    auto minFloatVal = std::min_element(constContentVals.begin(), constContentVals.end());
    auto newZP = static_cast<int>(*minFloatVal);

    auto newScale = 1.0 / (oldScale * rescaleFactor);
    mlir::Type newElemType = mlir::quant::UniformQuantizedType::get(
            oldElemType.getFlags(), storageType, oldElemType.getExpressedType(), newScale, newZP,
            oldElemType.getStorageTypeMin(), oldElemType.getStorageTypeMax());
    auto newType = oldType.changeTypeComponents(TypeComponents().setElementType(newElemType));

    newCstAttr = newCstAttr.transform().add(newZP).castElemType(storageType).get();
    rewriter.setInsertionPoint(constOp);
    newCstOp = rewriter.create<Const::DeclareOp>(constOp->getLoc(), newType, std::move(newCstAttr));
    return createNewDQ(rewriter, dequantize, newCstOp.getOutput());
}

class ConstDivisorRewriter final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    ConstDivisorRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<Const::DeclareOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Replaces this pattern:
//
//             const.Declare
//        ____________|_____________
//       |    ..      |        | .. |
//  IE.Divide .. IE.Divide    op .. op
//
// with
//
//        const.Declare'         const.Declare
//  [#const.ScalarMultInverse]         |
//         ______|______             __|__
//        |     ..      |           |  .. |
//  IE.Multiply .. IE.Multiply     op  .. op
//
// Every IE.Divide op that operates on floating point types is replaced by a IE.Multiply op.
// The constant input divisor is replaced by its reciprocal.
//
mlir::LogicalResult ConstDivisorRewriter::matchAndRewrite(Const::DeclareOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Const.Declare op at '{0}'", origOp->getLoc());
    // Checks if Const.DeclareOp has at least one IE.Divide user
    auto noneOfUsersAreDivide = llvm::none_of(origOp->getUsers(), [&](auto user) {
        return isDivideUser(origOp, user);
    });

    if (noneOfUsersAreDivide) {
        _log.trace("Ignore: Const.Declare op has no IE.Divide users");
        return mlir::failure();
    }

    auto newCstAttr = origOp.transformContentAttr().scalarMultInverse().get();
    auto newCstOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), newCstAttr.getType(), std::move(newCstAttr));
    // We replace the old Const.DeclareOp with the new one only for IE.Divide users
    rewriter.replaceUsesWithIf(origOp, newCstOp, [&](mlir::OpOperand& opOperand) {
        return isDivideUser(origOp, opOperand.getOwner());
    });

    for (auto userOp : llvm::make_early_inc_range(newCstOp->getUsers())) {
        // Casting is safe because newCstOp has only IE.Divide users
        auto divideOp = mlir::cast<IE::DivideOp>(userOp);
        rewriter.setInsertionPoint(divideOp);
        rewriter.replaceOpWithNewOp<IE::MultiplyOp>(divideOp, divideOp.getInput1(), newCstOp,
                                                    divideOp.getAutoBroadcastAttr(), nullptr, nullptr, nullptr,
                                                    nullptr);
    }
    return mlir::success();
}

// Convert:
// Input1(1x1024x1024)   Input2(1x1x1)
//             \            /
//           IE.Divide(1x1024x1024)

// To:

//                  Const(1x1x1)     Input2(1x1x1)
//                           \           /
// Input1(1x1024x1024)      IE.Divide(1x1x1)
//              \               /
//           IE.Multiply(1x1024x1024)
class NonConstDivisorRewriter final : public mlir::OpRewritePattern<IE::DivideOp> {
public:
    NonConstDivisorRewriter(mlir::MLIRContext* ctx, Logger log,
                            std::unique_ptr<IE::IConvertDivideToMultiplyBenefitStrategy> benefitStrategy)
            : mlir::OpRewritePattern<IE::DivideOp>(ctx), _log(log), _benefitStrategy(std::move(benefitStrategy)) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DivideOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    std::unique_ptr<IE::IConvertDivideToMultiplyBenefitStrategy> _benefitStrategy;
};

mlir::LogicalResult NonConstDivisorRewriter::matchAndRewrite(IE::DivideOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Divide op at '{0}'", origOp->getLoc());

    // Const divisor is handled in other rewriter
    if (mlir::isa_and_nonnull<Const::DeclareOp>(origOp.getInput2().getDefiningOp())) {
        return mlir::failure();
    }

    if (mlir::isa_and_nonnull<IE::DequantizeOp>(origOp.getInput2().getDefiningOp())) {
        return mlir::failure();
    }

    // InverseOp only support float type
    const auto divisorElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput2().getType()).getElementType();
    if (!mlir::isa<mlir::FloatType>(divisorElemType)) {
        return mlir::failure();
    }

    if (_benefitStrategy->isNonConstBeneficialConversion(origOp).failed()) {
        _log.trace("Non-const Divide conversion is not beneficial");
        return mlir::failure();
    }

    // The Divide SW kernel is optimized for specific scenarios:
    // 1. When the second input (divisor) is a scalar.
    // 2. When both inputs have identical shapes.
    // Empirical evidence suggests that converting operations is advantageous when the divisor is a scalar.
    // Current implementation does not perform conversion when both inputs share the same shape,
    // as this scenario is already optimized by the kernel.
    auto divisorShape = getShape(origOp.getInput2());
    auto outputShape = getShape(origOp.getOutput());
    if (divisorShape == outputShape) {
        return mlir::failure();
    }

    auto elemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput2().getType()).getElementType();
    if (!elemType.isF16() && !elemType.isF32()) {
        _log.trace("Unsupported data type");
        return mlir::failure();
    }

    auto ctx = rewriter.getContext();
    auto constLoc = appendLoc(origOp->getLoc(), "_inverse");
    mlir::Value constOp;
    if (elemType.isF16()) {
        const auto baseType = mlir::RankedTensorType::get(divisorShape, mlir::Float16Type::get(ctx));
        constOp = Const::createConst(rewriter, constLoc, baseType, ArrayRef(vpux::type::float16(1.f)));
    }
    if (elemType.isF32()) {
        const auto baseType = mlir::RankedTensorType::get(divisorShape, mlir::Float32Type::get(ctx));
        constOp = Const::createConst(rewriter, constLoc, baseType, ArrayRef(1.f));
    }
    auto divideOp = rewriter.create<IE::DivideOp>(appendLoc(origOp->getLoc(), "_divide"), constOp, origOp.getInput2(),
                                                  IE::AutoBroadcastType::NUMPY);

    auto multiplyOp =
            rewriter.create<IE::MultiplyOp>(takeOpLoc(origOp, "_multiply"), origOp.getInput1(), divideOp.getOutput(),
                                            origOp.getAutoBroadcast(), nullptr, nullptr, nullptr, nullptr);
    rewriter.replaceOp(origOp, multiplyOp.getOutput());
    _log.trace("Successfully replaced divide with multiply");

    return mlir::success();
}

class FakeQuantizeDivideRewriter final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    FakeQuantizeDivideRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Replaces this pattern:
//
//              const.Declare
//                    |
//             IE.FakeQuantize
//        ____________|____________
//       |    ..      |       | .. |
//  IE.Divide .. IE.Divide   op .. op
//
// with
//
//           const.Declare'               const.Declare
//    [#const.ScalarMultInverse]                |
//                  |                           |
//           IE.FakeQuantize'            IE.FakeQuantize
//   (with new in/out low/high params)          |
//           _______|______                   __|__
//          |              |                 | ..  |
//    IE.Multiply .. IE.Multiply            op ..  op
//
// Every IE.Divide op that operates on floating point types is replaced by a IE.Multiply op.
// IE.FakeQuantize input divisor is replaced by the new one with updated in/out low/high params and
// its constant input is replaced by its reciprocal.
//
mlir::LogicalResult FakeQuantizeDivideRewriter::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("Got IE.FakeQuantizeOp op at '{0}'", origOp->getLoc());
    // Checks if IE.FakeQuantize has at least one IE.Divide user
    auto noneOfUsersAreDivide = llvm::none_of(origOp->getUsers(), [&](auto user) {
        return isDivideUser(origOp, user);
    });

    if (noneOfUsersAreDivide) {
        _log.trace("Ignore: IE.FakeQuantize op has no IE.Divide users");
        return mlir::failure();
    }

    if (!mlir::isa_and_nonnull<Const::DeclareOp>(origOp.getInput().getDefiningOp())) {
        _log.trace("Ignore: IE.FakeQuantize op has no constant input");
        return mlir::failure();
    }

    auto constOp = mlir::cast<Const::DeclareOp>(origOp.getInput().getDefiningOp());
    const auto maybeNewFq = replaceWithNewFakeQuantizeOp(rewriter, constOp, origOp);
    if (mlir::failed(maybeNewFq)) {
        _log.trace("Ignore: IE.FakeQuantize input/output low/high params are not splat values");
        return mlir::failure();
    }

    const auto newFqOp = maybeNewFq.value();
    for (auto userOp : llvm::make_early_inc_range(newFqOp.getDefiningOp()->getUsers())) {
        // Casting is safe because newFqOp has only IE.Divide users (see replaceWithNewFakeQuantizeOp)
        auto divideOp = mlir::cast<IE::DivideOp>(userOp);
        // Insertion point was changed in replaceWithNewFakeQuantizeOp, we have to reset it manually here
        rewriter.setInsertionPoint(divideOp);
        rewriter.replaceOpWithNewOp<IE::MultiplyOp>(divideOp, divideOp.getInput1(), newFqOp,
                                                    divideOp.getAutoBroadcastAttr(), nullptr, nullptr, nullptr,
                                                    nullptr);
    }
    return mlir::success();
}

class DequantizeDivideRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeDivideRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Replaces this pattern:
//
//              const.Declare
//                    |
//             IE.Dequantize
//        ____________|____________
//       |    ..      |       | .. |
//  IE.Divide .. IE.Divide   op .. op
//
// with
//
//           const.Declare'               const.Declare
//    [#const.ScalarMultInverse]                |
//                  |                           |
//           IE.Dequantize'               IE.Dequantize
//          (with new input)                    |
//           _______|______                   __|__
//          |              |                 | ..  |
//    IE.Multiply .. IE.Multiply            op ..  op
//
// Every IE.Divide op that operates on floating point types is replaced by a IE.Multiply op.
// IE.Dequantize input divisor is replaced by the new one with updated constant input replaced by its reciprocal.
//
mlir::LogicalResult DequantizeDivideRewriter::matchAndRewrite(IE::DequantizeOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Got IE.DequantizeOp op at '{0}'", origOp->getLoc());
    // Checks if IE.Dequantize has at least one IE.Divide user
    auto noneOfUsersAreDivide = llvm::none_of(origOp->getUsers(), [&](auto user) {
        return isDivideUser(origOp, user);
    });

    if (noneOfUsersAreDivide) {
        _log.trace("Ignore: IE.Dequantize op has no IE.Divide users");
        return mlir::failure();
    }

    if (!mlir::isa_and_nonnull<Const::DeclareOp>(origOp.getInput().getDefiningOp())) {
        _log.trace("Ignore: IE.Dequantize op has no constant input");
        return mlir::failure();
    }

    auto constOp = mlir::cast<Const::DeclareOp>(origOp.getInput().getDefiningOp());
    const auto maybeNewDq = replaceWithNewDequantizeOp(rewriter, constOp, origOp, _log);
    if (mlir::failed(maybeNewDq)) {
        _log.trace("Ignore: IE.Dequantize is not of type UniformQuantizedType or input is not I8/U8");
        return mlir::failure();
    }

    const auto newDqOp = maybeNewDq.value();
    for (auto userOp : llvm::make_early_inc_range(newDqOp.getDefiningOp()->getUsers())) {
        // Casting is safe because newDqOp has only IE.Divide users (see replaceWithNewDequantizeOp)
        auto divideOp = mlir::cast<IE::DivideOp>(userOp);
        // Insertion point was changed in replaceWithNewDequantizeOp, we have to reset it manually here
        rewriter.setInsertionPoint(divideOp);
        rewriter.replaceOpWithNewOp<IE::MultiplyOp>(divideOp, divideOp.getInput1(), newDqOp,
                                                    divideOp.getAutoBroadcastAttr(), nullptr, nullptr, nullptr,
                                                    nullptr);
    }
    return mlir::success();
}

void ConvertDivideToMultiplyPass::safeRunOnFunc() {
    static_assert(IE::DivideOp::hasTrait<IE::EltwiseOp>(),
                  "This pass cannot replace IE.Divide with IE.Multiply when division is not element-wise: the "
                  "reciprocal must be calculated differently");

    auto func = getOperation();
    auto& ctx = getContext();

    auto nonConstBenefitStrategy = IE::createConvertDivideToMultiplyBenefitStrategy(func);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConstDivisorRewriter>(&ctx, _log);
    patterns.add<FakeQuantizeDivideRewriter>(&ctx, _log);
    patterns.add<DequantizeDivideRewriter>(&ctx, _log);
    patterns.add<NonConstDivisorRewriter>(&ctx, _log, std::move(nonConstBenefitStrategy));

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createConvertDivideToMultiplyPass(Logger log) {
    return std::make_unique<ConvertDivideToMultiplyPass>(log);
}
