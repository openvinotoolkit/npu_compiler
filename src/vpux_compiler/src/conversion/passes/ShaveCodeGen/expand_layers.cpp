//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/conversion/passes/ShaveCodeGen/Approximation.hpp"

#include <mlir/Dialect/Math/Transforms/Passes.h>
#include "mlir/Dialect/Math/Transforms/Approximation.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"

namespace vpux {
#define GEN_PASS_DECL_EXPANDLAYERS
#define GEN_PASS_DEF_EXPANDLAYERS
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

// Lowers a math::RoundOp for fp16 operands using round half-away-from-zero semantics
//
// This static function lowers a math::RoundOp when the operand is a 16-bit floating point (fp16)
// value. It implements the round half-away-from-zero behavior without promoting the value to a
// higher precision (such as fp32). Instead, the function directly manipulates the underlying
// 16-bit bitpattern of the fp16 value via bitwise and arithmetic operations.
//
// In half-away-from-zero rounding, when a value is exactly halfway between two representable
// numbers, it is rounded to the value that is further from zero. The function achieves this by:
//   - Converting the fp16 value to its 16-bit integer representation (via bitcast),
//   - Extracting and processing components such as the sign, exponent, and fraction,
//   - Applying a bias and shift operations to calculate the rounded integer value,
//   - Recombining the bits and converting the result back to an fp16 value.
//
// On success, the original operation is replaced with the newly computed rounded value, and the
// function returns mlir::success(). If the operation cannot be lowered (for example, if the operand
// is not of type fp16), the function returns mlir::failure().
static mlir::LogicalResult Roundfp16(mlir::math::RoundOp op, mlir::PatternRewriter& rewriter) {
    constexpr int FP16_BIAS = 15;      /* expo bias */
    constexpr int FP16_TOTALBITS = 16; /* total number of bits */
    constexpr int FP16_FRACTBITS = 10; /* number of explicit fraction bits */
    constexpr uint16_t FP16_GREATINT = static_cast<uint16_t>(
            (FP16_BIAS + FP16_FRACTBITS) << FP16_FRACTBITS);          /* big: all equal or above are integers */
    constexpr uint16_t unsignedminusone = static_cast<uint16_t>(-1);  // 0xFFFF

    using namespace mlir;

    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    Value operand = op.getOperand();
    Type opType = operand.getType();
    Type opEType = getElementTypeOrSelf(opType);

    if (!opEType.isF16()) {
        return rewriter.notifyMatchFailure(op, "not a round of f16.");
    }

    if (isa<ShapedType>(opType)) {
        return rewriter.notifyMatchFailure(op, "ShapedType not supported");
    }

    auto f16Type = operand.getType();
    auto i16Type = rewriter.getIntegerType(16);
    Value xInt = rewriter.create<arith::BitcastOp>(loc, i16Type, operand);

    Value fp16Bias = rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(FP16_BIAS));
    Value fractBits = rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(FP16_FRACTBITS));
    constexpr uint16_t TOTALBITS_ONE = static_cast<uint16_t>(FP16_TOTALBITS - 1);
    Value totalBitsMinusOne = rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(TOTALBITS_ONE));

    constexpr uint16_t signMaskValue = static_cast<uint16_t>(1u << TOTALBITS_ONE);
    Value signMask =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(static_cast<int16_t>(signMaskValue)));

    // xAbs = xInt & ~signMask
    constexpr uint16_t absMaskvalue = static_cast<uint16_t>(~(1u << FP16_BIAS));  // 0x7FFF
    Value absMask = rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(absMaskvalue));
    Value xAbs = rewriter.create<arith::AndIOp>(loc, xInt, absMask);

    // xAbsHalfPlus = (shortx)( (halfx)xAbs + 0.5 )
    Value xAbsF16 = rewriter.create<arith::BitcastOp>(loc, f16Type, xAbs);
    Value halfF16 = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(f16Type, 0.5));

    // xAbsF16 + 0.5.
    Value xAbsPlus = rewriter.create<arith::AddFOp>(loc, xAbsF16, halfF16);
    Value xAbsHalfPlus = rewriter.create<arith::BitcastOp>(loc, i16Type, xAbsPlus);

    // xExpo = (xAbsHalfPlus >> FP16_FRACTBITS) - FP16_BIAS.
    Value shifted = rewriter.create<arith::ShRSIOp>(loc, xAbsHalfPlus, fractBits);
    Value xExpo = rewriter.create<arith::SubIOp>(loc, shifted, fp16Bias);

    // FP16_TRUNCFRACT = (((unsigned)-1) << FP16_FRACTBITS)
    constexpr uint16_t fp16TruncFracvalue = static_cast<uint16_t>((unsignedminusone << FP16_FRACTBITS));
    Value fp16TruncFrac = rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(fp16TruncFracvalue));

    // truncMask = fp16TruncFrac >> xExpo (shift dinamic).
    Value truncMask = rewriter.create<arith::ShRSIOp>(loc, fp16TruncFrac, xExpo);

    // isSmall = (xAbs - (shortx)halves) >> (FP16_TOTALBITS - 1).
    Value halvesInt = rewriter.create<arith::BitcastOp>(loc, i16Type, halfF16);
    Value diffSmall = rewriter.create<arith::SubIOp>(loc, xAbs, halvesInt);
    Value isSmall = rewriter.create<arith::ShRSIOp>(loc, diffSmall, totalBitsMinusOne);

    // isGreat = ((FP16_GREATINT - 1) - xAbs) >> (FP16_TOTALBITS - 1).
    constexpr uint16_t greatIntMinusOnevalue = static_cast<uint16_t>(FP16_GREATINT - 1);
    Value greatIntMinusOne = rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(greatIntMinusOnevalue));
    Value diffGreat = rewriter.create<arith::SubIOp>(loc, greatIntMinusOne, xAbs);
    Value isGreat = rewriter.create<arith::ShRSIOp>(loc, diffGreat, totalBitsMinusOne);

    // isTrunc = ~(isGreat | isSmall).
    Value greatOrSmall = rewriter.create<arith::OrIOp>(loc, isGreat, isSmall);

    // Force sign extension of the constant:
    Value rawAllOnes = rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(unsignedminusone));
    Value shiftZero = rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(0));
    Value allOnesExtended = rewriter.create<arith::ShRSIOp>(loc, rawAllOnes, shiftZero);
    Value isTrunc = rewriter.create<arith::XOrIOp>(loc, greatOrSmall, allOnesExtended);

    // xTrunc = xAbsHalfPlus & truncMask.
    Value xTrunc = rewriter.create<arith::AndIOp>(loc, xAbsHalfPlus, truncMask);

    // xSign = xInt & signMask.
    Value xSign = rewriter.create<arith::AndIOp>(loc, xInt, signMask);

    // res = (isGreat & xInt) | xSign | (isTrunc & xTrunc)
    Value partGreat = rewriter.create<arith::AndIOp>(loc, isGreat, xInt);
    Value partTrunc = rewriter.create<arith::AndIOp>(loc, isTrunc, xTrunc);
    Value combined1 = rewriter.create<arith::OrIOp>(loc, partGreat, xSign);
    Value combined = rewriter.create<arith::OrIOp>(loc, combined1, partTrunc);

    Value res = rewriter.create<arith::BitcastOp>(loc, f16Type, combined);

    rewriter.replaceOp(op, res);
    return success();
}

// Lowers a math::RoundEvenOp for fp16 operands using round-even semantics
static mlir::LogicalResult RoundEvenfp16(mlir::math::RoundEvenOp op, mlir::PatternRewriter& rewriter) {
    constexpr int FP16_BIAS = 15;      /* expo bias */
    constexpr int FP16_TOTALBITS = 16; /* total number of bits */
    constexpr int FP16_FRACTBITS = 10; /* number of explicit fraction bits */
    constexpr uint16_t FP16_GREATINT = static_cast<uint16_t>(
            (FP16_BIAS + FP16_FRACTBITS) << FP16_FRACTBITS); /* big: all equal or above are integers */

    mlir::Location loc = op.getLoc();
    mlir::ImplicitLocOpBuilder b(loc, rewriter);
    mlir::Value operand = op.getOperand();
    mlir::Type opType = operand.getType();
    mlir::Type opEType = getElementTypeOrSelf(opType);

    if (!opEType.isF16()) {
        return rewriter.notifyMatchFailure(op, "not a round of f16.");
    }

    if (mlir::isa<mlir::ShapedType>(opType)) {
        return rewriter.notifyMatchFailure(op, "ShapedType not supported");
    }

    auto f16Type = operand.getType();
    auto i16Type = rewriter.getIntegerType(16);

    // const shortx signMask = (shortx)(1 << (FP16_TOTALBITS - 1));
    constexpr uint16_t TOTALBITS_ONE = static_cast<uint16_t>(FP16_TOTALBITS - 1);
    constexpr uint16_t signMaskValue = static_cast<uint16_t>(1u << TOTALBITS_ONE);
    mlir::Value signMask = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getI16IntegerAttr(static_cast<int16_t>(signMaskValue)));

    // halfx roundShift = (halfx)((shortx)FP16_GREATINT);
    mlir::Value roundShift = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(FP16_GREATINT));
    mlir::Value froundShift = rewriter.create<mlir::arith::BitcastOp>(loc, f16Type, roundShift);

    mlir::Value totalBitsMinusOne =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(TOTALBITS_ONE));
    constexpr uint16_t greatIntMinusOnevalue = static_cast<uint16_t>(FP16_GREATINT - 1);
    mlir::Value greatIntMinusOne =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(greatIntMinusOnevalue));
    auto allOnesAttr = rewriter.getIntegerAttr(i16Type, llvm::APInt::getAllOnes(i16Type.getIntOrFloatBitWidth()));
    mlir::Value allOnes = rewriter.create<mlir::arith::ConstantOp>(loc, allOnesAttr);

    mlir::Value xInt = rewriter.create<mlir::arith::BitcastOp>(loc, i16Type, operand);

    // xAbs = xInt & ~signMask
    constexpr uint16_t absMaskvalue = static_cast<uint16_t>(0x7FFF);  //~signMask
    mlir::Value absMask = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(absMaskvalue));
    mlir::Value xAbs = rewriter.create<mlir::arith::AndIOp>(loc, xInt, absMask);

    // xSign = xInt & signMask.
    mlir::Value xSign = rewriter.create<mlir::arith::AndIOp>(loc, xInt, signMask);

    // isGreat = ((shortx)(FP16_GREATINT - 1) - xAbs) >> (FP16_TOTALBITS - 1);
    mlir::Value diffGreat = rewriter.create<mlir::arith::SubIOp>(loc, greatIntMinusOne, xAbs);
    mlir::Value isGreat = rewriter.create<mlir::arith::ShRSIOp>(loc, diffGreat, totalBitsMinusOne);

    // halfx vround = v_sub(((halfx)xAbs + roundShift), roundShift);
    mlir::Value fxAbs = rewriter.create<mlir::arith::BitcastOp>(loc, f16Type, xAbs);
    mlir::Value sum = rewriter.create<mlir::arith::AddFOp>(loc, froundShift, fxAbs);
    mlir::Value vround = rewriter.create<mlir::arith::SubFOp>(loc, sum, froundShift);
    mlir::Value Ivround = rewriter.create<mlir::arith::BitcastOp>(loc, i16Type, vround);

    // halfx xres = (halfx)((~isGreat & (shortx)vround) | ((isGreat)&xInt) | xSign);
    mlir::Value isNotGreat = rewriter.create<mlir::arith::XOrIOp>(loc, i16Type, isGreat, allOnes);

    mlir::Value partOne = rewriter.create<mlir::arith::AndIOp>(loc, isNotGreat, Ivround);
    mlir::Value partTwo = rewriter.create<mlir::arith::AndIOp>(loc, isGreat, xInt);
    mlir::Value combined1 = rewriter.create<mlir::arith::OrIOp>(loc, partOne, partTwo);
    mlir::Value combined = rewriter.create<mlir::arith::OrIOp>(loc, combined1, xSign);

    mlir::Value res = rewriter.create<mlir::arith::BitcastOp>(loc, f16Type, combined);

    rewriter.replaceOp(op, res);
    return mlir::success();
}

using namespace vpux;

namespace {

//
// ExpandLayersPass
//

class ExpandLayersPass final : public impl::ExpandLayersBase<ExpandLayersPass> {
public:
    explicit ExpandLayersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ExpandLayersPass::safeRunOnModule() {
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    auto module = getOperation();

    auto vpuSwmoduleOp = module.lookupSymbol<mlir::ModuleOp>("VPU.SW");

    for (auto funcOp :
         llvm::make_early_inc_range(vpuSwmoduleOp.getOperation()->getRegion(0).getOps<mlir::func::FuncOp>())) {
        if (funcOp.getBlocks().size() == 0) {
            // Ignore functions which were not generated by ShaveCodeGen.
            continue;
        }

        mlir::RewritePatternSet patterns(&ctx);

        target.addIllegalOp<mlir::math::SinOp, mlir::math::CosOp>();
        target.addIllegalOp<mlir::math::ErfOp>();
        target.addIllegalOp<mlir::math::RoundEvenOp>();
        target.addIllegalOp<mlir::math::RoundOp>();
        target.addIllegalOp<mlir::math::FmaOp>();

        patterns.add<vpux::ShaveCodeGen::SinAndCosApproximation<true, mlir::math::SinOp>,
                     vpux::ShaveCodeGen::SinAndCosApproximation<false, mlir::math::CosOp>>(&ctx);
        patterns.add<mlir::math::ErfPolynomialApproximation>(&ctx);
        mlir::populateExpandRoundEvenPattern(patterns);
        mlir::populateExpandRoundFPattern(patterns);

        // Increase pattern benefit to take precedence over native mlir patterns.
        patterns.add(Roundfp16, mlir::PatternBenefit(100));
        patterns.add(RoundEvenfp16, mlir::PatternBenefit(100));
        mlir::populateExpandFmaFPattern(patterns);

        if (mlir::failed(
                    mlir::applyPatternsAndFoldGreedily(funcOp, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
            signalPassFailure();
            return;
        }
    }
}

}  // namespace

//
// createExpandLayersPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createExpandLayersPass(Logger log) {
    return std::make_unique<ExpandLayersPass>(log);
}
