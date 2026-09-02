//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/conversion/passes/ShaveCodeGen/Approximation.hpp"

#include <mlir/Dialect/Math/IR/Math.h>
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
    auto getUI16Attr = [&](uint16_t value) {
        return rewriter.getIntegerAttr(i16Type, llvm::APInt(16, value));
    };
    Value xInt = rewriter.create<arith::BitcastOp>(loc, i16Type, operand);

    Value fp16Bias = rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(FP16_BIAS));
    Value fractBits = rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(FP16_FRACTBITS));
    constexpr uint16_t TOTALBITS_ONE = static_cast<uint16_t>(FP16_TOTALBITS - 1);
    Value totalBitsMinusOne = rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(TOTALBITS_ONE));

    constexpr uint16_t signMaskValue = static_cast<uint16_t>(1u << TOTALBITS_ONE);
    Value signMask = rewriter.create<arith::ConstantOp>(loc, getUI16Attr(signMaskValue));

    // xAbs = xInt & ~signMask
    constexpr uint16_t absMaskvalue = static_cast<uint16_t>(~(1u << FP16_BIAS));  // 0x7FFF
    Value absMask = rewriter.create<arith::ConstantOp>(loc, getUI16Attr(absMaskvalue));
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
    Value fp16TruncFrac = rewriter.create<arith::ConstantOp>(loc, getUI16Attr(fp16TruncFracvalue));

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
    Value rawAllOnes = rewriter.create<arith::ConstantOp>(loc, getUI16Attr(unsignedminusone));
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
            loc, rewriter.getIntegerAttr(i16Type, llvm::APInt(16, signMaskValue)));

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

mlir::LogicalResult Asinhfp16(mlir::math::AsinhOp op, mlir::PatternRewriter& rewriter) {
    mlir::Location loc = op.getLoc();
    mlir::ImplicitLocOpBuilder b(loc, rewriter);
    mlir::Value input = op.getOperand();
    mlir::Type opType = input.getType();
    mlir::Type opEType = getElementTypeOrSelf(opType);

    if (!opEType.isF16()) {
        return rewriter.notifyMatchFailure(op, "not a round of f16.");
    }

    auto f32Type = mlir::Float32Type::get(rewriter.getContext());

    float ln2 = 0.69314718056f;
    float big = 1.8446744e+19f;

    float cp3 = -0.16666276752948760986328125f;
    float cp2 = 7.4845172464847564697265625e-2f;
    float cp1 = -4.26840074360370635986328125e-2f;
    float cp0 = 2.00918130576610565185546875e-2f;

    float cq6 = 0.481211841106414794921875f;
    float cq5 = 0.89442551136016845703125f;
    float cq4 = -0.178837835788726806640625f;
    float cq3 = -4.8278056085109710693359375e-2f;
    float cq2 = 7.5103588402271270751953125e-2f;
    float cq1 = -3.49466800689697265625e-2f;
    float cq0 = 5.84384240210056304931640625e-3f;

    mlir::Value x32 = rewriter.create<mlir::arith::ExtFOp>(loc, f32Type, input);

    auto poly0 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, cp0));
    auto poly1 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, cp1));
    auto poly2 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, cp2));
    auto poly3 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, cp3));

    auto pq0 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, cq0));
    auto pq1 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, cq1));
    auto pq2 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, cq2));
    auto pq3 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, cq3));
    auto pq4 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, cq4));
    auto pq5 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, cq5));
    auto pq6 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, cq6));

    auto vbig = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, big));
    auto vln = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, ln2));
    auto one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, 1.0f));
    auto halfone = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(f32Type, 0.5f));
    auto abs = rewriter.create<mlir::math::AbsFOp>(loc, x32);  // abs(x)

    // if |x| < 0.5; Asinh(x) = x * (1.0f + xx * (poly3 + xx * (poly2 + xx * (poly1 + xx * poly0))));
    auto x2 = rewriter.create<mlir::arith::MulFOp>(loc, x32, x32);
    auto p0 = rewriter.create<mlir::arith::MulFOp>(loc, x2, poly0);
    auto p1 = rewriter.create<mlir::arith::AddFOp>(loc, p0, poly1);
    auto p2 = rewriter.create<mlir::arith::MulFOp>(loc, x2, p1);
    auto p3 = rewriter.create<mlir::arith::AddFOp>(loc, p2, poly2);
    auto p4 = rewriter.create<mlir::arith::MulFOp>(loc, x2, p3);
    auto p5 = rewriter.create<mlir::arith::AddFOp>(loc, p4, poly3);
    auto p6 = rewriter.create<mlir::arith::MulFOp>(loc, x2, p5);
    auto p7 = rewriter.create<mlir::arith::AddFOp>(loc, one, p6);
    auto pu = rewriter.create<mlir::arith::MulFOp>(loc, x32, p7);
    auto is_small = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, abs, halfone);

    // if |x| < 1.0; Asinh(x) = copysignf(pq6 + xAbs * (pq5 + xAbs * (pq4 + xAbs * (pq3 + xAbs * (pq2 + xAbs * (pq1 +
    // xAbs * pq0))))),x)
    auto absminhalf = rewriter.create<mlir::arith::SubFOp>(loc, abs, halfone);
    auto q0 = rewriter.create<mlir::arith::MulFOp>(loc, absminhalf, pq0);
    auto q1 = rewriter.create<mlir::arith::AddFOp>(loc, q0, pq1);
    auto q2 = rewriter.create<mlir::arith::MulFOp>(loc, absminhalf, q1);
    auto q3 = rewriter.create<mlir::arith::AddFOp>(loc, q2, pq2);
    auto q4 = rewriter.create<mlir::arith::MulFOp>(loc, absminhalf, q3);
    auto q5 = rewriter.create<mlir::arith::AddFOp>(loc, q4, pq3);
    auto q6 = rewriter.create<mlir::arith::MulFOp>(loc, absminhalf, q5);
    auto q7 = rewriter.create<mlir::arith::AddFOp>(loc, q6, pq4);
    auto q8 = rewriter.create<mlir::arith::MulFOp>(loc, absminhalf, q7);
    auto q9 = rewriter.create<mlir::arith::AddFOp>(loc, q8, pq5);
    auto q10 = rewriter.create<mlir::arith::MulFOp>(loc, absminhalf, q9);
    auto q11 = rewriter.create<mlir::arith::AddFOp>(loc, q10, pq6);
    auto qu = rewriter.create<mlir::math::CopySignOp>(loc, q11, x32);
    auto is_one = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, abs, one);

    // copysignf(log(xAbs) + ln2, x);                              for |x| > big
    auto logf = rewriter.create<mlir::math::LogOp>(loc, abs, mlir::arith::FastMathFlags::afn);  // log(xAbs)
    auto tvalue = rewriter.create<mlir::arith::AddFOp>(loc, logf, vln);                         // log(xAbs) + ln2
    auto bigvalue = rewriter.create<mlir::math::CopySignOp>(loc, tvalue, x32);

    auto is_big = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, abs, vbig);

    // copysignf(log(xAbs + sqrtf(x*x + 1.0f)), x);       for |x| <= big
    auto fma = rewriter.create<mlir::math::FmaOp>(loc, abs, abs, one);                              // x^2+1
    auto sqrtVal = rewriter.create<mlir::math::SqrtOp>(loc, fma, mlir::arith::FastMathFlags::afn);  // sqrt(x^2 + 1)
    auto tsum = rewriter.create<mlir::arith::AddFOp>(loc, sqrtVal, abs);
    auto tmed = rewriter.create<mlir::math::LogOp>(loc, tsum, mlir::arith::FastMathFlags::afn);
    auto med = rewriter.create<mlir::math::CopySignOp>(loc, tmed, x32);

    auto vt1 = rewriter.create<mlir::arith::SelectOp>(loc, is_small, pu, qu);
    auto vt2 = rewriter.create<mlir::arith::SelectOp>(loc, is_one, vt1, med);
    auto res32 = rewriter.create<mlir::arith::SelectOp>(loc, is_big, bigvalue, vt2).getResult();

    mlir::Value res = rewriter.create<mlir::arith::TruncFOp>(loc, opType, res32).getResult();

    rewriter.replaceOp(op, res);
    return mlir::success();
}

// Acosh layer  acosh(x) = log(x + sqrt(x^2 - 1))

mlir::LogicalResult Acoshfp16(mlir::math::AcoshOp op, mlir::PatternRewriter& rewriter) {
    mlir::Location loc = op.getLoc();
    mlir::ImplicitLocOpBuilder b(loc, rewriter);
    mlir::Value input = op.getOperand();
    mlir::Type opType = input.getType();
    mlir::Type opEType = getElementTypeOrSelf(opType);

    if (!opEType.isF16()) {
        return rewriter.notifyMatchFailure(op, "not a round of f16.");
    }

    float ln2 = 0.6933594f;
    float big = 255.75f;
    auto nan16 = llvm::APFloat::getQNaN(llvm::APFloat::IEEEhalf());

    auto vbig = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), big));
    auto vln = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), ln2));
    auto one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 1.0f));
    auto mone = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), -1.0f));
    auto nan = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), nan16));
    auto onep2 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 1.2f));

    auto poly1 = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getFloatAttr(input.getType(), -0.08331299f));  // -1/12
    auto poly2 = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getFloatAttr(input.getType(), 0.018753052f));  // 3/160
    auto poly3 = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getFloatAttr(input.getType(), -0.005580902f));  // -5/896
    auto two = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 2.0f));

    // x < 1 return NaN
    auto is_small = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, input, one);

    // x < 1.2f
    auto xmin1 = rewriter.create<mlir::arith::AddFOp>(loc, input, mone);  // x - 1
    auto p0 = rewriter.create<mlir::arith::MulFOp>(loc, xmin1, poly3);
    auto p1 = rewriter.create<mlir::arith::AddFOp>(loc, p0, poly2);
    auto p2 = rewriter.create<mlir::arith::MulFOp>(loc, p1, xmin1);
    auto p3 = rewriter.create<mlir::arith::AddFOp>(loc, p2, poly1);
    auto p4 = rewriter.create<mlir::arith::MulFOp>(loc, p3, xmin1);
    auto pu = rewriter.create<mlir::arith::AddFOp>(loc, p4, one);

    auto v2u = rewriter.create<mlir::arith::MulFOp>(loc, xmin1, two);
    auto s2u = rewriter.create<mlir::math::SqrtOp>(loc, v2u, mlir::arith::FastMathFlags::afn);
    auto v_lt12 = rewriter.create<mlir::arith::MulFOp>(loc, s2u, pu);
    auto is_lt12 = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, input, onep2);

    // x < big, to avoid x*x overflows
    auto logf = rewriter.create<mlir::math::LogOp>(loc, input, mlir::arith::FastMathFlags::afn);  // log(x)
    auto bigvalue = rewriter.create<mlir::arith::AddFOp>(loc, logf, vln);                         // log(x) + ln2
    auto is_over = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, input, vbig);

    // rest
    auto xplus1 = rewriter.create<mlir::arith::AddFOp>(loc, input, one);      // x + 1
    auto mulsqrt = rewriter.create<mlir::arith::MulFOp>(loc, xmin1, xplus1);  // (x - 1)(x + 1)
    auto sqrtVal = rewriter.create<mlir::math::SqrtOp>(loc, mulsqrt,
                                                       mlir::arith::FastMathFlags::afn);  // sqrt((x - 1)(x + 1))
    auto addSqrt = rewriter.create<mlir::arith::AddFOp>(loc, input, sqrtVal);             // x + sqrt(x^2 - 1)
    auto rest = rewriter.create<mlir::math::LogOp>(loc, addSqrt);

    auto vsmall = rewriter.create<mlir::arith::SelectOp>(loc, is_lt12, v_lt12, rest);    // x < 1.2f
    auto over = rewriter.create<mlir::arith::SelectOp>(loc, is_over, bigvalue, vsmall);  // x > big
    auto res = rewriter.create<mlir::arith::SelectOp>(loc, is_small, nan, over);         // x < 1

    rewriter.replaceOp(op, res);
    return mlir::success();
}

// Lowers an arith::NegFOp. Fneg is not supported in movicompile
static mlir::LogicalResult NegF(mlir::arith::NegFOp op, mlir::PatternRewriter& rewriter) {
    mlir::Location loc = op->getLoc();
    mlir::Value input = op.getOperand();
    mlir::Type opType = input.getType();

    auto zeroConst = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(opType, 0.0));  // Const 0
    mlir::Value res = rewriter.create<mlir::arith::SubFOp>(loc, zeroConst, input);                       // -x = 0-x

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
        target.addIllegalOp<mlir::math::AsinOp, mlir::math::AcosOp>();
        target.addIllegalOp<mlir::math::AsinhOp, mlir::math::AcoshOp>();
        target.addIllegalOp<mlir::arith::NegFOp>();

        patterns.add<vpux::ShaveCodeGen::SinAndCosApproximation<true, mlir::math::SinOp>,
                     vpux::ShaveCodeGen::SinAndCosApproximation<false, mlir::math::CosOp>>(&ctx);
        patterns.add<mlir::math::ErfPolynomialApproximation>(&ctx);
        patterns.add<vpux::ShaveCodeGen::AsinPolynomialApproximation, vpux::ShaveCodeGen::AcosPolynomialApproximation>(
                &ctx);

        // Increase pattern benefit to take precedence over native mlir patterns.
        patterns.add(Roundfp16, mlir::PatternBenefit(100));
        patterns.add(RoundEvenfp16, mlir::PatternBenefit(100));
        patterns.add(NegF);
        patterns.add(Asinhfp16);
        patterns.add(Acoshfp16);

        const auto extendPatternsWith = [&patterns](StringRef name) {
            [[maybe_unused]] const auto consumed =
                    name.consume_front((mlir::math::MathDialect::getDialectNamespace() + ".").str());
            assert(consumed);
            mlir::math::populateExpansionPatterns(patterns, name);
        };

        extendPatternsWith(mlir::math::RoundEvenOp::getOperationName());
        extendPatternsWith(mlir::math::RoundOp::getOperationName());
        extendPatternsWith(mlir::math::FmaOp::getOperationName());

        if (mlir::failed(mlir::applyPatternsGreedily(funcOp, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
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
