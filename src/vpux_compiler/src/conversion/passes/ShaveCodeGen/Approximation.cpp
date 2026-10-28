//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// ============================================================================
// TEMPORARY WORKAROUND IMPLEMENTATION
//
// These files (Approximation.hpp and Approximation.cpp)
// serve as temporary storage for various functions and utilities required
// for our expansion work. Due to our current integration with an older version
// of LLVM which does not expose some of the new APIs (such as SinAndCosApproximation)
// via official headers, we have copied the necessary functionality into these files.
//
// In addition to the SinAndCosApproximation functionality, we plan to add and
// extend other functions here as needed during this transitional phase.
//
// NOTE:
//   - This is a temporary solution until we update LLVM and gain direct access to
//     the official implementations via provided headers.
//   - The code in these files is subject to change and will eventually be removed
//     or refactored once the LLVM update is complete.
//   - Please ensure that any new functions added to these files are clearly marked
//     as part of this temporary workaround, and are reviewed during the migration.
// ============================================================================

#include <cstddef>

#include <mlir/Dialect/Math/Transforms/Passes.h>
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/Approximation.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace mlir;
using namespace mlir::math;
using namespace mlir::vector;

namespace vpux::ShaveCodeGen {

// Helper to encapsulate a vector's shape (including scalable dims).
struct VectorShape {
    ArrayRef<int64_t> sizes;
    ArrayRef<bool> scalableFlags;
};

// Returns vector shape if the type is a vector, otherwise return nullopt.
static std::optional<VectorShape> vectorShape(Type type) {
    if (auto vectorType = dyn_cast<VectorType>(type)) {
        return VectorShape{vectorType.getShape(), vectorType.getScalableDims()};
    }
    return std::nullopt;
}

static std::optional<VectorShape> vectorShape(Value value) {
    return vectorShape(value.getType());
}

//----------------------------------------------------------------------------//
// Helper functions to create constants.
//----------------------------------------------------------------------------//

static inline Value f32Cst(ImplicitLocOpBuilder& builder, double value) {
    return arith::ConstantOp::create(builder, builder.getF32FloatAttr(value));
}

static inline Value i32Cst(ImplicitLocOpBuilder& builder, int32_t value) {
    return arith::ConstantOp::create(builder, builder.getI32IntegerAttr(value));
}

//----------------------------------------------------------------------------//
// Broadcast scalar types and values into vector types and values.
//----------------------------------------------------------------------------//

// Broadcasts scalar type into vector type (iff shape is non-scalar).
static Type broadcast(Type type, std::optional<VectorShape> shape) {
    assert(!isa<VectorType>(type) && "must be scalar type");
    return shape ? VectorType::get(shape->sizes, type, shape->scalableFlags) : type;
}

// Broadcasts scalar value into vector (iff shape is non-scalar).
static Value broadcast(ImplicitLocOpBuilder& builder, Value value, std::optional<VectorShape> shape) {
    assert(!isa<VectorType>(value.getType()) && "must be scalar value");
    auto type = broadcast(value.getType(), shape);
    return shape ? BroadcastOp::create(builder, type, value) : value;
}

static Value floatCst(ImplicitLocOpBuilder& builder, float value, Type elementType) {
    assert((elementType.isF16() || elementType.isF32()) && "x must be f16 or f32 type.");
    return arith::ConstantOp::create(builder, builder.getFloatAttr(elementType, value));
}

//----------------------------------------------------------------------------//
// Sin and Cos approximation.
//----------------------------------------------------------------------------//

#define TWO_OVER_PI 0.6366197723675813430755350534900574481378385829618257949906693762L
#define PI_OVER_2 1.5707963267948966192313216916397514420985846996875529104874722961L

// Approximates sin(x) or cos(x) by finding the best approximation polynomial in
// the reduced range [0, pi/2] for both sin(x) and cos(x). Then given y in the
// reduced range sin(x) will be computed as sin(y), -sin(y), cos(y) or -cos(y).
template <bool isSine, typename OpTy>
mlir::LogicalResult SinAndCosApproximation<isSine, OpTy>::matchAndRewrite(OpTy op,
                                                                          mlir::PatternRewriter& rewriter) const {
    using namespace mlir;
    using namespace mlir::math;

    static_assert(llvm::is_one_of<OpTy, math::SinOp, math::CosOp>::value,
                  "SinAndCosApproximation pattern expects math::SinOp or math::CosOp");

    if (!getElementTypeOrSelf(op.getOperand()).isF32()) {
        return rewriter.notifyMatchFailure(op, "unsupported operand type");
    }

    std::optional<VectorShape> shape = vectorShape(op.getOperand());

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    auto bcast = [&](Value value) -> Value {
        return broadcast(builder, value, shape);
    };
    auto mul = [&](Value a, Value b) -> Value {
        return arith::MulFOp::create(builder, a, b);
    };
    auto sub = [&](Value a, Value b) -> Value {
        return arith::SubFOp::create(builder, a, b);
    };
    auto floor = [&](Value a) {
        return math::FloorOp::create(builder, a);
    };

    auto i32Vec = broadcast(builder.getI32Type(), shape);
    auto fPToSingedInteger = [&](Value a) -> Value {
        return arith::FPToSIOp::create(builder, i32Vec, a);
    };

    auto modulo4 = [&](Value a) -> Value {
        return arith::AndIOp::create(builder, a, bcast(i32Cst(builder, 3)));
    };

    auto isEqualTo = [&](Value a, Value b) -> Value {
        return arith::CmpIOp::create(builder, arith::CmpIPredicate::eq, a, b);
    };

    auto isGreaterThan = [&](Value a, Value b) -> Value {
        return arith::CmpIOp::create(builder, arith::CmpIPredicate::sgt, a, b);
    };

    auto select = [&](Value cond, Value t, Value f) -> Value {
        return arith::SelectOp::create(builder, cond, t, f);
    };

    auto fmla = [&](Value a, Value b, Value c) {
        return math::FmaOp::create(builder, a, b, c);
    };

    auto bitwiseOr = [&](Value a, Value b) {
        return arith::OrIOp::create(builder, a, b);
    };

    Value twoOverPi = bcast(f32Cst(builder, (float)TWO_OVER_PI));
    Value piOverTwo = bcast(f32Cst(builder, (float)PI_OVER_2));

    Value x = op.getOperand();

    Value k = floor(mul(x, twoOverPi));

    Value y = sub(x, mul(k, piOverTwo));

    Value cstOne = bcast(f32Cst(builder, 1.0));
    Value cstNegativeOne = bcast(f32Cst(builder, -1.0));

    Value cstSC2 = bcast(f32Cst(builder, -0.16666667163372039794921875f));
    Value cstSC4 = bcast(f32Cst(builder, 8.333347737789154052734375e-3f));
    Value cstSC6 = bcast(f32Cst(builder, -1.9842604524455964565277099609375e-4f));
    Value cstSC8 = bcast(f32Cst(builder, 2.760012648650445044040679931640625e-6f));
    Value cstSC10 = bcast(f32Cst(builder, -2.50293279435709337121807038784027099609375e-8f));

    Value cstCC2 = bcast(f32Cst(builder, -0.5f));
    Value cstCC4 = bcast(f32Cst(builder, 4.166664183139801025390625e-2f));
    Value cstCC6 = bcast(f32Cst(builder, -1.388833043165504932403564453125e-3f));
    Value cstCC8 = bcast(f32Cst(builder, 2.47562347794882953166961669921875e-5f));
    Value cstCC10 = bcast(f32Cst(builder, -2.59630184018533327616751194000244140625e-7f));

    Value kMod4 = modulo4(fPToSingedInteger(k));

    Value kR0 = isEqualTo(kMod4, bcast(i32Cst(builder, 0)));
    Value kR1 = isEqualTo(kMod4, bcast(i32Cst(builder, 1)));
    Value kR2 = isEqualTo(kMod4, bcast(i32Cst(builder, 2)));
    Value kR3 = isEqualTo(kMod4, bcast(i32Cst(builder, 3)));

    Value sinuseCos = isSine ? bitwiseOr(kR1, kR3) : bitwiseOr(kR0, kR2);
    Value negativeRange = isSine ? isGreaterThan(kMod4, bcast(i32Cst(builder, 1))) : bitwiseOr(kR1, kR2);

    Value y2 = mul(y, y);

    Value base = select(sinuseCos, cstOne, y);
    Value cstC2 = select(sinuseCos, cstCC2, cstSC2);
    Value cstC4 = select(sinuseCos, cstCC4, cstSC4);
    Value cstC6 = select(sinuseCos, cstCC6, cstSC6);
    Value cstC8 = select(sinuseCos, cstCC8, cstSC8);
    Value cstC10 = select(sinuseCos, cstCC10, cstSC10);

    Value v1 = fmla(y2, cstC10, cstC8);
    Value v2 = fmla(y2, v1, cstC6);
    Value v3 = fmla(y2, v2, cstC4);
    Value v4 = fmla(y2, v3, cstC2);
    Value v5 = fmla(y2, v4, cstOne);
    Value v6 = mul(base, v5);

    Value approximation = select(negativeRange, mul(cstNegativeOne, v6), v6);

    rewriter.replaceOp(op, approximation);

    return success();
}

//----------------------------------------------------------------------------//
// Asin approximation.
//----------------------------------------------------------------------------//

// Approximates asin(x).
// This approximation is based on the following stackoverflow post:
// https://stackoverflow.com/a/42683455
LogicalResult AsinPolynomialApproximation::matchAndRewrite(math::AsinOp op, PatternRewriter& rewriter) const {
    Value operand = op.getOperand();
    Type elementType = getElementTypeOrSelf(operand);

    if (!(elementType.isF32() || elementType.isF16())) {
        return rewriter.notifyMatchFailure(op, "only f32 and f16 type is supported.");
    }
    std::optional<VectorShape> shape = vectorShape(operand);

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    auto bcast = [&](Value value) -> Value {
        return broadcast(builder, value, shape);
    };

    auto fma = [&](Value a, Value b, Value c) -> Value {
        return math::FmaOp::create(builder, a, b, c);
    };

    auto mul = [&](Value a, Value b) -> Value {
        return arith::MulFOp::create(builder, a, b);
    };

    auto sub = [&](Value a, Value b) -> Value {
        return arith::SubFOp::create(builder, a, b);
    };

    auto abs = [&](Value a) -> Value {
        return math::AbsFOp::create(builder, a);
    };

    auto sqrt = [&](Value a) -> Value {
        return math::SqrtOp::create(builder, a);
    };

    auto scopy = [&](Value a, Value b) -> Value {
        return math::CopySignOp::create(builder, a, b);
    };

    auto sel = [&](Value a, Value b, Value c) -> Value {
        return arith::SelectOp::create(builder, a, b, c);
    };

    Value abso = abs(operand);
    Value aa = mul(operand, operand);
    Value opp = sqrt(sub(bcast(floatCst(builder, 1.0, elementType)), aa));

    Value gt =
            arith::CmpFOp::create(builder, arith::CmpFPredicate::OGT, aa, bcast(floatCst(builder, 0.5, elementType)));

    Value x = sel(gt, opp, abso);

    // Asin(x) approximation for x = [-9/16, 9/16]:
    Value s = mul(x, x);
    Value q = mul(s, s);
    Value r = bcast(floatCst(builder, static_cast<float>(5.5579749017470502e-2), elementType));
    Value t = bcast(floatCst(builder, static_cast<float>(-6.2027913464120114e-2), elementType));

    r = fma(r, q, bcast(floatCst(builder, static_cast<float>(5.4224464349245036e-2), elementType)));
    t = fma(t, q, bcast(floatCst(builder, static_cast<float>(-1.1326992890324464e-2), elementType)));
    r = fma(r, q, bcast(floatCst(builder, static_cast<float>(1.5268872539397656e-2), elementType)));
    t = fma(t, q, bcast(floatCst(builder, static_cast<float>(1.0493798473372081e-2), elementType)));
    r = fma(r, q, bcast(floatCst(builder, static_cast<float>(1.4106045900607047e-2), elementType)));
    t = fma(t, q, bcast(floatCst(builder, static_cast<float>(1.7339776384962050e-2), elementType)));
    r = fma(r, q, bcast(floatCst(builder, static_cast<float>(2.2372961589651054e-2), elementType)));
    t = fma(t, q, bcast(floatCst(builder, static_cast<float>(3.0381912707941005e-2), elementType)));
    r = fma(r, q, bcast(floatCst(builder, static_cast<float>(4.4642857881094775e-2), elementType)));
    t = fma(t, q, bcast(floatCst(builder, static_cast<float>(7.4999999991367292e-2), elementType)));
    r = fma(r, s, t);
    r = fma(r, s, bcast(floatCst(builder, static_cast<float>(1.6666666666670193e-1), elementType)));
    t = mul(x, s);
    r = fma(r, t, x);

    Value rsub = sub(bcast(floatCst(builder, static_cast<float>(1.57079632679), elementType)), r);
    r = sel(gt, rsub, r);
    r = scopy(r, operand);

    rewriter.replaceOp(op, r);
    return success();
}

//----------------------------------------------------------------------------//
// Acos approximation.
//----------------------------------------------------------------------------//

// Approximates acos(x).
// This approximation is based on the following stackoverflow post:
// https://stackoverflow.com/a/42683455

LogicalResult AcosPolynomialApproximation::matchAndRewrite(math::AcosOp op, PatternRewriter& rewriter) const {
    Value operand = op.getOperand();
    Type elementType = getElementTypeOrSelf(operand);

    if (!(elementType.isF32() || elementType.isF16())) {
        return rewriter.notifyMatchFailure(op, "only f32 and f16 type is supported.");
    }
    std::optional<VectorShape> shape = vectorShape(operand);

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    auto bcast = [&](Value value) -> Value {
        return broadcast(builder, value, shape);
    };

    auto fma = [&](Value a, Value b, Value c) -> Value {
        return math::FmaOp::create(builder, a, b, c);
    };

    auto mul = [&](Value a, Value b) -> Value {
        return arith::MulFOp::create(builder, a, b);
    };

    Value negOperand = arith::NegFOp::create(builder, operand);
    Value zero = bcast(floatCst(builder, 0.0, elementType));
    Value half = bcast(floatCst(builder, 0.5, elementType));
    Value negOne = bcast(floatCst(builder, -1.0, elementType));
    Value selR = arith::CmpFOp::create(builder, arith::CmpFPredicate::OGT, operand, zero);
    Value r = arith::SelectOp::create(builder, selR, negOperand, operand);
    Value chkConst = bcast(floatCst(builder, -0.5625, elementType));
    Value firstPred = arith::CmpFOp::create(builder, arith::CmpFPredicate::OGT, r, chkConst);

    Value trueVal = fma(bcast(floatCst(builder, static_cast<float>(9.3282184640716537e-1), elementType)),
                        bcast(floatCst(builder, static_cast<float>(1.6839188885261840e+0), elementType)),
                        math::AsinOp::create(builder, r));

    Value falseVal = math::SqrtOp::create(builder, fma(half, r, half));
    falseVal = math::AsinOp::create(builder, falseVal);
    falseVal = mul(bcast(floatCst(builder, 2.0, elementType)), falseVal);

    r = arith::SelectOp::create(builder, firstPred, trueVal, falseVal);

    // Check whether the operand lies in between [-1.0, 0.0).
    Value greaterThanNegOne = arith::CmpFOp::create(builder, arith::CmpFPredicate::OGE, operand, negOne);

    Value lessThanZero = arith::CmpFOp::create(builder, arith::CmpFPredicate::OLT, operand, zero);

    Value betweenNegOneZero = arith::AndIOp::create(builder, greaterThanNegOne, lessThanZero);

    trueVal = fma(bcast(floatCst(builder, static_cast<float>(1.8656436928143307e+0), elementType)),
                  bcast(floatCst(builder, static_cast<float>(1.6839188885261840e+0), elementType)),
                  arith::NegFOp::create(builder, r));

    Value finalVal = arith::SelectOp::create(builder, betweenNegOneZero, trueVal, r);

    rewriter.replaceOp(op, finalVal);
    return success();
}

template struct SinAndCosApproximation<true, mlir::math::SinOp>;
template struct SinAndCosApproximation<false, mlir::math::CosOp>;

}  // namespace vpux::ShaveCodeGen
