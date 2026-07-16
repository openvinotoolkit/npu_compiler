//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/conversions.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/linalg_type_conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/Dialect/Quant/Transforms/Passes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
#define GEN_PASS_DECL_CONVERTELTWISELAYERS2MATH
#define GEN_PASS_DEF_CONVERTELTWISELAYERS2MATH
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// ConvertSWLayers2LinalgPass
//

class ConvertEltwiseLayers2MathPass final : public impl::ConvertEltwiseLayers2MathBase<ConvertEltwiseLayers2MathPass> {
public:
    explicit ConvertEltwiseLayers2MathPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

template <typename SrcOp>
mlir::Value emitLinalgRegion(SrcOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                             mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(op);
    VPUX_UNUSED(args);
    VPUX_UNUSED(resultTypes);
    VPUX_UNUSED(rewriter);
    VPUX_THROW("emitLinalgRegion not specialized for operator");
}

template <>
mlir::Value emitLinalgRegion<IE::LogOp>(IE::LogOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    return rewriter.create<mlir::math::LogOp>(op->getLoc(), resultTypes[0], args[0], mlir::arith::FastMathFlags::afn);
}

template <>
mlir::Value emitLinalgRegion<IE::ExpOp>(IE::ExpOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    return rewriter.create<mlir::math::ExpOp>(op->getLoc(), resultTypes[0], args[0], mlir::arith::FastMathFlags::afn);
}

template <>
mlir::Value emitLinalgRegion<IE::SqrtOp>(IE::SqrtOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    return rewriter.create<mlir::math::SqrtOp>(op->getLoc(), resultTypes[0], args[0], mlir::arith::FastMathFlags::afn);
}

template <>
mlir::Value emitLinalgRegion<IE::TanhOp>(IE::TanhOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    return rewriter.create<mlir::math::TanhOp>(op->getLoc(), resultTypes[0], args[0], mlir::arith::FastMathFlags::afn);
}

template <>
mlir::Value emitLinalgRegion<IE::AtanOp>(IE::AtanOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    return rewriter.create<mlir::math::AtanOp>(op->getLoc(), resultTypes[0], args[0], mlir::arith::FastMathFlags::afn);
}

// Bitwise layers

template <>
mlir::Value emitLinalgRegion<IE::BitwiseAndOp>(IE::BitwiseAndOp op, mlir::ValueRange args,
                                               llvm::ArrayRef<mlir::Type> resultTypes,
                                               mlir::PatternRewriter& rewriter) {
    return rewriter.create<mlir::arith::AndIOp>(op->getLoc(), resultTypes, args);
}

template <>
mlir::Value emitLinalgRegion<IE::BitwiseOrOp>(IE::BitwiseOrOp op, mlir::ValueRange args,
                                              llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    return rewriter.create<mlir::arith::OrIOp>(op->getLoc(), resultTypes, args);
}

template <>
mlir::Value emitLinalgRegion<IE::BitwiseXorOp>(IE::BitwiseXorOp op, mlir::ValueRange args,
                                               llvm::ArrayRef<mlir::Type> resultTypes,
                                               mlir::PatternRewriter& rewriter) {
    return rewriter.create<mlir::arith::XOrIOp>(op->getLoc(), resultTypes, args);
}

template <>
mlir::Value emitLinalgRegion<IE::BitwiseNotOp>(IE::BitwiseNotOp op, mlir::ValueRange args,
                                               llvm::ArrayRef<mlir::Type> resultTypes,
                                               mlir::PatternRewriter& rewriter) {
    // Do this as a xor with ~0 since there is no bitwise not operator in arith.
    auto allOnesAttr =
            rewriter.getIntegerAttr(resultTypes[0], llvm::APInt::getAllOnes(resultTypes[0].getIntOrFloatBitWidth()));
    auto allOnes = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), allOnesAttr);
    return rewriter.create<mlir::arith::XOrIOp>(op->getLoc(), resultTypes, args[0], allOnes);
}

// Logical and select layers

static mlir::Value emitNotEqualZero(mlir::Value val, mlir::PatternRewriter& rewriter) {
    // Emit a comparison of val to zero to determine its truthfulness (true if the
    // value is not equal to zero). This is used in the implementation of logical and
    // select layers for converting input values to bool (i1).
    if (mlir::isa<mlir::FloatType>(val.getType())) {
        auto zeroAttr = rewriter.getFloatAttr(val.getType(), 0.);
        mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), zeroAttr);
        return rewriter.create<mlir::arith::CmpFOp>(val.getLoc(), mlir::arith::CmpFPredicate::ONE, val, zero,
                                                    mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);
    }
    auto zeroAttr = rewriter.getIntegerAttr(val.getType(), 0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), zeroAttr);

    return rewriter.create<mlir::arith::CmpIOp>(val.getLoc(), mlir::arith::CmpIPredicate::ne, val, zero);
}

static mlir::Value convertFromBool(mlir::Value val, mlir::Type ty, mlir::PatternRewriter& rewriter) {
    if (mlir::isa<mlir::FloatType>(ty)) {
        return rewriter.create<mlir::arith::UIToFPOp>(val.getLoc(), ty, val);
    }
    return rewriter.create<mlir::arith::ExtUIOp>(val.getLoc(), ty, val);
}

template <>
mlir::Value emitLinalgRegion<IE::SelectOp>(IE::SelectOp op, mlir::ValueRange args,
                                           llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    auto compare = emitNotEqualZero(args[0], rewriter);
    return rewriter.create<mlir::arith::SelectOp>(op->getLoc(), resultTypes[0], compare, args[1], args[2]);
}

template <>
mlir::Value emitLinalgRegion<IE::LogicalNotOp>(IE::LogicalNotOp op, mlir::ValueRange args,
                                               llvm::ArrayRef<mlir::Type> resultTypes,
                                               mlir::PatternRewriter& rewriter) {
    auto compare = emitNotEqualZero(args[0], rewriter);
    auto oneAttr = rewriter.getIntegerAttr(compare.getType(), 1);
    mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), oneAttr);  // i1 1
    auto logicalRes = rewriter.create<mlir::arith::XOrIOp>(op->getLoc(), compare, one);
    return convertFromBool(logicalRes, resultTypes[0], rewriter);
}

template <typename LogicalOp>
mlir::Value emitEltwiseLogical(mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                               mlir::PatternRewriter& rewriter) {
    auto lhsCompare = emitNotEqualZero(args[0], rewriter);
    auto rhsCompare = emitNotEqualZero(args[1], rewriter);
    auto logicalRes = rewriter.create<LogicalOp>(op->getLoc(), lhsCompare, rhsCompare);
    return convertFromBool(logicalRes, resultTypes[0], rewriter);
}

template <>
mlir::Value emitLinalgRegion<IE::LogicalOrOp>(IE::LogicalOrOp op, mlir::ValueRange args,
                                              llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    return emitEltwiseLogical<mlir::arith::OrIOp>(op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion<IE::AndOp>(IE::AndOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    return emitEltwiseLogical<mlir::arith::AndIOp>(op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion<IE::LogicalXorOp>(IE::LogicalXorOp op, mlir::ValueRange args,
                                               llvm::ArrayRef<mlir::Type> resultTypes,
                                               mlir::PatternRewriter& rewriter) {
    return emitEltwiseLogical<mlir::arith::XOrIOp>(op, args, resultTypes, rewriter);
}

// Comparison layers

template <mlir::arith::CmpIPredicate signedP, mlir::arith::CmpIPredicate unsignedP, mlir::arith::CmpFPredicate fpP>
static mlir::Value emitCmp(mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                           mlir::PatternRewriter& rewriter) {
    auto elTy = mlir::cast<NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    mlir::Value cmpVal = nullptr;
    if (mlir::isa<mlir::FloatType>(elTy)) {
        cmpVal = rewriter.create<mlir::arith::CmpFOp>(
                op->getLoc(), fpP, args[0], args[1],
                mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);
    } else if (elTy.isSignedInteger()) {
        cmpVal = rewriter.create<mlir::arith::CmpIOp>(op->getLoc(), signedP, args[0], args[1]);
    } else {
        cmpVal = rewriter.create<mlir::arith::CmpIOp>(op->getLoc(), unsignedP, args[0], args[1]);
    }

    if (mlir::isa<mlir::FloatType>(resultTypes[0])) {
        return rewriter.create<mlir::arith::UIToFPOp>(op->getLoc(), resultTypes[0], cmpVal);
    }

    return rewriter.create<mlir::arith::ExtUIOp>(op->getLoc(), resultTypes[0], cmpVal);
}

template <>
mlir::Value emitLinalgRegion(IE::EqualOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                             mlir::PatternRewriter& rewriter) {
    return emitCmp<mlir::arith::CmpIPredicate::eq, mlir::arith::CmpIPredicate::eq, mlir::arith::CmpFPredicate::OEQ>(
            op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion(IE::NotEqualOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                             mlir::PatternRewriter& rewriter) {
    return emitCmp<mlir::arith::CmpIPredicate::ne, mlir::arith::CmpIPredicate::ne, mlir::arith::CmpFPredicate::ONE>(
            op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion(IE::LessOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                             mlir::PatternRewriter& rewriter) {
    return emitCmp<mlir::arith::CmpIPredicate::slt, mlir::arith::CmpIPredicate::ult, mlir::arith::CmpFPredicate::OLT>(
            op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion(IE::LessEqualOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                             mlir::PatternRewriter& rewriter) {
    return emitCmp<mlir::arith::CmpIPredicate::sle, mlir::arith::CmpIPredicate::ule, mlir::arith::CmpFPredicate::OLE>(
            op, args, resultTypes, rewriter);
}
template <>
mlir::Value emitLinalgRegion(IE::GreaterOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                             mlir::PatternRewriter& rewriter) {
    return emitCmp<mlir::arith::CmpIPredicate::sgt, mlir::arith::CmpIPredicate::ugt, mlir::arith::CmpFPredicate::OGT>(
            op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion(IE::GreaterEqualOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                             mlir::PatternRewriter& rewriter) {
    return emitCmp<mlir::arith::CmpIPredicate::sge, mlir::arith::CmpIPredicate::uge, mlir::arith::CmpFPredicate::OGE>(
            op, args, resultTypes, rewriter);
}

// Squared diff op

template <>
mlir::Value emitLinalgRegion<IE::SquaredDifferenceOp>(IE::SquaredDifferenceOp op, mlir::ValueRange args,
                                                      llvm::ArrayRef<mlir::Type> resultTypes,
                                                      mlir::PatternRewriter& rewriter) {
    auto loc = op->getLoc();
    if (mlir::isa<mlir::FloatType>(args[0].getType())) {
        auto sub = rewriter.create<mlir::arith::SubFOp>(loc, resultTypes, args);
        return rewriter.create<mlir::arith::MulFOp>(loc, sub.getType(), sub, sub);
    }
    auto sub = rewriter.create<mlir::arith::SubIOp>(loc, resultTypes, args);
    return rewriter.create<mlir::arith::MulIOp>(loc, sub.getType(), sub, sub);
}

// Clamp/ReLU/LeakyRelu

static mlir::Value emitClamp(mlir::Value val, mlir::Type elTy, double min, double max,
                             mlir::PatternRewriter& rewriter) {
    auto loc = val.getLoc();
    if (mlir::isa<mlir::IntegerType>(val.getType())) {
        auto bw = val.getType().getIntOrFloatBitWidth();
        bool isExact;
        bool isSigned = elTy.isSignedInteger();
        llvm::APFloat minF(min), maxF(max);
        llvm::APSInt minI(bw, /*isUnsigned=*/!isSigned);
        llvm::APSInt maxI(bw, /*isUnsigned=*/!isSigned);

        // From the openvino documentation:
        // Note: In case of integral numeric type, ceil is used to convert min from float to T and floor is used to
        // convert max from float to T.
        //
        // Convert the min/max values to the specified integer type. convertToInt
        // will give us MIN_INT or MAX_INT if the value in case of underflow or overflow.
        minF.convertToInteger(minI, llvm::APFloat::rmTowardPositive, &isExact);
        maxF.convertToInteger(maxI, llvm::APFloat::rmTowardNegative, &isExact);

        // Construct the min/max constants.
        auto minCt = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(val.getType(), minI));
        auto maxCt = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(val.getType(), maxI));

        auto cmpPred = isSigned ? mlir::arith::CmpIPredicate::sle : mlir::arith::CmpIPredicate::ule;

        // Compute max(input, min_limit).
        auto cmpMin = rewriter.create<mlir::arith::CmpIOp>(loc, cmpPred, val, minCt);
        auto lowClamped = rewriter.create<mlir::arith::SelectOp>(loc, val.getType(), cmpMin, minCt, val);

        // Compute min(max(input, min_limit), max_limit).
        auto cmpMax = rewriter.create<mlir::arith::CmpIOp>(loc, cmpPred, maxCt, lowClamped);
        return rewriter.create<mlir::arith::SelectOp>(loc, val.getType(), cmpMax, maxCt, lowClamped);
    }

    // Construct the min/max constants.
    auto minCt = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(val.getType(), min));
    auto maxCt = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(val.getType(), max));
    auto fmFlags = mlir::arith::FastMathFlags::nsz | mlir::arith::FastMathFlags::nnan;

    // Compute max(input, min_limit).
    auto cmpMin = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLE, val, minCt, fmFlags);
    auto lowClamped = rewriter.create<mlir::arith::SelectOp>(loc, val.getType(), cmpMin, minCt, val);

    // Compute min(max(input, min_limit), max_limit).
    auto cmpMax =
            rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLE, lowClamped, maxCt, fmFlags);
    return rewriter.create<mlir::arith::SelectOp>(loc, val.getType(), cmpMax, lowClamped, maxCt);
}

template <>
mlir::Value emitLinalgRegion<IE::ClampOp>(IE::ClampOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                          mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto elTy = mlir::cast<NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    return emitClamp(args[0], elTy, op.getMinAttr().getValueAsDouble(), op.getMaxAttr().getValueAsDouble(), rewriter);
}

static mlir::Value emitLeakyReLU(mlir::Value val, double slope, mlir::PatternRewriter& rewriter) {
    // Compute this as max(in, 0) + slope * min(in, 0)
    auto loc = val.getLoc();
    auto zeroAttr = rewriter.getFloatAttr(val.getType(), 0.);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), zeroAttr);

    auto cmp = rewriter.create<mlir::arith::CmpFOp>(val.getLoc(), mlir::arith::CmpFPredicate::OLE, val, zero,
                                                    mlir::arith::FastMathFlags::nsz | mlir::arith::FastMathFlags::nnan);
    auto max = rewriter.create<mlir::arith::SelectOp>(loc, val.getType(), cmp, zero, val);

    if (slope == 0.) {
        return max;
    }

    auto min = rewriter.create<mlir::arith::SelectOp>(loc, cmp, val, zero);
    auto slopeAttr = rewriter.getFloatAttr(val.getType(), slope);
    auto slopeCt = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), slopeAttr);
    auto mul = rewriter.create<mlir::arith::MulFOp>(loc, min, slopeCt);
    return rewriter.create<mlir::arith::AddFOp>(loc, max, mul);
}

template <>
mlir::Value emitLinalgRegion<IE::LeakyReluOp>(IE::LeakyReluOp op, mlir::ValueRange args,
                                              llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    return emitLeakyReLU(args[0], op.getNegativeSlopeAttr().getValueAsDouble(), rewriter);
}

template <>
mlir::Value emitLinalgRegion<IE::ReLUOp>(IE::ReLUOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    VPUX_UNUSED(op);
    return emitLeakyReLU(args[0], 0., rewriter);
}

// Elu

template <>
mlir::Value emitLinalgRegion<IE::EluOp>(IE::EluOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    auto loc = op.getLoc();
    auto input = args[0];

    auto elemType = mlir::cast<mlir::FloatType>(resultTypes[0]);

    // Create constants
    auto zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 0.0));
    auto one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 1.0));

    // Access alpha (x attribute)
    auto xAttr = op.getXAttr();
    const double alpha = xAttr.getValue().convertToDouble();
    auto alphaConst = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, alpha));

    // Use afn (approximate functions) flag for exp
    auto fmFlags = mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), mlir::arith::FastMathFlags::afn);
    auto fmFlagsMinMax = mlir::arith::FastMathFlagsAttr::get(
            rewriter.getContext(), mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);

    // Compute min and max using vector operations
    auto min = rewriter.create<mlir::arith::MinimumFOp>(loc, input, zero, fmFlagsMinMax);
    auto max = rewriter.create<mlir::arith::MaximumFOp>(loc, input, zero, fmFlagsMinMax);

    // Compute exponential of min (negative values only)
    auto expMin = rewriter.create<mlir::math::ExpOp>(loc, min, fmFlags);

    // Combine: max + alpha * (exp(min) - 1)
    auto expSub = rewriter.create<mlir::arith::SubFOp>(loc, expMin, one);
    auto scaled = rewriter.create<mlir::arith::MulFOp>(loc, alphaConst, expSub);

    return rewriter.create<mlir::arith::AddFOp>(loc, max, scaled);
}

// Gelu

template <>
mlir::Value emitLinalgRegion<IE::GeluOp>(IE::GeluOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto loc = op.getLoc();
    auto input = args[0];

    // Precomputed constants
    const double SQRT2_DIV_PI_VAL = 0.79788456080286535587989211986876f;
    const double MUL_FITTING_CONST_VAL = SQRT2_DIV_PI_VAL * 0.044715f;

    // Create constants with proper type
    auto sqrt2_div_pi =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), SQRT2_DIV_PI_VAL));
    auto mul_fitting_const = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getFloatAttr(input.getType(), MUL_FITTING_CONST_VAL));
    auto one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 1.0));
    auto half = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 0.5));

    // Use afn (approximate functions) flag for tanh
    auto fmFlags = mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), mlir::arith::FastMathFlags::afn);

    // Computes
    auto square = rewriter.create<mlir::arith::MulFOp>(loc, input, input);
    auto scaled_square = rewriter.create<mlir::arith::MulFOp>(loc, mul_fitting_const, square);
    auto scaled_base = rewriter.create<mlir::arith::AddFOp>(loc, sqrt2_div_pi, scaled_square);
    auto scaled_input = rewriter.create<mlir::arith::MulFOp>(loc, input, scaled_base);
    auto tanh_result = rewriter.create<mlir::math::TanhOp>(loc, scaled_input, fmFlags);
    auto tanh_plus_one = rewriter.create<mlir::arith::AddFOp>(loc, one, tanh_result);
    auto gelu_intermediate = rewriter.create<mlir::arith::MulFOp>(loc, input, tanh_plus_one);

    return rewriter.create<mlir::arith::MulFOp>(loc, gelu_intermediate, half);
}

// Selu

template <>
mlir::Value emitLinalgRegion<IE::SeluOp>(IE::SeluOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    auto loc = op.getLoc();

    // Get input value from args
    auto input = args[0];
    auto elemType = mlir::cast<mlir::FloatType>(resultTypes[0]);

    // Create constants
    auto zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 0.0));
    auto one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 1.0));

    // Get alpha and lambda from the operation's attributes
    auto alphaAttr = op.getAlphaValueAttr();
    auto lambdaAttr = op.getLambdaValueAttr();

    if (!alphaAttr || !lambdaAttr) {
        VPUX_THROW("SeluOp must have alphaValue and lambdaValue attributes");
    }

    // Extract double values from attributes
    double alphaVal = alphaAttr.getValue().convertToDouble();
    double lambdaVal = lambdaAttr.getValue().convertToDouble();

    auto alpha = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, alphaVal));
    auto lambda = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, lambdaVal));

    // Use afn (approximate functions) flag for exp
    auto fmFlags = mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), mlir::arith::FastMathFlags::afn);
    auto fmFlagsMinMax = mlir::arith::FastMathFlagsAttr::get(
            rewriter.getContext(), mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);

    // Compute min and max using vector operations
    auto min = rewriter.create<mlir::arith::MinimumFOp>(loc, input, zero, fmFlagsMinMax);
    auto max = rewriter.create<mlir::arith::MaximumFOp>(loc, input, zero, fmFlagsMinMax);

    // Compute exponential of min (negative values only)
    auto expMin = rewriter.create<mlir::math::ExpOp>(loc, min, fmFlags);

    // Compute SELU components
    auto exp_minus_one = rewriter.create<mlir::arith::SubFOp>(loc, expMin, one);
    auto alpha_multiply = rewriter.create<mlir::arith::MulFOp>(loc, alpha, exp_minus_one);
    auto sum_max = rewriter.create<mlir::arith::AddFOp>(loc, max, alpha_multiply);

    // Scale by lambda
    return rewriter.create<mlir::arith::MulFOp>(loc, lambda, sum_max);
}

// Prelu

template <>
mlir::Value emitLinalgRegion<IE::PReluOp>(IE::PReluOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                          mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto loc = op.getLoc();
    auto input = args[0];

    mlir::Type elemType = args[0].getType();
    auto zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 0.0));

    auto fmFlagsMinMax = mlir::arith::FastMathFlagsAttr::get(
            rewriter.getContext(), mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);

    // Computes PReLU activation: max(input, 0) + slope * min(input, 0)
    auto min = rewriter.create<mlir::arith::MinimumFOp>(loc, input, zero, fmFlagsMinMax);
    auto max = rewriter.create<mlir::arith::MaximumFOp>(loc, input, zero, fmFlagsMinMax);

    auto slopeMulMin = rewriter.create<mlir::arith::MulFOp>(loc, min, args[1]);
    return rewriter.create<mlir::arith::AddFOp>(loc, max, slopeMulMin);
}

// Add/Sub/Mul

template <typename SrcIOp, typename SrcFOp>
mlir::Value emitEltwiseWithPostOp(mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                  mlir::PatternRewriter& rewriter) {
    mlir::Value val = nullptr;
    if (mlir::isa<mlir::FloatType>(args[0].getType())) {
        val = rewriter.create<SrcFOp>(op->getLoc(), resultTypes, args);
    } else {
        val = rewriter.create<SrcIOp>(op->getLoc(), resultTypes, args);
    }
    if (auto postOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op)) {
        auto clampAttr = postOp.getClampAttr();
        auto postOpAttr = postOp.getPostOp();

        if (clampAttr != nullptr ||
            (postOpAttr != nullptr && !mlir::isa<IE::ReluAttr, IE::LeakyReluAttr>(postOpAttr))) {
            VPUX_THROW("Unsupported post-processing for operation '{0}' at '{1}'", op->getName(), op->getLoc());
        }

        if (postOpAttr != nullptr) {
            llvm::TypeSwitch<mlir::Attribute, void>(postOpAttr)
                    .Case<IE::ReluAttr>([&](auto attr) {
                        VPUX_UNUSED(attr);
                        val = emitLeakyReLU(val, 0., rewriter);
                    })
                    .template Case<IE::LeakyReluAttr>([&](auto attr) {
                        VPUX_UNUSED(attr);
                        val = emitLeakyReLU(val, attr.getNegativeSlope().getValueAsDouble(), rewriter);
                    });
        }
    }
    return val;
}

template <>
mlir::Value emitLinalgRegion<IE::MultiplyOp>(IE::MultiplyOp op, mlir::ValueRange args,
                                             llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    assert(op.getClampAttr() == nullptr && "Unexpected clamp attribute");
    return emitEltwiseWithPostOp<mlir::arith::MulIOp, mlir::arith::MulFOp>(op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion<IE::SubtractOp>(IE::SubtractOp op, mlir::ValueRange args,
                                             llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    assert(op.getClampAttr() == nullptr && "Unexpected clamp attribute");
    return emitEltwiseWithPostOp<mlir::arith::SubIOp, mlir::arith::SubFOp>(op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion<IE::AddOp>(IE::AddOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    assert(op.getClampAttr() == nullptr && "Unexpected clamp attribute");
    return emitEltwiseWithPostOp<mlir::arith::AddIOp, mlir::arith::AddFOp>(op, args, resultTypes, rewriter);
}

// Divide layers

template <>
mlir::Value emitLinalgRegion<IE::DivideOp>(IE::DivideOp op, mlir::ValueRange args,
                                           llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    auto elTy = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    auto loc = op->getLoc();
    if (mlir::isa<mlir::FloatType>(elTy)) {
        // Add the arcp fastmath flag to allow converting to a multiplication
        // with the reciprocal. This matters for broadcasting (allows converting
        // the division to a multiplication when doing an inner broadcast).
        // SW layers also make (explicit) use of this.
        auto attr = mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), mlir::arith::FastMathFlags::arcp);
        return rewriter.create<mlir::arith::DivFOp>(loc, resultTypes, args[0], args[1], attr);
    }
    if (elTy.isSignedInteger()) {
        return rewriter.create<mlir::arith::DivSIOp>(loc, resultTypes, args);
    }
    return rewriter.create<mlir::arith::DivUIOp>(loc, resultTypes, args);
}

// Min/Max layers
template <typename FPOp, typename SIOp, typename UIOp>
mlir::Value emitMinOrMax(mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                         mlir::PatternRewriter& rewriter) {
    auto elTy = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    auto loc = op->getLoc();
    if (mlir::isa<mlir::FloatType>(elTy)) {
        // The sw layer implementation doesn't handle NaNs or signed zeroes,
        // so it should be okay to have nnan/nsz.
        auto attr = mlir::arith::FastMathFlagsAttr::get(
                rewriter.getContext(), mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);
        return rewriter.create<FPOp>(loc, resultTypes, args[0], args[1], attr);
    }
    if (elTy.isSignedInteger()) {
        return rewriter.create<SIOp>(loc, resultTypes, args);
    }
    return rewriter.create<UIOp>(loc, resultTypes, args);
}

mlir::Value emitMin(mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                    mlir::PatternRewriter& rewriter) {
    return emitMinOrMax<mlir::arith::MinimumFOp, mlir::arith::MinSIOp, mlir::arith::MinUIOp>(op, args, resultTypes,
                                                                                             rewriter);
}

mlir::Value emitMax(mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                    mlir::PatternRewriter& rewriter) {
    return emitMinOrMax<mlir::arith::MaximumFOp, mlir::arith::MaxSIOp, mlir::arith::MaxUIOp>(op, args, resultTypes,
                                                                                             rewriter);
}

template <>
mlir::Value emitLinalgRegion<IE::MaximumOp>(IE::MaximumOp op, mlir::ValueRange args,
                                            llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    return emitMax(op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion<IE::MinimumOp>(IE::MinimumOp op, mlir::ValueRange args,
                                            llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    return emitMin(op, args, resultTypes, rewriter);
}

// Erf/Round/Sin/Cos layers

template <typename OpT>
mlir::Value emitF16ViaF32ThenTrunc(mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                   mlir::PatternRewriter& rewriter) {
    auto elTy = mlir::cast<NDTypeInterface>(op->getOperand(0).getType()).getElementType();

    // In case the type is f16 then we need to do a conversion to f32 and convert the result back to f16
    if (mlir::isa<mlir::Float16Type>(elTy)) {
        // f16 -> fp32
        auto f32Type = mlir::Float32Type::get(rewriter.getContext());
        auto extArg = rewriter.create<mlir::arith::ExtFOp>(args[0].getLoc(), f32Type, args[0]);

        // Creating the operation in fp32
        auto genOp = rewriter.create<OpT>(op->getLoc(), f32Type, extArg);
        auto f16Type = mlir::Float16Type::get(rewriter.getContext());
        return rewriter.create<mlir::arith::TruncFOp>(op->getLoc(), f16Type, genOp);
    }

    // Non-F16: direct call operation
    return rewriter.create<OpT>(op->getLoc(), resultTypes, args[0]);
}

// Erf layer
template <>
mlir::Value emitLinalgRegion<IE::ErfOp>(IE::ErfOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    return emitF16ViaF32ThenTrunc<mlir::math::ErfOp>(op, args, resultTypes, rewriter);
}

// Round layer
template <>
mlir::Value emitLinalgRegion<IE::RoundOp>(IE::RoundOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                          mlir::PatternRewriter& rewriter) {
    auto loc = op->getLoc();
    auto modeAttr = op.getMode();

    if (modeAttr == IE::RoundMode::HALF_TO_EVEN) {
        return rewriter.create<mlir::math::RoundEvenOp>(loc, resultTypes, args[0]);
    }
    return rewriter.create<mlir::math::RoundOp>(loc, resultTypes, args[0]);
}

// Sin layer
template <>
mlir::Value emitLinalgRegion<IE::SinOp>(IE::SinOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    return emitF16ViaF32ThenTrunc<mlir::math::SinOp>(op, args, resultTypes, rewriter);
}

// Cos layer
template <>
mlir::Value emitLinalgRegion<IE::CosOp>(IE::CosOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    return emitF16ViaF32ThenTrunc<mlir::math::CosOp>(op, args, resultTypes, rewriter);
}

// Cosh layer   cosh(x)=0.5×(exp(X)+exp(-X))

template <>
mlir::Value emitLinalgRegion<IE::CoshOp>(IE::CoshOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto loc = op->getLoc();
    auto input = args.front();
    auto zeroConst =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 0.0));  // Const 0
    auto halfConst =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 0.5));  // Const 0.5
    auto xExp = rewriter.create<mlir::math::ExpOp>(loc, input, mlir::arith::FastMathFlags::afn);         // exp(X)
    auto negX = rewriter.create<mlir::arith::SubFOp>(loc, zeroConst, input);                             // -x = 0-x
    auto expNegX = rewriter.create<mlir::math::ExpOp>(loc, negX, mlir::arith::FastMathFlags::afn);       // exp(-x)
    auto expSum = rewriter.create<mlir::arith::AddFOp>(loc, xExp, expNegX);  // exp(x) + exp(-x)

    return rewriter.create<mlir::arith::MulFOp>(loc, expSum, halfConst);
}

// tan(x) = sin(x) / cos(x)

template <>
mlir::Value emitLinalgRegion<IE::TanOp>(IE::TanOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto loc = op->getLoc();
    auto sinX = emitF16ViaF32ThenTrunc<mlir::math::SinOp>(op, args, resultTypes, rewriter);
    auto cosX = emitF16ViaF32ThenTrunc<mlir::math::CosOp>(op, args, resultTypes, rewriter);

    return rewriter.create<mlir::arith::DivFOp>(loc, sinX, cosX);
}

// atanh(x) = 0.5 * log((1 + x) / (1 - x))

template <>
mlir::Value emitLinalgRegion<IE::AtanhOp>(IE::AtanhOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                          mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto loc = op->getLoc();
    auto input = args.front();

    auto oneConst =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 1.0));  // Const 1
    auto halfConst =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 0.5));  // Const 0.5

    auto onePlusX = rewriter.create<mlir::arith::AddFOp>(loc, oneConst, input);   // 1 + x
    auto oneMinusX = rewriter.create<mlir::arith::SubFOp>(loc, oneConst, input);  // 1 - x

    auto divResult = rewriter.create<mlir::arith::DivFOp>(loc, onePlusX, oneMinusX);  // (1 + x) / (1 - x)
    auto logResult = rewriter.create<mlir::math::LogOp>(loc, divResult,
                                                        mlir::arith::FastMathFlags::afn);  // log((1 + x) / (1 - x))

    return rewriter.create<mlir::arith::MulFOp>(loc, logResult, halfConst);
}

// Sinh layer   sinh(x) = (exp(x) - exp(-x)) / 2

template <>
mlir::Value emitLinalgRegion<IE::SinhOp>(IE::SinhOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto loc = op->getLoc();
    auto input = args.front();
    auto zeroConst =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 0.0));  // Const 0
    auto halfConst =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 0.5));  // Const 0.5
    auto expX = rewriter.create<mlir::math::ExpOp>(loc, input, mlir::arith::FastMathFlags::afn);         // exp(x)
    auto negX = rewriter.create<mlir::arith::SubFOp>(loc, zeroConst, input);                             // -x = 0-x
    auto expNegX = rewriter.create<mlir::math::ExpOp>(loc, negX, mlir::arith::FastMathFlags::afn);       // exp (-x)
    auto diffExp = rewriter.create<mlir::arith::SubFOp>(loc, expX, expNegX);  // exp(x) - exp(-x)

    return rewriter.create<mlir::arith::MulFOp>(loc, diffExp, halfConst);
}

// Negative layer
template <>
mlir::Value emitLinalgRegion<IE::NegativeOp>(IE::NegativeOp op, mlir::ValueRange args,
                                             llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    auto loc = op.getLoc();

    if (mlir::isa<mlir::FloatType>(args[0].getType())) {
        auto zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(args[0].getType(), 0.));
        return rewriter.create<mlir::arith::SubFOp>(loc, resultTypes, mlir::ValueRange{zero, args[0]});
    }

    auto zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(args[0].getType(), 0));
    return rewriter.create<mlir::arith::SubIOp>(loc, resultTypes, mlir::ValueRange{zero, args[0]});
}

// Callback type for emitting the linalg body for an operation.
using EmitBodyCallback = std::function<mlir::Value(mlir::Operation*, mlir::ValueRange, llvm::ArrayRef<mlir::Type>,
                                                   mlir::PatternRewriter&)>;

enum class BroadcastType { NONE, NUMPY, PER_CHANNEL, UNSUPPORTED };

// Helper function to determine broadcast type for an operand
static BroadcastType getBroadcastType(mlir::Operation* op, mlir::Value operand, unsigned operandIndex) {
    // Check for PRelu-specific per-channel broadcast
    if (auto preluOp = mlir::dyn_cast<IE::PReluOp>(op)) {
        if (operandIndex == 1) {  // Slope operand
            auto inputType = mlir::cast<vpux::NDTypeInterface>(preluOp.getInput().getType());
            auto inputShape = inputType.getShape();

            // Get slope type and shape from the operand
            auto slopeType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
            auto slopeShape = slopeType.getShape();
            auto slopeRank = slopeType.getRank();

            // Per-channel broadcast applies only if slope is 1D and matches the channel dimension
            // The channel dimension is at index 1 in NCHW layout (the second dimension)
            if (slopeRank == 1 && inputShape.size() >= 2) {
                // Check if slope dimension matches the input's channel dimension
                if (slopeShape[Dim(0)] == inputShape[Dim(1)]) {
                    return BroadcastType::PER_CHANNEL;
                }
            }
            return BroadcastType::NUMPY;
        }
    }

    // Check for general broadcast attributes
    if (op->getOperands().size() > 1) {
        if (auto broadcastAttr = op->getAttrOfType<IE::AutoBroadcastTypeAttr>("auto_broadcast")) {
            switch (broadcastAttr.getValue()) {
            case IE::AutoBroadcastType::NUMPY:
                return BroadcastType::NUMPY;
            case IE::AutoBroadcastType::NONE_OR_EXPLICIT:
                return BroadcastType::NONE;
            default:
                return BroadcastType::UNSUPPORTED;
            }
        }
    }

    return BroadcastType::NONE;
}

static mlir::LogicalResult emitLinalgEltwiseHelper(mlir::Operation* op, EmitBodyCallback callback,
                                                   mlir::ValueRange convertedArgs, mlir::PatternRewriter& rewriter) {
    auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResultTypes().front());
    auto outputShape = mlir::cast<vpux::NDTypeInterface>(op->getResultTypes().front()).getShape();
    auto linalgResultElTy = ShaveCodeGen::getLinalgElementType(resultType, rewriter.getContext());

    // Reject the dynamic output type, at least for now.
    if (outputShape.isDynamic()) {
        return mlir::failure();
    }

    // Create the linalg affine maps. This will handle broadcasting (operand dimension size
    // is equal to 1 and is different than the result dimension) by using a 0 affine constant
    // in the affine map.
    auto rank = resultType.getRank();
    auto inverseOutputMap = mlir::inversePermutation(mlir::cast<vpux::NDTypeInterface>(op->getResultTypes().front())
                                                             .getDimsOrder()
                                                             .toAffineMap(rewriter.getContext()));

    // Helper function to create logical map for an operand
    auto createLogicalMap = [&](mlir::Value operand, unsigned operandIndex) -> mlir::AffineMap {
        auto operandRank = mlir::cast<mlir::RankedTensorType>(operand.getType()).getRank();
        auto operandShape = mlir::cast<vpux::NDTypeInterface>(operand.getType()).getShape();

        // Handle scalar operands
        if (operandRank == 0) {
            return mlir::AffineMap::get(rank, 0, {}, rewriter.getContext());
        }

        SmallVector<mlir::AffineExpr> affineExprs;
        BroadcastType broadcastType = getBroadcastType(op, operand, operandIndex);

        // Check for unsupported broadcast type
        if (broadcastType == BroadcastType::UNSUPPORTED) {
            return mlir::AffineMap();  // Return invalid map to indicate failure
        }

        // Handle per-channel broadcast
        if (broadcastType == BroadcastType::PER_CHANNEL) {
            // For per-channel broadcast, map to channel dimension (index 1 in logical NCHW order)
            affineExprs.push_back(rewriter.getAffineDimExpr(1));
            return mlir::AffineMap::get(rank, 0, affineExprs, rewriter.getContext());
        }

        // Default handling for all other operands and operations
        // The output rank should always be larger or equal than the operand rank
        // due to shape inference rules.
        assert(rank >= operandRank && "Unexpected input rank");
        for (auto it : llvm::enumerate(operandShape)) {
            // Match the input dimension to the output dimension index according
            // to numpy broadcasting rules. Note if the input and output shapes
            // have different ranks then the lower ranked shape (the input one)
            // is right aligned and filled with ones to the left to equalize
            // the ranks.
            auto outIdx = rank - operandRank + it.index();
            auto dimValue = it.value();
            auto outDim = outputShape.raw()[outIdx];
            // If the input dimension is equal to one and the output dimension is
            // not one then we are broadcasting.
            bool broadcastDim = false;
            if (broadcastType == BroadcastType::NUMPY) {
                broadcastDim = (dimValue == 1 && outDim != dimValue);
            }
            // Broadcasting across this dimension is equivalent to having a constant
            // zero expression in the affine map.
            auto affineExpr = broadcastDim ? rewriter.getAffineConstantExpr(0) : rewriter.getAffineDimExpr(outIdx);
            affineExprs.push_back(affineExpr);
        }

        return mlir::AffineMap::get(rank, 0, affineExprs, rewriter.getContext());
    };

    // Create affine maps for all operands
    SmallVector<mlir::AffineMap> affineMaps;
    for (auto operand : llvm::enumerate(op->getOperands())) {
        auto logicalMap = createLogicalMap(operand.value(), operand.index());

        // Check if we got an invalid map (unsupported broadcast)
        if (!logicalMap) {
            return mlir::failure();
        }
        // Compose affine maps to get the correct indexing for this operand
        // considering that the output tensor will have identity indexing.
        auto opMap = mlir::cast<vpux::NDTypeInterface>(operand.value().getType())
                             .getDimsOrder()
                             .toAffineMap(rewriter.getContext());
        auto finalMap = opMap.compose(logicalMap).compose(inverseOutputMap);
        affineMaps.push_back(finalMap);
    }

    // Add the affine map for the output tensor as well.
    affineMaps.push_back(rewriter.getMultiDimIdentityMap(rank));

    // Compute the output tensor shape with an identity layout which has a memory layout that matches our
    // original output tensor.
    auto ndResultTy = mlir::cast<NDTypeInterface>(resultType);
    auto dOrder = DimsOrder::fromPermutation(ndResultTy.getDimsOrder().toPermutation());
    auto dstShape = dOrder.toMemoryOrder(ndResultTy.getShape()).raw();
    mlir::Value outputTensor = rewriter.create<mlir::tensor::EmptyOp>(op->getLoc(), dstShape, linalgResultElTy);

    llvm::SmallVector<mlir::utils::IteratorType> loopAttrs(rank, mlir::utils::IteratorType::parallel);
    auto linalgOp = rewriter.create<mlir::linalg::GenericOp>(
            op->getLoc(), outputTensor.getType(), convertedArgs, outputTensor, affineMaps, loopAttrs,
            [&](mlir::OpBuilder& opBuilder, mlir::Location loc, mlir::ValueRange blockArgs) {
                mlir::Value opResult =
                        callback(op, blockArgs.take_front(op->getNumOperands()), {linalgResultElTy}, rewriter);
                opBuilder.create<mlir::linalg::YieldOp>(loc, opResult);
            });
    rewriter.replaceOp(op, linalgOp);

    return mlir::success();
}

template <typename SrcOp>
class IEEltwiseToLinalg : public mlir::OpConversionPattern<SrcOp> {
public:
    using mlir::OpConversionPattern<SrcOp>::OpConversionPattern;
    using OpAdaptor = typename mlir::OpConversionPattern<SrcOp>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        mlir::ValueRange convertedArgs = adaptor.getOperands();
        auto emitBody = [](mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> types,
                           mlir::PatternRewriter& rewriter) {
            return emitLinalgRegion<SrcOp>(mlir::cast<SrcOp>(op), args, types, rewriter);
        };
        return emitLinalgEltwiseHelper(op, emitBody, convertedArgs, rewriter);
    }
};

template <typename SrcOp>
class IEQuantizeDequantizeToQuant : public mlir::OpConversionPattern<SrcOp> {
public:
    using mlir::OpConversionPattern<SrcOp>::OpConversionPattern;
    using OpAdaptor = typename mlir::OpConversionPattern<SrcOp>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto quantType = ShaveCodeGen::normalizeType(op.getType());

        auto loc = op.getLoc();
        mlir::Operation* newOp = nullptr;

        if (mlir::isa<IE::QuantizeOp>(op)) {
            newOp = rewriter.create<mlir::quant::QuantizeCastOp>(loc, quantType, adaptor.getOperands()[0]);
        } else if (mlir::isa<IE::DequantizeOp>(op)) {
            newOp = rewriter.create<mlir::quant::DequantizeCastOp>(loc, quantType, adaptor.getOperands()[0]);
        } else {
            return mlir::failure();
        }

        rewriter.replaceOp(op, newOp->getResult(0));

        return mlir::success();
    }
};

// Convert layers

template <>
mlir::Value emitLinalgRegion<IE::ConvertOp>(IE::ConvertOp op, mlir::ValueRange args,
                                            llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    auto inputTy = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType()).getElementType();
    auto resultType = mlir::cast<mlir::RankedTensorType>(op.getOutput().getType());
    auto outputTy = mlir::cast<vpux::NDTypeInterface>(resultType).getElementType();
    auto loc = op.getLoc();

    // Float to Int
    if (mlir::isa<mlir::FloatType>(inputTy) && mlir::isa<mlir::IntegerType>(outputTy)) {
        if (outputTy.isSignedInteger()) {
            return rewriter.create<mlir::arith::FPToSIOp>(loc, resultTypes, args);
        }

        return rewriter.create<mlir::arith::FPToUIOp>(loc, resultTypes, args);
    }

    // Int to Float
    if (mlir::isa<mlir::IntegerType>(inputTy) && mlir::isa<mlir::FloatType>(outputTy)) {
        if (inputTy.isSignedInteger()) {
            return rewriter.create<mlir::arith::SIToFPOp>(loc, resultTypes, args);
        }

        return rewriter.create<mlir::arith::UIToFPOp>(loc, resultTypes, args);
    }

    // Float to Float
    if (mlir::isa<mlir::FloatType>(inputTy) && mlir::isa<mlir::FloatType>(outputTy)) {
        if (inputTy == outputTy) {
            return args[0];
        }

        if (outputTy.getIntOrFloatBitWidth() > inputTy.getIntOrFloatBitWidth()) {
            return rewriter.create<mlir::arith::ExtFOp>(loc, resultTypes, args);
        }

        return rewriter.create<mlir::arith::TruncFOp>(loc, resultTypes, args);
    }

    // Int to Int
    if (mlir::isa<mlir::IntegerType>(inputTy) && mlir::isa<mlir::IntegerType>(outputTy)) {
        if (outputTy.getIntOrFloatBitWidth() == inputTy.getIntOrFloatBitWidth()) {
            return args[0];
        }

        if (outputTy.getIntOrFloatBitWidth() > inputTy.getIntOrFloatBitWidth()) {
            if (inputTy.isSignedInteger()) {
                return rewriter.create<mlir::arith::ExtSIOp>(loc, resultTypes, args);
            }

            return rewriter.create<mlir::arith::ExtUIOp>(loc, resultTypes, args);
        }

        if (outputTy.getIntOrFloatBitWidth() < inputTy.getIntOrFloatBitWidth()) {
            return rewriter.create<mlir::arith::TruncIOp>(loc, resultTypes, args);
        }
    }

    VPUX_THROW("Unsupported convert type combination");
}

// Abs layer
template <>
mlir::Value emitLinalgRegion<IE::AbsOp>(IE::AbsOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    auto loc = op.getLoc();
    const auto elemType = args[0].getType();

    if (mlir::isa<mlir::FloatType>(elemType)) {
        return rewriter.create<mlir::math::AbsFOp>(loc, resultTypes, args);
    }

    if (mlir::isa<mlir::IntegerType>(elemType)) {
        return rewriter.create<mlir::math::AbsIOp>(loc, resultTypes, args);
    }

    VPUX_THROW("Unsupported element type '{0}' for IE::AbsOp", elemType);
}

// Sign layer
template <>
mlir::Value emitLinalgRegion<IE::SignOp>(IE::SignOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    // This algorithm computes the sign of a floating-point value by examining its sign bit.
    // To handle the case where there is -0, we check if the value is exactly zero
    // (including both +0 and -0) and return 0 in that case.
    VPUX_UNUSED(resultTypes);
    auto loc = op.getLoc();
    auto val = args[0];

    auto zeroAttr = rewriter.getFloatAttr(val.getType(), 0.0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), zeroAttr);
    auto negAttr = rewriter.getFloatAttr(val.getType(), -1.0);
    mlir::Value negOne = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), negAttr);
    auto posAttr = rewriter.getFloatAttr(val.getType(), 1.0);
    mlir::Value posOne = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), posAttr);

    auto bitWidth = val.getType().getIntOrFloatBitWidth();
    auto intTy = mlir::IntegerType::get(val.getContext(), bitWidth);
    auto casted = rewriter.create<mlir::arith::BitcastOp>(loc, intTy, val);
    llvm::APInt msbMaskVal = llvm::APInt::getSignMask(bitWidth);
    auto msbMask = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(intTy, msbMaskVal));
    auto signBit = rewriter.create<mlir::arith::AndIOp>(loc, casted, msbMask);

    auto shift = rewriter.create<mlir::arith::ShLIOp>(
            loc, casted, rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, intTy.getWidth()));

    auto isZero =
            rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, shift,
                                                 rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, intTy.getWidth()));

    auto signHandled = rewriter.create<mlir::arith::SelectOp>(
            loc,
            rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, signBit,
                                                 rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, intTy.getWidth())),
            negOne, posOne);

    return rewriter.create<mlir::arith::SelectOp>(loc, isZero, zero, signHandled);
}

// HSwish layer
template <>
mlir::Value emitLinalgRegion<IE::HSwishOp>(IE::HSwishOp op, mlir::ValueRange args,
                                           llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    // Compute this as x*((min(max(x+3,0),6))/6)
    VPUX_UNUSED(resultTypes);
    auto loc = op.getLoc();
    auto val = args[0];
    auto zeroAttr = rewriter.getFloatAttr(val.getType(), 0.0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), zeroAttr);
    auto threeAttr = rewriter.getFloatAttr(val.getType(), 3.0);
    mlir::Value three = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), threeAttr);
    auto sixAttr = rewriter.getFloatAttr(val.getType(), 6.0);
    mlir::Value six = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), sixAttr);
    auto divSixAttr = rewriter.getFloatAttr(val.getType(), 1.0 / 6.0);
    mlir::Value divSix = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), divSixAttr);
    auto fmFlags = mlir::arith::FastMathFlagsAttr::get(
            rewriter.getContext(), mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);

    auto add = rewriter.create<mlir::arith::AddFOp>(loc, val, three);
    auto max = rewriter.create<mlir::arith::MaximumFOp>(loc, add, zero, fmFlags);
    auto min = rewriter.create<mlir::arith::MinimumFOp>(loc, max, six, fmFlags);
    auto mul = rewriter.create<mlir::arith::MulFOp>(loc, min, divSix);

    return rewriter.create<mlir::arith::MulFOp>(loc, val, mul);
}

// HSigmoid Layer
template <>
mlir::Value emitLinalgRegion<IE::HSigmoidOp>(IE::HSigmoidOp op, mlir::ValueRange args,
                                             llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    // Compute this as (min(max(x+3,0),6))/6
    VPUX_UNUSED(resultTypes);
    auto loc = op.getLoc();
    auto val = args[0];
    auto zeroAttr = rewriter.getFloatAttr(val.getType(), 0.0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), zeroAttr);
    auto threeAttr = rewriter.getFloatAttr(val.getType(), 3.0);
    mlir::Value three = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), threeAttr);
    auto sixAttr = rewriter.getFloatAttr(val.getType(), 6.0);
    mlir::Value six = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), sixAttr);
    auto divSixAttr = rewriter.getFloatAttr(val.getType(), 1.0 / 6.0);
    mlir::Value divSix = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), divSixAttr);
    auto fmFlags = mlir::arith::FastMathFlagsAttr::get(
            rewriter.getContext(), mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);

    auto add = rewriter.create<mlir::arith::AddFOp>(loc, val, three);
    auto max = rewriter.create<mlir::arith::MaximumFOp>(loc, add, zero, fmFlags);
    auto min = rewriter.create<mlir::arith::MinimumFOp>(loc, max, six, fmFlags);

    return rewriter.create<mlir::arith::MulFOp>(loc, min, divSix);
}

// SoftPlus

static mlir::Value createSoftPlusComputation(mlir::Location loc, mlir::Value input, mlir::PatternRewriter& rewriter,
                                             bool isFP16Conversion) {
    // Threshold values from OpenVINO spec
    double threshold_val = isFP16Conversion ? 11.0f : 20.0f;

    auto inputType = input.getType();
    auto threshold = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(inputType, threshold_val));
    auto one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(inputType, 1.0));

    // Use fast math flags to enable backend optimizations
    auto fmFlags = mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), mlir::arith::FastMathFlags::afn |
                                                                                      mlir::arith::FastMathFlags::nnan |
                                                                                      mlir::arith::FastMathFlags::ninf);

    // Compute SoftPlus: log(1 + exp(input)) for values below threshold
    auto exp_result = rewriter.create<mlir::math::ExpOp>(loc, input, fmFlags);
    auto one_plus_exp = rewriter.create<mlir::arith::AddFOp>(loc, one, exp_result);
    auto softplus_result = rewriter.create<mlir::math::LogOp>(loc, one_plus_exp, fmFlags);

    // Use linear function for values above threshold (x >= threshold)
    auto is_below_threshold =
            rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, input, threshold);

    return rewriter.create<mlir::arith::SelectOp>(loc, is_below_threshold, softplus_result, input);
}

template <>
mlir::Value emitLinalgRegion<IE::SoftPlusOp>(IE::SoftPlusOp op, mlir::ValueRange args,
                                             llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto loc = op.getLoc();
    auto input = args[0];
    auto elTy = mlir::cast<NDTypeInterface>(op->getOperand(0).getType()).getElementType();

    // For FP16, use FP16->FP32->FP16 pattern
    if (mlir::isa<mlir::Float16Type>(elTy)) {
        auto f32Type = mlir::Float32Type::get(rewriter.getContext());
        auto extArg = rewriter.create<mlir::arith::ExtFOp>(loc, f32Type, input);

        // Create SoftPlus operation in fp32
        auto softplusF32 = createSoftPlusComputation(loc, extArg, rewriter, true);

        // Convert back to f16
        auto f16Type = mlir::Float16Type::get(rewriter.getContext());
        return rewriter.create<mlir::arith::TruncFOp>(loc, f16Type, softplusF32);
    }

    // Non-FP16: direct computation
    return createSoftPlusComputation(loc, input, rewriter, false);
}

// Mish

template <>
mlir::Value emitLinalgRegion<IE::MishOp>(IE::MishOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto loc = op.getLoc();
    auto input = args[0];

    // Create constants with proper type
    auto one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 1.0));

    // Afn (approximate functions) flag for exponential and tanh operations
    auto fmFlags = mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), mlir::arith::FastMathFlags::afn);

    // Mish(x) = x * tanh(ln(1 + e^x)), steps:
    // Compute e^x
    auto exp_x = rewriter.create<mlir::math::ExpOp>(loc, input, fmFlags);

    // Compute 1 + e^x
    auto one_plus_exp = rewriter.create<mlir::arith::AddFOp>(loc, one, exp_x);

    // Compute ln(1 + e^x)
    auto softplus = rewriter.create<mlir::math::LogOp>(loc, one_plus_exp, fmFlags);

    // Compute tanh(softplus)
    auto tanh_softplus = rewriter.create<mlir::math::TanhOp>(loc, softplus, fmFlags);

    // Compute x * tanh(softplus)
    return rewriter.create<mlir::arith::MulFOp>(loc, input, tanh_softplus);
}

enum class RoundMode { FLOOR, CEIL };

mlir::Value emitRoundRegion(mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                            mlir::PatternRewriter& rewriter, RoundMode mode) {
    VPUX_UNUSED(resultTypes);
    auto loc = op->getLoc();
    auto input = args[0];
    auto inputType = mlir::cast<mlir::FloatType>(input.getType());

    const int FP16_FRACTBITS = 10;
    const int FP16_BIAS = 15;
    const int FP16_TOTALBITS = 16;
    const int FP16_GREATINT = (FP16_BIAS + FP16_FRACTBITS) << FP16_FRACTBITS;  // 0x6400

    auto i16Type = rewriter.getIntegerType(16);

    auto signMask = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(i16Type, 1u << (FP16_TOTALBITS - 1)));                           // 0x8000
    auto ones = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(i16Type, 0x3C00));  // FP16(1.0)
    auto notSignMask = rewriter.create<mlir::arith::XOrIOp>(
            loc, signMask, rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(i16Type, -1)));

    auto allOnes = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(i16Type, 0xFFFF));
    auto zeroI16 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(i16Type, 0x0000));
    auto oneI16 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(i16Type, 0x0001));
    auto shiftSign =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(i16Type, FP16_TOTALBITS - 1));
    auto shiftFractBits =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(i16Type, FP16_FRACTBITS));
    auto biasVal = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(i16Type, FP16_BIAS));
    auto greatIntMinusOne = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(i16Type, (FP16_GREATINT - 1) & 0xFFFF));
    auto truncFractConst =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(i16Type, 0xFC00));  // (~0x03FF)

    auto xInt = rewriter.create<mlir::arith::BitcastOp>(loc, i16Type, input);
    // xSign = xInt & signMask
    auto xSign = rewriter.create<mlir::arith::AndIOp>(loc, xInt, signMask);
    // xAbs = xInt & ~signMask
    auto xAbs = rewriter.create<mlir::arith::AndIOp>(loc, xInt, notSignMask);
    // xExpo = (xAbs >> 10) - 15
    auto shifted = rewriter.create<mlir::arith::ShRSIOp>(loc, xAbs, shiftFractBits);
    auto xExpo = rewriter.create<mlir::arith::SubIOp>(loc, shifted, biasVal);
    // truncMask = (0xFC00 >> xExpo)
    auto truncMask = rewriter.create<mlir::arith::ShRSIOp>(loc, truncFractConst, xExpo);

    // isSign = xInt >> 15
    auto isSign = rewriter.create<mlir::arith::ShRSIOp>(loc, xInt, shiftSign);

    // isZero = (xAbs - 1) >> 15
    auto xAbsMinusOne = rewriter.create<mlir::arith::SubIOp>(loc, xAbs, oneI16);
    auto isZero = rewriter.create<mlir::arith::ShRSIOp>(loc, xAbsMinusOne, shiftSign);

    // isSmall = ((xAbs - ones) >> 15) & ~isZero
    auto xAbsMinusOnes = rewriter.create<mlir::arith::SubIOp>(loc, xAbs, ones);
    auto isSmallShifted = rewriter.create<mlir::arith::ShRSIOp>(loc, xAbsMinusOnes, shiftSign);
    auto notIsZero = rewriter.create<mlir::arith::XOrIOp>(loc, isZero, allOnes);
    auto isSmall = rewriter.create<mlir::arith::AndIOp>(loc, isSmallShifted, notIsZero);

    // isGreat = ((0x63FF - xAbs) >> 15)
    auto diff = rewriter.create<mlir::arith::SubIOp>(loc, greatIntMinusOne, xAbs);
    auto isGreat = rewriter.create<mlir::arith::ShRSIOp>(loc, diff, shiftSign);

    // isExact = isGreat | isZero
    auto isExact = rewriter.create<mlir::arith::OrIOp>(loc, isGreat, isZero);

    // isRoundMode (isFloor or isCeil)
    auto isExactOrSmall = rewriter.create<mlir::arith::OrIOp>(loc, isExact, isSmall);
    auto isRoundMode = rewriter.create<mlir::arith::XOrIOp>(loc, isExactOrSmall, allOnes);

    // xTrunc = (xAbs & truncMask)
    auto xTrunc = rewriter.create<mlir::arith::AndIOp>(loc, xAbs, truncMask);

    // isInexact = mask(xTrunc != xAbs)
    auto isEqual = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, xTrunc, xAbs);
    auto isInexact = rewriter.create<mlir::arith::SelectOp>(loc, isEqual, zeroI16, allOnes);

    mlir::Value adjustment;
    mlir::Value xSmall;

    if (mode == RoundMode::FLOOR) {
        // Floor: decrement for negative and inexact numbers
        auto adjustmentCond = rewriter.create<mlir::arith::AndIOp>(loc, isSign, isInexact);
        adjustment = rewriter.create<mlir::arith::AndIOp>(loc, adjustmentCond, ones);

        // xSmall for floor: -1.0 for negative and small numbers
        auto xSmallCond = rewriter.create<mlir::arith::AndIOp>(loc, isSign, isSmall);
        xSmall = rewriter.create<mlir::arith::AndIOp>(loc, xSmallCond, ones);
    } else {
        // Ceil: increment for positive and inexact numbers
        auto notIsSign = rewriter.create<mlir::arith::XOrIOp>(loc, isSign, allOnes);
        auto adjustmentCond = rewriter.create<mlir::arith::AndIOp>(loc, notIsSign, isInexact);
        adjustment = rewriter.create<mlir::arith::AndIOp>(loc, adjustmentCond, ones);

        // xSmall for ceil: +1.0 for positive and small numbers
        auto xSmallCond = rewriter.create<mlir::arith::AndIOp>(loc, notIsSign, isSmall);
        xSmall = rewriter.create<mlir::arith::AndIOp>(loc, xSmallCond, ones);
    }

    // xRounded = xTrunc + adjustment (FP16)
    auto xTruncFloat = rewriter.create<mlir::arith::BitcastOp>(loc, inputType, xTrunc);
    auto adjustmentFloat = rewriter.create<mlir::arith::BitcastOp>(loc, inputType, adjustment);
    auto xRoundedFloat = rewriter.create<mlir::arith::AddFOp>(loc, xTruncFloat, adjustmentFloat);
    auto xRounded = rewriter.create<mlir::arith::BitcastOp>(loc, i16Type, xRoundedFloat);

    // Final result
    auto exactPart = rewriter.create<mlir::arith::AndIOp>(loc, isExact, xInt);
    auto roundPart = rewriter.create<mlir::arith::AndIOp>(loc, isRoundMode, xRounded);
    auto smallPart = rewriter.create<mlir::arith::AndIOp>(loc, isSmall, xSmall);

    auto roundOrSmall = rewriter.create<mlir::arith::OrIOp>(loc, roundPart, smallPart);
    auto signOrRoundSmall = rewriter.create<mlir::arith::OrIOp>(loc, xSign, roundOrSmall);
    auto finalInt = rewriter.create<mlir::arith::OrIOp>(loc, exactPart, signOrRoundSmall);

    return rewriter.create<mlir::arith::BitcastOp>(loc, inputType, finalInt);
}

template <>
mlir::Value emitLinalgRegion<IE::FloorOp>(IE::FloorOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                          mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto loc = op.getLoc();
    auto input = args[0];
    auto elTy = mlir::cast<NDTypeInterface>(op->getOperand(0).getType()).getElementType();

    if (mlir::isa<mlir::Float16Type>(elTy)) {
        return emitRoundRegion(op, args, resultTypes, rewriter, RoundMode::FLOOR);
    }
    // For other floating-point types, use the math library
    return rewriter.create<mlir::math::FloorOp>(loc, input);
}

template <>
mlir::Value emitLinalgRegion<IE::CeilingOp>(IE::CeilingOp op, mlir::ValueRange args,
                                            llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto loc = op.getLoc();
    auto input = args[0];
    auto elTy = mlir::cast<NDTypeInterface>(op->getOperand(0).getType()).getElementType();

    if (mlir::isa<mlir::Float16Type>(elTy)) {
        return emitRoundRegion(op, args, resultTypes, rewriter, RoundMode::CEIL);
    }
    // For other floating-point types, use the math library
    return rewriter.create<mlir::math::CeilOp>(loc, input);
}

// Type conversion patterns
class YieldOpTypeConversion : public mlir::OpConversionPattern<IE::CGCYieldOp> {
public:
    using mlir::OpConversionPattern<IE::CGCYieldOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(IE::CGCYieldOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        rewriter.modifyOpInPlace(op, [&] {
            op->setOperands(adaptor.getOperands());
        });
        return mlir::success();
    }
};

// Asinh layer

template <>
mlir::Value emitLinalgRegion<IE::AsinhOp>(IE::AsinhOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                          mlir::PatternRewriter& rewriter) {
    return rewriter.create<mlir::math::AsinhOp>(op->getLoc(), resultTypes[0], args[0], mlir::arith::FastMathFlags::afn);
}

template <>
mlir::Value emitLinalgRegion<IE::AcoshOp>(IE::AcoshOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                          mlir::PatternRewriter& rewriter) {
    return rewriter.create<mlir::math::AcoshOp>(op->getLoc(), resultTypes[0], args[0], mlir::arith::FastMathFlags::afn);
}

template <>
mlir::Value emitLinalgRegion<IE::AsinOp>(IE::AsinOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    return rewriter.create<mlir::math::AsinOp>(op->getLoc(), resultTypes[0], args[0], mlir::arith::FastMathFlags::afn);
}

template <>
mlir::Value emitLinalgRegion<IE::AcosOp>(IE::AcosOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    return rewriter.create<mlir::math::AcosOp>(op->getLoc(), resultTypes[0], args[0], mlir::arith::FastMathFlags::afn);
}

// Remove quantized types from capsule block arguments and yielded values by inserting storage casts.
static void stripQuantFromCapsuleSignature(IE::CodeGenCapsuleOp capsuleOp) {
    auto getStorageTensorType = [](mlir::Type type) -> mlir::TensorType {
        auto rankedType = mlir::dyn_cast<mlir::RankedTensorType>(type);
        if (rankedType == nullptr) {
            return nullptr;
        }

        auto quantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(rankedType.getElementType());
        if (quantizedType == nullptr) {
            return nullptr;
        }

        return mlir::RankedTensorType::get(rankedType.getShape(), quantizedType.getStorageType(),
                                           rankedType.getEncoding());
    };

    auto* capsuleBlock = capsuleOp.getBody();
    auto builder = mlir::OpBuilder::atBlockBegin(capsuleBlock);

    for (auto blockArg : capsuleBlock->getArguments()) {
        auto storageTensorType = getStorageTensorType(blockArg.getType());
        if (storageTensorType == nullptr) {
            continue;
        }
        const auto originalType = blockArg.getType();
        blockArg.setType(storageTensorType);
        auto castBack = builder.create<mlir::quant::StorageCastOp>(capsuleOp.getLoc(), originalType, blockArg);
        blockArg.replaceAllUsesExcept(castBack.getResult(), castBack.getOperation());
    }

    auto yieldOp = mlir::cast<IE::CGCYieldOp>(capsuleBlock->getTerminator());
    builder.setInsertionPoint(yieldOp);
    for (unsigned operandIdx = 0; operandIdx < yieldOp->getNumOperands(); ++operandIdx) {
        auto operand = yieldOp->getOperand(operandIdx);
        auto storageTensorType = getStorageTensorType(operand.getType());
        if (storageTensorType == nullptr) {
            continue;
        }

        auto castToStorage = builder.create<mlir::quant::StorageCastOp>(yieldOp.getLoc(), storageTensorType, operand);
        yieldOp->setOperand(operandIdx, castToStorage.getResult());
    }
}

void ConvertEltwiseLayers2MathPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::RankedTensorType type) -> mlir::Type {
        return ShaveCodeGen::normalizeType(type);
    });
    const auto argConvert = [](mlir::OpBuilder&, mlir::RankedTensorType dstType, mlir::ValueRange inputs,
                               mlir::Location) -> mlir::Value {
        // Only capsule block arguments should need this and we should not be
        // able to reach this otherwise.
        auto input = inputs[0];
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(input)) {
            if (!mlir::isa<IE::CodeGenCapsuleOp>(blockArg.getOwner()->getParentOp())) {
                VPUX_THROW("Unexpected block argument conversion");
            }
            input.setType(dstType);
            return input;
        }
        VPUX_THROW("Unexpected type materialization required");
    };
    typeConverter.addTargetMaterialization(argConvert);

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                           mlir::math::MathDialect, mlir::func::FuncDialect, mlir::quant::QuantDialect>();

    target.addIllegalDialect<IE::IEDialect>();
    target.addLegalOp<IE::CodeGenCapsuleOp>();
    target.addDynamicallyLegalOp<IE::CGCYieldOp>([&](IE::CGCYieldOp op) {
        return typeConverter.isLegal(op);
    });

    mlir::RewritePatternSet patterns(&ctx);
    // Add type conversion patterns.
    patterns.add<YieldOpTypeConversion>(typeConverter, &ctx);

    // Mark quant cast ops as illegal - these are intermediate ops created by IEQuantizeDequantizeToQuant patterns
    // but will be lowered to math/arith by populateLowerQuantOpsPatterns().
    target.addIllegalOp<mlir::quant::QuantizeCastOp>();
    target.addIllegalOp<mlir::quant::DequantizeCastOp>();

    // Add element-wise patterns.
    patterns.add<IEEltwiseToLinalg<IE::MaximumOp>, IEEltwiseToLinalg<IE::MinimumOp>, IEEltwiseToLinalg<IE::DivideOp>,
                 IEEltwiseToLinalg<IE::LogOp>, IEEltwiseToLinalg<IE::ExpOp>, IEEltwiseToLinalg<IE::SqrtOp>,
                 IEEltwiseToLinalg<IE::TanhOp>, IEEltwiseToLinalg<IE::AtanOp>, IEEltwiseToLinalg<IE::BitwiseAndOp>,
                 IEEltwiseToLinalg<IE::BitwiseOrOp>, IEEltwiseToLinalg<IE::BitwiseXorOp>,
                 IEEltwiseToLinalg<IE::BitwiseNotOp>, IEEltwiseToLinalg<IE::LogicalOrOp>,
                 IEEltwiseToLinalg<IE::LogicalXorOp>, IEEltwiseToLinalg<IE::AndOp>, IEEltwiseToLinalg<IE::LogicalNotOp>,
                 IEEltwiseToLinalg<IE::SelectOp>, IEEltwiseToLinalg<IE::EqualOp>, IEEltwiseToLinalg<IE::NotEqualOp>,
                 IEEltwiseToLinalg<IE::LessOp>, IEEltwiseToLinalg<IE::LessEqualOp>, IEEltwiseToLinalg<IE::GreaterOp>,
                 IEEltwiseToLinalg<IE::GreaterEqualOp>, IEEltwiseToLinalg<IE::SquaredDifferenceOp>,
                 IEEltwiseToLinalg<IE::ErfOp>, IEEltwiseToLinalg<IE::RoundOp>, IEEltwiseToLinalg<IE::SinOp>,
                 IEEltwiseToLinalg<IE::CosOp>, IEEltwiseToLinalg<IE::PReluOp>, IEEltwiseToLinalg<IE::ReLUOp>,
                 IEEltwiseToLinalg<IE::LeakyReluOp>, IEEltwiseToLinalg<IE::ClampOp>, IEEltwiseToLinalg<IE::AddOp>,
                 IEEltwiseToLinalg<IE::MultiplyOp>, IEEltwiseToLinalg<IE::SubtractOp>, IEEltwiseToLinalg<IE::ConvertOp>,
                 IEEltwiseToLinalg<IE::TanOp>, IEEltwiseToLinalg<IE::AtanhOp>, IEEltwiseToLinalg<IE::SinhOp>,
                 IEEltwiseToLinalg<IE::CoshOp>, IEEltwiseToLinalg<IE::AbsOp>, IEEltwiseToLinalg<IE::NegativeOp>,
                 IEEltwiseToLinalg<IE::SignOp>, IEEltwiseToLinalg<IE::HSwishOp>, IEEltwiseToLinalg<IE::HSigmoidOp>,
                 IEEltwiseToLinalg<IE::EluOp>, IEEltwiseToLinalg<IE::GeluOp>, IEEltwiseToLinalg<IE::SeluOp>,
                 IEEltwiseToLinalg<IE::SoftPlusOp>, IEEltwiseToLinalg<IE::MishOp>, IEEltwiseToLinalg<IE::CeilingOp>,
                 IEEltwiseToLinalg<IE::FloorOp>, IEEltwiseToLinalg<IE::AsinhOp>, IEEltwiseToLinalg<IE::AcoshOp>,
                 IEEltwiseToLinalg<IE::AsinOp>, IEEltwiseToLinalg<IE::AcosOp>,
                 IEQuantizeDequantizeToQuant<IE::QuantizeOp>, IEQuantizeDequantizeToQuant<IE::DequantizeOp>>(
            typeConverter, &ctx);

    mlir::quant::populateLowerQuantOpsPatterns(patterns);
    ShaveCodeGen::populateIEReduceToLinalgPatterns(patterns, typeConverter);
    ShaveCodeGen::populateIEDataMovementToTensorPatterns(patterns, typeConverter);
    ShaveCodeGen::populateIEShapeManipulationToTensorPatterns(patterns, typeConverter);
    ShaveCodeGen::populateIESoftmaxToLinalgPatterns(patterns, typeConverter);
    mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    mlir::RewritePatternSet eltwisePatterns(&ctx);
    mlir::linalg::populateElementwiseToLinalgConversionPatterns(eltwisePatterns);
    mlir::FrozenRewritePatternSet frozenEltwisePatterns(std::move(eltwisePatterns));

    // E#172607 [ShaveCodeGen] Make Linalg lowering pass run on CodeGenCapsuleOps
    func->walk([&](IE::CodeGenCapsuleOp capsuleOp) -> mlir::WalkResult {
        if (mlir::failed(mlir::applyPartialConversion(capsuleOp, target, frozenPatterns))) {
            signalPassFailure();
            return mlir::WalkResult::interrupt();
        }

        // We're applying linalg conversion patterns separately using greedy rewrite
        // (without a type converter) to avoid having to mark all elementwise ops as
        // illegal in the main ConversionTarget.
        if (mlir::failed(mlir::applyPatternsGreedily(capsuleOp, frozenEltwisePatterns,
                                                     vpux::getDefaultGreedyRewriteConfig()))) {
            signalPassFailure();
            return mlir::WalkResult::interrupt();
        }

        stripQuantFromCapsuleSignature(capsuleOp);

        // CodeGenCapsule ops shouldn't be nested so return skip here.
        return mlir::WalkResult::skip();
    });
}

}  // namespace

//
// createConvertEltwiseLayers2MathPass
//

std::unique_ptr<mlir::Pass> ShaveCodeGen::createConvertEltwiseLayers2MathPass(Logger log) {
    return std::make_unique<ConvertEltwiseLayers2MathPass>(log);
}
