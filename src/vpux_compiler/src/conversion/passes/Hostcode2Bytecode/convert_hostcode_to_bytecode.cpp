//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/instructions.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/HostExec/IR/attributes.hpp"
#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/buffer.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/conditional.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/conversion.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/external.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/kernel_submission.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/register.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/IR/types.hpp"
#include "vpux/compiler/dialect/bytecode/utils/builders.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>

namespace vpux {
#define GEN_PASS_DECL_CONVERTHOSTCODETOBYTECODE
#define GEN_PASS_DEF_CONVERTHOSTCODETOBYTECODE
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

bool isAllOnesConst(mlir::Value value) {
    if (value == nullptr) {
        return false;
    }
    if (auto cstOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cstOp.getValueAttr())) {
            return intAttr.getValue().isAllOnes();
        }
    }
    return false;
}

class ArithConstantRewriter final : public mlir::OpConversionPattern<mlir::arith::ConstantOp> {
public:
    ArithConstantRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::ConstantOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::ConstantOp origOp, OpAdaptor /*adaptor*/,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        const auto value = bytecode::getI64ImmediateValueBits(origOp.getValueAttr());
        if (!value.has_value()) {
            return rewriter.notifyMatchFailure(origOp, [&](mlir::Diagnostic& diag) {
                diag << "unsupported constant attribute type or width: " << origOp.getValueAttr().getType();
            });
        }

        auto dstReg = bytecode::materializeI64ImmediateRegister(rewriter, origOp.getLoc(), *value);
        rewriter.replaceOp(origOp, dstReg);
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithAddIRewriter final : public mlir::OpConversionPattern<mlir::arith::AddIOp> {
public:
    ArithAddIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::AddIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::AddIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::AddI64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithMulIRewriter final : public mlir::OpConversionPattern<mlir::arith::MulIOp> {
public:
    ArithMulIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::MulIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::MulIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::MulI64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithMinSIRewriter final : public mlir::OpConversionPattern<mlir::arith::MinSIOp> {
public:
    ArithMinSIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::MinSIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::MinSIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::MinI64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithMaxSIRewriter final : public mlir::OpConversionPattern<mlir::arith::MaxSIOp> {
public:
    ArithMaxSIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::MaxSIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::MaxSIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::MaxI64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithCmpIRewriter final : public mlir::OpConversionPattern<mlir::arith::CmpIOp> {
public:
    ArithCmpIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::CmpIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::CmpIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        const auto pred = origOp.getPredicate();
        uint16_t flag = 0;
        switch (pred) {
        case mlir::arith::CmpIPredicate::eq:
            flag = intel_npu::vm::makeCmpFlag(intel_npu::vm::CmpPredicate::EQ, false);
            break;
        case mlir::arith::CmpIPredicate::ne:
            flag = intel_npu::vm::makeCmpFlag(intel_npu::vm::CmpPredicate::NE, false);
            break;
        case mlir::arith::CmpIPredicate::sgt:
            flag = intel_npu::vm::makeCmpFlag(intel_npu::vm::CmpPredicate::GT, true);
            break;
        case mlir::arith::CmpIPredicate::sge:
            flag = intel_npu::vm::makeCmpFlag(intel_npu::vm::CmpPredicate::GTE, true);
            break;
        case mlir::arith::CmpIPredicate::slt:
            flag = intel_npu::vm::makeCmpFlag(intel_npu::vm::CmpPredicate::LT, true);
            break;
        case mlir::arith::CmpIPredicate::sle:
            flag = intel_npu::vm::makeCmpFlag(intel_npu::vm::CmpPredicate::LTE, true);
            break;
        case mlir::arith::CmpIPredicate::ugt:
            flag = intel_npu::vm::makeCmpFlag(intel_npu::vm::CmpPredicate::GT, false);
            break;
        case mlir::arith::CmpIPredicate::uge:
            flag = intel_npu::vm::makeCmpFlag(intel_npu::vm::CmpPredicate::GTE, false);
            break;
        case mlir::arith::CmpIPredicate::ult:
            flag = intel_npu::vm::makeCmpFlag(intel_npu::vm::CmpPredicate::LT, false);
            break;
        case mlir::arith::CmpIPredicate::ule:
            flag = intel_npu::vm::makeCmpFlag(intel_npu::vm::CmpPredicate::LTE, false);
            break;
        default:
            return rewriter.notifyMatchFailure(origOp, "unsupported cmpi predicate");
        }
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::CmpI64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs(),
                                            rewriter.getI16IntegerAttr(flag));
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithSelectRewriter final : public mlir::OpConversionPattern<mlir::arith::SelectOp> {
public:
    ArithSelectRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::SelectOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::SelectOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::SelectOp>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getCondition(),
                                            adaptor.getTrueValue(), adaptor.getFalseValue());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithIndexCastRewriter final : public mlir::OpConversionPattern<mlir::arith::IndexCastOp> {
public:
    ArithIndexCastRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::IndexCastOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::IndexCastOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        // IntegerType and IndexType both convert to bytecode::RegisterType, so the cast is a no-op
        // at the bytecode level. Delegate to the rewriter so the conversion framework correctly
        // tracks the replacement; manually re-pointing uses and erasing the op leaves dangling
        // references inside the framework's mappings and triggers a use_empty() assertion.
        rewriter.replaceOp(origOp, adaptor.getIn());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithAddFRewriter final : public mlir::OpConversionPattern<mlir::arith::AddFOp> {
public:
    ArithAddFRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::AddFOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::AddFOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getType().isF64()) {
            return rewriter.notifyMatchFailure(origOp, "unsupported float type: only f64 is supported");
        }
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::AddF64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithSubFRewriter final : public mlir::OpConversionPattern<mlir::arith::SubFOp> {
public:
    ArithSubFRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::SubFOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::SubFOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getType().isF64()) {
            return rewriter.notifyMatchFailure(origOp, "unsupported float type: only f64 is supported");
        }
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::SubF64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithMulFRewriter final : public mlir::OpConversionPattern<mlir::arith::MulFOp> {
public:
    ArithMulFRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::MulFOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::MulFOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getType().isF64()) {
            return rewriter.notifyMatchFailure(origOp, "unsupported float type: only f64 is supported");
        }
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::MulF64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithDivFRewriter final : public mlir::OpConversionPattern<mlir::arith::DivFOp> {
public:
    ArithDivFRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::DivFOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::DivFOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getType().isF64()) {
            return rewriter.notifyMatchFailure(origOp, "unsupported float type: only f64 is supported");
        }
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::DivF64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithRemFRewriter final : public mlir::OpConversionPattern<mlir::arith::RemFOp> {
public:
    ArithRemFRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::RemFOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::RemFOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getType().isF64()) {
            return rewriter.notifyMatchFailure(origOp, "unsupported float type: only f64 is supported");
        }
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::RemF64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithMaximumFRewriter final : public mlir::OpConversionPattern<mlir::arith::MaximumFOp> {
public:
    ArithMaximumFRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::MaximumFOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::MaximumFOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getType().isF64()) {
            return rewriter.notifyMatchFailure(origOp, "unsupported float type: only f64 is supported");
        }
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::MaxF64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithMinimumFRewriter final : public mlir::OpConversionPattern<mlir::arith::MinimumFOp> {
public:
    ArithMinimumFRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::MinimumFOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::MinimumFOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getType().isF64()) {
            return rewriter.notifyMatchFailure(origOp, "unsupported float type: only f64 is supported");
        }
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::MinF64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class MathAbsFRewriter final : public mlir::OpConversionPattern<mlir::math::AbsFOp> {
public:
    MathAbsFRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::math::AbsFOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::math::AbsFOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getType().isF64()) {
            return rewriter.notifyMatchFailure(origOp, "unsupported float type: only f64 is supported");
        }
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::AbsF64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getOperand());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithNegFRewriter final : public mlir::OpConversionPattern<mlir::arith::NegFOp> {
public:
    ArithNegFRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::NegFOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::NegFOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getType().isF64()) {
            return rewriter.notifyMatchFailure(origOp, "unsupported float type: only f64 is supported");
        }
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::NegF64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getOperand());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class MathCeilRewriter final : public mlir::OpConversionPattern<mlir::math::CeilOp> {
public:
    MathCeilRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::math::CeilOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::math::CeilOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getType().isF64()) {
            return rewriter.notifyMatchFailure(origOp, "unsupported float type: only f64 is supported");
        }
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::CeilF64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getOperand());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class MathFloorRewriter final : public mlir::OpConversionPattern<mlir::math::FloorOp> {
public:
    MathFloorRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::math::FloorOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::math::FloorOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getType().isF64()) {
            return rewriter.notifyMatchFailure(origOp, "unsupported float type: only f64 is supported");
        }
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::FloorF64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getOperand());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class MathRoundEvenRewriter final : public mlir::OpConversionPattern<mlir::math::RoundEvenOp> {
public:
    MathRoundEvenRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::math::RoundEvenOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::math::RoundEvenOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getType().isF64()) {
            return rewriter.notifyMatchFailure(origOp, "unsupported float type: only f64 is supported");
        }
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::RoundF64Op>(
                origOp.getLoc(), dstRegOp.getResult(), adaptor.getOperand(),
                rewriter.getI16IntegerAttr(static_cast<int16_t>(intel_npu::vm::RoundingMode::RNE)));
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class MathRoundRewriter final : public mlir::OpConversionPattern<mlir::math::RoundOp> {
public:
    MathRoundRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::math::RoundOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::math::RoundOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getType().isF64()) {
            return rewriter.notifyMatchFailure(origOp, "unsupported float type: only f64 is supported");
        }
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::RoundF64Op>(
                origOp.getLoc(), dstRegOp.getResult(), adaptor.getOperand(),
                rewriter.getI16IntegerAttr(static_cast<int16_t>(intel_npu::vm::RoundingMode::RNA)));
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class MathTruncRewriter final : public mlir::OpConversionPattern<mlir::math::TruncOp> {
public:
    MathTruncRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::math::TruncOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::math::TruncOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getType().isF64()) {
            return rewriter.notifyMatchFailure(origOp, "unsupported float type: only f64 is supported");
        }
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::RoundF64Op>(
                origOp.getLoc(), dstRegOp.getResult(), adaptor.getOperand(),
                rewriter.getI16IntegerAttr(static_cast<int16_t>(intel_npu::vm::RoundingMode::RTZ)));
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithSubIRewriter final : public mlir::OpConversionPattern<mlir::arith::SubIOp> {
public:
    ArithSubIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::SubIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::SubIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::SubI64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithDivSIRewriter final : public mlir::OpConversionPattern<mlir::arith::DivSIOp> {
public:
    ArithDivSIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::DivSIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::DivSIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::DivI64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithDivUIRewriter final : public mlir::OpConversionPattern<mlir::arith::DivUIOp> {
public:
    ArithDivUIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::DivUIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::DivUIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::DivU64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithMinUIRewriter final : public mlir::OpConversionPattern<mlir::arith::MinUIOp> {
public:
    ArithMinUIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::MinUIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::MinUIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::MinU64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithMaxUIRewriter final : public mlir::OpConversionPattern<mlir::arith::MaxUIOp> {
public:
    ArithMaxUIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::MaxUIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::MaxUIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::MaxU64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithRemUIRewriter final : public mlir::OpConversionPattern<mlir::arith::RemUIOp> {
public:
    ArithRemUIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::RemUIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::RemUIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::RemU64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithAddUIExtendedRewriter final : public mlir::OpConversionPattern<mlir::arith::AddUIExtendedOp> {
public:
    ArithAddUIExtendedRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::AddUIExtendedOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::AddUIExtendedOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getOverflow().use_empty()) {
            return rewriter.notifyMatchFailure(origOp, "overflow result of arith.addui_extended is used; "
                                                       "only the sum result is supported");
        }
        auto sumRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::AddU64Op>(origOp.getLoc(), sumRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        // The overflow result is unused; provide a dead register to satisfy the conversion framework.
        auto overflowRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.replaceOp(origOp, {sumRegOp.getResult(), overflowRegOp.getResult()});
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithMulUIExtendedRewriter final : public mlir::OpConversionPattern<mlir::arith::MulUIExtendedOp> {
public:
    ArithMulUIExtendedRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::MulUIExtendedOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::MulUIExtendedOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getHigh().use_empty()) {
            return rewriter.notifyMatchFailure(origOp, "high result of arith.mului_extended is used; "
                                                       "only the low result is supported");
        }
        auto lowRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::MulU64Op>(origOp.getLoc(), lowRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        // The high result is unused; provide a dead register to satisfy the conversion framework.
        auto highRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.replaceOp(origOp, {lowRegOp.getResult(), highRegOp.getResult()});
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithRemSIRewriter final : public mlir::OpConversionPattern<mlir::arith::RemSIOp> {
public:
    ArithRemSIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::RemSIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::RemSIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::RemI64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class MathAbsIRewriter final : public mlir::OpConversionPattern<mlir::math::AbsIOp> {
public:
    MathAbsIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::math::AbsIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::math::AbsIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::AbsI64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getOperand());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class BitwiseAndIRewriter final : public mlir::OpConversionPattern<mlir::arith::AndIOp> {
public:
    BitwiseAndIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::AndIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::AndIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::And64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class BitwiseNotIRewriter final : public mlir::OpConversionPattern<mlir::arith::XOrIOp> {
public:
    BitwiseNotIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::XOrIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::XOrIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        // Lower xori(value, all-ones) to unary bytecode.not.i64.
        // Practical effect: x ^ -1 becomes not x.
        mlir::Value src;
        if (isAllOnesConst(origOp.getLhs())) {
            src = adaptor.getRhs();
        } else if (isAllOnesConst(origOp.getRhs())) {
            src = adaptor.getLhs();
        } else {
            return mlir::failure();
        }

        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::Not64Op>(origOp.getLoc(), dstRegOp.getResult(), src);
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class BitwiseXorIRewriter final : public mlir::OpConversionPattern<mlir::arith::XOrIOp> {
public:
    BitwiseXorIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::XOrIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::XOrIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        // Keep the all-ones form out of the binary xor lowering.
        // Practical effect: a ^ b stays xor a, b, while x ^ -1 is handled as not x.
        if (isAllOnesConst(origOp.getLhs()) || isAllOnesConst(origOp.getRhs())) {
            return mlir::failure();
        }

        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::Xor64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class BitwiseOrIRewriter final : public mlir::OpConversionPattern<mlir::arith::OrIOp> {
public:
    BitwiseOrIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::OrIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::OrIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::Or64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class BitwiseShLIRewriter final : public mlir::OpConversionPattern<mlir::arith::ShLIOp> {
public:
    BitwiseShLIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::ShLIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::ShLIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::Sll64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class BitwiseShrUIRewriter final : public mlir::OpConversionPattern<mlir::arith::ShRUIOp> {
public:
    BitwiseShrUIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::ShRUIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::ShRUIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::Srl64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class BitwiseShrSIRewriter final : public mlir::OpConversionPattern<mlir::arith::ShRSIOp> {
public:
    BitwiseShrSIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::ShRSIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::ShRSIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::Sra64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithExtSIRewriter final : public mlir::OpConversionPattern<mlir::arith::ExtSIOp> {
public:
    ArithExtSIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::ExtSIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::ExtSIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        const auto loc = origOp.getLoc();
        const auto srcWidth = mlir::cast<mlir::IntegerType>(origOp.getIn().getType()).getWidth();
        if (srcWidth == 1) {
            // i1 values are stored as 0 or 1 (zero-extended) in the 64-bit register.
            // Sign-extending i1 must map 0 -> 0 and 1 -> -1 (all ones).
            // Computing `0 - val` achieves this: 0 - 0 = 0, 0 - 1 = -1.
            auto zeroReg = bytecode::materializeI64ImmediateRegister(rewriter, loc, 0);
            auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(loc);
            rewriter.create<bytecode::SubI64Op>(loc, dstRegOp.getResult(), zeroReg, adaptor.getIn());
            rewriter.replaceOp(origOp, dstRegOp.getResult());
            return mlir::success();
        }
        // Integers in bytecode registers are always stored sign-extended to 64 bits,
        // so sign-extension between integer types is a no-op at the register level.
        rewriter.replaceOp(origOp, adaptor.getIn());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithTruncIRewriter final : public mlir::OpConversionPattern<mlir::arith::TruncIOp> {
public:
    ArithTruncIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::TruncIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::TruncIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        const auto loc = origOp.getLoc();
        const auto srcWidth = mlir::cast<mlir::IntegerType>(origOp.getIn().getType()).getWidth();
        const auto dstWidth = mlir::cast<mlir::IntegerType>(origOp.getType()).getWidth();
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(loc);
        if (srcWidth == 16 && dstWidth == 8) {
            rewriter.create<bytecode::ConvertI16ToI8Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 32 && dstWidth == 8) {
            rewriter.create<bytecode::ConvertI32ToI8Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 32 && dstWidth == 16) {
            rewriter.create<bytecode::ConvertI32ToI16Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 64 && dstWidth == 8) {
            rewriter.create<bytecode::ConvertI64ToI8Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 64 && dstWidth == 16) {
            rewriter.create<bytecode::ConvertI64ToI16Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 64 && dstWidth == 32) {
            rewriter.create<bytecode::ConvertI64ToI32Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else {
            return rewriter.notifyMatchFailure(origOp, "unsupported trunci width combination");
        }
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithSIToFPRewriter final : public mlir::OpConversionPattern<mlir::arith::SIToFPOp> {
public:
    ArithSIToFPRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::SIToFPOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::SIToFPOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        const auto loc = origOp.getLoc();
        const auto srcWidth = mlir::cast<mlir::IntegerType>(origOp.getIn().getType()).getWidth();
        const auto dstWidth = mlir::cast<mlir::FloatType>(origOp.getType()).getWidth();
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(loc);
        if (srcWidth == 8 && dstWidth == 32) {
            rewriter.create<bytecode::ConvertI8ToF32Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 8 && dstWidth == 64) {
            rewriter.create<bytecode::ConvertI8ToF64Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 16 && dstWidth == 32) {
            rewriter.create<bytecode::ConvertI16ToF32Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 16 && dstWidth == 64) {
            rewriter.create<bytecode::ConvertI16ToF64Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 32 && dstWidth == 32) {
            rewriter.create<bytecode::ConvertI32ToF32Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 32 && dstWidth == 64) {
            rewriter.create<bytecode::ConvertI32ToF64Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 64 && dstWidth == 32) {
            rewriter.create<bytecode::ConvertI64ToF32Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 64 && dstWidth == 64) {
            rewriter.create<bytecode::ConvertI64ToF64Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else {
            return rewriter.notifyMatchFailure(origOp, "unsupported sitofp width combination");
        }
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithFPToSIRewriter final : public mlir::OpConversionPattern<mlir::arith::FPToSIOp> {
public:
    ArithFPToSIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::FPToSIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::FPToSIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        const auto loc = origOp.getLoc();
        const auto srcWidth = mlir::cast<mlir::FloatType>(origOp.getIn().getType()).getWidth();
        const auto dstWidth = mlir::cast<mlir::IntegerType>(origOp.getType()).getWidth();
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(loc);
        if (srcWidth == 32 && dstWidth == 8) {
            rewriter.create<bytecode::ConvertF32ToI8Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 32 && dstWidth == 16) {
            rewriter.create<bytecode::ConvertF32ToI16Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 32 && dstWidth == 32) {
            rewriter.create<bytecode::ConvertF32ToI32Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 32 && dstWidth == 64) {
            rewriter.create<bytecode::ConvertF32ToI64Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 64 && dstWidth == 8) {
            rewriter.create<bytecode::ConvertF64ToI8Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 64 && dstWidth == 16) {
            rewriter.create<bytecode::ConvertF64ToI16Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 64 && dstWidth == 32) {
            rewriter.create<bytecode::ConvertF64ToI32Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else if (srcWidth == 64 && dstWidth == 64) {
            rewriter.create<bytecode::ConvertF64ToI64Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        } else {
            return rewriter.notifyMatchFailure(origOp, "unsupported fptosi width combination");
        }
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithExtFRewriter final : public mlir::OpConversionPattern<mlir::arith::ExtFOp> {
public:
    ArithExtFRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::ExtFOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::ExtFOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!mlir::cast<mlir::FloatType>(origOp.getIn().getType()).isF32() || !origOp.getType().isF64()) {
            return rewriter.notifyMatchFailure(origOp,
                                               "unsupported extf width combination: only f32->f64 is supported");
        }
        const auto loc = origOp.getLoc();
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(loc);
        rewriter.create<bytecode::ConvertF32ToF64Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithTruncFRewriter final : public mlir::OpConversionPattern<mlir::arith::TruncFOp> {
public:
    ArithTruncFRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::TruncFOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::TruncFOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!mlir::cast<mlir::FloatType>(origOp.getIn().getType()).isF64() || !origOp.getType().isF32()) {
            return rewriter.notifyMatchFailure(origOp,
                                               "unsupported truncf width combination: only f64->f32 is supported");
        }
        const auto loc = origOp.getLoc();
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(loc);
        rewriter.create<bytecode::ConvertF64ToF32Op>(loc, dstRegOp.getResult(), adaptor.getIn());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class FuncCallOpRewriter final : public mlir::OpConversionPattern<mlir::func::CallOp> {
public:
    FuncCallOpRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log,
                       const std::unordered_map<std::string, int>& funcNameToIndex)
            : mlir::OpConversionPattern<mlir::func::CallOp>(typeConverter, ctx),
              _log(log),
              _funcNameToIndex(funcNameToIndex) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::func::CallOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        const auto calleeSymbol = origOp.getCallee();

        // Use the pre-built index map: functions are erased from the module during conversion,
        // so searching moduleOp at rewrite time would fail for callees processed earlier
        const auto it = _funcNameToIndex.find(calleeSymbol.str());
        if (it == _funcNameToIndex.end()) {
            _log.error("Function {0} not found or not a host compile function", calleeSymbol);
            return mlir::failure();
        }
        const int funcIndex = it->second;

        // TODO(E#214401): Consider passing the callee symbol instead of the function index
        auto functionIndexRegister = bytecode::materializeI64ImmediateRegister(rewriter, origOp.getLoc(), funcIndex);

        // Calling convention: `call rs, N, rN..., M, rM...`.
        // `N` destination registers are written by the callee's `retv` in source order
        const size_t numResults = origOp.getNumResults();
        SmallVector<mlir::Value> destRegs;
        for (size_t i = 0; i < numResults; ++i) {
            auto destReg = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
            destRegs.push_back(destReg.getResult());
        }

        // `M` arguments are emitted in source order and mapped by the VM to callee parameter
        // registers `G + i`, where `G = num_general_registers - M` for the callee frame
        rewriter.create<bytecode::CallOp>(origOp.getLoc(), functionIndexRegister, mlir::ValueRange(destRegs),
                                          adaptor.getOperands());

        rewriter.replaceOp(origOp, destRegs);

        return mlir::success();
    }

private:
    Logger _log;
    const std::unordered_map<std::string, int>& _funcNameToIndex;
};

class ReturnRewriter final : public mlir::OpConversionPattern<mlir::func::ReturnOp> {
public:
    ReturnRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::func::ReturnOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::func::ReturnOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (adaptor.getOperands().empty()) {
            rewriter.replaceOpWithNewOp<bytecode::RetOp>(origOp);
        } else {
            rewriter.replaceOpWithNewOp<bytecode::RetVOp>(origOp, adaptor.getOperands());
        }
        return mlir::success();
    }

private:
    Logger _log;
};

class AssertRewriter final : public mlir::OpConversionPattern<mlir::cf::AssertOp> {
public:
    AssertRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::cf::AssertOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::cf::AssertOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        rewriter.replaceOpWithNewOp<bytecode::ExtAssertOp>(origOp, adaptor.getArg(), adaptor.getMsg());
        return mlir::success();
    }

private:
    Logger _log;
};

class MemRefAllocRewriter final : public mlir::OpConversionPattern<mlir::memref::AllocOp> {
public:
    MemRefAllocRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::memref::AllocOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::AllocOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!adaptor.getSymbolOperands().empty()) {
            return rewriter.notifyMatchFailure(origOp, "memref.alloc with symbol operands is not yet supported");
        }
        const auto offset = origOp.getType().getStridesAndOffset().second;
        if (mlir::ShapedType::isDynamic(offset)) {
            return rewriter.notifyMatchFailure(origOp, "memref.alloc with dynamic offset is not supported");
        }
        if (offset != 0) {
            return rewriter.notifyMatchFailure(origOp, "memref.alloc with non-zero offset is not supported");
        }
        _log.trace("Lower memref.alloc of type {0} to bytecode.ext.buffer.create", origOp.getType());
        auto destinationRegister = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc()).getResult();
        rewriter.create<bytecode::ExtBufferCreateOp>(origOp.getLoc(), destinationRegister, origOp.getType(),
                                                     adaptor.getDynamicSizes());
        rewriter.replaceOp(origOp, destinationRegister);
        return mlir::success();
    }

private:
    Logger _log;
};

class MemRefSubViewRewriter final : public mlir::OpConversionPattern<mlir::memref::SubViewOp> {
public:
    MemRefSubViewRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::memref::SubViewOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::SubViewOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto sourceType = origOp.getSourceType();
        auto resultType = origOp.getResult().getType();
        if (sourceType.getRank() != resultType.getRank()) {
            return rewriter.notifyMatchFailure(origOp, "rank-reducing memref.subview is not yet supported");
        }

        const auto loc = origOp.getLoc();
        _log.trace("Lower memref.subview to bytecode.buffer.subview");

        // Each rank slot becomes one bytecode register. Static slots are materialized via set_imm;
        // dynamic slots reuse the already type-converted SSA operand from the adaptor (which the
        // type converter has mapped from `index` to bytecode::RegisterType). The adaptor exposes
        // only the dynamic operands in natural order, matching OffsetSizeAndStrideOpInterface.
        const auto zipRegisters = [&](mlir::ArrayRef<int64_t> staticValues,
                                      mlir::ValueRange dynamicValues) -> SmallVector<mlir::Value> {
            SmallVector<mlir::Value> registers;
            registers.reserve(staticValues.size());
            size_t dynIdx = 0;
            for (auto staticValue : staticValues) {
                if (mlir::ShapedType::isDynamic(staticValue)) {
                    registers.push_back(dynamicValues[dynIdx++]);
                } else {
                    registers.push_back(bytecode::materializeI64ImmediateRegister(rewriter, loc, staticValue));
                }
            }
            return registers;
        };

        auto offsetRegisters = zipRegisters(origOp.getStaticOffsets(), adaptor.getOffsets());
        auto sizeRegisters = zipRegisters(origOp.getStaticSizes(), adaptor.getSizes());
        auto strideRegisters = zipRegisters(origOp.getStaticStrides(), adaptor.getStrides());
        auto destinationRegister = rewriter.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();
        rewriter.create<bytecode::BufferSubviewOp>(loc, destinationRegister, adaptor.getSource(), offsetRegisters,
                                                   sizeRegisters, strideRegisters);
        rewriter.replaceOp(origOp, destinationRegister);
        return mlir::success();
    }

private:
    Logger _log;
};

class MemRefCastRewriter final : public mlir::OpConversionPattern<mlir::memref::CastOp> {
public:
    MemRefCastRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::memref::CastOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::CastOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        // memref.cast only refines/loosens static shape/stride information. After lowering to bytecode,
        // the buffer descriptor lives at runtime in a `!bytecode.Register`, so the cast is a no-op.
        // Forward the converted source register directly to avoid emitting a dead chain of
        // unrealized_conversion_cast / memref.cast ops that survive the conversion.
        _log.trace("Drop memref.cast at {0}; bytecode descriptors are runtime values", origOp.getLoc());
        rewriter.replaceOp(origOp, adaptor.getSource());
        return mlir::success();
    }

private:
    Logger _log;
};

class MemRefReinterpretCastRewriter final : public mlir::OpConversionPattern<mlir::memref::ReinterpretCastOp> {
public:
    MemRefReinterpretCastRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::memref::ReinterpretCastOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::ReinterpretCastOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        // Dynamic offsets are unsupported
        if (!adaptor.getOffsets().empty()) {
            return rewriter.notifyMatchFailure(origOp, "memref.reinterpret_cast with dynamic offsets is not supported");
        }
        // buffer.view does not support offsets; reject anything with non-zero static offsets.
        for (auto offset : origOp.getStaticOffsets()) {
            if (offset != 0) {
                return rewriter.notifyMatchFailure(
                        origOp, "memref.reinterpret_cast with non-zero static offset is not supported");
            }
        }

        const auto loc = origOp.getLoc();
        _log.trace("Lower memref.reinterpret_cast to bytecode.ext.buffer.view");

        // Materialize each dimension slot as a register: static slots become set_imm;
        // dynamic slots reuse the already type-converted SSA operand from the adaptor.
        const auto zipRegisters = [&](mlir::ArrayRef<int64_t> staticValues,
                                      mlir::ValueRange dynamicValues) -> std::optional<SmallVector<mlir::Value>> {
            SmallVector<mlir::Value> registers;
            registers.reserve(staticValues.size());
            size_t dynIdx = 0;
            for (auto staticValue : staticValues) {
                if (mlir::ShapedType::isDynamic(staticValue)) {
                    if (dynIdx >= dynamicValues.size()) {
                        return std::nullopt;
                    }
                    registers.push_back(dynamicValues[dynIdx++]);
                } else {
                    registers.push_back(bytecode::materializeI64ImmediateRegister(rewriter, loc, staticValue));
                }
            }
            return registers;
        };

        auto sizeRegisters = zipRegisters(origOp.getStaticSizes(), adaptor.getSizes());
        if (!sizeRegisters.has_value()) {
            return rewriter.notifyMatchFailure(origOp, "dynamic size count mismatch in memref.reinterpret_cast");
        }
        auto strideRegisters = zipRegisters(origOp.getStaticStrides(), adaptor.getStrides());
        if (!strideRegisters.has_value()) {
            return rewriter.notifyMatchFailure(origOp, "dynamic stride count mismatch in memref.reinterpret_cast");
        }
        auto destinationRegister = rewriter.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();
        auto elemType = origOp.getResult().getType().getElementType();
        auto byteOffsetRegister = bytecode::materializeI64ImmediateRegister(rewriter, loc, 0);
        rewriter.create<bytecode::ExtBufferViewOp>(loc, destinationRegister, adaptor.getSource(), byteOffsetRegister,
                                                   mlir::TypeAttr::get(elemType), *sizeRegisters, *strideRegisters);
        rewriter.replaceOp(origOp, destinationRegister);
        return mlir::success();
    }

private:
    Logger _log;
};

class MemRefViewRewriter final : public mlir::OpConversionPattern<mlir::memref::ViewOp> {
public:
    MemRefViewRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::memref::ViewOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::ViewOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        const auto loc = origOp.getLoc();
        _log.trace("Lower memref.view to bytecode.ext.buffer.view");

        const auto resultType = mlir::cast<mlir::MemRefType>(origOp.getResult().getType());
        const auto rank = resultType.getRank();
        const auto elemType = resultType.getElementType();

        // memref.view gives one size operand per dynamic result dimension; materialize static extents as
        // immediates and splice the runtime operands into their slots, preserving dimension order.
        const auto dynamicSizes = adaptor.getSizes();
        SmallVector<mlir::Value> sizeRegisters;
        sizeRegisters.reserve(rank);
        size_t dynIdx = 0;
        for (const auto dim : resultType.getShape()) {
            if (mlir::ShapedType::isDynamic(dim)) {
                VPUX_THROW_UNLESS(dynIdx < dynamicSizes.size(),
                                  "memref.view has fewer dynamic size operands ({0}) than dynamic result dimensions",
                                  dynamicSizes.size());
                sizeRegisters.push_back(dynamicSizes[dynIdx++]);
            } else {
                sizeRegisters.push_back(bytecode::materializeI64ImmediateRegister(rewriter, loc, dim));
            }
        }
        VPUX_THROW_UNLESS(dynIdx == dynamicSizes.size(),
                          "memref.view has {0} dynamic size operands but only {1} dynamic result dimensions",
                          dynamicSizes.size(), dynIdx);

        SmallVector<mlir::Value> strideRegisters(rank);
        if (rank > 0) {
            strideRegisters[rank - 1] = bytecode::materializeI64ImmediateRegister(rewriter, loc, 1);
            for (int64_t i = rank - 2; i >= 0; --i) {
                auto dstReg = rewriter.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();
                rewriter.create<bytecode::MulI64Op>(loc, dstReg, sizeRegisters[i + 1], strideRegisters[i + 1]);
                strideRegisters[i] = dstReg;
            }
        }

        auto destinationRegister = rewriter.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();
        rewriter.create<bytecode::ExtBufferViewOp>(loc, destinationRegister, adaptor.getSource(),
                                                   adaptor.getByteShift(), mlir::TypeAttr::get(elemType), sizeRegisters,
                                                   strideRegisters);
        rewriter.replaceOp(origOp, destinationRegister);
        return mlir::success();
    }

private:
    Logger _log;
};

class MemRefDimRewriter final : public mlir::OpConversionPattern<mlir::memref::DimOp> {
public:
    MemRefDimRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::memref::DimOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::DimOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        // Static-shape queries with constant indices fold to arith.constant before this pattern runs;
        // any memref.dim that survives the fold (dynamic source dim or dynamic index) reads the runtime
        // buffer descriptor through bytecode.buffer.get_dim.
        const auto loc = origOp.getLoc();
        auto destinationRegister = rewriter.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();
        rewriter.create<bytecode::BufferGetDimOp>(loc, destinationRegister, adaptor.getSource(), adaptor.getIndex());
        rewriter.replaceOp(origOp, destinationRegister);
        return mlir::success();
    }

private:
    Logger _log;
};

class MemRefStoreRewriter final : public mlir::OpConversionPattern<mlir::memref::StoreOp> {
public:
    MemRefStoreRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::memref::StoreOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::StoreOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        // Scalar store values without a bytecode register conversion keep the original type in the adaptor.
        // Notify about this early with a clean match-failure rather than letting it fall through to the
        // bytecode op verifier.
        if (!mlir::isa<bytecode::RegisterType>(adaptor.getValue().getType())) {
            return rewriter.notifyMatchFailure(origOp,
                                               "memref.store value type is not convertible to a bytecode register");
        }
        rewriter.create<bytecode::BufferStoreOp>(origOp.getLoc(), adaptor.getMemref(), adaptor.getValue(),
                                                 adaptor.getIndices());
        rewriter.eraseOp(origOp);
        return mlir::success();
    }

private:
    Logger _log;
};

struct CommandListCreationState {
    bytecode::VirtualGeneralRegisterOp dstCmdListReg;
    bytecode::CmdListExecOp lastCmdListExecOp;

    void setLastRegister(bytecode::VirtualGeneralRegisterOp reg) {
        dstCmdListReg = reg;
    }
    bytecode::VirtualGeneralRegisterOp getLastRegister() const {
        return dstCmdListReg;
    }

    void setLastCmdListExecOp(bytecode::CmdListExecOp cmdListExecOp) {
        lastCmdListExecOp = cmdListExecOp;
    }
    bytecode::CmdListExecOp getLastCmdListExecOp() const {
        return lastCmdListExecOp;
    }

    // Set the HostSyncFlag of the last CmdListExecOp to 1
    // to ensure the command list is synchronized with the host before execution.
    void finalize() {
        if (lastCmdListExecOp != nullptr) {
            lastCmdListExecOp.setFlag(1);
        }
    }
};

class AsyncCreateGroupRewriter final : public mlir::OpConversionPattern<mlir::async::CreateGroupOp> {
public:
    AsyncCreateGroupRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx,
                             CommandListCreationState& cmdListCreationState, const Logger& log)
            : mlir::OpConversionPattern<mlir::async::CreateGroupOp>(typeConverter, ctx),
              _log(log),
              _cmdListCreationState(cmdListCreationState) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::async::CreateGroupOp origOp, OpAdaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (origOp->hasAttr("no_reset_cmdlist") == false) {
            auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
            rewriter.create<bytecode::CmdListCreateOp>(origOp.getLoc(), dstRegOp.getResult());
            rewriter.replaceOp(origOp, dstRegOp.getResult());

            _cmdListCreationState.setLastRegister(dstRegOp);
        } else {
            // no need to create a new command list, reuse the last one created
            // by the previous async.create_group op.
            auto dstRegOp = _cmdListCreationState.getLastRegister();
            if (!dstRegOp) {
                return rewriter.notifyMatchFailure(origOp,
                                                   "no_reset_cmdlist is set but no previous CmdListCreateOp exists");
            }
            rewriter.replaceOp(origOp, dstRegOp.getResult());
        }
        return mlir::success();
    }

private:
    Logger _log;
    CommandListCreationState& _cmdListCreationState;
};

class AsyncAwaitAllRewriter final : public mlir::OpConversionPattern<mlir::async::AwaitAllOp> {
public:
    AsyncAwaitAllRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx,
                          CommandListCreationState& cmdListCreationState, const Logger& log)
            : mlir::OpConversionPattern<mlir::async::AwaitAllOp>(typeConverter, ctx),
              _log(log),
              _cmdListCreationState(cmdListCreationState) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::async::AwaitAllOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        const auto isBoolAttrTrue = [&](mlir::StringRef name) {
            auto attr = mlir::dyn_cast_or_null<mlir::BoolAttr>(origOp->getAttr(name));
            return attr && attr.getValue();
        };

        if (isBoolAttrTrue("barrier")) {
            // E-221988
            // add create a CmdListAddBarrierOp when CmdListAddBarrierOp is implemented
            // For now, our target model does not require barrier addition
        } else if (!isBoolAttrTrue("noop")) {
            rewriter.create<bytecode::CmdListCloseOp>(appendLoc(origOp.getLoc(), "_close"), adaptor.getOperand());
            uint64_t hostSyncFlag = 0;
            auto cmdListExecOp =
                    rewriter.create<bytecode::CmdListExecOp>(appendLoc(origOp.getLoc(), "_exec"), adaptor.getOperand(),
                                                             rewriter.getI16IntegerAttr(hostSyncFlag));
            _cmdListCreationState.setLastCmdListExecOp(cmdListExecOp);
        }
        rewriter.eraseOp(origOp);
        return mlir::success();
    }

private:
    Logger _log;
    CommandListCreationState& _cmdListCreationState;
};

class AsyncExecuteWithNestedCallRewriter final : public mlir::OpConversionPattern<mlir::async::ExecuteOp> {
public:
    AsyncExecuteWithNestedCallRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::async::ExecuteOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::async::ExecuteOp origOp, OpAdaptor /*adaptor*/,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getBodyResults().empty()) {
            return rewriter.notifyMatchFailure(origOp,
                                               "async.execute returns values; only token-returning ops are handled");
        }

        auto bodyOps = origOp.getBody()->without_terminator();
        if (std::distance(bodyOps.begin(), bodyOps.end()) != 1) {
            return rewriter.notifyMatchFailure(
                    origOp,
                    "async.execute body must contain exactly one op (Core::NestedCallOp) besides the terminator");
        }
        auto nestedCall = mlir::dyn_cast<Core::NestedCallOp>(*bodyOps.begin());
        if (!nestedCall) {
            return rewriter.notifyMatchFailure(origOp, "async.execute body does not contain a Core::NestedCallOp");
        }

        // Resolve the kernel name from the callee symbol; existence is validated by verifySymbolUses.
        auto callee = nestedCall.getCallee();
        auto kernelName = callee.getLeafReference().getValue();
        auto parentModule = origOp->getParentOfType<mlir::ModuleOp>();
        bytecode::KernelSectionOp kernelSection;
        parentModule.walk([&](bytecode::KernelSectionOp op) {
            kernelSection = op;
            return mlir::WalkResult::interrupt();
        });
        if (!kernelSection) {
            return rewriter.notifyMatchFailure(origOp, "no bytecode.kernel_section found in module");
        }

        bool kernelFound = false;
        for (auto kernelOp : kernelSection.getContent().getOps<bytecode::KernelOp>()) {
            if (kernelOp.getSymName() == kernelName) {
                kernelFound = true;
                break;
            }
        }

        if (!kernelFound) {
            return rewriter.notifyMatchFailure(origOp, [&](mlir::Diagnostic& diag) {
                diag << "kernel '" << kernelName << "' not found in bytecode.kernel_section";
            });
        }

        const auto loc = origOp.getLoc();

        // Split NestedCall operands into inputs and outputs using the convention:
        //   operands = [inputs..., outputs...], results = outputs
        const auto numResults = nestedCall.getNumResults();
        const auto numOperands = nestedCall.getArgOperands().size();
        if (numOperands < numResults) {
            return rewriter.notifyMatchFailure(origOp, "nested call has fewer operands than results");
        }
        const auto numInputs = numOperands - numResults;
        SmallVector<mlir::Value> nestedCallOperands(nestedCall.getArgOperands().begin(),
                                                    nestedCall.getArgOperands().end());
        ArrayRef<mlir::Value> inputOperands(nestedCallOperands.data(), numInputs);
        ArrayRef<mlir::Value> outputOperands(nestedCallOperands.data() + numInputs,
                                             nestedCallOperands.size() - numInputs);

        // Remap each operand to the converted RegisterType value.
        const auto asRegister = [&](mlir::Value val) -> mlir::Value {
            if (auto remapped = rewriter.getRemappedValue(val)) {
                return remapped;
            }
            return rewriter
                    .create<mlir::UnrealizedConversionCastOp>(loc, bytecode::RegisterType::get(rewriter.getContext()),
                                                              val)
                    .getResult(0);
        };

        SmallVector<mlir::Value> convertedInputs, convertedOutputs;
        for (auto val : inputOperands) {
            convertedInputs.push_back(asRegister(val));
        }
        for (auto val : outputOperands) {
            convertedOutputs.push_back(asRegister(val));
        }

        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(loc);
        const auto kernelRef = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), kernelName);
        rewriter.create<bytecode::KernelCreateOp>(loc, dstRegOp.getResult(), kernelRef, convertedInputs,
                                                  convertedOutputs);

        rewriter.replaceOp(origOp, dstRegOp.getResult());

        bytecode::StringSectionOp stringSection;
        parentModule.walk([&](bytecode::StringSectionOp op) {
            stringSection = op;
            return mlir::WalkResult::interrupt();
        });

        bool kernelNameFound = false;
        if (!stringSection) {
            // create a string section here
            mlir::OpBuilder sectionBuilder(kernelSection);
            stringSection = sectionBuilder.create<bytecode::StringSectionOp>(sectionBuilder.getUnknownLoc(),
                                                                             bytecode::STRING_SECTION_NAME);
            stringSection.getContent().emplaceBlock();
        } else {
            for (auto stringOp : stringSection.getContent().getOps<bytecode::StringOp>()) {
                if (stringOp.getSymName() == kernelName) {
                    kernelNameFound = true;
                    break;
                }
            }
        }

        if (!kernelNameFound) {
            // adding kernel name to string section
            auto stringBuilder = mlir::OpBuilder::atBlockEnd(&stringSection.getContent().getBlocks().front());
            auto kernelNameAttr = mlir::StringAttr::get(stringBuilder.getContext(), kernelName);
            stringBuilder.create<bytecode::StringOp>(stringBuilder.getUnknownLoc(), kernelNameAttr, kernelNameAttr);
        }

        return mlir::success();
    }

private:
    Logger _log;
};

class AsyncAddToGroupRewriter final : public mlir::OpConversionPattern<mlir::async::AddToGroupOp> {
public:
    AsyncAddToGroupRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::async::AddToGroupOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::async::AddToGroupOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        if (!origOp.getResult().use_empty()) {
            // async.add_to_group is lowered to cmd_list.add_kernel, and cmd_list.add_kernel has no result value
            return rewriter.notifyMatchFailure(
                    origOp, "async.add_to_group lowering only supports ops whose returned result is unused; "
                            "remove uses of the result before Hostcode2Bytecode conversion");
        }

        rewriter.create<bytecode::CmdListAddKernelOp>(origOp.getLoc(), adaptor.getGroup(), adaptor.getOperand(),
                                                      mlir::ValueRange{}, mlir::ValueRange{});
        rewriter.eraseOp(origOp);
        return mlir::success();
    }

private:
    Logger _log;
};

struct PreAllocatedRegisters {
    llvm::DenseMap<mlir::Value, mlir::Value> argToReg;
    mlir::Value condBrOneReg;
    llvm::DenseMap<mlir::Operation*, mlir::Value> switchCompRegs;
};

// Emit bytecode.set ops to copy each adapted operand into the canonical VGR for the
// corresponding destination block argument. Called before every jump that passes values
// to a non-entry block.
// E#216236 - A branch that swaps or rotates destination block arguments requires more complex handling
static void emitBlockArgSetup(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, mlir::Block* destBlock,
                              mlir::ValueRange adaptedOperands,
                              const llvm::DenseMap<mlir::Value, mlir::Value>& argToReg) {
    const auto regType = bytecode::RegisterType::get(rewriter.getContext());
    for (auto [arg, operand] : llvm::zip(destBlock->getArguments(), adaptedOperands)) {
        const auto it = argToReg.find(arg);
        VPUX_THROW_UNLESS(it != argToReg.end(),
                          "No canonical register found for block argument during branch lowering");
        VPUX_THROW_UNLESS(operand.getType() == regType, "Branch operand must have RegisterType after type conversion");
        rewriter.create<bytecode::SetOp>(loc, it->second, operand);
    }
}

class CfBranchRewriter final : public mlir::OpConversionPattern<mlir::cf::BranchOp> {
public:
    CfBranchRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx,
                     const llvm::DenseMap<mlir::Value, mlir::Value>& argToReg, const Logger& log)
            : mlir::OpConversionPattern<mlir::cf::BranchOp>(typeConverter, ctx), _argToReg(argToReg), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::cf::BranchOp brOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        const auto loc = brOp.getLoc();
        emitBlockArgSetup(rewriter, loc, brOp.getDest(), adaptor.getDestOperands(), _argToReg);
        rewriter.replaceOpWithNewOp<bytecode::JmpOp>(brOp, brOp.getDest());
        return mlir::success();
    }

private:
    const llvm::DenseMap<mlir::Value, mlir::Value>& _argToReg;
    Logger _log;
};

class CfCondBranchRewriter final : public mlir::OpConversionPattern<mlir::cf::CondBranchOp> {
public:
    CfCondBranchRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx,
                         const llvm::DenseMap<mlir::Value, mlir::Value>& argToReg, mlir::Value condBrOneReg,
                         const Logger& log)
            : mlir::OpConversionPattern<mlir::cf::CondBranchOp>(typeConverter, ctx),
              _argToReg(argToReg),
              _condBrOneReg(condBrOneReg),
              _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::cf::CondBranchOp condBrOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        const auto loc = condBrOp.getLoc();
        const auto condReg = adaptor.getCondition();
        VPUX_THROW_UNLESS(condReg.getType() == bytecode::RegisterType::get(rewriter.getContext()),
                          "cf.cond_br condition must have RegisterType after type conversion");

        auto* trueDest = condBrOp.getTrueDest();
        auto* falseDest = condBrOp.getFalseDest();
        auto* currentBlock = condBrOp->getBlock();
        auto* region = currentBlock->getParent();

        // Each non-entry block argument is backed by a pre-allocated canonical register
        // (see preallocateRegisters): one register per argument, shared across all uses
        // of that argument throughout the function.
        //
        // A naive lowering would emit the argument-copy sequences for both edges before
        // the JE in the current block. This is unsafe when the two edges pass different
        // values through the same canonical register. For example, if the true edge
        // passes %new_sum and the false edge passes %sum, and both map to the same
        // canonical register, the true-edge write (canonical = %new_sum) overwrites the
        // register before the false-edge read (canonical = %sum) can use it — silently
        // producing the wrong value at the false destination.
        //
        // Two dedicated setup blocks fix this: the JE branches first, then each setup
        // block runs only on its own execution path, so the two copy sequences are
        // mutually exclusive and can never interfere with each other.
        //
        // ^false_setup is placed immediately after currentBlock so JE's not-taken
        // fallthrough lands on it without an extra jump.
        const auto afterCurrent = std::next(currentBlock->getIterator());
        auto* falseSetupBlock = rewriter.createBlock(region, afterCurrent);
        auto* trueSetupBlock = rewriter.createBlock(region, std::next(falseSetupBlock->getIterator()));

        rewriter.setInsertionPoint(condBrOp);
        rewriter.replaceOpWithNewOp<bytecode::JEOp>(condBrOp, condReg, _condBrOneReg, trueSetupBlock, falseSetupBlock);

        // Fill ^false_setup: copy false-edge args then jump to falseDest.
        rewriter.setInsertionPointToEnd(falseSetupBlock);
        emitBlockArgSetup(rewriter, loc, falseDest, adaptor.getFalseDestOperands(), _argToReg);
        rewriter.create<bytecode::JmpOp>(loc, falseDest);

        // Fill ^true_setup: copy true-edge args then jump to trueDest.
        rewriter.setInsertionPointToEnd(trueSetupBlock);
        emitBlockArgSetup(rewriter, loc, trueDest, adaptor.getTrueDestOperands(), _argToReg);
        rewriter.create<bytecode::JmpOp>(loc, trueDest);

        return mlir::success();
    }

private:
    const llvm::DenseMap<mlir::Value, mlir::Value>& _argToReg;
    mlir::Value _condBrOneReg;
    Logger _log;
};

class CfSwitchRewriter final : public mlir::OpConversionPattern<mlir::cf::SwitchOp> {
public:
    CfSwitchRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx,
                     const llvm::DenseMap<mlir::Value, mlir::Value>& argToReg,
                     const llvm::DenseMap<mlir::Operation*, mlir::Value>& switchCompRegs, const Logger& log)
            : mlir::OpConversionPattern<mlir::cf::SwitchOp>(typeConverter, ctx),
              _argToReg(argToReg),
              _switchCompRegs(switchCompRegs),
              _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::cf::SwitchOp switchOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        const auto loc = switchOp.getLoc();
        const auto condReg = adaptor.getFlag();
        auto* defaultDest = switchOp.getDefaultDestination();
        const auto defaultOps = adaptor.getDefaultOperands();
        const auto caseValuesAttr = switchOp.getCaseValues();

        // Zero-case: only a default arm — emit blockArgSetup + JMP.
        if (!caseValuesAttr || caseValuesAttr->empty()) {
            emitBlockArgSetup(rewriter, loc, defaultDest, defaultOps, _argToReg);
            rewriter.replaceOpWithNewOp<bytecode::JmpOp>(switchOp, defaultDest);
            return mlir::success();
        }

        const auto caseDests = switchOp.getCaseDestinations();
        const SmallVector<mlir::ValueRange> caseOps(adaptor.getCaseOperands());
        const SmallVector<mlir::APInt> caseValues(caseValuesAttr->getValues<mlir::APInt>());
        const size_t numCases = caseValues.size();

        const auto compRegIt = _switchCompRegs.find(switchOp.getOperation());
        VPUX_THROW_UNLESS(compRegIt != _switchCompRegs.end(),
                          "No pre-allocated comparison register found for cf.switch op");
        const auto compReg = compRegIt->second;

        auto* currentBlock = switchOp->getBlock();
        auto* region = currentBlock->getParent();

        // Physical block order after currentBlock:
        //   check1..checkN-1, def_setup, case0_setup..caseN-1_setup
        auto insertPos = std::next(currentBlock->getIterator());

        SmallVector<mlir::Block*> checkBlocks;
        checkBlocks.reserve(numCases - 1);
        for (size_t i = 1; i < numCases; ++i) {
            checkBlocks.push_back(rewriter.createBlock(region, insertPos));
            insertPos = std::next(checkBlocks.back()->getIterator());
        }

        auto* defSetupBlock = rewriter.createBlock(region, insertPos);
        insertPos = std::next(defSetupBlock->getIterator());

        SmallVector<mlir::Block*> caseSetupBlocks;
        caseSetupBlocks.reserve(numCases);
        for (size_t i = 0; i < numCases; ++i) {
            caseSetupBlocks.push_back(rewriter.createBlock(region, insertPos));
            insertPos = std::next(caseSetupBlocks.back()->getIterator());
        }

        // Emit set_imm + je for one check; falseDest is always the next physical block.
        const auto emitCheck = [&](mlir::Block* inBlock, size_t caseIdx, mlir::Block* fallthroughBlock) {
            rewriter.setInsertionPointToEnd(inBlock);
            rewriter.create<bytecode::SetImmOp>(
                    loc, compReg, rewriter.getI64IntegerAttr(bytecode::apIntToI64ImmediateBits(caseValues[caseIdx])));
            rewriter.create<bytecode::JEOp>(loc, condReg, compReg, caseSetupBlocks[caseIdx], fallthroughBlock);
        };

        // Fill currentBlock (case 0); fallthroughDest = checkBlocks[0] or defSetupBlock if N==1.
        rewriter.setInsertionPoint(switchOp);
        {
            auto* fallthroughBlock = checkBlocks.empty() ? defSetupBlock : checkBlocks[0];
            rewriter.create<bytecode::SetImmOp>(
                    loc, compReg, rewriter.getI64IntegerAttr(bytecode::apIntToI64ImmediateBits(caseValues[0])));
            rewriter.replaceOpWithNewOp<bytecode::JEOp>(switchOp, condReg, compReg, caseSetupBlocks[0],
                                                        fallthroughBlock);
        }

        // Fill check blocks for cases 1..N-1.
        for (size_t i = 0; i < checkBlocks.size(); ++i) {
            auto* fallthroughBlock = (i + 1 < checkBlocks.size()) ? checkBlocks[i + 1] : defSetupBlock;
            emitCheck(checkBlocks[i], i + 1, fallthroughBlock);
        }

        // Fill def_setup block.
        rewriter.setInsertionPointToEnd(defSetupBlock);
        emitBlockArgSetup(rewriter, loc, defaultDest, defaultOps, _argToReg);
        rewriter.create<bytecode::JmpOp>(loc, defaultDest);

        // Fill case-setup blocks: each is only reached when its JE fires.
        for (size_t i = 0; i < numCases; ++i) {
            rewriter.setInsertionPointToEnd(caseSetupBlocks[i]);
            emitBlockArgSetup(rewriter, loc, caseDests[i], caseOps[i], _argToReg);
            rewriter.create<bytecode::JmpOp>(loc, caseDests[i]);
        }

        return mlir::success();
    }

private:
    const llvm::DenseMap<mlir::Value, mlir::Value>& _argToReg;
    const llvm::DenseMap<mlir::Operation*, mlir::Value>& _switchCompRegs;
    Logger _log;
};

}  // namespace

// disable pipelined command list recording for host compile inference exec function
// for dynamic batch support for now.
void preprocessingAsyncOps(mlir::func::FuncOp funcOp) {
    const bool isHostCompileInferenceExecFunc = vpux::HostExec::isHostCompileInferenceExecFunc(funcOp);
    if (!isHostCompileInferenceExecFunc || !funcOp->hasAttr("disable_pipelined_cmdlist_recording")) {
        return;
    }

    bool preprocessingRequired = false;
    auto disablePipelinedCmdListAttr = funcOp->getAttr("disable_pipelined_cmdlist_recording");
    if (auto attr = mlir::dyn_cast<mlir::BoolAttr>(disablePipelinedCmdListAttr)) {
        preprocessingRequired = attr.getValue();
    }

    if (!preprocessingRequired) {
        return;
    }

    mlir::OpBuilder builder(funcOp);
    auto trueAttr = builder.getBoolAttr(true);
    if (vpux::HostExec::isHostCompileInferenceExecFunc(funcOp)) {
        funcOp.walk([&](mlir::async::CreateGroupOp op) {
            op->setAttr("no_reset_cmdlist", trueAttr);
        });
        // await all should be converted into barrier to
        // synchronize between sub graphs (e.g., transpose + the last of the model)
        auto awaitAllOps = funcOp.getOps<mlir::async::AwaitAllOp>();
        auto awaitAllOpCount = static_cast<size_t>(std::distance(awaitAllOps.begin(), awaitAllOps.end()));
        for (auto [index, awaitAllOp] : enumerate(awaitAllOps)) {
            if (index == (awaitAllOpCount - 1)) {
                // the last await all should be converted into noop
                // to avoid unnecessary barrier at the end of the function
                awaitAllOp->setAttr("noop", trueAttr);
            } else {
                // need to add a barrier for other await all to ensure the synchronization
                // between sub graphs (e.g., transpose + the last of the model)
                awaitAllOp->setAttr("barrier", trueAttr);
            }
        }
        // add a create group op at the beginning of the function
        auto& entryBlock = funcOp.getBody().front();
        builder.setInsertionPointToStart(&entryBlock);
        auto constAttr = builder.getIntegerAttr(builder.getIndexType(), 1);
        auto groupSize = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), constAttr);
        auto group = builder.create<mlir::async::CreateGroupOp>(builder.getUnknownLoc(), groupSize);

        // add a await all op before every func.return in the function. Returns may live in
        // blocks other than the entry block when the function contains control flow, and every
        // exit path needs the await/close+exec so the shared command list is executed.
        SmallVector<mlir::func::ReturnOp> returnOps;
        funcOp.walk([&](mlir::func::ReturnOp op) {
            returnOps.push_back(op);
        });
        VPUX_THROW_UNLESS(!returnOps.empty(), "No func.return found in function {0}", funcOp.getSymName());
        for (auto retOp : returnOps) {
            builder.setInsertionPoint(retOp);
            builder.create<mlir::async::AwaitAllOp>(retOp.getLoc(), group);
        }
    }
}

namespace vpux {

class ConvertHostcodeToBytecodePass final : public impl::ConvertHostcodeToBytecodeBase<ConvertHostcodeToBytecodePass> {
public:
    explicit ConvertHostcodeToBytecodePass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final {
        auto moduleOp = getOperation();

        auto hostCompileFunctions = [&]() {
            SmallVector<mlir::func::FuncOp> hostCompileFunctions;
            for (auto funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
                if (config::isPureHostCompileFunc(funcOp)) {
                    hostCompileFunctions.push_back(funcOp);
                }
            }
            return hostCompileFunctions;
        }();
        if (hostCompileFunctions.empty()) {
            _log.debug("No host compile functions found.");
            return;
        }
        for (auto funcOp : hostCompileFunctions) {
            _log.trace("Found host compile function: {0}", funcOp.getSymName());
        }

        // Build name->index map before any function is erased. FuncCallOpRewriter uses this map
        // to resolve callee indices; searching the module during conversion would fail for
        // functions that have already been processed and erased
        std::unordered_map<std::string, int> funcNameToIndex;
        for (size_t i = 0; i < hostCompileFunctions.size(); ++i) {
            _log.trace("Registering function {0} with index {1}", hostCompileFunctions[i].getSymName(), i);
            funcNameToIndex[hostCompileFunctions[i].getSymName().str()] = i;
        }

        auto funcSection = prepareFuncSection(moduleOp);
        for (auto funcOp : hostCompileFunctions) {
            CommandListCreationState cmdListCreationState;
            preprocessingAsyncOps(funcOp);
            if (mlir::failed(convertFuncToBytecode(funcOp, funcSection, funcNameToIndex, cmdListCreationState))) {
                _log.error("Failed to convert function {0} to bytecode", funcOp.getName());
                signalPassFailure();
                return;
            }
            cmdListCreationState.finalize();
        }
    }

    static PreAllocatedRegisters preallocateRegisters(mlir::func::FuncOp funcOp) {
        PreAllocatedRegisters result;
        auto& body = funcOp.getBody();
        auto& entryBlock = body.front();

        mlir::OpBuilder builder(funcOp.getContext());
        builder.setInsertionPointToStart(&entryBlock);

        // Canonical VGRs for non-entry block arguments.
        for (auto& block : body) {
            if (&block == &entryBlock) {
                continue;
            }
            for (auto arg : block.getArguments()) {
                auto regOp = builder.create<bytecode::VirtualGeneralRegisterOp>(funcOp.getLoc());
                result.argToReg[arg] = regOp.getResult();
            }
        }

        // One shared const-1 register for all cf.cond_br comparisons (JE fires when condReg == 1).
        bool hasCondBr = false;
        body.walk([&hasCondBr](mlir::cf::CondBranchOp) {
            hasCondBr = true;
            return mlir::WalkResult::interrupt();
        });
        if (hasCondBr) {
            auto oneReg = builder.create<bytecode::VirtualGeneralRegisterOp>(funcOp.getLoc());
            builder.create<bytecode::SetImmOp>(funcOp.getLoc(), oneReg.getResult(), builder.getI64IntegerAttr(1));
            result.condBrOneReg = oneReg.getResult();
        }

        // One comparison VGR per cf.switch op with at least one case (reused for all case value
        // comparisons in that switch). Zero-case switches jump directly to default and need no VGR.
        body.walk([&](mlir::cf::SwitchOp switchOp) {
            const auto caseValuesAttr = switchOp.getCaseValues();
            if (!caseValuesAttr || caseValuesAttr->empty()) {
                return;
            }
            auto compReg = builder.create<bytecode::VirtualGeneralRegisterOp>(funcOp.getLoc());
            result.switchCompRegs[switchOp.getOperation()] = compReg.getResult();
        });

        return result;
    }

    // Introduce an empty function sections into the module operation
    bytecode::FuncSectionOp prepareFuncSection(mlir::ModuleOp moduleOp) {
        mlir::OpBuilder builder(&getContext());
        builder.setInsertionPointToEnd(moduleOp.getBody());
        const auto loc = mlir::NameLoc::get(mlir::StringAttr::get(&getContext(), bytecode::FUNCTION_SECTION_NAME));
        auto funcSection = bytecode::FuncSectionOp::create(builder, loc, bytecode::FUNCTION_SECTION_NAME);
        funcSection.getContent().emplaceBlock();
        return funcSection;
    }

    // Convert a function to bytecode operations and store the new bytecode function in the provided function section
    // The original function is erased after conversion
    mlir::LogicalResult convertFuncToBytecode(mlir::func::FuncOp funcOp, bytecode::FuncSectionOp funcSection,
                                              const std::unordered_map<std::string, int>& funcNameToIndex,
                                              CommandListCreationState& cmdListCreationState) {
        mlir::TypeConverter typeConverter;
        typeConverter.addConversion([](mlir::IntegerType type) -> mlir::Type {
            return bytecode::RegisterType::get(type.getContext());
        });
        typeConverter.addConversion([](mlir::IndexType type) -> mlir::Type {
            return bytecode::RegisterType::get(type.getContext());
        });
        typeConverter.addConversion([](mlir::async::GroupType type) -> mlir::Type {
            return bytecode::RegisterType::get(type.getContext());
        });
        typeConverter.addConversion([](mlir::async::TokenType type) -> mlir::Type {
            return bytecode::RegisterType::get(type.getContext());
        });
        typeConverter.addConversion([](mlir::async::ValueType type) -> mlir::Type {
            return bytecode::RegisterType::get(type.getContext());
        });
        typeConverter.addConversion([](mlir::MemRefType type) -> mlir::Type {
            return bytecode::RegisterType::get(type.getContext());
        });
        typeConverter.addConversion([](mlir::BaseMemRefType type) -> mlir::Type {
            return bytecode::RegisterType::get(type.getContext());
        });
        typeConverter.addConversion([](mlir::FloatType type) -> mlir::Type {
            return bytecode::RegisterType::get(type.getContext());
        });
        const auto materialize = [&](mlir::OpBuilder& builder, mlir::Type resultType, mlir::ValueRange inputs,
                                     mlir::Location loc) -> mlir::Value {
            if (inputs.size() != 1) {
                return mlir::Value();
            }
            if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(inputs.front())) {
                // Only function parameters (entry block args) map to VirtualParameterRegisterOp.
                // Non-entry block args are handled by the canonical VGR assigned in preallocateRegisters.
                auto* region = blockArg.getOwner()->getParent();
                if (blockArg.getOwner() == &region->front()) {
                    return builder.create<bytecode::VirtualParameterRegisterOp>(loc, blockArg.getArgNumber());
                }
            }
            return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
        };
        typeConverter.addTargetMaterialization(materialize);
        typeConverter.addSourceMaterialization(materialize);

        // Pre-pass: allocate canonical VGRs for non-entry block arguments, one shared const-1
        // register for all cf.cond_br comparisons, and one comparison VGR per cf.switch op —
        // all placed at the entry block so the register allocator finds them in its single-block scan.
        const auto preAllocRegs = preallocateRegisters(funcOp);

        auto ctx = &getContext();
        mlir::ConversionTarget target(*ctx);
        target.addIllegalDialect<mlir::arith::ArithDialect>();
        target.addIllegalDialect<mlir::cf::ControlFlowDialect>();
        target.addIllegalOp<mlir::cf::BranchOp>();
        target.addIllegalOp<mlir::func::ReturnOp>();
        target.addIllegalOp<mlir::async::CreateGroupOp>();
        target.addIllegalOp<mlir::async::AddToGroupOp>();
        target.addIllegalOp<mlir::async::AwaitAllOp>();
        target.addIllegalOp<mlir::async::ExecuteOp>();
        target.addIllegalOp<mlir::func::CallOp>();
        target.addIllegalOp<mlir::memref::AllocOp>();
        target.addIllegalOp<mlir::memref::SubViewOp>();
        target.addIllegalOp<mlir::memref::ViewOp>();
        target.addIllegalOp<mlir::memref::ReinterpretCastOp>();
        target.addIllegalOp<mlir::memref::CastOp>();
        target.addIllegalOp<mlir::memref::DimOp>();
        target.addIllegalOp<mlir::memref::StoreOp>();
        target.addLegalDialect<bytecode::BytecodeDialect>();

        mlir::RewritePatternSet patterns(ctx);
        patterns.add<ArithConstantRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithAddIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithMulIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithMinSIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithMaxSIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithCmpIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithSelectRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithIndexCastRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithAddFRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithSubFRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithMulFRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithDivFRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithRemFRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithMaximumFRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithMinimumFRewriter>(typeConverter, ctx, _log);
        patterns.add<MathAbsFRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithNegFRewriter>(typeConverter, ctx, _log);
        patterns.add<MathCeilRewriter>(typeConverter, ctx, _log);
        patterns.add<MathFloorRewriter>(typeConverter, ctx, _log);
        patterns.add<MathRoundEvenRewriter>(typeConverter, ctx, _log);
        patterns.add<MathRoundRewriter>(typeConverter, ctx, _log);
        patterns.add<MathTruncRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithSubIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithDivSIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithDivUIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithMinUIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithMaxUIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithRemUIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithAddUIExtendedRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithMulUIExtendedRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithRemSIRewriter>(typeConverter, ctx, _log);
        patterns.add<MathAbsIRewriter>(typeConverter, ctx, _log);
        patterns.add<BitwiseAndIRewriter>(typeConverter, ctx, _log);
        patterns.add<BitwiseNotIRewriter>(typeConverter, ctx, _log);
        patterns.add<BitwiseXorIRewriter>(typeConverter, ctx, _log);
        patterns.add<BitwiseOrIRewriter>(typeConverter, ctx, _log);
        patterns.add<BitwiseShLIRewriter>(typeConverter, ctx, _log);
        patterns.add<BitwiseShrUIRewriter>(typeConverter, ctx, _log);
        patterns.add<BitwiseShrSIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithExtSIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithTruncIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithSIToFPRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithFPToSIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithExtFRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithTruncFRewriter>(typeConverter, ctx, _log);
        patterns.add<FuncCallOpRewriter>(typeConverter, ctx, _log, funcNameToIndex);
        patterns.add<ReturnRewriter>(typeConverter, ctx, _log);
        patterns.add<AssertRewriter>(typeConverter, ctx, _log);
        patterns.add<MemRefAllocRewriter>(typeConverter, ctx, _log);
        patterns.add<MemRefSubViewRewriter>(typeConverter, ctx, _log);
        patterns.add<MemRefViewRewriter>(typeConverter, ctx, _log);
        patterns.add<MemRefReinterpretCastRewriter>(typeConverter, ctx, _log);
        patterns.add<MemRefCastRewriter>(typeConverter, ctx, _log);
        patterns.add<MemRefDimRewriter>(typeConverter, ctx, _log);
        patterns.add<MemRefStoreRewriter>(typeConverter, ctx, _log);
        patterns.add<AsyncCreateGroupRewriter>(typeConverter, ctx, cmdListCreationState, _log);
        patterns.add<AsyncAddToGroupRewriter>(typeConverter, ctx, _log);
        patterns.add<AsyncAwaitAllRewriter>(typeConverter, ctx, cmdListCreationState, _log);
        patterns.add<AsyncExecuteWithNestedCallRewriter>(typeConverter, ctx, _log);
        patterns.add<CfBranchRewriter>(typeConverter, ctx, preAllocRegs.argToReg, _log);
        patterns.add<CfCondBranchRewriter>(typeConverter, ctx, preAllocRegs.argToReg, preAllocRegs.condBrOneReg, _log);
        patterns.add<CfSwitchRewriter>(typeConverter, ctx, preAllocRegs.argToReg, preAllocRegs.switchCompRegs, _log);

        if (mlir::failed(mlir::applyPartialConversion(funcOp, target, std::move(patterns)))) {
            return errorAt(funcOp, "Failed to apply conversion patterns");
        }

        // Post-pass: replace all uses of non-entry block arguments with their canonical VGRs.
        // The type materializer may have produced unrealized_cast(blockArg → Register) ops during
        // conversion; after replaceAllUsesWith those become Register→Register identity casts.
        // Remove them so the resulting bytecode is clean.
        for (auto& [arg, reg] : preAllocRegs.argToReg) {
            mlir::Value argVal = arg;
            argVal.replaceAllUsesWith(reg);
        }
        SmallVector<mlir::UnrealizedConversionCastOp> identityCasts;
        funcOp.getBody().walk([&identityCasts](mlir::UnrealizedConversionCastOp castOp) {
            if (castOp->getNumOperands() == 1 && castOp->getNumResults() == 1 &&
                castOp->getOperand(0).getType() == castOp->getResult(0).getType()) {
                identityCasts.push_back(castOp);
            }
        });
        for (auto castOp : identityCasts) {
            castOp.getResult(0).replaceAllUsesWith(castOp->getOperand(0));
            castOp.erase();
        }

        // Erase non-entry block arguments — their uses were replaced above.
        // Region::getArguments() covers only the entry block, so iterate all blocks explicitly.
        for (auto& block : funcOp.getBody()) {
            if (&block == &funcOp.getBody().front()) {
                continue;
            }
            for (unsigned i = block.getNumArguments(); i-- > 0;) {
                block.eraseArgument(i);
            }
        }

        mlir::OpBuilder builder(&getContext());
        builder.setInsertionPointToEnd(&funcSection.getContent().getBlocks().front());
        auto bytecodeFuncOp =
                bytecode::ExtFuncOp::create(builder, funcOp->getLoc(), funcOp.getName(), funcOp.getFunctionType());
        bytecodeFuncOp.getBody().takeBody(funcOp.getBody());

        for (auto arg : bytecodeFuncOp.getBody().getArguments() | reversed) {
            bytecodeFuncOp.getBody().eraseArgument(arg.getArgNumber());
        }
        funcOp->erase();

        return mlir::success();
    }
};

}  // namespace vpux

std::unique_ptr<mlir::Pass> vpux::bytecode::createConvertHostcodeToBytecodePass(const Logger& log) {
    return std::make_unique<ConvertHostcodeToBytecodePass>(log);
}
