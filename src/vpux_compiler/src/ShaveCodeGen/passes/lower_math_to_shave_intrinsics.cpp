//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"

#include "vpux/utils/logger/logger.hpp"

#include "vpux/compiler/ShaveCodeGen/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/TypeSwitch.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_LOWERMATHTOSHAVEINTRINSICS
#define GEN_PASS_DEF_LOWERMATHTOSHAVEINTRINSICS
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

struct BuiltinsCache {
    mlir::func::FuncOp tanhIntrinsicF16 = nullptr;
    mlir::func::FuncOp tanhLibcallF32 = nullptr;
    mlir::func::FuncOp atanIntrinsicF16 = nullptr;
    mlir::func::FuncOp atanLibcallF32 = nullptr;
};

//
// LowerMathToShaveIntrinsicsBase
//

class LowerMathToShaveIntrinsicsPass final :
        public ShaveCodeGen::impl::LowerMathToShaveIntrinsicsBase<LowerMathToShaveIntrinsicsPass> {
public:
    explicit LowerMathToShaveIntrinsicsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

class TanhOpLowering : public mlir::OpRewritePattern<mlir::math::TanhOp> {
public:
    TanhOpLowering(mlir::MLIRContext* context, BuiltinsCache& bCache)
            : mlir::OpRewritePattern<mlir::math::TanhOp>(context), bCache(bCache) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::math::TanhOp op, mlir::PatternRewriter& rewriter) const final {
        if (mlir::dyn_cast<mlir::VectorType>(op.getType()) || mlir::dyn_cast<mlir::TensorType>(op.getType())) {
            return rewriter.notifyMatchFailure(op, "non-scalar operations are not supported");
        }
        auto elementType = op.getResult().getType();
        auto swModule = op->getParentOfType<mlir::ModuleOp>();
        mlir::func::FuncOp funcOp = getTanh(elementType, swModule);

        if (!funcOp) {
            return rewriter.notifyMatchFailure(op, "unsupported element type");
        }
        rewriter.replaceOpWithNewOp<mlir::func::CallOp>(op, funcOp, op.getOperand());
        return mlir::success();
    }

private:
    mlir::func::FuncOp getTanhFunc(mlir::Type elementType, mlir::ModuleOp swModule) const {
        constexpr StringRef f16Name = "llvm.shave.sau.tanh.f16.l.r";
        constexpr StringRef f32Name = "tanhf";

        mlir::OpBuilder builder(getContext());
        builder.setInsertionPointToEnd(&swModule->getRegion(0).front());
        StringRef funcName = elementType.isF16() ? f16Name : f32Name;
        mlir::FunctionType funcType = mlir::FunctionType::get(builder.getContext(), {elementType}, elementType);
        auto funcOp = builder.create<mlir::func::FuncOp>(swModule.getLoc(), funcName, funcType,
                                                         mlir::StringAttr::get(builder.getContext(), "private"),
                                                         nullptr, nullptr);
        funcOp->setAttr(ShaveCodeGen::IntrinsicAttrName, mlir::UnitAttr::get(builder.getContext()));
        return funcOp;
    }

    mlir::func::FuncOp getTanh(mlir::Type elementType, mlir::ModuleOp swModule) const {
        if (elementType.isF16()) {
            if (bCache.tanhIntrinsicF16 != nullptr) {
                return bCache.tanhIntrinsicF16;
            }
            bCache.tanhIntrinsicF16 = getTanhFunc(elementType, swModule);
            return bCache.tanhIntrinsicF16;
        }
        if (elementType.isF32()) {
            if (bCache.tanhLibcallF32 != nullptr) {
                return bCache.tanhLibcallF32;
            }
            bCache.tanhLibcallF32 = getTanhFunc(elementType, swModule);
            return bCache.tanhLibcallF32;
        }

        return nullptr;
    }

    BuiltinsCache& bCache;
};

class AtanOpLowering : public mlir::OpRewritePattern<mlir::math::AtanOp> {
public:
    AtanOpLowering(mlir::MLIRContext* context, BuiltinsCache& bCache)
            : mlir::OpRewritePattern<mlir::math::AtanOp>(context), bCache(bCache) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::math::AtanOp op, mlir::PatternRewriter& rewriter) const final {
        if (mlir::dyn_cast<mlir::VectorType>(op.getType()) || mlir::dyn_cast<mlir::TensorType>(op.getType())) {
            return rewriter.notifyMatchFailure(op, "non-scalar operations are not supported");
        }
        auto elementType = op.getResult().getType();
        auto swModule = op->getParentOfType<mlir::ModuleOp>();
        mlir::func::FuncOp funcOp = getAtan(elementType, swModule);

        if (!funcOp) {
            return rewriter.notifyMatchFailure(op, "unsupported element type");
        }
        if (elementType.isF16()) {
            // Only for fp16, range reduction for [-1,1] and Atan
            auto loc = op->getLoc();
            auto input = op.getOperand();

            //
            //  atan(x)=| atan(x),                 -1 <= x <= 1
            //          | sgn(x)(pi/2-atan(1/|x|),    |x| > 1
            //

            auto zeroConst = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(elementType, 0.0));
            auto halfpiConst = rewriter.create<mlir::arith::ConstantOp>(
                    loc, rewriter.getFloatAttr(elementType, llvm::numbers::pi / 2));
            auto oneConst = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(elementType, 1.0));
            auto m_oneConst = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(elementType, -1.0));

            auto negX = rewriter.create<mlir::arith::SubFOp>(loc, zeroConst, input);
            auto cmp0 = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, input, zeroConst);
            auto abs = rewriter.create<mlir::arith::SelectOp>(loc, cmp0, negX, input);
            auto inv = rewriter.create<mlir::arith::DivFOp>(loc, oneConst, abs);
            auto cmp1 = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, abs, oneConst);
            auto sel0 = rewriter.create<mlir::arith::SelectOp>(loc, cmp1, inv, input);
            auto atn = rewriter.create<mlir::func::CallOp>(loc, funcOp, mlir::ValueRange{sel0.getResult()});
            mlir::Value atanVal = atn.getResult(0);
            auto halfPiMinus = rewriter.create<mlir::arith::SubFOp>(loc, halfpiConst, atanVal);
            auto sign = rewriter.create<mlir::arith::SelectOp>(loc, cmp0, m_oneConst, oneConst);
            auto scaled = rewriter.create<mlir::arith::MulFOp>(loc, halfPiMinus, sign);
            auto res = rewriter.create<mlir::arith::SelectOp>(loc, cmp1, scaled, atanVal);

            rewriter.replaceOp(op, res.getResult());
        } else {
            rewriter.replaceOpWithNewOp<mlir::func::CallOp>(op, funcOp, op.getOperand());
        }

        return mlir::success();
    }

private:
    mlir::func::FuncOp getAtanFunc(mlir::Type elementType, mlir::ModuleOp swModule) const {
        constexpr StringRef f16Name = "llvm.shave.sau.atn.f16.l.r";
        constexpr StringRef f32Name = "atanf";

        mlir::OpBuilder builder(getContext());
        builder.setInsertionPointToEnd(&swModule->getRegion(0).front());
        StringRef funcName = elementType.isF16() ? f16Name : f32Name;
        mlir::FunctionType funcType = mlir::FunctionType::get(builder.getContext(), {elementType}, elementType);
        auto funcOp = builder.create<mlir::func::FuncOp>(swModule.getLoc(), funcName, funcType,
                                                         mlir::StringAttr::get(builder.getContext(), "private"),
                                                         nullptr, nullptr);
        funcOp->setAttr(ShaveCodeGen::IntrinsicAttrName, mlir::UnitAttr::get(builder.getContext()));
        return funcOp;
    }

    mlir::func::FuncOp getAtan(mlir::Type elementType, mlir::ModuleOp swModule) const {
        if (elementType.isF16()) {
            if (bCache.atanIntrinsicF16 != nullptr) {
                return bCache.atanIntrinsicF16;
            }
            bCache.atanIntrinsicF16 = getAtanFunc(elementType, swModule);
            return bCache.atanIntrinsicF16;
        }
        if (elementType.isF32()) {
            if (bCache.atanLibcallF32 != nullptr) {
                return bCache.atanLibcallF32;
            }
            bCache.atanLibcallF32 = getAtanFunc(elementType, swModule);
            return bCache.atanLibcallF32;
        }

        return nullptr;
    }

    BuiltinsCache& bCache;
};

void LowerMathToShaveIntrinsicsPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto _swModule = VPUIP::getVPUSWModule(moduleOp, _log);
    auto& ctx = getContext();

    BuiltinsCache bCache;

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<TanhOpLowering>(&ctx, bCache);
    patterns.add<AtanOpLowering>(&ctx, bCache);

    if (mlir::failed(mlir::applyPatternsGreedily(_swModule, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
        return;
    }
}

}  // namespace

//
// createLowerMathToShaveIntrinsicsPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createLowerMathToShaveIntrinsicsPass(Logger log) {
    return std::make_unique<LowerMathToShaveIntrinsicsPass>(log);
}
