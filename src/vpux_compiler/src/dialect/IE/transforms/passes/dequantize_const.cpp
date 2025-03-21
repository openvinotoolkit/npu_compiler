//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <algorithm>

namespace vpux::IE {
#define GEN_PASS_DECL_DEQUANTIZECONST
#define GEN_PASS_DEF_DEQUANTIZECONST
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {
bool isMixedPrecisionOp(mlir::Operation* op) {
    if (!mlir::isa_and_nonnull<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::MultiplyOp>(op)) {
        return false;
    }
    auto activationType = mlir::cast<NDTypeInterface>(op->getOperand(0).getType());
    auto weightsType = mlir::cast<NDTypeInterface>(op->getOperand(1).getType());
    return activationType.getElementType().isF16() && weightsType.getElementType().isF16();
}

bool isAsymmetricQuant(mlir::quant::QuantizedType quantType) {
    if (quantType == nullptr) {
        return false;
    }
    SmallVector<int64_t> zeroPoints = {};
    if (auto perAxis = quantType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        zeroPoints = to_small_vector(perAxis.getZeroPoints());
    } else if (auto perTensor = quantType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        zeroPoints.push_back(perTensor.getZeroPoint());
    }
    const auto qMin = quantType.getStorageTypeMin();
    const auto qMax = quantType.getStorageTypeMax();
    // u8: min = 0, max = 255
    // zp = (255 + 0 + 1) / 2 = 128
    // i8: min = -128, max = 127
    // zp = (127 + -128 + 1) / 2 = 0
    // u4: min = 0, max = 15
    // zp = (15 + 0 + 1) / 2 = 8
    // i4: min = -8, max = 7
    // zp = (7 + -8 + 1) / 2 = 0
    const int64_t targetZeroPoint = (qMax + qMin + 1) / 2;
    const auto isAsymmetricZeroPoint = [targetZeroPoint](const int64_t zp) -> bool {
        return zp != targetZeroPoint;
    };
    return std::any_of(zeroPoints.begin(), zeroPoints.end(), isAsymmetricZeroPoint);
}
};  // namespace

namespace {

//
// DequantizeConst
//

class DequantizeConst final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeConst(mlir::MLIRContext* ctx, bool _enableRuntimeDequantization, int64_t runtimeDequantizationLimit,
                    Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx),
              _log(log),
              _enableRuntimeDequantization{_enableRuntimeDequantization},
              _runtimeDequantizationLimit{runtimeDequantizationLimit} {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp dCastOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _enableRuntimeDequantization;
    int64_t _runtimeDequantizationLimit;
};

mlir::LogicalResult DequantizeConst::matchAndRewrite(IE::DequantizeOp dCastOp, mlir::PatternRewriter& rewriter) const {
    auto inputConst = dCastOp.getInput().getDefiningOp<Const::DeclareOp>();
    if (inputConst == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got DequantizeCast Operation '{0}' with Constant input '{1}'", dCastOp->getLoc(), inputConst.getLoc());

    auto isBias = [&](mlir::Operation* op) {
        if (!mlir::isa_and_nonnull<IE::ConvolutionOp, IE::GroupConvolutionOp>(op)) {
            return false;
        }
        return op->getNumOperands() > 2 && op->getOperand(2) == dCastOp.getResult();
    };

    const auto qType = inputConst.getType().cast<vpux::NDTypeInterface>();
    const auto qElemType = qType.getElementType().cast<mlir::quant::QuantizedType>();
    auto users = dCastOp.getOutput().getUsers();
    if (_enableRuntimeDequantization && isAsymmetricQuant(qElemType) && qElemType.getStorageTypeIntegralWidth() == 8 &&
        std::all_of(users.begin(), users.end(), isMixedPrecisionOp) &&
        !std::any_of(users.begin(), users.end(), isBias) &&
        qType.getTotalAllocSize().count() > _runtimeDequantizationLimit) {
        // _runtimeDequantizationLimit is configured by flag default is 512 kb
        _log.trace("Keeping dequantize for asymmetric weights");
        return mlir::failure();
    }

    const auto outType = dCastOp.getType().cast<vpux::NDTypeInterface>();
    const auto newConstType = outType.changeElemType(qElemType.getExpressedType());
    auto newConstAttr = inputConst.transformContentAttr().dequantize().get();
    rewriter.replaceOpWithNewOp<Const::DeclareOp>(dCastOp, newConstType, std::move(newConstAttr))
            ->setLoc(inputConst->getLoc());

    return mlir::success();
}

//
// DequantizeConstPass
//

class DequantizeConstPass final : public IE::impl::DequantizeConstBase<DequantizeConstPass> {
public:
    explicit DequantizeConstPass(int64_t runtimeDequantizationLimit, bool enableRuntimeDequantization, Logger log)
            : _enableRuntimeDequantization(enableRuntimeDequantization),
              _runtimeDequantizationLimit{runtimeDequantizationLimit} {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    bool _enableRuntimeDequantization;
    int64_t _runtimeDequantizationLimit;
};

mlir::LogicalResult DequantizeConstPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (enableRuntimeDequant.hasValue()) {
        _enableRuntimeDequantization = enableRuntimeDequant.getValue();
    }
    if (runtimeDequantizationLimit.hasValue()) {
        _runtimeDequantizationLimit = runtimeDequantizationLimit.getValue();
    }

    return mlir::success();
}

void DequantizeConstPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DequantizeConst>(&ctx, _enableRuntimeDequantization, _runtimeDequantizationLimit, _log);
    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDequantizeConstPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDequantizeConstPass(const int64_t runtimeDequantizationLimit,
                                                                bool enableRuntimeDequantization, Logger log) {
    return std::make_unique<DequantizeConstPass>(runtimeDequantizationLimit, enableRuntimeDequantization, log);
}
