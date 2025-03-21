//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_MERGEFAKEQUANT
#define GEN_PASS_DEF_MERGEFAKEQUANT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// MergeQuantDequant
//

class MergeQuantDequant final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    MergeQuantDequant(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp dequantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MergeQuantDequant::matchAndRewrite(IE::DequantizeOp dequantizeOp,
                                                       mlir::PatternRewriter& rewriter) const {
    auto quantizeOp = dequantizeOp.getInput().getDefiningOp<IE::QuantizeOp>();
    if (quantizeOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got Quantize ('{0}') -> Dequantize ('{1}') pair", quantizeOp.getLoc(), dequantizeOp.getLoc());

    const auto quantizeType = dequantizeOp.getInput().getType().cast<vpux::NDTypeInterface>();

    int64_t levels = 0;
    mlir::RankedTensorType attrType;
    mlir::DenseElementsAttr rMinAttr, rMaxAttr;
    getFakeQuantParams(quantizeType, levels, attrType, rMinAttr, rMaxAttr);

    mlir::IntegerAttr levelsAttr = nullptr;
    mlir::TypeAttr lowFpTypeAttr = nullptr;

    if (const auto quantizeStorageType =
                quantizeType.getElementType().dyn_cast<mlir::quant::QuantizedType>().getStorageType();
        quantizeStorageType.isa<mlir::Float8E4M3FNType>() || quantizeStorageType.isa<mlir::Float8E5M2Type>()) {
        lowFpTypeAttr = mlir::TypeAttr::get(quantizeStorageType);
    } else {
        levelsAttr = getIntAttr(dequantizeOp.getContext(), levels);
    }

    auto rMinOp = rewriter.create<Const::DeclareOp>(dequantizeOp.getLoc(), attrType, Const::ContentAttr::get(rMinAttr));
    auto rMaxOp = rewriter.create<Const::DeclareOp>(dequantizeOp.getLoc(), attrType, Const::ContentAttr::get(rMaxAttr));

    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(dequantizeOp, quantizeOp.getInput(), rMinOp.getOutput(),
                                                    rMaxOp.getOutput(), rMinOp.getOutput(), rMaxOp.getOutput(),
                                                    levelsAttr, lowFpTypeAttr, IE::AutoBroadcastType::NUMPY);

    return mlir::success();
}

//
// MergeQuantCastDequant
//

class MergeQuantCastDequant final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    MergeQuantCastDequant(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp dequantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MergeQuantCastDequant::matchAndRewrite(IE::DequantizeOp dequantizeOp,
                                                           mlir::PatternRewriter& rewriter) const {
    auto quantizeCastOp = dequantizeOp.getInput().getDefiningOp<IE::QuantizeCastOp>();
    if (quantizeCastOp == nullptr) {
        return mlir::failure();
    }

    auto quantizeOp = quantizeCastOp.getInput().getDefiningOp<IE::QuantizeOp>();
    if (quantizeOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got Quantize ('{0}') -> QuantizeCast ('{1}') -> Dequantize ('{2}') ops", quantizeOp.getLoc(),
               quantizeCastOp.getLoc(), dequantizeOp.getLoc());

    const auto inputQuantizeType = quantizeCastOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outputQuantizeCastType = dequantizeOp.getInput().getType().cast<vpux::NDTypeInterface>();

    int64_t inLevels = 0, outLevels = 0;
    mlir::RankedTensorType inAttrType, outAttrType;
    mlir::DenseElementsAttr inMinAttr, inMaxAttr, outMinAttr, outMaxAttr;
    getFakeQuantParams(inputQuantizeType, inLevels, inAttrType, inMinAttr, inMaxAttr);
    getFakeQuantParams(outputQuantizeCastType, outLevels, outAttrType, outMinAttr, outMaxAttr);

    mlir::IntegerAttr levelsAttr = nullptr;
    mlir::TypeAttr lowFpTypeAttr = nullptr;

    if (const auto quantizeStorageType =
                outputQuantizeCastType.getElementType().dyn_cast<mlir::quant::QuantizedType>().getStorageType();
        quantizeStorageType.isa<mlir::Float8E4M3FNType>() || quantizeStorageType.isa<mlir::Float8E5M2Type>()) {
        lowFpTypeAttr = mlir::TypeAttr::get(quantizeStorageType);
    } else {
        if (inLevels != outLevels) {
            return mlir::failure();
        }

        levelsAttr = getIntAttr(dequantizeOp.getContext(), inLevels);
    }

    auto inMinOp =
            rewriter.create<Const::DeclareOp>(dequantizeOp.getLoc(), inAttrType, Const::ContentAttr::get(inMinAttr));
    auto inMaxOp =
            rewriter.create<Const::DeclareOp>(dequantizeOp.getLoc(), inAttrType, Const::ContentAttr::get(inMaxAttr));
    auto outMinOp =
            rewriter.create<Const::DeclareOp>(dequantizeOp.getLoc(), outAttrType, Const::ContentAttr::get(outMinAttr));
    auto outMaxOp =
            rewriter.create<Const::DeclareOp>(dequantizeOp.getLoc(), outAttrType, Const::ContentAttr::get(outMaxAttr));

    // lowFpType in not needed (nullptr), only levels are given
    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(dequantizeOp, quantizeOp.getInput(), inMinOp.getOutput(),
                                                    inMaxOp.getOutput(), outMinOp.getOutput(), outMaxOp.getOutput(),
                                                    levelsAttr, lowFpTypeAttr, IE::AutoBroadcastType::NUMPY);

    return mlir::success();
}

//
// MergeFakeQuantPass
//

class MergeFakeQuantPass final : public IE::impl::MergeFakeQuantBase<MergeFakeQuantPass> {
public:
    explicit MergeFakeQuantPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void MergeFakeQuantPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MergeQuantDequant>(&ctx, _log);
    patterns.add<MergeQuantCastDequant>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMergeFakeQuantPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMergeFakeQuantPass(Logger log) {
    return std::make_unique<MergeFakeQuantPass>(log);
}
