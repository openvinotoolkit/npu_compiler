//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <utility>

namespace vpux::IE {
#define GEN_PASS_DECL_LEGALIZEREIFYRESULTSHAPESRESIDUALS
#define GEN_PASS_DEF_LEGALIZEREIFYRESULTSHAPESRESIDUALS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

mlir::Value convertIndexToSI64(mlir::PatternRewriter& rewriter, mlir::Value index) {
    auto ctx = rewriter.getContext();
    auto loc = index.getLoc();

    auto indexCast = rewriter.create<mlir::arith::IndexCastOp>(appendLoc(loc, "to_index"), getInt64Type(ctx), index);
    auto fromElements = rewriter.create<mlir::tensor::FromElementsOp>(appendLoc(loc, "to_i64"), indexCast.getResult());

    auto elemSize = getElemTypeSize(fromElements.getType());
    auto signedType = mlir::IntegerType::get(ctx, elemSize.count(), mlir::IntegerType::Signed);
    auto signedTensorType = mlir::RankedTensorType::get(getShape(fromElements), signedType);

    return rewriter.create<mlir::tensor::BitcastOp>(appendLoc(loc, "to_si64"), signedTensorType, fromElements);
}

mlir::Value convertSignedIntValueToIndex(mlir::PatternRewriter& rewriter, mlir::Value tensorSI) {
    auto ctx = rewriter.getContext();
    auto loc = tensorSI.getLoc();

    auto elemSize = getElemTypeSize(tensorSI.getType());
    auto signlessType = mlir::IntegerType::get(ctx, elemSize.count(), mlir::IntegerType::Signless);
    auto signlessTensorType = mlir::RankedTensorType::get(getShape(tensorSI), signlessType);

    auto bitCast = rewriter.create<mlir::tensor::BitcastOp>(appendLoc(loc, "to_si64"), signlessTensorType, tensorSI);

    auto extractOpLoc = appendLoc(loc, "to_i64");
    auto zeroIndexOp = rewriter.create<mlir::arith::ConstantIndexOp>(extractOpLoc, 0);
    auto toElementsOp = rewriter.create<mlir::tensor::ExtractOp>(extractOpLoc, signlessType, bitCast.getResult(),
                                                                 mlir::ValueRange{zeroIndexOp.getResult()});

    return rewriter.create<mlir::arith::IndexCastOp>(appendLoc(loc, "to_index"), rewriter.getIndexType(),
                                                     toElementsOp.getResult());
}

mlir::Value convertArithConstToConstantOp(mlir::PatternRewriter& rewriter, mlir::arith::ConstantOp constOp) {
    auto ctx = rewriter.getContext();

    auto constValue = constOp.getValue();
    if (auto constIntAttr = mlir::dyn_cast<mlir::IntegerAttr>(constValue)) {
        auto baseType = mlir::RankedTensorType::get({1}, getSInt64Type(ctx));
        auto values = SmallVector<int64_t>{constIntAttr.getInt()};
        return Const::createConst<int64_t>(rewriter, constOp.getLoc(), baseType, values);
    }

    VPUX_THROW("Failed to parse Constant op at '{0}'", constOp->getLoc());
}

using BuildOpFuncType = std::function<mlir::Operation*(const mlir::Location, const mlir::Value, const mlir::Value,
                                                       mlir::PatternRewriter&)>;

mlir::Operation* buildDivideOp(const mlir::Location loc, const mlir::Value lhs, const mlir::Value rhs,
                               mlir::PatternRewriter& rewriter) {
    return rewriter.create<IE::DivideOp>(appendLoc(loc, "div"), lhs, rhs, IE::AutoBroadcastType::NONE_OR_EXPLICIT);
}

mlir::Operation* buildAddOp(const mlir::Location loc, const mlir::Value lhs, const mlir::Value rhs,
                            mlir::PatternRewriter& rewriter) {
    return rewriter.create<IE::AddOp>(appendLoc(loc, "add"), lhs, rhs, IE::AutoBroadcastType::NONE_OR_EXPLICIT,
                                      /*post_op=*/nullptr,
                                      /*clamp=*/nullptr,
                                      /*outputPadding=*/nullptr,
                                      /*inputPadding=*/nullptr);
}

//
// TensorBitCastRewriter
//

class TensorDimRewriter final : public mlir::OpRewritePattern<mlir::tensor::DimOp> {
public:
    TensorDimRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::tensor::DimOp>(ctx), _log(std::move(log)) {
        this->setDebugName("TensorBitCastRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(mlir::tensor::DimOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TensorDimRewriter::matchAndRewrite(mlir::tensor::DimOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    auto indexOperand = origOp.getIndex();
    auto constant = indexOperand.getDefiningOp<mlir::arith::ConstantIndexOp>();
    if (constant == nullptr) {
        return mlir::failure();
    }

    auto ctx = origOp.getContext();
    auto loc = origOp.getLoc();

    auto indexValue = parseIntAttr<int64_t>(constant.getValue());

    auto shapeOfLoc = appendLoc(loc, "shape_of_for_dim_{0}", indexValue);
    auto shapeOfOp = rewriter.create<IE::ShapeOfOp>(shapeOfLoc, origOp.getSource(), getSInt64Type(ctx));

    auto offsets = getIntArrayAttr(ctx, SmallVector<int64_t>{indexValue});
    auto sizes = getIntArrayAttr(ctx, SmallVector<int64_t>{1});

    auto sliceLoc = appendLoc(loc, "dim_{0}", indexValue);
    auto sliceOp = rewriter.create<IE::SliceOp>(sliceLoc, shapeOfOp, offsets, sizes);

    auto index = convertSignedIntValueToIndex(rewriter, sliceOp.getResult());

    rewriter.replaceOp(origOp, index);

    return mlir::success();
}

//
// BinaryOpRewriter
//

template <typename BinaryOp>
class BinaryOpRewriter final : public mlir::OpRewritePattern<BinaryOp> {
public:
    BinaryOpRewriter(mlir::MLIRContext* ctx, Logger log, const BuildOpFuncType& builder)
            : mlir::OpRewritePattern<BinaryOp>(ctx), _log(std::move(log)), _builder(builder) {
        this->setDebugName("BinaryOpRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(BinaryOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    const BuildOpFuncType _builder = nullptr;
};

template <typename BinaryOp>
mlir::LogicalResult BinaryOpRewriter<BinaryOp>::matchAndRewrite(BinaryOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    auto loc = origOp.getLoc();

    auto operands = origOp.getOperands();
    VPUX_THROW_UNLESS(operands.size() == 2, "A binary op must have 2 operands, got '{0}'", operands.size());

    auto newOperands = SmallVector<mlir::Value>();
    for (auto operand : operands) {
        if (auto constOp = operand.template getDefiningOp<mlir::arith::ConstantOp>()) {
            auto newOperand = convertArithConstToConstantOp(rewriter, constOp);
            newOperands.push_back(newOperand);
        } else if (mlir::isa<mlir::IndexType>(operand.getType())) {
            auto newOperand = convertIndexToSI64(rewriter, operand);
            newOperands.push_back(newOperand);
        } else {
            VPUX_THROW("Unsupported operand type: '{0}'", operand.getType());
        }
    }

    auto newOp = _builder(loc, newOperands[0], newOperands[1], rewriter);
    auto index = convertSignedIntValueToIndex(rewriter, newOp->getResult(0));

    rewriter.replaceOp(origOp, index);

    return mlir::success();
}

class FoldFromElementsOpWithExtractOp final : public mlir::OpRewritePattern<mlir::tensor::FromElementsOp> {
public:
    using OpRewritePattern<mlir::tensor::FromElementsOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::tensor::FromElementsOp fromElements,
                                        mlir::PatternRewriter& rewriter) const final {
        if (fromElements.getNumOperands() != 1) {
            return mlir::failure();
        }

        auto extractOp = fromElements.getOperand(0).getDefiningOp<mlir::tensor::ExtractOp>();
        if (extractOp == nullptr) {
            return mlir::failure();
        }

        auto tensor = extractOp.getTensor();
        auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(tensor.getType());
        if (tensorType == nullptr || tensorType.getRank() != 1 || tensorType.getDimSize(0) != 1) {
            return mlir::failure();
        }
        // NOTE: extractOp.getIndices() must have 0th index or operation is malformed

        rewriter.replaceOp(fromElements, tensor);

        return mlir::success();
    }
};

//
// LegalizeReifyResultShapesResiduals
//

class LegalizeReifyResultShapesResiduals final :
        public IE::impl::LegalizeReifyResultShapesResidualsBase<LegalizeReifyResultShapesResiduals> {
public:
    explicit LegalizeReifyResultShapesResiduals(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void LegalizeReifyResultShapesResiduals::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<TensorDimRewriter>(&ctx, _log);
    patterns.add<BinaryOpRewriter<mlir::arith::DivSIOp>>(&ctx, _log, buildDivideOp);
    patterns.add<BinaryOpRewriter<mlir::arith::AddIOp>>(&ctx, _log, buildAddOp);

    mlir::arith::IndexCastOp::getCanonicalizationPatterns(patterns, &ctx);
    patterns.add<FoldFromElementsOpWithExtractOp>(&ctx);
    mlir::tensor::BitcastOp::getCanonicalizationPatterns(patterns, &ctx);

    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> IE::createLegalizeReifyResultShapesResidualsPass(Logger log) {
    return std::make_unique<LegalizeReifyResultShapesResiduals>(log);
}
