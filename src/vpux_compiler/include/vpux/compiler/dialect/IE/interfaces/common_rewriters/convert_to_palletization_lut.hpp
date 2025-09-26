//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>

namespace vpux {
namespace IE {

// Returns false when type conversion is required;
bool isLegalTensorElemForPalletization(mlir::Type elementType, const bool convertOnlyAsymmetric,
                                       const bool allowPerChannelZp);

//
// ConvolutionLUTRewriter
//

using IsLegalTensorElemFunctor = std::function<bool(mlir::Type)>;
using ChangeWeightTypeToLUTFunctor = std::function<mlir::quant::QuantizedType(mlir::quant::QuantizedType, mlir::Type)>;

template <typename ConcreteOp>
class ConvolutionLUTRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ConvolutionLUTRewriter(mlir::MLIRContext* ctx, const IsLegalTensorElemFunctor& isLegal,
                           const ChangeWeightTypeToLUTFunctor& changeWgtType, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx),
              _isLegalTensorElem(isLegal),
              _changeWeightTypeToLUT(changeWgtType),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;
    mlir::FailureOr<mlir::Value> getPalletizedFilter(IE::DequantizeOp filterDequantOp, mlir::Type actType,
                                                     mlir::PatternRewriter& rewriter) const;

private:
    const IsLegalTensorElemFunctor _isLegalTensorElem;
    const ChangeWeightTypeToLUTFunctor _changeWeightTypeToLUT;
    Logger _log;
};

template <typename ConcreteOp>
mlir::FailureOr<mlir::Value> ConvolutionLUTRewriter<ConcreteOp>::getPalletizedFilter(
        IE::DequantizeOp filterDequantOp, mlir::Type actType, mlir::PatternRewriter& rewriter) const {
    const auto inputType = filterDequantOp.getInput().getType();
    const auto origQuantType =
            mlir::dyn_cast<mlir::quant::QuantizedType>(mlir::cast<vpux::NDTypeInterface>(inputType).getElementType());

    if (_isLegalTensorElem(origQuantType)) {
        _log.nest().trace("Weight type '{0}' is already legal for the transformation", origQuantType);
        return mlir::failure();
    }

    const auto newType =
            mlir::cast<vpux::NDTypeInterface>(inputType).changeElemType(_changeWeightTypeToLUT(origQuantType, actType));
    const auto newQuantType = mlir::cast<mlir::quant::QuantizedType>(newType.getElementType());

    _log.nest().trace("QuantizeCast from '{0}' to '{1}' for non const", origQuantType, newQuantType);

    auto quantizeCastOp =
            rewriter.create<IE::QuantizeCastOp>(filterDequantOp.getLoc(), filterDequantOp.getInput(), newQuantType);
    return quantizeCastOp.getOutput();
}

template <typename ConcreteOp>
mlir::LogicalResult ConvolutionLUTRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    // operation legality is already checked, so actType is legal and can be either fp16 or fp8
    auto inputDequantOp = mlir::dyn_cast_or_null<IE::DequantizeOp>(origOp.getInput().getDefiningOp());
    const auto actType =
            inputDequantOp == nullptr
                    ? mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getElementType()
                    : mlir::cast<vpux::NDTypeInterface>(inputDequantOp.getInput().getType()).getElementType();

    auto filterDequantOp = origOp.getFilter().template getDefiningOp<IE::DequantizeOp>();
    auto validFilter = getPalletizedFilter(filterDequantOp, actType, rewriter);
    if (mlir::failed(validFilter)) {
        return mlir::failure();
    }

    auto newDequantizeOp = rewriter.create<IE::DequantizeOp>(origOp->getLoc(), validFilter.value(),
                                                             filterDequantOp.getDstElemTypeAttr());
    mlir::IRMapping mapper;
    mapper.map(origOp.getFilter(), newDequantizeOp);
    auto newOp = rewriter.clone(*origOp, mapper);
    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

}  // namespace IE
}  // namespace vpux
