//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

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
    mlir::FailureOr<mlir::Value> getPalletizedFilter(mlir::Value quantizedFilter, mlir::Type actType,
                                                     mlir::PatternRewriter& rewriter) const;

private:
    const IsLegalTensorElemFunctor _isLegalTensorElem;
    const ChangeWeightTypeToLUTFunctor _changeWeightTypeToLUT;
    Logger _log;
};

template <typename ConcreteOp>
mlir::FailureOr<mlir::Value> ConvolutionLUTRewriter<ConcreteOp>::getPalletizedFilter(
        mlir::Value quantizedFilter, mlir::Type actType, mlir::PatternRewriter& rewriter) const {
    const auto inputType = quantizedFilter.getType();
    const auto origQuantType =
            mlir::dyn_cast<mlir::quant::QuantizedType>(mlir::cast<vpux::NDTypeInterface>(inputType).getElementType());

    if (origQuantType == nullptr || _isLegalTensorElem(origQuantType)) {
        _log.nest().trace("Weight type '{0}' is already legal for the transformation", origQuantType);
        return mlir::failure();
    }

    const auto newQuantType = _changeWeightTypeToLUT(origQuantType, actType);
    if (newQuantType == nullptr) {
        return mlir::failure();
    }

    _log.nest().trace("QuantizeCast from '{0}' to '{1}'", origQuantType, newQuantType);

    auto quantizeCastOp = IE::QuantizeCastOp::create(
            rewriter, appendLoc(quantizedFilter.getLoc(), "quantize_cast_filter"), quantizedFilter, newQuantType);
    return quantizeCastOp.getOutput();
}

template <typename ConcreteOp>
mlir::LogicalResult ConvolutionLUTRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    // operation legality is already checked, so actType is legal and can be either fp16 or fp8
    auto actValue = origOp->getOperand(0);
    auto inputDequantOp = mlir::dyn_cast_if_present<IE::DequantizeOp>(actValue.getDefiningOp());
    const auto actType =
            inputDequantOp == nullptr
                    ? mlir::cast<vpux::NDTypeInterface>(actValue.getType()).getElementType()
                    : mlir::cast<vpux::NDTypeInterface>(inputDequantOp.getInput().getType()).getElementType();

    auto weightValue = origOp->getOperand(1);
    auto weightDefiningOp = weightValue.getDefiningOp();

    auto maybeFilterDequantOp = mlir::dyn_cast_if_present<IE::DequantizeOp>(weightDefiningOp);
    auto maybeFilterDynamicDequantOp = mlir::dyn_cast_if_present<IE::DynamicDequantizeOp>(weightDefiningOp);

    auto quantizedWeight = (maybeFilterDequantOp != nullptr || maybeFilterDynamicDequantOp != nullptr)
                                   ? weightDefiningOp->getOperand(0)
                                   : weightValue;

    auto palletizedWeight = getPalletizedFilter(quantizedWeight, actType, rewriter);
    if (mlir::failed(palletizedWeight)) {
        return mlir::failure();
    }

    mlir::Value mappedValue = palletizedWeight.value();
    if (maybeFilterDequantOp != nullptr) {
        mappedValue = IE::DequantizeOp::create(rewriter, takeOpLoc(origOp, "dequantize_filter"),
                                               palletizedWeight.value(), maybeFilterDequantOp.getDstElemTypeAttr());
    } else if (maybeFilterDynamicDequantOp != nullptr) {
        mappedValue = IE::DynamicDequantizeOp::create(rewriter, takeOpLoc(origOp, "dynamic_dequantize_filter"),
                                                      palletizedWeight.value(), maybeFilterDynamicDequantOp.getScale(),
                                                      maybeFilterDynamicDequantOp.getZp(),
                                                      maybeFilterDynamicDequantOp.getDstElemTypeAttr());
    }

    mlir::IRMapping mapper;
    mapper.map(weightValue, mappedValue);
    auto newOp = rewriter.clone(*origOp, mapper);
    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

}  // namespace IE
}  // namespace vpux
