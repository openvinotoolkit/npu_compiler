//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/impl/weights_dequantize_to_dynamic_dequantize_strategy.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

using namespace vpux;

namespace {

class WeightsDequantizeToDynamicDequantizeConstRewriter final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    WeightsDequantizeToDynamicDequantizeConstRewriter(mlir::MLIRContext* ctx, Logger log,
                                                      mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<Const::DeclareOp>(ctx, benefit), _log(log) {
        setDebugName("WeightsDequantizeToDynamicDequantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("Got {0} at `{1}`.", origOp->getName(), origOp->getLoc());

        auto maybeWdInfo = IE::WeightsDequantizeStructureInfo::create(origOp, _log.nest());
        if (mlir::failed(maybeWdInfo)) {
            _log.trace("Failed to match WeightsDequantize structure");
            return mlir::failure();
        }
        auto wdInfo = maybeWdInfo.value();
        if (wdInfo.getDynamicScale()) {
            _log.trace("Can't create DynamicDequantize with dynamic scale");
            return mlir::failure();
        }
        if (!wdInfo.isKVcachedPattern() && IE::getTrueElemType(origOp).isInteger(2)) {
            // Force to use DynamicDequantize for u2 WaC groupwise prefill model
            _log.trace("Got u2 weights-as-constant groupwise prefill pattern.");
            return mlir::failure();
        }

        const auto users = origOp->getUsers();
        if (users.empty()) {
            _log.trace("No users of the matched DeclareOp");
            return mlir::failure();
        }

        auto chainOps = wdInfo.getOpChain();
        auto multiplyOpIt = std::find_if(chainOps.begin(), chainOps.end(), [](mlir::Operation* op) {
            return mlir::dyn_cast<IE::MultiplyOp>(op);
        });
        if (multiplyOpIt == chainOps.end()) {
            _log.trace("No MultiplyOp in the matched structure");
            return mlir::failure();
        }
        auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(*multiplyOpIt);
        if (!multiplyOp) {
            _log.trace("Failed to cast to MultiplyOp");
            return mlir::failure();
        }

        const auto loc = multiplyOp.getLoc();

        const auto baseInputElemType = origOp.getContentAttr().getBaseContent().getElementType();

        const auto isUnsigned = baseInputElemType.isUnsignedInteger();

        auto typeRange = getStorageParams(baseInputElemType);
        if (mlir::failed(typeRange)) {
            return mlir::failure();
        }
        const auto minValue = static_cast<int64_t>(std::get<0>(*typeRange));
        const auto maxValue = static_cast<int64_t>(std::get<1>(*typeRange));

        const auto quantElemType = mlir::quant::UniformQuantizedType::get(
                (isUnsigned) ? 0 : mlir::quant::QuantizationFlags::Signed, baseInputElemType, rewriter.getF16Type(),
                1.0, 0, minValue, maxValue);

        const auto newTensorType =
                mlir::dyn_cast<vpux::NDTypeInterface>(origOp.getType()).changeElemType(quantElemType);

        auto newContentAttr = origOp.getContentAttr().transform().castElemType(quantElemType).get();

        auto newDeclareOp = rewriter.create<Const::DeclareOp>(loc, newTensorType, newContentAttr);

        const auto dstType = mlir::dyn_cast<vpux::NDTypeInterface>(multiplyOp.getOutput().getType()).getElementType();

        auto dynamicDequant = rewriter.create<IE::DynamicDequantizeOp>(appendLoc(loc, "ddq"), newDeclareOp,
                                                                       wdInfo.getStaticScale(), nullptr, dstType);

        rewriter.replaceAllOpUsesWith(multiplyOp, dynamicDequant.getResult());

        return mlir::success();
    }

private:
    Logger _log;
};

}  // namespace

//
// WeightsDequantizeToDynamicDequantizeStrategy
//

void IE::arch50xx::WeightsDequantizeToDynamicDequantizeStrategy::registerRewriters(RewriterRegistry& registry,
                                                                                   Logger& log) const {
    registry.registerRewriterSet("weights-dequantize-to-dynamic-dequantize", [&]() {
        registry.registerRewriter<WeightsDequantizeToDynamicDequantizeConstRewriter>(
                "weights-dequantize-to-dynamic-dequantize-const", log, _benefitLevels[_index]);
        IE::registerConvertOpRewriters(registry);
    });
}
