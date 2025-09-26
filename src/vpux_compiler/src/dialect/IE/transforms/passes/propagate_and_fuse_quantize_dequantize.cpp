//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEANDFUSEQUANTIZEDEQUANTIZE
#define GEN_PASS_DEF_PROPAGATEANDFUSEQUANTIZEDEQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseDequantizeWithMultiplier
//

class FuseDequantizeWithMultiplier final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    FuseDequantizeWithMultiplier(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
        setDebugName("FuseDequantizeWithMultiplier");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

std::optional<double> getMultiplierFromUser(IE::DequantizeOp dequantizeOp) {
    auto userOp = *dequantizeOp.getOutput().getUsers().begin();
    mlir::Value constInput;
    if (auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(userOp)) {
        if (multiplyOp.getPostOp() || multiplyOp.getClamp()) {
            return std::nullopt;
        }
        constInput =
                multiplyOp.getInput1() == dequantizeOp.getOutput() ? multiplyOp.getInput2() : multiplyOp.getInput1();
    } else if (auto dwConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(userOp)) {
        if (dwConvOp.getBias() || dwConvOp.getPostOp() || dwConvOp.getClamp()) {
            return std::nullopt;
        }
        if (!IE::isEltwiseGroupConv(dwConvOp)) {
            return std::nullopt;
        }
        if (dwConvOp.getInput() != dequantizeOp.getOutput()) {
            return std::nullopt;
        }

        // DW conv may have precision change.
        auto inType = mlir::dyn_cast<vpux::NDTypeInterface>(dwConvOp.getInput().getType());
        auto outType = mlir::dyn_cast<vpux::NDTypeInterface>(dwConvOp.getOutput().getType());
        if (inType.getElementType() != outType.getElementType()) {
            return std::nullopt;
        }

        constInput = dwConvOp.getFilter();
    } else {
        return std::nullopt;
    }

    auto constOp = mlir::dyn_cast_or_null<Const::DeclareOp>(constInput.getDefiningOp());
    if (constOp == nullptr || !IE::isBaseContentSplat(constOp)) {
        return std::nullopt;
    }

    return vpux::IE::getConst(constOp).front();
}

mlir::LogicalResult FuseDequantizeWithMultiplier::matchAndRewrite(IE::DequantizeOp dequantizeOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    if (!dequantizeOp->hasOneUse()) {
        return matchFailed(_log, rewriter, dequantizeOp, "dequantize has more users");
    }

    auto validMultiplier = getMultiplierFromUser(dequantizeOp);
    if (!validMultiplier.has_value()) {
        return matchFailed(_log, rewriter, dequantizeOp, "could not get multiplier from user");
    }

    auto multiplier = validMultiplier.value();
    auto inType = mlir::dyn_cast_or_null<vpux::NDTypeInterface>(dequantizeOp.getInput().getType());
    mlir::quant::QuantizedType dstType;

    if (const auto perTensorQuantileQType =
                mlir::dyn_cast<mlir::quant::QuantileQuantizedType>(inType.getElementType())) {
        auto scale = perTensorQuantileQType.getScale();
        scale *= multiplier;
        dstType = mlir::quant::QuantileQuantizedType::get(
                perTensorQuantileQType.getFlags(), perTensorQuantileQType.getStorageType(),
                perTensorQuantileQType.getQuantileType(), perTensorQuantileQType.getExpressedType(),
                perTensorQuantileQType.getQuantiles(), scale, perTensorQuantileQType.getZeroPoint(),
                perTensorQuantileQType.getStorageTypeMin(), perTensorQuantileQType.getStorageTypeMax());
    } else if (const auto perAxisQuantileQType =
                       mlir::dyn_cast_or_null<mlir::quant::QuantileQuantizedPerAxisType>(inType.getElementType())) {
        auto scales = perAxisQuantileQType.getScales();
        SmallVector<double> newScales(scales.size());
        std::transform(scales.begin(), scales.end(), newScales.begin(), [multiplier](double x) {
            return x * multiplier;
        });

        dstType = mlir::quant::QuantileQuantizedPerAxisType::get(
                perAxisQuantileQType.getFlags(), perAxisQuantileQType.getStorageType(),
                perAxisQuantileQType.getQuantileType(), perAxisQuantileQType.getExpressedType(),
                perAxisQuantileQType.getQuantiles(), newScales, perAxisQuantileQType.getZeroPoints(),
                perAxisQuantileQType.getQuantizedDimension(), perAxisQuantileQType.getStorageTypeMin(),
                perAxisQuantileQType.getStorageTypeMax());
    } else if (const auto uniformType =
                       mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(inType.getElementType())) {
        auto scale = uniformType.getScale();
        scale *= multiplier;
        dstType = mlir::quant::UniformQuantizedType::getChecked(
                dequantizeOp.getLoc(), uniformType.isSigned(), uniformType.getStorageType(),
                uniformType.getExpressedType(), scale, uniformType.getZeroPoint(), uniformType.getStorageTypeMin(),
                uniformType.getStorageTypeMax());
    } else if (const auto perAxisType =
                       mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(inType.getElementType())) {
        auto scales = perAxisType.getScales();
        SmallVector<double> newScales(scales.size());
        std::transform(scales.begin(), scales.end(), newScales.begin(), [multiplier](double x) {
            return x * multiplier;
        });
        dstType = mlir::quant::UniformQuantizedPerAxisType::getChecked(
                dequantizeOp.getLoc(), perAxisType.isSigned(), perAxisType.getStorageType(),
                perAxisType.getExpressedType(), newScales, perAxisType.getZeroPoints(),
                perAxisType.getQuantizedDimension(), perAxisType.getStorageTypeMin(), perAxisType.getStorageTypeMax());
    } else {
        return matchFailed(_log, rewriter, dequantizeOp, "unsupported quantize type");
    }

    auto quantizeCastOp = rewriter.create<IE::QuantizeCastOp>(appendLoc(dequantizeOp.getLoc(), "_quantizecast"),
                                                              dequantizeOp.getInput(), dstType);
    auto newDequantizeOp = rewriter.create<IE::DequantizeOp>(dequantizeOp.getLoc(), quantizeCastOp.getOutput(),
                                                             dequantizeOp.getDstElemType());
    auto userOp = *dequantizeOp.getOutput().getUsers().begin();
    rewriter.replaceOp(userOp, newDequantizeOp.getOutput());
    return mlir::success();
}

//
// PropagateQuantize
//

class PropagateQuantize final : public mlir::OpInterfaceRewritePattern<IE::ElemTypeInfoOpInterface> {
public:
    PropagateQuantize(mlir::MLIRContext* ctx, Logger log, bool seOpsEnabled)
            : mlir::OpInterfaceRewritePattern<IE::ElemTypeInfoOpInterface>(ctx),
              _log(log),
              _seOpsEnabled(seOpsEnabled) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ElemTypeInfoOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _seOpsEnabled;
};

/* This rewriter searches for pattern:
fp_tensor -> [ElemTypeInfoOpInterface] -> fp_tensor -> [Quantize]        -> quantized_tensor
and replaces it with
fp_tensor -> [Quantize] -> quantized_tensor -> [ElemTypeInfoOpInterface] -> quantized_tensor */
mlir::LogicalResult PropagateQuantize::matchAndRewrite(IE::ElemTypeInfoOpInterface origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("Got layer: {0}", origOp);
    auto layer = mlir::cast<IE::LayerOpInterface>(origOp.getOperation());

    // 1. Get the first quantizeOp.
    auto quantizeOp = mlir::dyn_cast<IE::QuantizeOp>(*(layer->getUsers().begin()));
    if (quantizeOp == nullptr) {
        return mlir::failure();
    }

    // 2. Check that every user is Quantize op ant they are the same.
    const auto isSameQuantize = [&](mlir::Operation* user) {
        if (auto currentQuantize = mlir::dyn_cast<IE::QuantizeOp>(user)) {
            return currentQuantize.getDstElemType() == quantizeOp.getDstElemType();
        }

        return false;
    };

    if (!llvm::all_of(layer->getUsers(), isSameQuantize)) {
        return mlir::failure();
    }

    // 3. Check that operation supports quantization params propagation.
    if (auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(origOp.getOperation());
        layerWithPostOp != nullptr && layerWithPostOp.getPostOp() != nullptr) {
        // A quantization-agnostic operation is no longer quantization-agnostic after it is fused with a post-op
        // (because post-op's are not quantization-agnostic). Since most post-op's will be fused by this time, this
        // check is here to prevent the propagation of output quantization through both the ElemTypeInfoOp and its
        // post-op. (At this time MaxPool seems to be the only operation which is both a IE::ElemTypeInfoOpInterface and
        // a IE::LayerWithPostOpInterface)
        return mlir::failure();
    }

    const auto quantizedElemType = mlir::cast<vpux::NDTypeInterface>(quantizeOp.getOutput().getType()).getElementType();
    auto elemTypeInfo = origOp.getElemTypeInfo();
    for (size_t outputInd = 0; outputInd < layer->getNumResults(); outputInd++) {
        elemTypeInfo.setOutput(outputInd, quantizedElemType);
    }

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    // 4. Particular check for SE pointers
    if (!vpux::IE::isSupportedElemTypeInfoCase(origOp.getOperation(), _seOpsEnabled, logCb)) {
        return mlir::failure();
    }

    origOp.inferElemTypeInfoUp(elemTypeInfo);

    if (!mlir::isa<mlir::quant::QuantizedType>(elemTypeInfo.getInput(0))) {
        return matchFailed(rewriter, origOp, "Operation does not support quantization params propagation");
    }

    for (size_t outputInd = 0; outputInd < layer->getNumResults(); outputInd++) {
        if (elemTypeInfo.getOutput(outputInd) != quantizedElemType) {
            return matchFailed(rewriter, origOp, "Operation does not support quantization params propagation");
        }
    }

    // All checks passed. Rewrite the sub-graph.
    rewriter.startOpModification(origOp);
    rewriter.setInsertionPoint(origOp);

    // 1. Create new Quantize ops, place them on each input of current operation.
    for (auto& operand : origOp->getOpOperands()) {
        auto newQuantize =
                rewriter.create<IE::QuantizeOp>(quantizeOp->getLoc(), operand.get(), elemTypeInfo.getInput(0));
        // Update input of Operation. NewQuant -> current Op.
        operand.set(newQuantize.getOutput());
    }

    // 2. Infer return types, set output type of operation to inferred quantized type.
    mlir::SmallVector<mlir::Type> inferredTypes;
    auto op = mlir::cast<mlir::InferTypeOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(
            op.inferReturnTypes(getContext(), op->getLoc(), origOp->getOperands(), op->getAttrDictionary(),  // operands
                                op->getPropertiesStorage(), op->getRegions(), inferredTypes)
                    .succeeded(),
            "New type inference failed for '{0}'", op);
    for (auto result : origOp->getResults()) {
        result.setType(inferredTypes[0]);
    }

    // 3. remove old Quantize ops.
    for (auto result : origOp->getResults()) {
        for (auto user : llvm::make_early_inc_range(result.getUsers())) {
            rewriter.replaceOp(user, result);
        }
    }

    // Rewrite done.
    rewriter.finalizeOpModification(origOp);
    return mlir::success();
}

//
// PropagateDequantize
//

class PropagateDequantize final : public mlir::OpInterfaceRewritePattern<IE::ElemTypeInfoOpInterface> {
public:
    PropagateDequantize(mlir::MLIRContext* ctx, Logger log, bool seOpsEnabled)
            : mlir::OpInterfaceRewritePattern<IE::ElemTypeInfoOpInterface>(ctx),
              _log(log),
              _seOpsEnabled(seOpsEnabled) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ElemTypeInfoOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _seOpsEnabled;
};

/* This rewriter searches for pattern:
quantized_tensor -> [Dequantize] -> fp_tensor -> [ElemTypeInfoOpInterface]                  -> fp_tensor
and replaces it with
quantized_tensor -> [ElemTypeInfoOpInterface] -> quantized_tensor(inferred) -> [Dequantize] -> fp_tensor */
mlir::LogicalResult PropagateDequantize::matchAndRewrite(IE::ElemTypeInfoOpInterface origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Got layer: {0}", origOp);

    auto layer = mlir::cast<IE::LayerOpInterface>(origOp.getOperation());

    // 1. All inputs are Dequantize ops with same destination element type
    SmallVector<IE::DequantizeOp> dequantizeOps;
    auto allInputsDequantize = llvm::all_of(layer.getInputs(), [&](mlir::Value input) {
        auto dequantizeOp = input.getDefiningOp<IE::DequantizeOp>();
        if (dequantizeOp == nullptr) {
            return false;
        }

        dequantizeOps.push_back(dequantizeOp);
        return true;
    });

    if (!allInputsDequantize) {
        return matchFailed(rewriter, origOp, "Not all inputs are Dequantize op");
    }

    auto firstDequantizeOp = dequantizeOps[0];
    auto differentDstElemType = llvm::any_of(drop_begin(dequantizeOps), [&](IE::DequantizeOp dequantizeOp) {
        return dequantizeOp.getDstElemType() != firstDequantizeOp.getDstElemType();
    });

    if (differentDstElemType) {
        return matchFailed(rewriter, origOp, "Dequantize inputs have different destination element type");
    }

    // 2. Check if operation supports quantization params propagation.
    if (auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(origOp.getOperation());
        layerWithPostOp != nullptr && layerWithPostOp.getPostOp() != nullptr) {
        // A quantization-agnostic operation is no longer quantization-agnostic after it is fused with a post-op
        // (because post-op's are not quantization-agnostic). Since most post-op's will be fused by this time, this
        // check is here to prevent the propagation of input quantization through both the ElemTypeInfoOp and its
        // post-op. (At this time MaxPool seems to be the only operation which is both a IE::ElemTypeInfoOpInterface and
        // a IE::LayerWithPostOpInterface)
        return mlir::failure();
    }

    auto elemTypeInfo = origOp.getElemTypeInfo();

    SmallVector<mlir::Type> originalTypes;
    for (auto idx : irange(dequantizeOps.size())) {
        auto dequantizeOp = dequantizeOps[idx];

        const auto quantizedElemType =
                mlir::cast<vpux::NDTypeInterface>(dequantizeOp.getInput().getType()).getElementType();
        elemTypeInfo.setInput(idx, quantizedElemType);
        originalTypes.push_back(quantizedElemType);
    }

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    // 3. Particular check for SE pointers
    if (!vpux::IE::isSupportedElemTypeInfoCase(origOp.getOperation(), _seOpsEnabled, logCb)) {
        return mlir::failure();
    }

    origOp.inferElemTypeInfo(elemTypeInfo);

    const auto typesAreOriginal = llvm::all_of(irange(originalTypes.size()), [&](size_t idx) {
        return elemTypeInfo.getInput(idx) == originalTypes[idx];
    });

    if (!typesAreOriginal) {
        return matchFailed(rewriter, origOp, "Operation does not support quantization params propagation");
    }

    for (size_t outputInd = 0; outputInd < layer->getNumResults(); outputInd++) {
        if (!mlir::isa<mlir::quant::QuantizedType>(elemTypeInfo.getOutput(outputInd))) {
            return matchFailed(rewriter, origOp, "Operation does not support quantization params propagation: {0}",
                               elemTypeInfo.getOutput(outputInd));
        }
    }

    // 4. Rewrite the sub-graph.
    rewriter.startOpModification(origOp);

    const auto inputs = origOp->getOpOperands();
    for (auto idx : irange(inputs.size())) {
        auto& input = inputs[idx];

        input.set(dequantizeOps[idx].getInput());
    }

    // infer return type
    mlir::SmallVector<mlir::Type> inferredTypes;
    auto op = mlir::cast<mlir::InferTypeOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(op.inferReturnTypes(getContext(), op->getLoc(), op->getOperands(), op->getAttrDictionary(),
                                          op->getPropertiesStorage(), op->getRegions(), inferredTypes)
                              .succeeded(),
                      "New type inference failed for '{0}'", op);

    for (unsigned int outputInd = 0; outputInd < layer->getNumResults(); outputInd++) {
        origOp->getResult(outputInd).setType(inferredTypes[outputInd]);

        auto output = origOp->getOpResult(outputInd);
        rewriter.setInsertionPointAfter(origOp);
        auto newLoc = appendLoc(origOp->getLoc(), "_propagated_Dequantize '{0}'", outputInd);
        auto newDequant = rewriter.create<IE::DequantizeOp>(newLoc, output, firstDequantizeOp.getDstElemType());
        _log.trace("Added new Dequantize op: '{0}' at index '{1}'", newDequant, outputInd);
        output.replaceAllUsesExcept(newDequant.getOutput(), llvm::SmallPtrSet<mlir::Operation*, 1>{newDequant});
        _log.trace("All uses of current layer have been replaced with new Dequantize op at index '{0}'", outputInd);
    }

    rewriter.finalizeOpModification(origOp);
    return mlir::success();
}

class PropagateAndFuseQuantizeDequantizePass final :
        public IE::impl::PropagateAndFuseQuantizeDequantizeBase<PropagateAndFuseQuantizeDequantizePass> {
public:
    explicit PropagateAndFuseQuantizeDequantizePass(const bool seOpsEnabled, Logger log): _seOpsEnabled(seOpsEnabled) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _seOpsEnabled;
};

mlir::LogicalResult PropagateAndFuseQuantizeDequantizePass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (seOpsEnabled.hasValue()) {
        _seOpsEnabled = seOpsEnabled.getValue();
    }

    return mlir::success();
}

void PropagateAndFuseQuantizeDequantizePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseDequantizeWithMultiplier>(&ctx, _log);
    patterns.add<PropagateQuantize>(&ctx, _log.nest(), _seOpsEnabled);
    patterns.add<PropagateDequantize>(&ctx, _log.nest(), _seOpsEnabled);

    auto config = getDefaultGreedyRewriteConfig();
    config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;
    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createPropagateAndFuseQuantizeDequantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateAndFuseQuantizeDequantizePass(const bool seOpsEnabled,
                                                                                   Logger log) {
    return std::make_unique<PropagateAndFuseQuantizeDequantizePass>(seOpsEnabled, log);
}
