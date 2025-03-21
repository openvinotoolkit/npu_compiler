//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEQUANTIZEDEQUANTIZE
#define GEN_PASS_DEF_PROPAGATEQUANTIZEDEQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

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
        layerWithPostOp != nullptr && layerWithPostOp.getPostOp().has_value()) {
        // A quantization-agnostic operation is no longer quantization-agnostic after it is fused with a post-op
        // (because post-op's are not quantization-agnostic). Since most post-op's will be fused by this time, this
        // check is here to prevent the propagation of output quantization through both the ElemTypeInfoOp and its
        // post-op. (At this time MaxPool seems to be the only operation which is both a IE::ElemTypeInfoOpInterface and
        // a IE::LayerWithPostOpInterface)
        return mlir::failure();
    }

    const auto quantizedElemType = quantizeOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType();
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

    if (!elemTypeInfo.getInput(0).isa<mlir::quant::QuantizedType>()) {
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
        layerWithPostOp != nullptr && layerWithPostOp.getPostOp().has_value()) {
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

        const auto quantizedElemType = dequantizeOp.getInput().getType().cast<vpux::NDTypeInterface>().getElementType();
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
        if (!elemTypeInfo.getOutput(outputInd).isa<mlir::quant::QuantizedType>()) {
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

class PropagateQuantizeDequantizePass final :
        public IE::impl::PropagateQuantizeDequantizeBase<PropagateQuantizeDequantizePass> {
public:
    explicit PropagateQuantizeDequantizePass(const bool seOpsEnabled, Logger log): _seOpsEnabled(seOpsEnabled) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _seOpsEnabled;
};

mlir::LogicalResult PropagateQuantizeDequantizePass::initialize(mlir::MLIRContext* ctx) {
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

void PropagateQuantizeDequantizePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PropagateQuantize>(&ctx, _log.nest(), _seOpsEnabled);
    patterns.add<PropagateDequantize>(&ctx, _log.nest(), _seOpsEnabled);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createPropagateQuantizeDequantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateQuantizeDequantizePass(const bool seOpsEnabled, Logger log) {
    return std::make_unique<PropagateQuantizeDequantizePass>(seOpsEnabled, log);
}
