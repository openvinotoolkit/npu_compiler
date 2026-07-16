//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/propagate_and_fuse_quantize_dequantize.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEANDFUSEQUANTIZEDEQUANTIZE
#define GEN_PASS_DEF_PROPAGATEANDFUSEQUANTIZEDEQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

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

bool isValidToPropagateQuantize(mlir::Operation* op, bool seOpsEnabled, mlir::Type& quantizedElemType, Logger log) {
    const auto isSameQuantize = [&](mlir::Operation* user) {
        if (auto currentQuantize = mlir::dyn_cast<IE::QuantizeOp>(user)) {
            return currentQuantize.getDstElemType() == quantizedElemType;
        }

        return false;
    };
    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };
    // 1. Check the prevOp is ElemTypeInfoOp
    auto elemTypeInfoOp = mlir::dyn_cast<IE::ElemTypeInfoOpInterface>(op);
    if (!elemTypeInfoOp) {
        log.trace("Not an ElemTypeInfoOp {0}", op->getResult(0));
        return false;
    }
    // 2. Check that every user is Quantize op and they are the same.
    auto layer = mlir::cast<IE::LayerOpInterface>(op);
    // Only check quantize dst element type for multiple users
    // Direct case like op1 -> op2 passed.
    if (!layer->hasOneUse() && !llvm::all_of(layer->getUsers(), isSameQuantize)) {
        log.trace("The users of Operation {0} should all be the same quantizeOp when users number > 1", elemTypeInfoOp);
        return false;
    }
    // 3. Check that operation supports quantization params propagation.
    if (auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(elemTypeInfoOp.getOperation());
        layerWithPostOp != nullptr && layerWithPostOp.hasPPE()) {
        // A quantization-agnostic operation is no longer quantization-agnostic after it is fused with a post-op
        // (because post-op's are not quantization-agnostic). Since most post-op's will be fused by this time, this
        // check is here to prevent the propagation of output quantization through both the ElemTypeInfoOp and its
        // post-op. (At this time MaxPool seems to be the only operation which is both a IE::ElemTypeInfoOpInterface
        // and a IE::LayerWithPostOpInterface)
        log.trace("Operation {0} does not support quantization params propagation: layer has post op", elemTypeInfoOp);
        return false;
    }

    // 4. Particular check for SE pointers
    if (!vpux::IE::isSupportedElemTypeInfoCase(elemTypeInfoOp.getOperation(), seOpsEnabled, logCb)) {
        log.trace("Operation {0} does not support SE pointers", elemTypeInfoOp);
        return false;
    }

    // 5. Check quantization params propagation
    auto elemTypeInfo = elemTypeInfoOp.getElemTypeInfo();
    for (size_t outputInd = 0; outputInd < layer->getNumResults(); outputInd++) {
        elemTypeInfo.setOutput(outputInd, quantizedElemType);
    }
    elemTypeInfoOp.inferElemTypeInfoUp(elemTypeInfo);

    if (!mlir::isa<mlir::quant::QuantizedType>(elemTypeInfo.getInput(0))) {
        log.trace("Operation {0} does not support quantization params propagation: input cannot be quantized",
                  elemTypeInfoOp);
        return false;
    }
    for (size_t outputInd = 0; outputInd < layer->getNumResults(); outputInd++) {
        if (elemTypeInfo.getOutput(outputInd) != quantizedElemType) {
            log.trace("Operation {0} does not support quantization params propagation", elemTypeInfoOp);
            return false;
        }
    }
    quantizedElemType = elemTypeInfo.getInput(0);
    return true;
}

}  // namespace

namespace vpux::IE {

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

    if (const auto perTensorUniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(inType.getElementType())) {
        auto scales = perTensorUniformType.getScale();
        scales *= multiplier;
        if (const auto quantileStorageType =
                    mlir::dyn_cast<vpux::type::QuantileType>(perTensorUniformType.getStorageType())) {
            dstType = mlir::quant::UniformQuantizedType::get(
                    perTensorUniformType.getFlags(), quantileStorageType, perTensorUniformType.getExpressedType(),
                    scales, perTensorUniformType.getZeroPoint(), perTensorUniformType.getStorageTypeMin(),
                    perTensorUniformType.getStorageTypeMax());
        } else {
            dstType = mlir::quant::UniformQuantizedType::getChecked(
                    dequantizeOp.getLoc(), perTensorUniformType.isSigned(), perTensorUniformType.getStorageType(),
                    perTensorUniformType.getExpressedType(), scales, perTensorUniformType.getZeroPoint(),
                    perTensorUniformType.getStorageTypeMin(), perTensorUniformType.getStorageTypeMax());
        }
    } else if (const auto perAxisUniformType =
                       mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(inType.getElementType())) {
        auto scales = perAxisUniformType.getScales();
        SmallVector<double> newScales(scales.size());
        std::transform(scales.begin(), scales.end(), newScales.begin(), [multiplier](double x) {
            return x * multiplier;
        });
        if (const auto quantileStorageType =
                    mlir::dyn_cast<vpux::type::QuantileType>(perAxisUniformType.getStorageType())) {
            dstType = mlir::quant::UniformQuantizedPerAxisType::get(
                    perAxisUniformType.getFlags(), quantileStorageType, perAxisUniformType.getExpressedType(),
                    newScales, perAxisUniformType.getZeroPoints(), perAxisUniformType.getQuantizedDimension(),
                    perAxisUniformType.getStorageTypeMin(), perAxisUniformType.getStorageTypeMax());
        } else {
            dstType = mlir::quant::UniformQuantizedPerAxisType::getChecked(
                    dequantizeOp.getLoc(), perAxisUniformType.isSigned(), perAxisUniformType.getStorageType(),
                    perAxisUniformType.getExpressedType(), newScales, perAxisUniformType.getZeroPoints(),
                    perAxisUniformType.getQuantizedDimension(), perAxisUniformType.getStorageTypeMin(),
                    perAxisUniformType.getStorageTypeMax());
        }
    } else {
        return matchFailed(_log, rewriter, dequantizeOp, "unsupported quantize type");
    }

    auto quantizeCastOp = rewriter.create<IE::QuantizeCastOp>(appendLoc(dequantizeOp.getLoc(), "quantizecast"),
                                                              dequantizeOp.getInput(), dstType);
    auto newDequantizeOp = rewriter.create<IE::DequantizeOp>(appendLoc(dequantizeOp.getLoc(), "dequantize"),
                                                             quantizeCastOp.getOutput(), dequantizeOp.getDstElemType());
    auto userOp = *dequantizeOp.getOutput().getUsers().begin();
    rewriter.replaceOp(userOp, newDequantizeOp.getOutput());
    return mlir::success();
}

/* This rewriter searches for pattern:
fp_tensor -> [ElemTypeInfoOpInterface] -> fp_tensor -> [Quantize]        -> quantized_tensor
and replaces it with
fp_tensor -> [Quantize] -> quantized_tensor -> [ElemTypeInfoOpInterface] -> quantized_tensor */

mlir::LogicalResult PropagateQuantize::matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("PropagateQuantize Got layer: {0}", origOp);

    // 1. Check the parentOp is ElemTypeInfoOpInterface
    auto quantizedElemType = origOp.getDstElemType();
    auto prevOp = origOp.getOperand().getDefiningOp();
    mlir::Operation* firstUser = nullptr;
    while (prevOp) {
        if (!isValidToPropagateQuantize(prevOp, _seOpsEnabled, quantizedElemType, _log)) {
            break;
        }
        firstUser = prevOp;
        // Not backward for multiple operands
        if (prevOp->getOperands().size() > 1) {
            break;
        }
        prevOp = prevOp->getOperand(0).getDefiningOp();
    }
    if (!firstUser) {
        return mlir::failure();
    }

    // All checks passed. Rewrite the sub-graph.
    rewriter.startOpModification(firstUser);
    rewriter.setInsertionPoint(firstUser);

    // 1. Create new Quantize ops, place them on each input of current operation.
    auto firstElemTypeInfoOp = mlir::dyn_cast<IE::ElemTypeInfoOpInterface>(firstUser);
    for (auto [idx, operand] : llvm::enumerate(firstElemTypeInfoOp->getOpOperands())) {
        auto newLoc = appendLoc(firstElemTypeInfoOp->getLoc(), "propagated_Quantize '{0}'", idx);
        auto newQuantize = rewriter.create<IE::QuantizeOp>(newLoc, operand.get(), quantizedElemType);
        // Update input of Operation. NewQuant -> current Op.
        operand.set(newQuantize.getOutput());
    }
    // Rewrite done.
    rewriter.finalizeOpModification(firstElemTypeInfoOp);

    // 2. Infer return types, set output type of operation to inferred quantized type.
    auto lastElemTypeInfoOp = mlir::dyn_cast<IE::ElemTypeInfoOpInterface>(origOp.getOperand().getDefiningOp());
    for (auto elemTypeInfoOp = firstElemTypeInfoOp;;) {
        rewriter.startOpModification(elemTypeInfoOp);
        mlir::SmallVector<mlir::Type> inferredTypes;
        auto op = mlir::cast<mlir::InferTypeOpInterface>(elemTypeInfoOp.getOperation());
        VPUX_THROW_UNLESS(op.inferReturnTypes(getContext(), op->getLoc(), elemTypeInfoOp->getOperands(),
                                              op->getAttrDictionary(),  // operands
                                              op->getPropertiesStorage(), op->getRegions(), inferredTypes)
                                  .succeeded(),
                          "New type inference failed for '{0}'", op);
        VPUX_THROW_UNLESS(elemTypeInfoOp == lastElemTypeInfoOp || elemTypeInfoOp->hasOneUse(),
                          "Only support infer interim for 1 user elemTypeInfoOp '{0}'", elemTypeInfoOp);
        for (auto result : elemTypeInfoOp->getResults()) {
            result.setType(inferredTypes[0]);
        }
        rewriter.finalizeOpModification(elemTypeInfoOp);
        if (elemTypeInfoOp == lastElemTypeInfoOp) {
            break;
        }
        elemTypeInfoOp = mlir::dyn_cast<IE::ElemTypeInfoOpInterface>(*(elemTypeInfoOp->getUsers().begin()));
    }

    // 3. remove old Quantize ops.
    rewriter.startOpModification(lastElemTypeInfoOp);
    rewriter.setInsertionPoint(lastElemTypeInfoOp);
    for (auto result : lastElemTypeInfoOp->getResults()) {
        for (auto user : llvm::make_early_inc_range(result.getUsers())) {
            rewriter.replaceOp(user, result);
        }
    }
    // Rewrite done.
    rewriter.finalizeOpModification(lastElemTypeInfoOp);
    _log.trace("Successfully propagated QuantizeOp.");
    return mlir::success();
}

}  // namespace vpux::IE

namespace {

/* This rewriter searches for pattern:
quantized_tensor -> [Dequantize] -> fp_tensor -> [ElemTypeInfoOpInterface]                  -> fp_tensor
and replaces it with
quantized_tensor -> [ElemTypeInfoOpInterface] -> quantized_tensor(inferred) -> [Dequantize] -> fp_tensor */
class PropagateAndFuseQuantizeDequantizePass final :
        public IE::impl::PropagateAndFuseQuantizeDequantizeBase<PropagateAndFuseQuantizeDequantizePass> {
public:
    explicit PropagateAndFuseQuantizeDequantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
};

void PropagateAndFuseQuantizeDequantizePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto config = getDefaultGreedyRewriteConfig();
    auto func = getOperation();
    auto moduleOp = getModuleOp(func);
    const auto seOpsEnabled = config::hasEnableSEPtrsOperations(moduleOp);

    mlir::RewritePatternSet pqPatterns(&ctx);
    pqPatterns.add<IE::PropagateQuantize>(&ctx, _log.nest(), seOpsEnabled);
    if (mlir::failed(applyPatternsGreedily(func, std::move(pqPatterns), config))) {
        signalPassFailure();
    }

    mlir::RewritePatternSet patterns(&ctx);
    const auto& strategyFactory = IE::getIEStrategyFactory(&ctx);
    auto strategy = strategyFactory->getPropagateAndFuseQuantizeDequantizeStrategy(seOpsEnabled);
    strategy->addPatterns(patterns, _log);
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), config))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createPropagateAndFuseQuantizeDequantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateAndFuseQuantizeDequantizePass(Logger log) {
    return std::make_unique<PropagateAndFuseQuantizeDequantizePass>(log);
}
