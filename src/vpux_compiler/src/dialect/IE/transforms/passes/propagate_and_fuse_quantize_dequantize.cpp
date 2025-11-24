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
    auto newDequantizeOp = rewriter.create<IE::DequantizeOp>(appendLoc(dequantizeOp.getLoc(), "_dequantize"),
                                                             quantizeCastOp.getOutput(), dequantizeOp.getDstElemType());
    auto userOp = *dequantizeOp.getOutput().getUsers().begin();
    rewriter.replaceOp(userOp, newDequantizeOp.getOutput());
    return mlir::success();
}

//
// PropagateQuantize
//

class PropagateQuantize final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    PropagateQuantize(mlir::MLIRContext* ctx, Logger log, bool seOpsEnabled)
            : mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log), _seOpsEnabled(seOpsEnabled) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _seOpsEnabled;
};

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
        layerWithPostOp != nullptr && layerWithPostOp.getPostOp() != nullptr) {
        // A quantization-agnostic operation is no longer quantization-agnostic after it is fused with a post-op
        // (because post-op's are not quantization-agnostic). Since most post-op's will be fused by this time, this
        // check is here to prevent the propagation of output quantization through both the ElemTypeInfoOp and its
        // post-op. (At this time MaxPool seems to be the only operation which is both a IE::ElemTypeInfoOpInterface
        // and a IE::LayerWithPostOpInterface)
        log.trace("Operation {0} does not support quantization params propagation", elemTypeInfoOp);
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
        log.trace("Operation {0} does not support quantization params propagation", elemTypeInfoOp);
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
    auto elemTypeInfo = firstElemTypeInfoOp.getElemTypeInfo();
    auto firstLayer = mlir::cast<IE::LayerOpInterface>(firstUser);
    for (size_t outputInd = 0; outputInd < firstLayer->getNumResults(); outputInd++) {
        elemTypeInfo.setOutput(outputInd, quantizedElemType);
    }
    firstElemTypeInfoOp.inferElemTypeInfoUp(elemTypeInfo);
    for (auto [idx, operand] : llvm::enumerate(firstElemTypeInfoOp->getOpOperands())) {
        auto newLoc = appendLoc(firstElemTypeInfoOp->getLoc(), "_propagated_Quantize '{0}'", idx);
        auto newQuantize = rewriter.create<IE::QuantizeOp>(newLoc, operand.get(), elemTypeInfo.getInput(0));
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
    return mlir::success();
}

//
// PropagateDequantize
//

class PropagateDequantize final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    PropagateDequantize(mlir::MLIRContext* ctx, Logger log, bool seOpsEnabled)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log), _seOpsEnabled(seOpsEnabled) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _seOpsEnabled;
};

bool isValidToPropagateDequantize(mlir::Operation* user, bool seOpsEnabled, mlir::Type& quantizedElemType,
                                  mlir::Type& origDstElemType, Logger log) {
    auto elemTypeInfoOp = mlir::dyn_cast<IE::ElemTypeInfoOpInterface>(user);
    if (!elemTypeInfoOp) {
        return false;
    }

    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };
    const auto isSameDequantize = [&](mlir::Value input) {
        if (auto currentDequantize = input.getDefiningOp<IE::DequantizeOp>()) {
            return currentDequantize.getDstElemType() == origDstElemType;
        }

        return false;
    };
    auto layer = mlir::cast<IE::LayerOpInterface>(user);
    // 1. All inputs are Dequantize ops with same destination element type
    if (layer->getOperands().size() > 1 && !llvm::all_of(layer.getInputs(), isSameDequantize)) {
        log.trace("The inputs of Operation {0} should all be the same dequantizeOp when Op operands > 1",
                  elemTypeInfoOp);
        return false;
    }
    // 2. Check if operation supports quantization params propagation.
    if (auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(user);
        layerWithPostOp != nullptr && layerWithPostOp.getPostOp() != nullptr) {
        // A quantization-agnostic operation is no longer quantization-agnostic after it is fused with a post-op
        // (because post-op's are not quantization-agnostic). Since most post-op's will be fused by this time, this
        // check is here to prevent the propagation of input quantization through both the ElemTypeInfoOp and its
        // post-op. (At this time MaxPool seems to be the only operation which is both a IE::ElemTypeInfoOpInterface
        // and a IE::LayerWithPostOpInterface)
        log.trace("Operation {0} does not support quantization params propagation", elemTypeInfoOp);
        return false;
    }
    // 3. Particular check for SE pointers
    if (!vpux::IE::isSupportedElemTypeInfoCase(elemTypeInfoOp.getOperation(), seOpsEnabled, logCb)) {
        log.trace("Operation {0} does not support SE pointers", elemTypeInfoOp);
        return false;
    }
    // 4. Check whether elemTypeInfoOp all input dequantizeOps parameters are consistent
    auto elemTypeInfo = elemTypeInfoOp.getElemTypeInfo();

    SmallVector<mlir::Type> originalTypes;
    for (auto [idx, input] : llvm::enumerate(layer.getInputs())) {
        if (layer->getOperands().size() > 1) {
            auto dequantizeOp = input.getDefiningOp<IE::DequantizeOp>();
            quantizedElemType = mlir::cast<vpux::NDTypeInterface>(dequantizeOp.getInput().getType()).getElementType();
        }
        elemTypeInfo.setInput(idx, quantizedElemType);
        originalTypes.push_back(quantizedElemType);
    }
    elemTypeInfoOp.inferElemTypeInfo(elemTypeInfo);
    const auto typesAreOriginal = llvm::all_of(irange(originalTypes.size()), [&](size_t idx) {
        return elemTypeInfo.getInput(idx) == originalTypes[idx];
    });

    if (!typesAreOriginal) {
        log.trace("Operation {0} does not support quantization params propagation", elemTypeInfoOp);
        return false;
    }
    // 5. Check whether elemTypeInfoOp all output parameters are consistent
    for (size_t outputInd = 0; outputInd < layer->getNumResults(); outputInd++) {
        if (!mlir::isa<mlir::quant::QuantizedType>(elemTypeInfo.getOutput(outputInd))) {
            log.trace("Operation {0} does not support quantization params propagation", elemTypeInfoOp);
            return false;
        }
    }
    if (layer->getOperands().size() == 1) {
        quantizedElemType = elemTypeInfo.getOutput(0);
        origDstElemType = elemTypeInfoOp.getElemTypeInfo().getOutput(0);
    }
    return true;
}

/* This rewriter searches for pattern:
quantized_tensor -> [Dequantize] -> fp_tensor -> [ElemTypeInfoOpInterface]                  -> fp_tensor
and replaces it with
quantized_tensor -> [ElemTypeInfoOpInterface] -> quantized_tensor(inferred) -> [Dequantize] -> fp_tensor */
mlir::LogicalResult PropagateDequantize::matchAndRewrite(IE::DequantizeOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("PropagateDequantize Got layer: {0}", origOp);

    auto users = origOp->getUsers();
    mlir::Operation* lastUser = nullptr;
    auto origDstElemType = origOp.getDstElemType();
    auto quantizedElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getElementType();
    // store orig users for DequantizeOp to avoid update missing and duplicate same users
    std::unordered_set<mlir::Operation*> origUsers;
    if (origOp->hasOneUse()) {
        // DequantizeOp -> Op1 -> Op2 -> Op3, find Op3
        mlir::Operation* nextUser = *(users.begin());
        while (nextUser) {
            if (!isValidToPropagateDequantize(nextUser, _seOpsEnabled, quantizedElemType, origDstElemType, _log)) {
                break;
            }
            if (nextUser == *(users.begin())) {
                origUsers.insert(nextUser);
            }
            lastUser = nextUser;
            // Not forward for multiple users
            if (!nextUser->hasOneUse()) {
                break;
            }
            nextUser = *(nextUser->getUsers().begin());
        }
    } else {
        for (auto user : users) {
            if (isValidToPropagateDequantize(user, _seOpsEnabled, quantizedElemType, origDstElemType, _log)) {
                origUsers.insert(user);
            }
        }
    }
    if (!origUsers.size()) {
        return mlir::failure();
    }

    // 4. Rewrite the sub-graph.
    for (auto user : origUsers) {
        rewriter.startOpModification(user);
        // remove all dequantize op operands
        const auto inputs = user->getOpOperands();
        auto layer = mlir::cast<IE::LayerOpInterface>(user);
        for (auto idx : irange(inputs.size())) {
            auto& input = inputs[idx];
            auto dequantizeOp = layer.getInputs()[idx].getDefiningOp<IE::DequantizeOp>();
            input.set(dequantizeOp.getInput());
        }
        // infer return type and insert new Dequantize Op
        auto currentUser = user;
        while (currentUser) {
            mlir::SmallVector<mlir::Type> inferredTypes;
            auto op = mlir::cast<mlir::InferTypeOpInterface>(currentUser);
            VPUX_THROW_UNLESS(
                    op.inferReturnTypes(getContext(), op->getLoc(), op->getOperands(), op->getAttrDictionary(),
                                        op->getPropertiesStorage(), op->getRegions(), inferredTypes)
                            .succeeded(),
                    "New type inference failed for '{0}'", op);
            auto layer = mlir::cast<IE::LayerOpInterface>(currentUser);
            auto isToInsert = !lastUser || currentUser == lastUser;
            for (unsigned int outputInd = 0; outputInd < layer->getNumResults(); outputInd++) {
                auto dstElemType = mlir::dyn_cast<vpux::NDTypeInterface>(currentUser->getResult(outputInd).getType())
                                           .getElementType();
                currentUser->getResult(outputInd).setType(inferredTypes[outputInd]);
                if (isToInsert) {
                    auto output = currentUser->getOpResult(outputInd);
                    rewriter.setInsertionPointAfter(currentUser);
                    auto newLoc = appendLoc(currentUser->getLoc(), "_propagated_Dequantize '{0}'", outputInd);
                    auto newDequant = rewriter.create<IE::DequantizeOp>(newLoc, output, dstElemType);
                    _log.trace("Added new Dequantize op: '{0}' at index '{1}'", newDequant, outputInd);
                    output.replaceAllUsesExcept(newDequant.getOutput(),
                                                llvm::SmallPtrSet<mlir::Operation*, 1>{newDequant});
                    _log.trace("All uses of current layer have been replaced with new Dequantize op at index '{0}'",
                               outputInd);
                }
            }

            if (isToInsert) {
                break;
            }
            currentUser = *(currentUser->getUsers().begin());
        }
        rewriter.finalizeOpModification(user);
    }

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
    auto config = getDefaultGreedyRewriteConfig();
    auto func = getOperation();

    mlir::RewritePatternSet pqPatterns(&ctx);
    pqPatterns.add<PropagateQuantize>(&ctx, _log.nest(), _seOpsEnabled);
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(pqPatterns), config))) {
        signalPassFailure();
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseDequantizeWithMultiplier>(&ctx, _log);
    patterns.add<PropagateDequantize>(&ctx, _log.nest(), _seOpsEnabled);

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
