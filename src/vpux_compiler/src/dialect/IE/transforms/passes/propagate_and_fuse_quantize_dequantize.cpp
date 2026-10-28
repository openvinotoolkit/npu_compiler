//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/propagate_and_fuse_quantize_dequantize.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

#include "vpux/compiler/core/types/quantile_types/types.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/ADT/TypeSwitch.h>
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
    if (auto multiplyOp = mlir::dyn_cast_if_present<IE::MultiplyOp>(userOp)) {
        if (multiplyOp.getPostOp() || multiplyOp.getClamp()) {
            return std::nullopt;
        }
        constInput =
                multiplyOp.getInput1() == dequantizeOp.getOutput() ? multiplyOp.getInput2() : multiplyOp.getInput1();
    } else if (auto dwConvOp = mlir::dyn_cast_if_present<IE::GroupConvolutionOp>(userOp)) {
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

    auto constOp = mlir::dyn_cast_if_present<Const::DeclareOp>(constInput.getDefiningOp());
    if (constOp == nullptr || !IE::isBaseContentSplat(constOp)) {
        return std::nullopt;
    }

    return vpux::IE::getConst(constOp).front();
}

bool isValidToPropagateQuantize(mlir::Operation* op, bool seOpsEnabled, mlir::Type& quantizedElemType, Logger log) {
    const auto isSameQuantize = [&](mlir::Operation* user) {
        if (auto currentQuantize = mlir::dyn_cast_if_present<IE::QuantizeOp>(user)) {
            return currentQuantize.getDstElemType() == quantizedElemType;
        }

        return false;
    };
    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };
    // 1. Check the prevOp is ElemTypeInfoOp
    auto elemTypeInfoOp = mlir::dyn_cast_if_present<IE::ElemTypeInfoOpInterface>(op);
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

// Whether splitting Quantize across multiOperandOp's operands pays off: either the quantized
// type is narrower than origElemType, or an operand traces back to a Dequantize that can fold
// or fuse with the inserted Quantize.
bool shouldSplitQuantizeAcrossOperands(mlir::Operation* multiOperandOp, mlir::Type quantizedElemType,
                                       mlir::Type origElemType, bool seOpsEnabled, Logger log) {
    if (vpux::getElemTypeSize(quantizedElemType) < vpux::getElemTypeSize(origElemType)) {
        return true;
    }

    const auto reachesFusableDequantize = [&](mlir::Value operand) {
        auto* op = operand.getDefiningOp();
        auto hopElemType = quantizedElemType;
        while (op != nullptr) {
            if (auto dequantizeOp = mlir::dyn_cast<IE::DequantizeOp>(op)) {
                auto inType = mlir::dyn_cast<vpux::NDTypeInterface>(dequantizeOp.getInput().getType());
                return inType != nullptr && inType.getElementType() == hopElemType &&
                       dequantizeOp.getOutput().hasOneUse();
            }
            if (op->getNumOperands() != 1 || op->getNumResults() != 1 || !op->getResult(0).hasOneUse() ||
                !isValidToPropagateQuantize(op, seOpsEnabled, hopElemType, log)) {
                break;
            }
            op = op->getOperand(0).getDefiningOp();
        }
        return false;
    };

    const auto operands = multiOperandOp->getOperands();
    const auto fusableCount = llvm::count_if(operands, reachesFusableDequantize);
    const auto totalCount = static_cast<int64_t>(operands.size());
    if (2 * fusableCount >= totalCount) {
        return true;
    }

    log.trace("Splitting Quantize at '{0}' is not beneficial: no storage size reduction and only {1}/{2} "
              "operands reach a fusable Dequantize",
              multiOperandOp->getLoc(), fusableCount, totalCount);
    return false;
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
                    mlir::dyn_cast<mlir::quant::QuantileType>(perTensorUniformType.getStorageType())) {
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
                    mlir::dyn_cast<mlir::quant::QuantileType>(perAxisUniformType.getStorageType())) {
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

    auto quantizeCastOp = IE::QuantizeCastOp::create(rewriter, appendLoc(dequantizeOp.getLoc(), "quantizecast"),
                                                     dequantizeOp.getInput(), dstType);
    auto newDequantizeOp = IE::DequantizeOp::create(rewriter, appendLoc(dequantizeOp.getLoc(), "dequantize"),
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
    const auto origElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getElementType();
    auto prevOp = origOp.getOperand().getDefiningOp();
    mlir::Operation* firstUser = nullptr;
    while (prevOp) {
        // Snapshot the type valid at the previously accepted op, in case prevOp is a
        // rejected multi-operand op and this needs to be restored.
        const auto quantizedElemTypeBeforeThisOp = quantizedElemType;
        if (!isValidToPropagateQuantize(prevOp, _seOpsEnabled, quantizedElemType, _log)) {
            break;
        }
        // Not backward for multiple operands, unless splitting pays off.
        if (prevOp->getOperands().size() > 1) {
            if (shouldSplitQuantizeAcrossOperands(prevOp, quantizedElemType, origElemType, _seOpsEnabled, _log)) {
                firstUser = prevOp;
            } else {
                quantizedElemType = quantizedElemTypeBeforeThisOp;
            }
            break;
        }
        firstUser = prevOp;
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
        auto newQuantize = IE::QuantizeOp::create(rewriter, newLoc, operand.get(), quantizedElemType);
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

// PropagateDynamicDequantize

namespace {
// Adjusts a cloned SliceOp so it correctly slices a broadcast-compatible scale tensor: dims where
// the scale is broadcast (size 1) get offset 0 and size 1, other dims keep the offset/size copied
// from the original (activation-side) Slice this op was cloned from.
void adjustClonedSliceForScaleShape(IE::SliceOp sliceOp, ShapeRef scaleShape) {
    auto offsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
    auto sizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
    VPUX_THROW_UNLESS(offsets.size() == scaleShape.size(), "Slice rank '{0}' does not match scale rank '{1}'",
                      offsets.size(), scaleShape.size());
    for (auto idx : irange(sizes.size())) {
        if (scaleShape[Dim(idx)] == 1) {
            offsets[idx] = 0;
            sizes[idx] = 1;
        }
    }
    auto* ctx = sliceOp->getContext();
    sliceOp->setAttr("static_offsets", getIntArrayAttr(ctx, offsets));
    sliceOp->setAttr("static_sizes", getIntArrayAttr(ctx, sizes));
}

// Predicts the scale shape after propagating (without cloning any IR) through one hop of the
// activation chain, mirroring the transformations mapNonActivationInput applies for real. This
// keeps the scale-shape tracker used by isValidToPropagateDequant in sync with the multi-hop
// chain, instead of always comparing against the original (un-transformed) scale shape.
// Returns std::nullopt if the hop cannot be propagated onto the scale path.
std::optional<Shape> computeNextScaleShape(mlir::Operation* hopOp, ShapeRef currentScaleShape,
                                           const DimsOrder& scaleDimsOrder) {
    return mlir::TypeSwitch<mlir::Operation*, std::optional<Shape>>(hopOp)
            .Case<IE::AffineReshapeOp>([&](IE::AffineReshapeOp affineReshapeOp) -> std::optional<Shape> {
                const auto newScaleShape = computeShapeValueFromAffineReshape(affineReshapeOp, currentScaleShape);
                if (!newScaleShape.has_value()) {
                    return std::nullopt;
                }
                // The reshape must preserve the total number of scale elements. A mismatch means a
                // group dimension is being merged with a non-group dimension, which is semantically
                // invalid for per-group DynamicDequantize.
                const auto origElems = vpux::details::calcTotalShapeSize(currentScaleShape.raw());
                const auto newElems = vpux::details::calcTotalShapeSize(newScaleShape.value());
                if (origElems != newElems) {
                    return std::nullopt;
                }
                return Shape(newScaleShape.value());
            })
            .Case<IE::TransposeOp>([&](IE::TransposeOp transposeOp) -> std::optional<Shape> {
                if (!transposeOp.getOrderValue().has_value()) {
                    return std::nullopt;
                }
                const auto perm = DimsOrder::fromAffineMap(transposeOp.getOrderValue().value());
                if (static_cast<size_t>(perm.numDims()) != currentScaleShape.size()) {
                    return std::nullopt;
                }
                Shape newScaleShape(currentScaleShape.size());
                for (auto idx : irange(perm.numDims())) {
                    newScaleShape[Dim(idx)] = currentScaleShape[perm.dimAt(idx)];
                }
                return newScaleShape;
            })
            .Case<IE::PermuteCastOp>([&](IE::PermuteCastOp permuteCastOp) -> std::optional<Shape> {
                // PermuteCastOp is always a trivial permutation (view-like, no data movement), so
                // treating the scale's own layout as matching its tracked logical dim order, the
                // output shape can be derived directly from mem_perm/dst_order the same way
                // IE::PermuteCastOp::inferReturnTypeComponents does for the activation path.
                const auto dstOrder = DimsOrder::fromAffineMap(permuteCastOp.getDstOrder());
                if (static_cast<size_t>(dstOrder.numDims()) != currentScaleShape.size()) {
                    return std::nullopt;
                }

                const auto& srcOrder = scaleDimsOrder;
                const auto scaleShapeInMemOrder = srcOrder.toMemoryOrder(currentScaleShape);

                // Match IE::PermuteCastOp verification: the mem-permutation must be trivial for the scale too.
                if (!vpux::isTrivialPermute(scaleShapeInMemOrder, permuteCastOp.getMemPerm())) {
                    return std::nullopt;
                }

                const auto scaleShapePermutedInMemOrder =
                        vpux::applyPerm(scaleShapeInMemOrder, permuteCastOp.getMemPerm());
                return Shape(dstOrder.toLogicalOrder(scaleShapePermutedInMemOrder).raw());
            })
            .Case<IE::SliceOp>([&](IE::SliceOp sliceOp) -> std::optional<Shape> {
                const auto sizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
                if (currentScaleShape.size() != sizes.size()) {
                    return std::nullopt;
                }
                Shape newScaleShape(currentScaleShape.raw());
                for (auto idx : irange(sizes.size())) {
                    if (currentScaleShape[Dim(idx)] != 1) {
                        newScaleShape[Dim(idx)] = sizes[idx];
                    }
                }
                return newScaleShape;
            })
            .Default([&](mlir::Operation*) -> std::optional<Shape> {
                // Other ElemTypeInfoOpInterface ops may change the scale shape.
                // Avoid propagating.
                return std::nullopt;
            });
}
}  // namespace

void PropagateDynamicDequantize::mapNonActivationInput(IE::DynamicDequantizeOp op, mlir::Operation* crtUser,
                                                       mlir::IRMapping& mapper, mlir::PatternRewriter& rewriter,
                                                       NonActivationState& state) const {
    if (op.getScale() == nullptr) {
        return;
    }
    VPUX_THROW_WHEN(state.secondaryValue == nullptr, "Scale-path value was not initialized for '{0}'", op);

    // Clone this hop onto the scale path starting from the scale value produced by the previous
    // hop (or the original scale for the first hop), not from the original scale directly -
    // otherwise multi-hop chains would re-derive every hop from the un-transformed scale shape.
    const auto currentScaleValue = state.secondaryValue;
    const auto currentScaleShape = getBoundedShape(currentScaleValue);
    mapper.map(crtUser->getOperand(0), currentScaleValue);
    auto scaleCurrentUser = rewriter.clone(*crtUser, mapper);
    extendOpLoc(scaleCurrentUser, "scale");
    if (mlir::isa<IE::AffineReshapeOp>(scaleCurrentUser)) {
        auto newShapeValue = computeShapeValueFromAffineReshape(mlir::cast<IE::AffineReshapeOp>(scaleCurrentUser),
                                                                currentScaleShape);
        // newShapeValue is guaranteed valid: isValidToPropagateDequantize pre-checked this.
        scaleCurrentUser->setAttr("shape_value", getIntArrayAttr(op->getContext(), newShapeValue.value()));
    } else if (auto slice = mlir::dyn_cast_if_present<IE::SliceOp>(scaleCurrentUser)) {
        adjustClonedSliceForScaleShape(slice, currentScaleShape);
    }
    vpux::inferReturnTypes(scaleCurrentUser, vpux::InferShapedTypeMode::ALL);

    state.secondaryValue = scaleCurrentUser->getResult(0);

    mapper.clear();
    mapper.map(op.getScale(), state.secondaryValue);
}

mlir::LogicalResult PropagateDynamicDequantize::validateOpParams(IE::DynamicDequantizeOp op,
                                                                 mlir::PatternRewriter& rewriter,
                                                                 NonActivationState& state) const {
    if (op.getZp() != nullptr) {
        return matchFailed(_log, rewriter, op,
                           "Only support unrolled DynamicDequantizeOp without zero point, but got one");
    }

    auto scale = op.getScale();
    if (scale == nullptr) {
        return mlir::success();
    }

    // Reset the scale-path state: it evolves hop-by-hop for the duration of this match.
    state.secondaryValue = scale;
    state.secondaryShape = getBoundedShape(scale).toValues();

    auto numOfNonOneDims = getNonOneDim(getBoundedShape(scale));
    auto hasOneUseDynamicDequantizeOp = op->hasOneUse();
    auto childOp = *(op->getUsers().begin());
    if ((numOfNonOneDims.size() != 1 || !hasOneUseDynamicDequantizeOp) &&
        mlir::isa_and_present<IE::TransposeOp>(childOp)) {
        return matchFailed(_log, rewriter, op, "Only support unrolled DynamicDequantizeOp, but got {0}",
                           numOfNonOneDims);
    }

    return mlir::success();
}

bool PropagateDynamicDequantize::isValidToPropagateDequant(IE::DynamicDequantizeOp op, mlir::Operation* nextUser,
                                                           NonActivationState& state) const {
    auto origDstElemType = op.getDstElemType();
    auto quantizedElemType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType()).getElementType();
    if (!isValidToPropagateDequantize<IE::DynamicDequantizeOp>(nextUser, _seOpsEnabled, quantizedElemType,
                                                               origDstElemType, _log)) {
        return false;
    }

    if (op.getScale() == nullptr) {
        return true;
    }

    // Verify the CURRENT (possibly already hop-transformed) scale shape can be propagated through
    // this hop before modifying IR, and advance the state so subsequent hops in the same chain
    // are checked against the up-to-date scale shape rather than the original one.
    const auto scaleDimsOrder = mlir::cast<NDTypeInterface>(op.getScale().getType()).getDimsOrder();
    const auto newScaleShape = computeNextScaleShape(nextUser, state.secondaryShape, scaleDimsOrder);
    if (!newScaleShape.has_value()) {
        _log.trace("Scale shape cannot be propagated through {0}", nextUser);
        return false;
    }
    state.secondaryShape = newScaleShape.value();

    return true;
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
