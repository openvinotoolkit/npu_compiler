//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZETILEOP
#define GEN_PASS_DEF_OPTIMIZETILEOP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

IE::AutoBroadcastType getBroadCastType(mlir::Operation* op) {
    return llvm::TypeSwitch<mlir::Operation*, IE::AutoBroadcastType>(op)
            .Case<IE::MultiplyOp>([&](auto multiply) {
                return multiply.getAutoBroadcast();
            })

            .Case<IE::AddOp>([&](auto add) {
                return add.getAutoBroadcast();
            })
            .Default([&](auto) -> IE::AutoBroadcastType {
                VPUX_THROW("Unexpected operation type at '{0}'", op);
            });
}

class FoldTileOpRewriter final : public mlir::OpRewritePattern<IE::TileOp> {
public:
    FoldTileOpRewriter(mlir::MLIRContext* ctx, const Logger& log): mlir::OpRewritePattern<IE::TileOp>(ctx), _log(log) {
        setDebugName("FoldTileOpRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const Logger& _log;
};

mlir::LogicalResult FoldTileOpRewriter::matchAndRewrite(IE::TileOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.use_empty()) {
        return mlir::failure();
    }
    auto ctx = getContext();

    auto origInputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto origInputShape = origInputType.getShape();
    const auto inputHasStaticShape = mlir::cast<mlir::ShapedType>(origOp.getInput().getType()).hasStaticShape();

    // If the tile op is used as input for op like eltwise or multiply, and its size is too big to fit into CMX, which
    // means that the op will be tiled into multiple small ones. And it will cost lots of time before executing the
    // eltwise/multiply op. So it will be performant if it can be fused into the post op.
    auto hasLargeSingleChannelInput = inputHasStaticShape &&
                                      origInputType.getTotalAllocSize() > vpux::VPU::getTotalCMXSize(origOp) &&
                                      origInputShape.size() == 4 && origInputShape[Dims4D::Act::C] == 1;

    // Handle bias/scale-like inputs: shape NxCx1x1 tiled only in H/W dims (N and C are unchanged).
    // The eltwise op uses NUMPY broadcast, so removing the Tile and passing the NxCx1x1
    // input directly produces the same result via implicit broadcasting.
    // Exclude totalSize==1 (scalar) to avoid interfering with the scalar fold path below.
    //
    // Guard: only fold when the tiled output is large enough that the broadcast DMA overhead dominates.
    // For small tensors, the DMA is cheap and the DPU NCE Eltwise (equal-sized inputs) is faster
    // than a SW broadcast Add (SHAVE kernel dispatch overhead ~10-50us regardless of size).
    // See VPU::NCEInvariant::isLargeEnoughForDPUOverSHAVE for the shared threshold heuristic
    // (also used by DecomposeMVN's forceDecompose gate and MVNLayoutInfoOpModelForSW's NHWC guard).
    auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto moduleOp = origOp->getParentOfType<mlir::ModuleOp>();
    const auto numTiles = config::hasTileExecutor(moduleOp) ? config::getTileExecutor(moduleOp).getCount() : int64_t{1};
    const auto outputShape = getShape(origOp.getOutput());
    // Note: IE.TileOp always preserves rank and repeats by a static attribute, so a static
    // 4D input (inputHasStaticShape / origInputShape.size() == 4, checked below) guarantees a
    // static 4D output as well; no separate output static-shape/rank check is needed here.
    //
    // Guard: skip folding when the tiled output remains 1D (H==1 or W==1). On such tensors the
    // SHAVE broadcast kernel is slower than the original Tile + NCE Eltwise (equal-shape DPU op).
    const auto outputSizeBytes = outputType.getTotalAllocSize().count();
    const auto isConstInput = origOp.getInput().getDefiningOp<Const::DeclareOp>() != nullptr;
    const auto isFoldBeneficial =
            vpux::VPU::NCEInvariant::isLargeEnoughForDPUOverSHAVE(origOp, outputSizeBytes, numTiles);
    const auto hasSpatiallyBroadcastedInput =
            inputHasStaticShape && origInputShape.totalSize() != 1 && origInputShape.size() == 4 &&
            origInputShape[Dims4D::Act::H] == 1 && origInputShape[Dims4D::Act::W] == 1 &&
            outputShape[Dims4D::Act::N] == origInputShape[Dims4D::Act::N] &&
            outputShape[Dims4D::Act::C] == origInputShape[Dims4D::Act::C] && outputShape[Dims4D::Act::H] != 1 &&
            outputShape[Dims4D::Act::W] != 1 && isFoldBeneficial;

    // Force-fold the Tile when it feeds a bias-Add for a static-weight GroupConv and the bias
    // itself is also a compile-time constant. Non-constant bias must be kept so the downstream
    // equal-shape NCE.Eltwise path fires instead of the slower SW broadcast Add for small tensors.
    //
    // NOTE: uses origOp.getOutput()'s direct user, not outputUserOp. outputUserOp is computed
    // after walking through any view ops between the TileOp and the eltwise. Here we check the
    // TileOp's immediate successor to identify the GroupConv + Add pattern before any view ops
    // are skipped. The hasFoldableUser guard below prevents the fold when a view op sits between
    // the TileOp and the AddOp, so this check is consistent with the rest of the fold logic.
    const auto isSpatialBroadcastFromGroupConv = [&]() -> bool {
        if (!origOp->hasOneUse()) {
            return false;
        }
        auto* tileUser = *origOp.getOutput().getUsers().begin();
        auto addOp = mlir::dyn_cast<IE::AddOp>(tileUser);
        if (addOp == nullptr) {
            return false;
        }
        if (addOp.getInputPaddingAttr() != nullptr || addOp.getOutputPaddingAttr() != nullptr ||
            addOp.getScale() != nullptr) {
            return false;
        }
        auto nonTileOperand =
                (addOp.getInput1().getDefiningOp() == origOp.getOperation()) ? addOp.getInput2() : addOp.getInput1();
        auto groupConvOp = nonTileOperand.getDefiningOp<IE::GroupConvolutionOp>();
        if (groupConvOp == nullptr || groupConvOp.getBias() != nullptr) {
            return false;
        }
        if (addOp.getOutput().getType() != groupConvOp.getOutput().getType()) {
            return false;
        }
        if (!groupConvOp.getOutput().hasOneUse()) {
            return false;
        }
        if (groupConvOp.getPostOpAttr() != nullptr) {
            return false;
        }
        if (!mlir::isa<mlir::FloatType>(origInputType.getElementType())) {
            return false;
        }
        return groupConvOp.getFilter().getDefiningOp<Const::DeclareOp>() != nullptr &&
               origOp.getInput().getDefiningOp<Const::DeclareOp>() != nullptr;
    }();

    if ((!inputHasStaticShape || origInputShape.totalSize() != 1) && !hasLargeSingleChannelInput &&
        !hasSpatiallyBroadcastedInput && !isSpatialBroadcastFromGroupConv) {
        return mlir::failure();
    }

    if (!origOp->hasOneUse()) {
        return mlir::failure();
    }

    const auto isFoldableViewOp = [](mlir::Operation* viewOp) {
        if (!mlir::isa<IE::ReshapeOp, IE::AffineReshapeOp, IE::ShapeCastOp>(viewOp)) {
            return false;
        }
        if (!viewOp->hasOneUse()) {
            return false;
        }
        return true;
    };

    auto outputValue = mlir::cast<mlir::Value>(origOp.getOutput());
    auto outputUserOp = *(outputValue.getUsers().begin());
    while (isFoldableViewOp(outputUserOp)) {
        outputValue = outputUserOp->getResult(0);
        outputUserOp = *(outputValue.getUsers().begin());
    }

    // For the large single channel input or spatial broadcast input, don't fold TileOp if the output is used by
    // FoldableViewOp, since the compiler may not be able to back infer the new output shape
    auto hasFoldableUser = isFoldableViewOp(*origOp->getUsers().begin());
    if ((hasLargeSingleChannelInput || hasSpatiallyBroadcastedInput || isSpatialBroadcastFromGroupConv) &&
        hasFoldableUser) {
        return mlir::failure();
    }

    // More ops which support auto broadcast may also apply here!
    if (!mlir::isa_and_nonnull<IE::MultiplyOp, IE::AddOp>(outputUserOp)) {
        return mlir::failure();
    }

    // Can't fold TileOp if the layer has precision convert like from fp16 to fp32
    auto userInType = mlir::cast<vpux::NDTypeInterface>(outputUserOp->getOperand(0).getType());
    auto userOutType = mlir::cast<vpux::NDTypeInterface>(outputUserOp->getResult(0).getType());
    if (userInType.getElementType() != userOutType.getElementType()) {
        return mlir::failure();
    }

    // Can't fold TileOp if the layer has post operation, except for the spatial broadcast case where only H/W
    // dims are expanded: the eltwise op's NUMPY broadcast handles the smaller input shape transparently, so the
    // post op result is unaffected.
    if (!hasSpatiallyBroadcastedInput && !isSpatialBroadcastFromGroupConv) {
        if (auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(outputUserOp)) {
            if (layerWithPostOp.hasPPE()) {
                return mlir::failure();
            }
        }
    }

    _log.trace("Folding TileOp at '{0}'", origOp.getLoc());

    const auto broadCastType = getBroadCastType(outputUserOp);

    if (hasLargeSingleChannelInput || hasSpatiallyBroadcastedInput || isSpatialBroadcastFromGroupConv) {
        // Validate that substituting origInputShape (pre-Tile) for the Tile result is broadcastable
        // against the other eltwise operand and still produces the original output shape.
        // Guard: if both operands are NxCx1x1 after substitution, broadcasting gives NxCx1x1,
        // not the original NxCxHxW -- in that case the fold is invalid.
        const auto lhsIsTileOp = outputUserOp->getOperand(0).getDefiningOp() == origOp;
        const auto lhsShape = lhsIsTileOp ? origInputShape : getShape(outputUserOp->getOperand(0));
        const auto rhsShape = lhsIsTileOp ? getShape(outputUserOp->getOperand(1)) : origInputShape;
        const auto outShape = IE::broadcastEltwiseShape(lhsShape, rhsShape, broadCastType, outputUserOp->getLoc());
        if (mlir::failed(outShape) || ShapeRef(outShape.value()) != getShape(outputUserOp->getResult(0))) {
            return mlir::failure();
        }

        // Skip fold for non-constant spatial broadcast when the other Add input is from an NCE
        // Convolution. The equal-shape NCE Add enables DPU eltwise scheduling chained with
        // the upstream conv; replacing it with a SHAVE broadcast breaks that chain and regresses
        // performance. Non-constant inputs from non-Conv producers (e.g. per-channel scale GroupConv
        // with a non-constant filter in AdaIN conditioners) do not have an NCE chain to preserve,
        // so they fold normally.
        if (hasSpatiallyBroadcastedInput && !isConstInput) {
            const auto otherOperand = lhsIsTileOp ? outputUserOp->getOperand(1) : outputUserOp->getOperand(0);
            auto* defOp = otherOperand.getDefiningOp();
            bool skipFold = mlir::isa_and_nonnull<IE::ConvolutionOp>(defOp);
            if (!skipFold) {
                if (auto groupConv = mlir::dyn_cast_if_present<IE::GroupConvolutionOp>(defOp)) {
                    // Guard GroupConv only when its filter is static: a non-constant-filter GroupConv
                    // (e.g. per-channel scale/multiply) is not an NCE conv, so it does not benefit
                    // from equal-shape NCE Add chaining.
                    skipFold = groupConv.getFilter().getDefiningOp<Const::DeclareOp>() != nullptr;
                }
            }
            if (skipFold) {
                return mlir::failure();
            }

            // When the eltwise has a fused post-op (PPE) and the bias is non-constant, fold the Tile
            // for performance (avoids large HxW DMA expansion) but strip the post-op from the
            // eltwise and insert a standalone activation op after it.
            //
            // Without this, the broadcast Add (signal + bias_1xCx1x1) with post_op falls through
            // to a SHAVE SW kernel that does not implement the post_op attribute — silently
            // dropping the activation (e.g. LeakyRelu) and causing accuracy regression.
            // Stripping it and re-emitting as a standalone op preserves the numerical result.
            if (auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(outputUserOp)) {
                if (layerWithPostOp.hasPPE()) {
                    auto addOp = mlir::dyn_cast<IE::AddOp>(outputUserOp);
                    if (addOp == nullptr) {
                        return mlir::failure();
                    }
                    const auto postOpAttr = addOp.getPostOpAttr();
                    const auto clampAttr = addOp.getClampAttr();

                    rewriter.setInsertionPointAfter(addOp);

                    // Recreate the Add without post_op/clamp, passing the pre-Tile bias (e.g.
                    // 1xCx1x1) directly so the new Add has the correct broadcast shapes up front.
                    const auto biasInput = origOp.getInput();
                    const auto newAddInput1 = lhsIsTileOp ? biasInput : addOp.getInput1();
                    const auto newAddInput2 = lhsIsTileOp ? addOp.getInput2() : biasInput;
                    auto newAdd = rewriter.create<IE::AddOp>(addOp.getLoc(), addOp.getOutput().getType(), newAddInput1,
                                                             newAddInput2, addOp.getScale(),
                                                             addOp.getAutoBroadcastAttr(), /*post_op=*/nullptr,
                                                             /*clamp=*/nullptr, addOp.getStaticScaleAttr(),
                                                             addOp.getOutputPaddingAttr(), addOp.getInputPaddingAttr());

                    // Insert standalone activation(s) matching the stripped post_op/clamp.
                    // Each emitted op gets an appendLoc suffix; the original addOp.getLoc() is
                    // preserved on the new Add above.
                    mlir::Value activationOut;
                    if (const auto lrelu = mlir::dyn_cast_if_present<IE::LeakyReluAttr>(postOpAttr)) {
                        activationOut = rewriter.create<IE::LeakyReluOp>(appendLoc(addOp.getLoc(), "lrelu"),
                                                                         newAdd.getOutput(), lrelu.getNegativeSlope())
                                                ->getResult(0);
                    } else if (mlir::isa_and_present<IE::ReluAttr>(postOpAttr)) {
                        activationOut =
                                rewriter.create<IE::ReLUOp>(appendLoc(addOp.getLoc(), "relu"), newAdd.getOutput())
                                        ->getResult(0);
                    } else if (postOpAttr == nullptr) {
                        // Clamp-only (no post_op): hasPPE() => clampAttr is set.
                        const auto min = clampAttr.getAs<mlir::FloatAttr>("min").getValueAsDouble();
                        const auto max = clampAttr.getAs<mlir::FloatAttr>("max").getValueAsDouble();
                        activationOut =
                                rewriter.create<IE::ClampOp>(appendLoc(addOp.getLoc(), "clamp"), newAdd.getOutput(),
                                                             vpux::getFPAttr(ctx, min), vpux::getFPAttr(ctx, max))
                                        ->getResult(0);
                    } else {
                        // Unrecognised post_op (with or without clamp) — erase the new Add and leave
                        // the original untouched rather than silently dropping the unknown activation.
                        rewriter.eraseOp(newAdd);
                        return mlir::failure();
                    }

                    // When both post_op and clamp are present, apply the clamp after the activation.
                    if (clampAttr != nullptr && postOpAttr != nullptr) {
                        const auto min = clampAttr.getAs<mlir::FloatAttr>("min").getValueAsDouble();
                        const auto max = clampAttr.getAs<mlir::FloatAttr>("max").getValueAsDouble();
                        activationOut =
                                rewriter.create<IE::ClampOp>(appendLoc(addOp.getLoc(), "clamp"), activationOut,
                                                             vpux::getFPAttr(ctx, min), vpux::getFPAttr(ctx, max))
                                        ->getResult(0);
                    }

                    rewriter.replaceAllOpUsesWith(addOp, activationOut);
                    rewriter.replaceAllOpUsesWith(origOp, origOp.getInput());
                    return mlir::success();
                }
            }
        }

        rewriter.replaceOp(origOp, origOp.getInput());
        return mlir::success();
    }

    // Scalar case: verify that substituting the all-ones replacementShape preserves the eltwise output shape.
    const auto replacementShape =
            SmallVector<int64_t>(mlir::cast<vpux::NDTypeInterface>(outputValue.getType()).getRank(), 1);
    const auto existingOutShape = to_small_vector(getShape(outputUserOp->getResult(0)));
    const auto lhsIsReplaced = outputUserOp->getOperand(0) == outputValue;
    const auto newLhsShape = lhsIsReplaced ? replacementShape : to_small_vector(getShape(outputUserOp->getOperand(0)));
    const auto newRhsShape = lhsIsReplaced ? to_small_vector(getShape(outputUserOp->getOperand(1))) : replacementShape;
    const auto newOutShape = IE::broadcastEltwiseShape(ArrayRef<int64_t>(newLhsShape), ArrayRef<int64_t>(newRhsShape),
                                                       broadCastType, outputUserOp->getLoc());
    if (mlir::failed(newOutShape) || newOutShape.value() != existingOutShape) {
        return mlir::failure();
    }

    auto newReshapeOp = rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), origOp.getInput(),
                                                             getIntArrayAttr(ctx, replacementShape));
    rewriter.replaceAllUsesWith(outputValue, newReshapeOp);
    return mlir::success();
}

//
// FuseTileConvertRewrite
//
// Pattern: Convert -> Tile -> Convert
//
// Benefits:
// 1. Reduces data size for Tile DMA
// 2. Convert operates on smaller tensor before Tile expansion
//

class FuseTileConvertRewrite final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    FuseTileConvertRewrite(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        setDebugName("FuseTileConvertRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const final;

private:
    const Logger& _log;
};

mlir::LogicalResult FuseTileConvertRewrite::matchAndRewrite(IE::ConvertOp convertOp,
                                                            mlir::PatternRewriter& rewriter) const {
    if (convertOp.use_empty()) {
        return mlir::failure();
    }
    _log.trace("FuseTileConvertRewrite: Got '{0}' at '{1}'", convertOp->getName(), convertOp->getLoc());
    auto nestedLogger = _log.nest();

    // Check if input is from TileOp
    auto tileOp = convertOp.getInput().getDefiningOp<IE::TileOp>();
    if (tileOp == nullptr) {
        nestedLogger.trace("ConvertOp does not have TileOp input");
        return mlir::failure();
    }

    if (!tileOp.getResult().hasOneUse()) {
        nestedLogger.trace("TileOp has multiple users");
        return mlir::failure();
    }

    // Check if TileOp's input is from another ConvertOp
    auto prevConvertOp = tileOp.getInput().getDefiningOp<IE::ConvertOp>();
    if (prevConvertOp == nullptr) {
        nestedLogger.trace("TileOp does not have ConvertOp input");
        return mlir::failure();
    }

    if (!prevConvertOp.getResult().hasOneUse()) {
        nestedLogger.trace("Previous ConvertOp has multiple users");
        return mlir::failure();
    }

    // Get the final destination element type from the second ConvertOp
    const auto finalDstElemType = convertOp.getDstElemType();

    // Check if the optimization reduces data size for Tile operation
    // Original: Convert(A->B) -> Tile(B) -> Convert(B->C)
    // Optimized: Convert(A->C) -> Tile(C)
    // Only beneficial when sizeof(C) <= sizeof(B), i.e., Tile operates on smaller or equal data
    const auto tileInputType = mlir::cast<vpux::NDTypeInterface>(tileOp.getInput().getType());
    const auto intermediateElemBitWidth = tileInputType.getElemTypeSize().count();
    const auto finalElemBitWidth = vpux::getElemTypeSize(finalDstElemType).count();

    if (finalElemBitWidth > intermediateElemBitWidth) {
        nestedLogger.trace("Optimization would increase Tile data size: intermediate {0} bits -> final {1} bits",
                           intermediateElemBitWidth, finalElemBitWidth);
        return mlir::failure();
    }

    // Get the original input to the first ConvertOp
    auto originalInput = prevConvertOp.getInput();

    auto newConvertOp = rewriter.create<IE::ConvertOp>(prevConvertOp.getLoc(), originalInput,
                                                       mlir::TypeAttr::get(finalDstElemType));

    auto tileOutType = mlir::cast<vpux::NDTypeInterface>(tileOp.getOutput().getType());
    auto newTileOutType = mlir::cast<mlir::RankedTensorType>(tileOutType.changeElemType(finalDstElemType));
    auto newTileOp = rewriter.create<IE::TileOp>(tileOp.getLoc(), newTileOutType, newConvertOp.getOutput(),
                                                 tileOp.getRepeatsValuesAttr());

    rewriter.replaceOp(convertOp, newTileOp.getOutput());

    nestedLogger.trace("Successfully fused Convert -> Tile -> Convert pattern");
    return mlir::success();
}

//
// FuseGroupConvWithBiasAdd
//
// Pattern: GroupConv(input, weight) -> Add(groupconv_out, bias_1xCx1x1)
//          where bias_1xCx1x1 may arrive via Tile(bias) or directly.
//
// When this pattern matches, it folds the bias into GroupConvolutionOp (weight-table bias),
// replacing the Add and eliminating any preceding Tile on the bias path.
//
// Fires for: (a) static filter + static bias -- fuse unconditionally (compile-time constant bias);
//            (b) non-constant filter + static bias -- fuse when platform supports the new weight-table
//                bias format (the bias must still be static: downstream NCE lowering always
//                materializes the weight-table bias from a compile-time Const::DeclareOp).
// Does NOT fire for: static filter + non-constant bias (e.g. MVN scale + conditioning-network bias) --
// the equal-shape NCE.Eltwise path is faster for small tensors in that case; non-constant bias with a
// non-constant filter, regardless of platform; or when the Add carries a scale operand/static_scale
// attribute, since the weight-table bias slot cannot represent a post-bias scale.
//

class FuseGroupConvWithBiasAdd final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    FuseGroupConvWithBiasAdd(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
        setDebugName("FuseGroupConvWithBiasAdd");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const Logger& _log;
};

mlir::LogicalResult FuseGroupConvWithBiasAdd::matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.use_empty()) {
        return mlir::failure();
    }
    _log.trace("FuseGroupConvWithBiasAdd: Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto lhsGroupConv = origOp.getInput1().getDefiningOp<IE::GroupConvolutionOp>();
    auto rhsGroupConv = origOp.getInput2().getDefiningOp<IE::GroupConvolutionOp>();

    IE::GroupConvolutionOp groupConvOp;
    mlir::Value biasTileValue;

    if (lhsGroupConv != nullptr && rhsGroupConv == nullptr) {
        groupConvOp = lhsGroupConv;
        biasTileValue = origOp.getInput2();
    } else if (rhsGroupConv != nullptr && lhsGroupConv == nullptr) {
        groupConvOp = rhsGroupConv;
        biasTileValue = origOp.getInput1();
    } else {
        return mlir::failure();
    }

    if (groupConvOp.getBias() != nullptr) {
        _log.nest().trace("GroupConvolutionOp already has bias");
        return mlir::failure();
    }

    if (!groupConvOp.getOutput().hasOneUse()) {
        _log.nest().trace("GroupConvolutionOp has multiple users");
        return mlir::failure();
    }

    if (groupConvOp.getPostOpAttr() != nullptr) {
        _log.nest().trace("GroupConvolutionOp already has post-op");
        return mlir::failure();
    }

    // Do not fuse if the Add changes the output type or applies padding; those semantics
    // cannot be preserved when the Add is removed and replaced by the GroupConvolution bias.
    if (origOp.getOutput().getType() != groupConvOp.getOutput().getType() || origOp.getInputPaddingAttr() != nullptr ||
        origOp.getOutputPaddingAttr() != nullptr || origOp.getScale() != nullptr ||
        origOp.getStaticScaleAttr() != nullptr) {
        _log.nest().trace("Add output type or padding attributes are incompatible with fusion");
        return mlir::failure();
    }

    // Accept bias either from a preceding TileOp (original pattern) or directly when it is
    // already NxCx1x1 -- the latter happens when FoldTileOpRewriter already removed the Tile.
    auto tileOp = biasTileValue.getDefiningOp<IE::TileOp>();
    mlir::Value directBias;
    if (tileOp != nullptr) {
        if (!tileOp.getOutput().hasOneUse()) {
            _log.nest().trace("TileOp has multiple users");
            return mlir::failure();
        }
        directBias = tileOp.getInput();
    } else {
        directBias = biasTileValue;
    }

    // Bias must be 4D with N==1, H==1, W==1 (1xCx1x1). N!=1 would require per-batch bias
    // folding which flat-indexing into the weight table does not support.
    // IE.GroupConvolution requires a floating-point bias; reject quantized/integral types.
    const auto directBiasType = mlir::cast<vpux::NDTypeInterface>(directBias.getType());
    if (!mlir::isa<mlir::FloatType>(directBiasType.getElementType())) {
        _log.nest().trace("Bias element type is not floating point");
        return mlir::failure();
    }
    auto directBiasShape = directBiasType.getShape();
    if (directBiasShape.size() != 4 || directBiasShape[Dims4D::Act::N] != 1 || directBiasShape[Dims4D::Act::H] != 1 ||
        directBiasShape[Dims4D::Act::W] != 1) {
        _log.nest().trace("Bias is not a 4D spatial broadcast shape (1xCx1x1)");
        return mlir::failure();
    }
    const auto groupConvOutputShape = mlir::cast<vpux::NDTypeInterface>(groupConvOp.getOutput().getType()).getShape();
    if (directBiasShape[Dims4D::Act::C] != groupConvOutputShape[Dims4D::Act::C]) {
        _log.nest().trace("Bias channel dimension ({0}) does not match GroupConvolution output channels ({1})",
                          directBiasShape[Dims4D::Act::C], groupConvOutputShape[Dims4D::Act::C]);
        return mlir::failure();
    }

    const bool isStaticFilter = groupConvOp.getFilter().getDefiningOp<Const::DeclareOp>() != nullptr;
    const bool isStaticBias = directBias.getDefiningOp<Const::DeclareOp>() != nullptr;

    // Static-weight GroupConv with non-constant bias (e.g. MVN scale + conditioning-network bias):
    // keep the Add so equal-shape NCE.Eltwise handles it -- faster than SW broadcast Add for small
    // tensors than converting to a weight-table bias with a runtime value.
    if (isStaticFilter && !isStaticBias) {
        _log.nest().trace("Static-weight GroupConv with non-constant bias, skipping");
        return mlir::failure();
    }

    // For non-constant-weight GroupConv, the platform must support the new weight-table bias format.
    // The bias itself must still be static: downstream NCE lowering (DepthConvToNCE) always
    // materializes the weight-table bias from a compile-time Const::DeclareOp, regardless of
    // weight-table format, so a runtime bias value cannot be represented here.
    if (!isStaticFilter) {
        if (!isStaticBias) {
            _log.nest().trace(
                    "Non-constant filter with non-constant bias, skipping (NCE lowering requires a const bias)");
            return mlir::failure();
        }
        const auto arch = config::getArch(origOp.getOperation());
        if (!VPU::MPEEngineConfig::isNewWeightTableFormatSupportedWithDwOps(arch)) {
            // On older platforms the NCE DW weight table doesn't support non-constant bias.
            // Do not detile -- let FoldTileOpRewriter decide whether to keep or remove the Tile:
            //   small tensors: Tile stays -> equal-shape Add -> NCE.Eltwise (faster than SW broadcast Add)
            //   large tensors: FoldTileOpRewriter already folded the Tile -> SW broadcast Add
            return mlir::failure();
        }
    }

    _log.trace("Fusing GroupConv + Add(bias) at '{0}'", origOp.getLoc());

    // Transfer the Add's post-op / clamp attributes to the new GroupConvolutionOp. This is
    // semantically correct: at hardware level the weight-table bias is added in the NCE
    // accumulator stage (before the PPE fires), so the PPE activation acts on the already-biased
    // output -- identical to the original graph where PPE fired on the output of the Add.
    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            origOp, groupConvOp.getInput(), groupConvOp.getFilter(), directBias, groupConvOp.getStridesAttr(),
            groupConvOp.getPadsBeginAttr(), groupConvOp.getPadsEndAttr(), groupConvOp.getDilationsAttr(),
            groupConvOp.getGroupsAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(),
            groupConvOp.getOutputPaddingAttr(), groupConvOp.getInputPaddingAttr());

    rewriter.eraseOp(groupConvOp);

    return mlir::success();
}

//
// OptimizeTileOpPass
//

class OptimizeTileOpPass final : public IE::impl::OptimizeTileOpBase<OptimizeTileOpPass> {
public:
    explicit OptimizeTileOpPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void OptimizeTileOpPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FoldTileOpRewriter>(&ctx, _log);
    patterns.add<FuseTileConvertRewrite>(&ctx, _log);
    patterns.add<FuseGroupConvWithBiasAdd>(&ctx, _log);

    auto func = getOperation();
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createOptimizeTileOpPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeTileOpPass(Logger log) {
    return std::make_unique<OptimizeTileOpPass>(log);
}
