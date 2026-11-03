//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/overlap_distribution_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEADDTOCONVBIAS
#define GEN_PASS_DEF_FUSEADDTOCONVBIAS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Returns the combined PPE scale factor coming from a Convolution's optional per-channel 'scale'
// operand and 'static_scale' attribute (GroupConvolution has neither of these, so it is always
// 1.0). Returns std::nullopt if 'scale' is present but is not a uniform compile-time constant, in
// which case the factor cannot be divided out of the bias here. Requires 'scale' to be a *direct*
// Const::DeclareOp rather than using IE::getConstParentOp, which walks through value-changing
// producers such as IE.Negative/IE.FakeQuantize without evaluating them - e.g. for
// Negative(Const<1.0>), it would return the unnegated 1.0 instead of the actual -1.0.
template <typename ConvOpTy>
std::optional<double> getConvScaleFactor(ConvOpTy convOp) {
    if constexpr (std::is_same_v<ConvOpTy, IE::ConvolutionOp>) {
        double scale = convOp.getStaticScaleAttr() != nullptr ? convOp.getStaticScaleAttr().getValueAsDouble() : 1.0;
        if (convOp.getScale() != nullptr) {
            auto scaleConst = convOp.getScale().template getDefiningOp<Const::DeclareOp>();
            if (scaleConst == nullptr) {
                return std::nullopt;
            }
            const auto scaleContent = scaleConst.getContentAttr().fold();
            if (!scaleContent.isSplat()) {
                return std::nullopt;
            }
            scale *= scaleContent.template getSplatValue<double>();
        }
        return scale;
    } else {
        return 1.0;
    }
}

//
// FuseAddToConvBiasPass
//
// IE.Add(ConvOutput, Bias) -> IE.Convolution(..., bias=Bias)
// (also applies to IE.GroupConvolution)
//
// Eliminates a standalone Add by feeding its non-convolution operand directly into the
// Convolution's (or GroupConvolution's) own (currently empty) bias operand, as long as:
//   - the Convolution has no bias yet, has a single use (the Add), and is NCE-eligible
//   - the "bias" operand - after looking through a chain of pure view ops (AffineReshape/
//     ShapeCast/PermuteCast/QuantizeCast/LayoutCast) and at most one IE.Tile left behind by
//     an earlier broadcast-materialization pass (BroadcastInputForAdd/FuseInefficientTileForAdd/
//     BroadcastInputForMultiply) - is broadcastable per-channel, i.e. shape [1, OC, 1, 1]
//   - that value's definition either already dominates the Convolution, or can be hoisted
//     above it (the value's whole defining chain is free of side effects and ultimately only
//     depends on values that dominate the Convolution)
//   - the target architecture supports a non-constant (runtime) weight-table bias, i.e.
//     VPU::getVPUStrategyFactory(...)->useNewWeightTableFormat() returns true for this op
//     (see convert_IE_to_VPU_NCE.cpp's ConvToNCE). On older architectures
//     this predicate is false, so the pass is a no-op there instead of emitting IR that would
//     later fail to lower to VPU.NCE.Convolution.
//
// Placed in buildFinalTransformationPipeline, after BroadcastInputForMultiply/
// FuseInefficientTileForAdd: by that point any bias-like Add operand that needed broadcasting
// has already been processed by those passes, so an IE.Tile is the only broadcast wrapper left
// to look through (rather than, say, an IE.Broadcast). The Tile itself may in turn be wrapped
// in further view ops inserted while balancing the Add's operand shapes/layouts.
//
// This avoids materializing a separate NCE.Eltwise (or SW-kernel) Add entirely: the bias is
// added for free as part of the Convolution's own DPU pass.
//

class FuseAddToConvBiasPass final : public IE::impl::FuseAddToConvBiasBase<FuseAddToConvBiasPass> {
public:
    explicit FuseAddToConvBiasPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    bool isFusableBiasValue(mlir::Value bias, int64_t OC) const;
    mlir::Value lookThroughViewOpsAndTile(mlir::Value bias, int64_t OC) const;
    mlir::Value bypassF16BiasConvert(mlir::Value bias, int64_t OC, double scaleFactor, mlir::Operation* convOp,
                                     const mlir::DominanceInfo& domInfo) const;
    bool canHoistBiasChain(mlir::Value bias, mlir::Operation* convOp, const mlir::DominanceInfo& domInfo) const;
    bool hoistBiasChainIfNeeded(mlir::Value bias, mlir::Operation* convOp, const mlir::DominanceInfo& domInfo) const;
    mlir::Value rescaleBiasByScale(mlir::Value bias, double scale, mlir::Operation* convOp) const;
    template <typename ConvOpTy>
    bool tryFuseConv(IE::AddOp addOp, mlir::Value activationInput, mlir::Value biasInput,
                     const mlir::DominanceInfo& domInfo) const;
    bool tryFuse(IE::AddOp addOp, mlir::Value activationInput, mlir::Value biasInput,
                 const mlir::DominanceInfo& domInfo) const;
    void fuseAddIntoConvBias(IE::AddOp addOp, const mlir::DominanceInfo& domInfo) const;
};

bool FuseAddToConvBiasPass::isFusableBiasValue(mlir::Value bias, int64_t OC) const {
    const auto biasType = mlir::cast<vpux::NDTypeInterface>(bias.getType());
    if (!mlir::isa<mlir::FloatType>(biasType.getElementType())) {
        return false;
    }

    const auto biasShape = biasType.getShape();
    if (!biasShape.isStatic() || biasShape.size() != 4) {
        return false;
    }

    return biasShape[Dims4D::Act::N] == 1 && biasShape[Dims4D::Act::H] == 1 && biasShape[Dims4D::Act::W] == 1 &&
           biasShape[Dims4D::Act::C] == OC;
}

// Walks back from 'bias' through a chain of pure view ops (AffineReshape/ShapeCast/PermuteCast/
// QuantizeCast/LayoutCast) and at most one IE.Tile, stopping as soon as a value with the
// expected per-channel shape [1, OC, 1, 1] is found. Returns that value, or the original 'bias'
// unchanged if no such value is found along the chain (the caller re-validates the result with
// isFusableBiasValue).
mlir::Value FuseAddToConvBiasPass::lookThroughViewOpsAndTile(mlir::Value bias, int64_t OC) const {
    mlir::Value current = bias;
    bool tileSeen = false;
    while (current != nullptr) {
        if (isFusableBiasValue(current, OC)) {
            return current;
        }
        auto* viewOp = current.getDefiningOp();
        if (auto tileOp = mlir::dyn_cast_or_null<IE::TileOp>(viewOp)) {
            if (tileSeen) {
                break;
            }
            tileSeen = true;
            current = tileOp.getInput();
        } else if (mlir::isa_and_present<IE::AffineReshapeOp, IE::ShapeCastOp, IE::PermuteCastOp, IE::QuantizeCastOp,
                                         IE::LayoutCastOp>(viewOp)) {
            current = viewOp->getOperand(0);
        } else {
            break;
        }
    }
    return bias;
}

// Rescale a constant bias when the Convolution has a non-unit scale. Otherwise, if the
// per-channel bias was converted from f32 to f16 before the Add, rebuild its pure view chain
// from the original f32 value to avoid a later f16-to-f32 conversion for runtime bias.
mlir::Value FuseAddToConvBiasPass::bypassF16BiasConvert(mlir::Value bias, int64_t OC, double scaleFactor,
                                                        mlir::Operation* convOp,
                                                        const mlir::DominanceInfo& domInfo) const {
    if (!isDoubleEqual(scaleFactor, 1.0)) {
        return rescaleBiasByScale(bias, scaleFactor, convOp);
    }
    if (!isFusableBiasValue(bias, OC) || !mlir::cast<vpux::NDTypeInterface>(bias.getType()).getElementType().isF16()) {
        return bias;
    }

    SmallVector<mlir::Operation*> viewChain;
    auto current = bias;
    mlir::Value f32Source;
    while (true) {
        if (auto convertOp = current.getDefiningOp<IE::ConvertOp>()) {
            const auto sourceType = mlir::cast<vpux::NDTypeInterface>(convertOp.getInput().getType());
            const auto resultType = mlir::cast<vpux::NDTypeInterface>(convertOp.getOutput().getType());
            if (sourceType.getElementType().isF32() && resultType.getElementType().isF16()) {
                f32Source = convertOp.getInput();
            }
            break;
        }

        auto* viewOp = current.getDefiningOp();
        if (mlir::isa_and_present<IE::AffineReshapeOp, IE::ReshapeOp, IE::ShapeCastOp, IE::PermuteCastOp,
                                  IE::LayoutCastOp>(viewOp)) {
            viewChain.push_back(viewOp);
            current = viewOp->getOperand(0);
            continue;
        }
        return bias;
    }

    if (f32Source == nullptr || !domInfo.properlyDominates(f32Source, convOp)) {
        return bias;
    }

    mlir::OpBuilder builder(convOp);
    auto rebuilt = f32Source;
    for (auto viewIt = viewChain.rbegin(); viewIt != viewChain.rend(); ++viewIt) {
        auto* viewOp = *viewIt;
        const auto loc = appendLoc(viewOp->getLoc(), "bias_f32");
        if (auto affineReshape = mlir::dyn_cast<IE::AffineReshapeOp>(viewOp)) {
            rebuilt = IE::AffineReshapeOp::create(builder, loc, rebuilt, affineReshape.getDimMappingAttr(),
                                                  affineReshape.getShapeValueAttr());
        } else if (auto reshape = mlir::dyn_cast<IE::ReshapeOp>(viewOp)) {
            rebuilt = IE::ReshapeOp::create(builder, loc, rebuilt, reshape.getShapeValueAttr());
        } else if (auto shapeCast = mlir::dyn_cast<IE::ShapeCastOp>(viewOp)) {
            rebuilt = IE::ShapeCastOp::create(builder, loc, rebuilt, shapeCast.getShapeAttr());
        } else if (auto permuteCast = mlir::dyn_cast<IE::PermuteCastOp>(viewOp)) {
            rebuilt = IE::PermuteCastOp::create(builder, loc, rebuilt, permuteCast.getDstOrder(),
                                                permuteCast.getMemPerm());
        } else if (auto layoutCast = mlir::dyn_cast<IE::LayoutCastOp>(viewOp)) {
            rebuilt = IE::LayoutCastOp::create(builder, loc, rebuilt, layoutCast.getDstOrderAttr());
        }
    }

    return isFusableBiasValue(rebuilt, OC) ? rebuilt : bias;
}

bool FuseAddToConvBiasPass::canHoistBiasChain(mlir::Value bias, mlir::Operation* convOp,
                                              const mlir::DominanceInfo& domInfo) const {
    if (domInfo.properlyDominates(bias, convOp)) {
        return true;
    }
    auto* op = bias.getDefiningOp();
    if (op == nullptr || op == convOp || !mlir::isMemoryEffectFree(op)) {
        return false;
    }
    return llvm::all_of(op->getOperands(), [&](mlir::Value operand) {
        return canHoistBiasChain(operand, convOp, domInfo);
    });
}

// The bias chain found by lookThroughViewOpsAndTile may sit textually after 'convOp' (e.g. the
// Add's operands were computed independently and happen to be ordered after the Convolution).
// If every op in that chain is free of side effects and ultimately only depends on values that
// already dominate 'convOp', it is safe to hoist the whole chain to right before 'convOp' instead
// of giving up on the fusion.
bool FuseAddToConvBiasPass::hoistBiasChainIfNeeded(mlir::Value bias, mlir::Operation* convOp,
                                                   const mlir::DominanceInfo& domInfo) const {
    if (domInfo.properlyDominates(bias, convOp)) {
        return true;
    }

    auto* op = bias.getDefiningOp();
    if (op == nullptr || op == convOp || !mlir::isMemoryEffectFree(op)) {
        return false;
    }

    for (auto operand : op->getOperands()) {
        if (!hoistBiasChainIfNeeded(operand, convOp, domInfo)) {
            return false;
        }
    }

    op->moveBefore(convOp);
    return true;
}

// HW adds the weight-table bias to the accumulator before applying the Convolution's own
// per-channel scale (scale operand / static_scale), so fusing in the bias as-is would turn
// 'scale * conv + bias' into 'scale * (conv + bias)'. Compensate with 'bias / scale', folded into
// the constant, matching the equivalent rescale already done when fusing IE.ScaleShift into a
// Convolution's bias (see convolution_utils.cpp). Returns a null Value for a non-constant bias:
// inserting a runtime IE.Divide there would turn a free fusion into an extra NCE/SW op. Requires
// 'bias' to be a *direct* Const::DeclareOp (see getConvScaleFactor() for why IE::getConstParentOp
// is not used here): rescaling the constant found behind a value-changing producer would silently
// drop that producer's effect from the result.
mlir::Value FuseAddToConvBiasPass::rescaleBiasByScale(mlir::Value bias, double scale, mlir::Operation* convOp) const {
    auto constOp = bias.getDefiningOp<Const::DeclareOp>();
    if (constOp == nullptr) {
        return nullptr;
    }

    auto contentAttr = constOp.transformContentAttr().rescale(1 / scale).get();
    mlir::OpBuilder builder(convOp);
    return Const::DeclareOp::create(builder, appendLoc(constOp.getLoc(), "rescaled"), constOp.getType(), contentAttr)
            .getOutput();
}

template <typename ConvOpTy>
bool FuseAddToConvBiasPass::tryFuseConv(IE::AddOp addOp, mlir::Value activationInput, mlir::Value biasInput,
                                        const mlir::DominanceInfo& domInfo) const {
    auto convOutput = activationInput;
    while (convOutput != nullptr) {
        auto* viewOp = convOutput.getDefiningOp();
        if (!mlir::isa_and_present<IE::AffineReshapeOp, IE::ShapeCastOp, IE::PermuteCastOp, IE::LayoutCastOp>(viewOp)) {
            break;
        }
        if (!viewOp->hasOneUse()) {
            return false;
        }
        convOutput = viewOp->getOperand(0);
    }
    if (convOutput == nullptr) {
        return false;
    }

    auto convOp = convOutput.getDefiningOp<ConvOpTy>();
    if (convOp == nullptr || convOp.getBias() != nullptr) {
        return false;
    }
    // The bias would be added before the activation/clamp instead of after it.
    if (convOp.getPostOpAttr() != nullptr || convOp.getClampAttr() != nullptr) {
        _log.nest().trace("Convolution at '{0}' has a post-op or clamp, skipping", convOp->getLoc());
        return false;
    }
    if (addOp.getScale() != nullptr || addOp.getStaticScaleAttr() != nullptr) {
        if constexpr (std::is_same_v<ConvOpTy, IE::ConvolutionOp>) {
            if (convOp.getScale() != nullptr || convOp.getStaticScaleAttr() != nullptr ||
                (addOp.getScale() != nullptr && !canHoistBiasChain(addOp.getScale(), convOp, domInfo))) {
                return false;
            }
        } else {
            return false;
        }
    }
    if (!convOp->hasOneUse()) {
        _log.nest().trace("Convolution at '{0}' has more than one use, skipping", convOp->getLoc());
        return false;
    }
    if (activationInput.getType() != convOp.getOutput().getType() ||
        activationInput.getType() != addOp.getOutput().getType()) {
        return false;
    }
    auto convWithPostOp = mlir::cast<IE::LayerWithPostOpInterface>(convOp.getOperation());
    const auto addPostOp = addOp.getPostOpAttr();
    const auto logCb = [&](const formatv_object_base& msg) {
        _log.nest().trace("{0}", msg.str());
    };
    if (addPostOp != nullptr && !convWithPostOp.isSupportedPostOpAttr(addPostOp, logCb)) {
        return false;
    }
    const auto fusedClamp = IE::getFusedClamp(_log, convOp.getClampAttr(), addPostOp, addOp.getClampAttr(),
                                              convWithPostOp, addOp.getOutput().getType());
    if (!fusedClamp.has_value()) {
        return false;
    }
    // Bias must not be (transitively) produced by the Convolution itself - e.g. IE.Add(x, x).
    if (biasInput.getDefiningOp() == convOp.getOperation()) {
        return false;
    }
    // A constant bias is rescaled per-channel by the filter's quantized scale (NCESparsity::getBiasFunc)
    // before it is stored in the weight table; a runtime bias is wired in as-is (see
    // createRuntimeBiasForWeightTable), so it would be numerically wrong when input or filter is quantized.
    const auto inputElemType = mlir::cast<vpux::NDTypeInterface>(convOp.getInput().getType()).getElementType();
    const auto filterElemType = mlir::cast<vpux::NDTypeInterface>(convOp.getFilter().getType()).getElementType();
    if (mlir::isa<mlir::quant::QuantizedType>(inputElemType) || mlir::isa<mlir::quant::QuantizedType>(filterElemType)) {
        _log.nest().trace("Convolution at '{0}' has quantized input/filter, skipping", convOp->getLoc());
        return false;
    }
    // HW adds bias to the accumulator before applying the Convolution's own per-channel scale
    // (scale operand / static_scale); getConvScaleFactor() returns std::nullopt when 'scale' is a
    // non-constant or non-uniform operand, in which case there is no single value to divide the
    // bias by here.
    const auto scaleFactor = getConvScaleFactor(convOp);
    if (!scaleFactor.has_value()) {
        _log.nest().trace("Convolution at '{0}' has a non-constant/non-uniform scale operand, skipping",
                          convOp->getLoc());
        return false;
    }
    if (isDoubleEqual(*scaleFactor, 0.0)) {
        _log.nest().trace("Convolution at '{0}' has a zero scale, skipping", convOp->getLoc());
        return false;
    }

    const auto OC = getShape(convOp.getOutput())[Dims4D::Act::C];
    if constexpr (std::is_same_v<ConvOpTy, IE::ConvolutionOp>) {
        if (addOp.getScale() != nullptr) {
            const auto scaleType = mlir::cast<vpux::NDTypeInterface>(addOp.getScale().getType());
            const auto scaleShape = scaleType.getShape();
            if ((!scaleType.getElementType().isF16() && !scaleType.getElementType().isF32()) ||
                !scaleShape.isStatic() || scaleShape.size() != 4 || scaleShape[Dims4D::Act::N] != OC ||
                scaleShape[Dims4D::Act::C] != 1 || scaleShape[Dims4D::Act::H] != 1 || scaleShape[Dims4D::Act::W] != 1) {
                return false;
            }
        }
    }

    // Look through any view ops and at most one IE.Tile that broadcasts a per-channel value up
    // to the full output shape. Such a Tile (and surrounding reshapes/layout casts balancing the
    // Add's operand shapes) may be left behind by BroadcastInputForAdd (and not cleaned up by
    // FuseInefficientTileForAdd, which only restores innermost-memory-dim broadcasts) upstream in
    // buildFinalTransformationPipeline. The pre-Tile value is what we actually want to wire into
    // the Convolution's bias operand - not the materialized full-size tensor.
    const auto realBias = lookThroughViewOpsAndTile(biasInput, OC);
    if (realBias.getDefiningOp() == convOp.getOperation() || getShape(realBias) == getShape(convOp.getOutput())) {
        return false;
    }
    if (!canHoistBiasChain(realBias, convOp.getOperation(), domInfo)) {
        _log.nest().trace("Bias value cannot be made to dominate the Convolution, skipping");
        return false;
    }
    if (!isDoubleEqual(*scaleFactor, 1.0) && realBias.template getDefiningOp<Const::DeclareOp>() == nullptr) {
        _log.nest().trace("Convolution at '{0}' has a scale and a non-constant bias, skipping", convOp->getLoc());
        return false;
    }

    const auto biasForConv = bypassF16BiasConvert(realBias, OC, *scaleFactor, convOp.getOperation(), domInfo);
    if (biasForConv == nullptr || !isFusableBiasValue(biasForConv, OC)) {
        return false;
    }

    // The bias is wired directly into the existing Convolution operand list, so its definition
    // must dominate the Convolution - hoisting the (side-effect-free) bias chain above it first
    // if it does not already.
    if (!hoistBiasChainIfNeeded(biasForConv, convOp.getOperation(), domInfo)) {
        _log.nest().trace("Bias value cannot be made to dominate the Convolution, skipping");
        return false;
    }
    if constexpr (std::is_same_v<ConvOpTy, IE::ConvolutionOp>) {
        if (addOp.getScale() != nullptr && !hoistBiasChainIfNeeded(addOp.getScale(), convOp, domInfo)) {
            return false;
        }
    }

    // GroupConvolution's strategy selects the new weight-table format only when a
    // non-constant bias is already present. Assign first, then roll back if the op is not
    // supported with that bias.
    convOp.getBiasMutable().assign(biasForConv);
    if (!VPU::getVPUStrategyFactory(convOp->getContext())->useNewWeightTableFormat(convOp, false)) {
        convOp.getBiasMutable().assign(nullptr);
        _log.nest().trace("Architecture or NCE invariants do not support the fused runtime bias, skipping");
        return false;
    }
    if (addPostOp != nullptr) {
        convOp.setPostOpAttr(addPostOp);
    }
    if (*fusedClamp != nullptr) {
        convOp.setClampAttr(*fusedClamp);
    }
    if constexpr (std::is_same_v<ConvOpTy, IE::ConvolutionOp>) {
        if (addOp.getScale() != nullptr) {
            convOp.getScaleMutable().assign(addOp.getScale());
        }
        if (addOp.getStaticScaleAttr() != nullptr) {
            convOp.setStaticScaleAttr(addOp.getStaticScaleAttr());
        }
    }
    if (VPU::NCEInvariant::isSupported(convOp, _log.nest()).failed()) {
        convOp.getBiasMutable().assign(nullptr);
        convOp.setPostOpAttr(nullptr);
        convOp.setClampAttr(nullptr);
        if constexpr (std::is_same_v<ConvOpTy, IE::ConvolutionOp>) {
            convOp.getScaleMutable().assign(nullptr);
            convOp.setStaticScaleAttr(nullptr);
        }
        _log.nest().trace("Architecture or NCE invariants do not support the fused runtime bias, skipping");
        return false;
    }

    _log.trace("Fusing '{0}' into the bias of '{1}'", addOp->getLoc(), convOp->getLoc());

    addOp.getOutput().replaceAllUsesWith(activationInput);
    addOp.erase();
    return true;
}

bool FuseAddToConvBiasPass::tryFuse(IE::AddOp addOp, mlir::Value activationInput, mlir::Value biasInput,
                                    const mlir::DominanceInfo& domInfo) const {
    if (tryFuseConv<IE::ConvolutionOp>(addOp, activationInput, biasInput, domInfo)) {
        return true;
    }
    return tryFuseConv<IE::GroupConvolutionOp>(addOp, activationInput, biasInput, domInfo);
}

void FuseAddToConvBiasPass::fuseAddIntoConvBias(IE::AddOp addOp, const mlir::DominanceInfo& domInfo) const {
    _log.trace("Got '{0}' at '{1}'", addOp->getName(), addOp->getLoc());

    if (addOp.getAutoBroadcast() != IE::AutoBroadcastType::NUMPY) {
        return;
    }

    const auto input1 = addOp.getInput1();
    const auto input2 = addOp.getInput2();

    if (tryFuse(addOp, input1, input2, domInfo)) {
        return;
    }
    tryFuse(addOp, input2, input1, domInfo);
}

void FuseAddToConvBiasPass::safeRunOnFunc() {
    auto func = getOperation();
    const mlir::DominanceInfo domInfo(func);

    SmallVector<IE::AddOp> addOps;
    func.walk([&](IE::AddOp addOp) {
        addOps.push_back(addOp);
    });

    for (auto addOp : addOps) {
        fuseAddIntoConvBias(addOp, domInfo);
    }
}

}  // namespace

//
// createFuseAddToConvBiasPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseAddToConvBiasPass(const Logger& log) {
    return std::make_unique<FuseAddToConvBiasPass>(log);
}
