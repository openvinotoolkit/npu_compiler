//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <limits>
#include <memory>

namespace vpux::VPU::arch60xx {
#define GEN_PASS_DECL_EXTRACTREDUCEMAXFROMEXPSTABILIZED
#define GEN_PASS_DEF_EXTRACTREDUCEMAXFROMEXPSTABILIZED
#include "vpux/compiler/NPU60XX/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU::arch60xx

using namespace vpux;

namespace {

bool requiresCTiling(VPU::NCEConvolutionOp convOp) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(convOp.getOutput().getType());
    auto outTileShape = outputType.getShape().toValues();
    outTileShape[Dims4D::Act::H] = 1;
    outTileShape[Dims4D::Act::W] = 1;
    const TileInfo outTile(outTileShape);

    auto tileTypes = VPU::getTileTypes(convOp, outTile, std::nullopt, std::nullopt);
    VPUX_THROW_WHEN(tileTypes.size() < 3, "Unexpected tile types count {0} for '{1}'", tileTypes.size(), convOp);

    auto reduceShape = tileTypes[2].getShape().toValues();
    reduceShape[Dims4D::Act::C] = 1;
    reduceShape[Dims4D::Act::H] = 1;
    reduceShape[Dims4D::Act::W] = 1;
    const auto reduceType = tileTypes[2].changeShape(reduceShape);

    SmallVector<vpux::NDTypeInterface> operands{reduceType};
    operands.append(tileTypes.begin(), tileTypes.end());
    const auto OC = tileTypes[2].getShape()[Dims4D::Act::C];

    const auto requiredCMX = VPU::getRequiredCMXSizeForNCEOps(convOp.getOperation(), operands, OC);
    const auto availableCMX = VPU::getTotalCMXFragmentationAwareSize(convOp.getOperation());
    return requiredCMX >= availableCMX;
}

mlir::LogicalResult rebuildWeightsTable(mlir::PatternRewriter& rewriter, VPU::NCEConvolutionOp convOp,
                                        int64_t padSize) {
    auto scale = convOp.getWeightTableScale();
    auto bias = convOp.getWeightTableBias();
    if (scale == nullptr || bias == nullptr) {
        return mlir::failure();
    }

    auto scaleConstOp = scale.getDefiningOp<Const::DeclareOp>();
    auto biasConstOp = bias.getDefiningOp<Const::DeclareOp>();
    if (scaleConstOp == nullptr || biasConstOp == nullptr) {
        return mlir::failure();
    }

    auto scaleContent = scaleConstOp.getContent();
    auto scaleValues = SmallVector<float>(scaleContent.getValues<float>());
    auto biasContent = biasConstOp.getContent();
    auto biasValues = SmallVector<float>(biasContent.getValues<float>());
    if (padSize <= 0 || scaleValues.size() != biasValues.size() || static_cast<size_t>(padSize) > biasValues.size()) {
        return mlir::failure();
    }

    auto ppeAttr = mlir::cast<VPU::PPEFpAttr>(convOp.getPpeAttr());
    const auto scaleAttr = ppeAttr.getScale();
    const auto biasAttr = ppeAttr.getBias();
    // `padSize` marks trailing channels that must be ignored by Softmax. Extracting ReduceMax requires those channels
    // have to be neutralized by ReduceMax's producer. For a NCE producer, Set scale=1.0f and bias=fp32min for the
    // trailing channels so that the final value is fp32min.
    for (size_t i = 0; i < scaleValues.size(); ++i) {
        if (i < scaleValues.size() - padSize) {
            scaleValues[i] = scaleAttr == nullptr ? scaleValues[i] : static_cast<float>(scaleAttr.getValueAsDouble());
            biasValues[i] = biasAttr == nullptr ? biasValues[i] : static_cast<float>(biasAttr.getValueAsDouble());
        } else {
            scaleValues[i] = 1.0f;
            biasValues[i] = std::numeric_limits<float>::lowest();
        }
    }

    auto neutralPpeAttr = VPU::PPEFpAttr::get(convOp.getContext(), ppeAttr.getMode(), ppeAttr.getClampLow(),
                                              ppeAttr.getClampHigh(), /*scale=*/nullptr, ppeAttr.getPreluAlpha(),
                                              /*bias=*/nullptr, ppeAttr.getAdder(), ppeAttr.getIn1Mult(),
                                              ppeAttr.getIn2Mult(), ppeAttr.getSprlut());
    rewriter.modifyOpInPlace(convOp, [&]() {
        convOp.setPpeAttr(neutralPpeAttr);
    });

    rewriter.setInsertionPoint(convOp);
    auto scaleType = mlir::cast<mlir::ShapedType>(scale.getType());
    const auto newScaleContent = Const::ContentAttr::get(Const::createConstContent(scaleType, ArrayRef(scaleValues)));
    auto newScale =
            Const::DeclareOp::create(rewriter, appendLoc(convOp.getLoc(), "_new_scale"), scaleType, newScaleContent);

    auto biasType = mlir::cast<mlir::ShapedType>(bias.getType());
    const auto newBiasContent = Const::ContentAttr::get(Const::createConstContent(biasType, ArrayRef(biasValues)));
    auto newBias =
            Const::DeclareOp::create(rewriter, appendLoc(convOp.getLoc(), "_new_bias"), biasType, newBiasContent);

    convOp.getWeightTableScale().replaceUsesWithIf(newScale, [&](mlir::OpOperand& operand) {
        return operand.getOwner() == convOp.getOperation();
    });

    convOp.getWeightTableBias().replaceUsesWithIf(newBias, [&](mlir::OpOperand& operand) {
        return operand.getOwner() == convOp.getOperation();
    });

    return mlir::success();
}

class ExtractReduceMaxFromExpStabilizedRewriter final : public mlir::OpRewritePattern<VPU::SoftMaxOp> {
public:
    ExtractReduceMaxFromExpStabilizedRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::SoftMaxOp>(ctx), _log(log) {
        this->setDebugName("ExtractReduceMaxFromExpStabilizedRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ExtractReduceMaxFromExpStabilizedRewriter::matchAndRewrite(VPU::SoftMaxOp origOp,
                                                                               mlir::PatternRewriter& rewriter) const {
    if (!origOp->hasAttr("SkipNormalization") || !origOp.getInput().hasOneUse()) {
        return mlir::failure();
    }
    auto ctx = this->getContext();
    auto convOp = mlir::dyn_cast_if_present<VPU::NCEConvolutionOp>(origOp.getInput().getDefiningOp());
    if (convOp == nullptr) {
        _log.debug("extract failed '{0}' at '{1}', unsupported producer op", origOp->getName(), origOp->getLoc());
        return mlir::failure();
    }

    const auto axis = origOp.getAxisIndAttr().getInt();
    if (axis != Dims4D::Act::C.ind()) {
        _log.debug("extract failed '{0}' at '{1}', origOp axis is not C", origOp->getName(), origOp->getLoc());
        return mlir::failure();
    }

    // Track E#215367. Support tiling on reduction axis
    auto convOutShape = getShape(convOp.getOutput());
    if (requiresCTiling(convOp) || convOutShape[Dim(axis)] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        _log.debug("extract failed '{0}' at '{1}', producer convolution requires tiling on C", origOp->getName(),
                   origOp->getLoc());
        return mlir::failure();
    }

    int64_t padSize = 0;
    if (origOp.getPadSize().has_value()) {
        padSize = origOp.getPadSizeAttr().getInt();
    }

    if (padSize > 0 && mlir::failed(rebuildWeightsTable(rewriter, convOp, padSize))) {
        _log.debug("extract failed '{0}' at '{1}', failed to rebuild weights table", origOp->getName(),
                   origOp->getLoc());
        return mlir::failure();
    }

    // Infer ReduceMax output type from the conv input (C reduced with keep_dims=true).
    const auto inputNdType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto reducedShape = inputNdType.getShape().toValues();
    reducedShape[Dims4D::Act::C] = 1;
    const auto reduceXyMaxType = inputNdType.changeShape(reducedShape);

    // Fuse ReduceMax into Convolution
    rewriter.setInsertionPointAfter(convOp);
    const auto axesValue = getIntArrayAttr(ctx, ArrayRef(SmallVector<int64_t>{Dims4D::Act::C.ind()}));
    auto newConvOp = VPU::NCEConvolutionOp::create(
            rewriter, convOp.getLoc(), convOp.getOutput().getType(), reduceXyMaxType,
            /*reduce_xy_min*/ nullptr,
            /*reduce_tensor_min_max*/ nullptr, convOp.getInput(), convOp.getFilter(), convOp.getWeightsTable(),
            convOp.getWeightTableScale(), convOp.getWeightTableBias(), convOp.getWeightZeroPoints(),
            convOp.getStridesAttr(), convOp.getPadAttr(), convOp.getPpeAttr(), convOp.getMpeEngineAttr(),
            convOp.getRawFilterShape(), convOp.getStaticRawFilterShape(), convOp.getMultiClusterStrategyAttr(),
            convOp.getOutputPaddingAttr(), convOp.getInputPaddingAttr(), /*s2dd2s_config*/ nullptr, axesValue);

    auto newSoftMaxOp = VPU::SoftMaxOp::create(
            rewriter, origOp.getLoc(), origOp.getOutput().getType(), newConvOp.getOutput(), newConvOp.getReduceXyMax(),
            origOp.getAxisIndAttr(), origOp.getPadSizeAttr(), origOp.getDstElemTypeAttr(), origOp.getMaskAwareAttr(),
            origOp.getMultiClusterStrategyAttr());
    newSoftMaxOp->setAttr("SkipNormalization", rewriter.getUnitAttr());

    const auto origLoc = origOp->getLoc();
    rewriter.replaceOp(origOp, newSoftMaxOp.getOutput());
    _log.debug("extract and fuse succeeded at '{0}'", origLoc);

    return mlir::success();
}

class ExtractReduceMaxFromExpStabilizedPass final :
        public VPU::arch60xx::impl::ExtractReduceMaxFromExpStabilizedBase<ExtractReduceMaxFromExpStabilizedPass> {
public:
    explicit ExtractReduceMaxFromExpStabilizedPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final {
        auto& ctx = getContext();
        auto func = getOperation();

        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<ExtractReduceMaxFromExpStabilizedRewriter>(&ctx, _log.nest());
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::arch60xx::createExtractReduceMaxFromExpStabilizedPass(const Logger& log) {
    return std::make_unique<ExtractReduceMaxFromExpStabilizedPass>(log);
}
