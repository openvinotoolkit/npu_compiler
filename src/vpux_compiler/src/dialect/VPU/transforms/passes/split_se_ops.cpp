//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/roll_utils.hpp"

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/se_attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_roll_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_SPLITSEOPS
#define GEN_PASS_DEF_SPLITSEOPS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

bool doesFitIntoCMX(mlir::Operation* op, NDTypeInterface inputType, NDTypeInterface outputType, int64_t seTableH,
                    int64_t seTableW) {
    auto arch = config::getArch(op);
    auto sparsityConstraint = VPU::getSparsityConstraint(arch);
    const auto inShape = inputType.getShape();
    const auto inputC = inShape[Dims4D::Act::C];

    const auto seSize = VPU::getSESize(inputC, sparsityConstraint);
    const auto seDepth = inputC / seSize;
    const auto sepType = mlir::IntegerType::get(op->getContext(), 32);

    const auto inputDataSize = inputType.getTotalAllocSize().count();
    const auto inputSMSize = (seTableH * seTableW * inputC) / CHAR_BIT;
    const auto inputSESize = (seTableH * seTableW * seDepth * sepType.getWidth()) / CHAR_BIT;
    const auto outputSize = outputType.getTotalAllocSize().count();
    const auto requiredCMX = inputDataSize + inputSMSize + inputSESize + outputSize;
    return requiredCMX < VPU::getTotalCMXSize(op).count();
}

//
// SplitInterpolate
//

class SplitInterpolate final : public mlir::OpRewritePattern<VPU::InterpolateOp> {
public:
    SplitInterpolate(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::InterpolateOp>(ctx), _log(log) {
        setDebugName("SplitInterpolate");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isLegalAndBenifitSplitInterpolate(NDTypeInterface inputType, NDTypeInterface outputType,
                                           VPU::NCEInterpolateModeAttr modeAttr,
                                           IE::InterpolateCoordModeAttr coordModeAttr) const;

    Logger _log;
};

bool SplitInterpolate::isLegalAndBenifitSplitInterpolate(NDTypeInterface inputType, NDTypeInterface outputType,
                                                         VPU::NCEInterpolateModeAttr modeAttr,
                                                         IE::InterpolateCoordModeAttr coordModeAttr) const {
    const auto inputElemType = inputType.getElementType();
    // If NCEInterpolate has a quantized type, splitting might cause accuracy issues
    if (mlir::isa<mlir::quant::QuantizedType>(inputElemType)) {
        return false;
    }

    auto potentialScales = VPU::getNCEInterpolateScales(inputType, outputType, coordModeAttr);
    VPUX_THROW_UNLESS(potentialScales.has_value(), "Cannot get scales of NCE Interpolate");
    const auto scales = potentialScales.value();

    const auto factors = VPU::getNCEInterpolateFactors(scales, modeAttr, coordModeAttr);

    const bool isHalfPixelCoordMode = coordModeAttr.getValue() == IE::InterpolateCoordMode::HALF_PIXEL ||
                                      coordModeAttr.getValue() == IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL;

    const bool isBilinearHalfPixel = modeAttr.getValue() == VPU::NCEInterpolateMode::BILINEAR && isHalfPixelCoordMode;

    const auto areFactorsLarge = llvm::all_of(factors, [&isBilinearHalfPixel](const auto factor) {
        // TODO: E156480 - accuracy issue seen for one model without the SE split for Interp Bilinear half pixel
        if (isBilinearHalfPixel) {
            return factor >= 2;
        }
        return factor >= 4;
    });

    return areFactorsLarge;
}

mlir::LogicalResult SplitInterpolate::matchAndRewrite(VPU::InterpolateOp origOp,
                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    if (!VPU::NCEInterpolateOp::isSupported(origOp, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true,
                                            /*checkBatch=*/true)) {
        return matchFailed(rewriter, origOp, "It is not NCEInterpolateOp");
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto modeAttr = VPU::getNCEInterpolateModeAttr(origOp.getAttr().getMode());
    const auto coordModeAttr = origOp.getAttr().getCoordMode();

    if (!isLegalAndBenifitSplitInterpolate(inputType, outputType, modeAttr, coordModeAttr)) {
        return matchFailed(rewriter, origOp, "It is not beneficial to split");
    }

    const auto sizesAttr = origOp.getSizesAttrAttr();
    const auto scalesAttr = origOp.getScalesAttrAttr();
    const auto axes = IE::getInterpAxesVal(origOp.getLoc(), origOp.getAxes(), origOp.getAxesAttrAttr(), inputType);
    const auto shapeCalcMode = origOp.getAttr().getShapeCalcMode().getValue();

    auto createSingleDimInterpOp = [&](Dim dim, mlir::Value inputVal, const vpux::NDTypeInterface newOutputType) {
        auto dimPtr = std::find(axes.begin(), axes.end(), dim.ind());
        VPUX_THROW_WHEN(dimPtr == axes.end(), "Cannot find Dim [{0}] in Interpolate axes attribution [{1}]", dim, axes);
        auto dimIdx = std::distance(axes.begin(), dimPtr);

        auto newSizesAttr = sizesAttr;
        auto newScalesAttr = scalesAttr;
        if (shapeCalcMode == IE::InterpolateCalcMode::SIZES) {
            const auto inputShape = getShape(inputVal);
            const auto sizes = parseIntArrayAttr<int64_t>(sizesAttr);
            auto newSizes = SmallVector<double>(sizes.size(), 1.0);
            for (auto axis : axes | indexed) {
                newSizes[axis.index()] = dim == Dim(axis.value()) ? sizes[axis.index()] : inputShape[Dim(axis.value())];
            }
            newSizesAttr = getIntArrayAttr(getContext(), newSizes);
        }

        if (shapeCalcMode == IE::InterpolateCalcMode::SCALES) {
            const auto scales = parseFPArrayAttr<double>(scalesAttr);
            auto newScales = SmallVector<double>(scales.size(), 1.0);
            newScales[dimIdx] = scales[dimIdx];
            newScalesAttr = getFPArrayAttr(getContext(), newScales);
        }

        auto newLoc = appendLoc(origOp.getLoc(), "_interpolate_on_Dim_{0}", dim.ind());
        if (newOutputType != nullptr) {
            return rewriter
                    .create<VPU::InterpolateOp>(
                            newLoc, newOutputType, inputVal, origOp.getSizes(), origOp.getScales(), origOp.getAxes(),
                            origOp.getCoordinates(), origOp.getLambdas(), newSizesAttr, newScalesAttr,
                            origOp.getAxesAttrAttr(), origOp.getTileOffsetAttrAttr(),
                            origOp.getInitialInputDimsAttrAttr(), origOp.getInitialOutputDimsAttrAttr(),
                            origOp.getInitialInputOffsetAttrAttr(), origOp.getInitialOutputOffsetAttrAttr(),
                            origOp.getMultiClusterStrategyAttr(), origOp.getAttr(), origOp.getOutputPaddingAttr(),
                            origOp.getInputPaddingAttr())
                    .getOutput();
        }
        return rewriter
                .create<VPU::InterpolateOp>(
                        newLoc, inputVal, origOp.getSizes(), origOp.getScales(), origOp.getAxes(),
                        origOp.getCoordinates(), origOp.getLambdas(), newSizesAttr, newScalesAttr,
                        origOp.getAxesAttrAttr(), origOp.getTileOffsetAttrAttr(), origOp.getInitialInputDimsAttrAttr(),
                        origOp.getInitialOutputDimsAttrAttr(), origOp.getInitialInputOffsetAttrAttr(),
                        origOp.getInitialOutputOffsetAttrAttr(), origOp.getMultiClusterStrategyAttr(), origOp.getAttr(),
                        origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr())
                .getOutput();
    };

    // Set the output type manually for the last operation, to ensure that the layout of the data is preserved
    auto interpolateW = createSingleDimInterpOp(Dims4D::Act::W, origOp.getInput(), nullptr);
    auto interpolateH = createSingleDimInterpOp(Dims4D::Act::H, interpolateW, outputType);

    _log.nest().trace("[{0}] Split successful", getDebugName());
    rewriter.replaceOp(origOp, interpolateH);
    return mlir::success();
}

//
// SplitRoll
//

class SplitRoll final : public mlir::OpRewritePattern<VPU::RollOp> {
public:
    SplitRoll(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::RollOp>(ctx), _log(log) {
        setDebugName("SplitRoll");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::RollOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isLegalAndBenifitSplitRoll(VPU::RollOp origOp, NDTypeInterface inputType, NDTypeInterface outputType,
                                    ArrayRef<int64_t> shift) const;

    Logger _log;
};

bool SplitRoll::isLegalAndBenifitSplitRoll(VPU::RollOp origOp, NDTypeInterface inputType, NDTypeInterface outputType,
                                           ArrayRef<int64_t> shift) const {
    const auto inShape = inputType.getShape();
    const auto seTableH = inShape[Dims4D::Act::H];
    const auto seTableW = inShape[Dims4D::Act::W];

    const auto fitInCmx = doesFitIntoCMX(origOp, inputType, outputType, seTableH, seTableW);
    const auto rollAtTwoDim = shift.size() == 2 && shift[0] != 0 && shift[1] != 0;
    return !fitInCmx && rollAtTwoDim;
}

mlir::LogicalResult SplitRoll::matchAndRewrite(VPU::RollOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    if (!VPU::isSupportedSEPRoll(origOp, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true)) {
        return matchFailed(rewriter, origOp, "It is not fit for NCE");
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getData().getType());
    const auto inputShape = inputType.getShape();
    auto shiftAndAxesOrFail =
            IE::getShiftAndAxesForRollOp(origOp.getLoc(), origOp.getShift(), origOp.getAxes(), inputShape);
    if (mlir::failed(shiftAndAxesOrFail)) {
        return mlir::failure();
    }
    const auto shiftAndAxes = shiftAndAxesOrFail.value();
    const auto shift = shiftAndAxes.shift;
    const auto axes = shiftAndAxes.axes;

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    if (!isLegalAndBenifitSplitRoll(origOp, inputType, outputType, shift)) {
        return matchFailed(rewriter, origOp, "It is not beneficial to split");
    }

    const auto newAxesElems = checked_cast<int64_t>(axes.size());
    const auto axesDimOrder = mlir::cast<vpux::NDTypeInterface>(origOp.getAxes().getType()).getDimsOrder();
    const auto newAxesType = mlir::RankedTensorType::get(
            ArrayRef(newAxesElems), mlir::cast<vpux::NDTypeInterface>(origOp.getAxes().getType()).getElementType(),
            getTensorAttr(rewriter.getContext(), axesDimOrder, nullptr));
    const auto newAxesValue = Const::createConst(rewriter, origOp.getAxes().getLoc(), newAxesType,
                                                 ArrayRef({Dims4D::Act::H.ind(), Dims4D::Act::W.ind()}));

    const auto shiftDimOrder = mlir::cast<vpux::NDTypeInterface>(origOp.getShift().getType()).getDimsOrder();
    const auto shiftLoc = origOp.getShift().getLoc();

    auto createSingleDimRollOp = [&](Dim dim, ArrayRef<int32_t> newShift, mlir::Value inputVal) {
        const auto newShiftElems = checked_cast<int64_t>(newShift.size());
        const auto newShiftType = mlir::RankedTensorType::get(
                ArrayRef(newShiftElems),
                mlir::cast<vpux::NDTypeInterface>(origOp.getShift().getType()).getElementType(),
                getTensorAttr(rewriter.getContext(), shiftDimOrder, nullptr));
        const auto shiftValue = Const::createConst(rewriter, shiftLoc, newShiftType, newShift);
        auto newLoc = appendLoc(origOp.getLoc(), "_roll_on_Dim_{0}", dim.ind());
        return rewriter.create<VPU::RollOp>(newLoc, inputVal, shiftValue, newAxesValue).getOutput();
    };

    auto rollW = createSingleDimRollOp(Dims4D::Act::W, {0, checked_cast<int32_t>(shift[VPU::SE_ROLL_SPATIAL_W])},
                                       origOp.getData());
    auto rollH =
            createSingleDimRollOp(Dims4D::Act::H, {checked_cast<int32_t>(shift[VPU::SE_ROLL_SPATIAL_H]), 0}, rollW);

    rewriter.replaceOp(origOp, rollH);
    return mlir::success();
}

//
// SplitSEOpsPass
//

class SplitSEOpsPass final : public VPU::impl::SplitSEOpsBase<SplitSEOpsPass> {
public:
    explicit SplitSEOpsPass(const bool seOpsEnabled, const bool seExperimentalOpsEnabled, Logger log)
            : _seOpsEnabled(seOpsEnabled), _seExperimentalOpsEnabled(seExperimentalOpsEnabled), _log(log) {
        _log.setName(Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _seOpsEnabled;
    bool _seExperimentalOpsEnabled;
    Logger _log;
};

mlir::LogicalResult SplitSEOpsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (seOpsEnabled.hasValue()) {
        _seOpsEnabled = seOpsEnabled.getValue();
    }
    if (seExperimentalOpsEnabled.hasValue()) {
        _seExperimentalOpsEnabled = seExperimentalOpsEnabled.getValue();
    }

    return mlir::success();
}

void SplitSEOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    if (!_seOpsEnabled) {
        return;
    }

    patterns.add<SplitInterpolate>(&ctx, _log);
    patterns.add<SplitRoll>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createSplitSEOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSplitSEOpsPass(const bool seOpsEnabled,
                                                            const bool seExperimentalOpsEnabled, Logger log) {
    return std::make_unique<SplitSEOpsPass>(seOpsEnabled, seExperimentalOpsEnabled, log);
}
