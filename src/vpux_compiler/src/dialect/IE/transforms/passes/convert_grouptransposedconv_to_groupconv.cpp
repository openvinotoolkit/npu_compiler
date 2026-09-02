//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/transposed_convolution_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTGROUPTRANSPOSEDCONVTOGROUPCONV
#define GEN_PASS_DEF_CONVERTGROUPTRANSPOSEDCONVTOGROUPCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool shouldConvertGroupTransposedConvToGroupConv(IE::GroupTransposedConvolutionOp groupTransposedConv,
                                                 bool enableSEPTransposedConv, Logger log) {
    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };

    log.trace("Got '{0}' at '{1}'", groupTransposedConv->getName(), groupTransposedConv->getLoc());
    if (enableSEPTransposedConv) {
        auto seOp = mlir::dyn_cast<IE::SEOpInterface>(groupTransposedConv.getOperation());
        if (seOp && seOp.isSupported(logCb)) {
            log.nest(1).trace("GroupTransposedConvolutionOp can be executed using SEP");
            return false;
        }
    }

    if (mlir::failed(IE::canConvertGroupTransposedConvToGroupConv(groupTransposedConv))) {
        log.nest(1).trace("GroupTransposedConvolutionOp cannot be converted.");
        return false;
    }

    return true;
}

//
// GroupTransposedConvConverter
//

class GroupTransposedConvConverter final : public mlir::OpRewritePattern<IE::GroupTransposedConvolutionOp> {
public:
    GroupTransposedConvConverter(mlir::MLIRContext* ctx, bool enableSEPTransposedConv, Logger log)
            : mlir::OpRewritePattern<IE::GroupTransposedConvolutionOp>(ctx),
              _enableSEPTransposedConv(enableSEPTransposedConv),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupTransposedConvolutionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableSEPTransposedConv;
    Logger _log;
};

mlir::LogicalResult GroupTransposedConvConverter::matchAndRewrite(IE::GroupTransposedConvolutionOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    if (!shouldConvertGroupTransposedConvToGroupConv(origOp, _enableSEPTransposedConv, _log)) {
        return mlir::failure();
    }

    _log.trace("Got GroupTransposedConvolutionOp layer at '{0}'", origOp->getLoc());

    auto padsOutput = Shape(parseIntArrayAttr<int64_t>(origOp.getSpatialOutputPadding()));

    const auto featureShape = getShape(origOp.getInput());
    if (featureShape.size() != 4) {
        return matchFailed(rewriter, origOp,
                           "Only 2D GroupTransposedConvolutionOp is supported, expected 4D feature but got {0}",
                           featureShape.size());
    }

    const auto outputShape = getShape(origOp.getOutput());
    if (outputShape.size() != 4) {
        return matchFailed(rewriter, origOp,
                           "Only 2D GroupTransposedConvolutionOp is supported, expected 4D output shape but got {0}",
                           outputShape.size());
    }

    auto filterType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType());
    auto origFilterShape = to_small_vector(filterType.getShape());
    if (origFilterShape.size() != 5) {
        return matchFailed(rewriter, origOp,
                           "Only 2D GroupTransposedConvolutionOp is supported, expected 5D filter but got {0}",
                           origFilterShape.size());
    }

    auto dwConvFilter = origOp.getFilter().getDefiningOp();
    if (dwConvFilter == nullptr) {
        return matchFailed(rewriter, origOp, "GroupTransposedConvolutionOp has no filter Op");
    }

    // convert filter shape from 5D to 4D
    auto groups = origFilterShape[IE::GROUP_TRANSPOSED_CONV_GROUPS_DIM_INDEX];
    origFilterShape[IE::GROUP_TRANSPOSED_CONV_C_OUT_DIM_INDEX] *= groups;
    origFilterShape.erase(origFilterShape.begin());

    const auto filter4DShapeAttr = getIntArrayAttr(rewriter.getContext(), origFilterShape);

    const auto postOp = origOp.getPostOpAttr();
    const auto clampOp = origOp.getClampAttr();
    const auto outputChannels = origOp.getOutputPaddingAttr();
    const auto inputChannels = origOp.getInputPaddingAttr();

    auto dilations = origOp.getDilationsAttr();

    const auto stridesVector = Shape(parseIntArrayAttr<int64_t>(origOp.getStrides()));
    const auto dilationsVector = Shape(parseIntArrayAttr<int64_t>(origOp.getDilations()));
    const auto padsBeginVec = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()));
    const auto padsEndVec = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsEnd()));
    const auto inputC = featureShape[Dims4D::Act::C];
    const auto outputC = outputShape[Dims4D::Act::C];
    const bool isStride1 = stridesVector[Dims4D::Strides::Y] == 1 && stridesVector[Dims4D::Strides::X] == 1;
    const bool isDilated = dilationsVector[Dims4D::Dilation::Y] > 1 || dilationsVector[Dims4D::Dilation::X] > 1;
    const bool isDepthwise = (groups == inputC) && (groups == outputC);
    const bool noOutputPadding = padsOutput[Dims4D::PadsOutput::Y] == 0 && padsOutput[Dims4D::PadsOutput::X] == 0;
    const bool useDirectDepthwisePath =
            isStride1 && isDilated && isDepthwise && noOutputPadding && origOp.getOutputShape() == nullptr;

    if (useDirectDepthwisePath) {
        const auto dilY = dilationsVector[Dims4D::Dilation::Y];
        const auto dilX = dilationsVector[Dims4D::Dilation::X];
        const auto kY = origFilterShape[Dims4D::Filter::KY.ind()];
        const auto kX = origFilterShape[Dims4D::Filter::KX.ind()];
        const auto effKYMinus1 = (kY - 1) * dilY;
        const auto effKXMinus1 = (kX - 1) * dilX;
        const auto newPadTop = effKYMinus1 - padsBeginVec[Dims4D::PadsBegin::Top];
        const auto newPadLeft = effKXMinus1 - padsBeginVec[Dims4D::PadsBegin::Left];
        const auto newPadBottom = effKYMinus1 - padsEndVec[Dims4D::PadsEnd::Bottom];
        const auto newPadRight = effKXMinus1 - padsEndVec[Dims4D::PadsEnd::Right];
        if (newPadTop < 0 || newPadLeft < 0 || newPadBottom < 0 || newPadRight < 0) {
            _log.trace("Direct depthwise-dilated GroupConv conversion skipped at '{0}' due to negative equivalent "
                       "padding; falling back to generic upsampling-based conversion",
                       origOp.getLoc());
        } else {
            auto directPadsBegin = getIntArrayAttr(getContext(), SmallVector<int64_t>{newPadTop, newPadLeft});
            auto directPadsEnd = getIntArrayAttr(getContext(), SmallVector<int64_t>{newPadBottom, newPadRight});
            auto directStrides = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});
            auto reshaped4DFilter = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_filter"),
                                                                         dwConvFilter->getResult(0), filter4DShapeAttr);
            // LegalizeDilatedConvolution runs before this pass in the pipeline, so we must expand
            // dilation here explicitly instead of relying on that pass to do it later.  E#222712
            auto expandedFilter = rewriter.create<IE::ExpandDilatedOp>(takeOpLoc(origOp, "expand_dilated_filter"),
                                                                       reshaped4DFilter, dilations);
            auto noDilations = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});
            auto directResult = rewriter.create<IE::GroupConvolutionOp>(origOp->getLoc(), origOp.getInput(),
                                                                        expandedFilter.getResult(), nullptr,
                                                                        directStrides, directPadsBegin, directPadsEnd,
                                                                        noDilations, getIntAttr(rewriter, groups),
                                                                        postOp, clampOp, outputChannels, inputChannels)
                                        .getOutput();
            const auto origOpLoc = origOp.getLoc();
            rewriter.replaceOp(origOp, directResult);
            _log.trace("Replaced GroupTransposedConvolutionOp at '{0}' with direct depthwise-dilated GroupConv",
                       origOpLoc);
            return mlir::success();
        }
    }

    auto featureUpScale = IE::createUpsampling(rewriter, takeOpLoc(origOp, "upscale_in"), origOp, padsOutput, true);
    if (mlir::failed(featureUpScale)) {
        _log.nest().trace("Failed to create Upsampling for {0}", origOp->getLoc());
        return mlir::failure();
    }
    auto paddingOutput = featureUpScale.value();

    auto strides = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});
    auto padsBegin = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto padsEnd = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});

    if (padsOutput[Dims4D::PadsOutput::Y] > 0) {
        paddingOutput = IE::createPadding(rewriter, takeOpLoc(origOp, "height"), paddingOutput, Dims4D::Act::H,
                                          padsOutput[Dims4D::PadsOutput::Y], nullptr);
    }
    if (padsOutput[Dims4D::PadsOutput::X] > 0) {
        paddingOutput = IE::createPadding(rewriter, takeOpLoc(origOp, "width"), paddingOutput, Dims4D::Act::W,
                                          padsOutput[Dims4D::PadsOutput::X], nullptr);
    }

    auto reshaped4DFilter = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_filter"),
                                                                 dwConvFilter->getResult(0), filter4DShapeAttr);

    auto resultOP =
            rewriter.create<IE::GroupConvolutionOp>(origOp->getLoc(), paddingOutput, reshaped4DFilter, nullptr, strides,
                                                    padsBegin, padsEnd, dilations, getIntAttr(rewriter, groups), postOp,
                                                    clampOp, outputChannels, inputChannels)
                    .getOutput();

    const auto origOpLoc = origOp.getLoc();
    rewriter.replaceOp(origOp, resultOP);

    _log.trace("Replaced GroupTransposedConvolutionOp at '{0}' with 'IE::GroupConvolutionOp' (2D)", origOpLoc);

    return mlir::success();
}

//
// ConvertGroupTransposedConvToGroupConvPass
//

class ConvertGroupTransposedConvToGroupConvPass final :
        public IE::impl::ConvertGroupTransposedConvToGroupConvBase<ConvertGroupTransposedConvToGroupConvPass> {
public:
    explicit ConvertGroupTransposedConvToGroupConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertGroupTransposedConvToGroupConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    const auto func = getOperation();
    const auto moduleOp = getModuleOp(func);
    const auto enableSEPtrsOps = config::hasEnableSEPtrsOperations(moduleOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GroupTransposedConvConverter>(&ctx, enableSEPtrsOps, _log);

    walkAndApplyPatterns(getOperation(), std::move(patterns));
}

}  // namespace

//
// createConvertGroupTransposedConvToGroupConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGroupTransposedConvToGroupConvPass(Logger log) {
    return std::make_unique<ConvertGroupTransposedConvToGroupConvPass>(log);
}
