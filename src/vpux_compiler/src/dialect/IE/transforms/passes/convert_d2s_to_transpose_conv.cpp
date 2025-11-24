//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/interfaces/d2s_to_transposed_conv_verifier.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/core/strides.hpp>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTDEPTH2SPACETOTRANSPOSEDCONV
#define GEN_PASS_DEF_CONVERTDEPTH2SPACETOTRANSPOSEDCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertDepth2SpaceToTransposedConv
//

class ConvertDepth2SpaceToTransposedConv final : public mlir::OpRewritePattern<IE::DepthToSpaceOp> {
public:
    ConvertDepth2SpaceToTransposedConv(mlir::MLIRContext* ctx,
                                       std::unique_ptr<IE::D2SToTransposedConvVerifierBase>& benefitVerifier,
                                       Logger log)
            : mlir::OpRewritePattern<IE::DepthToSpaceOp>(ctx), _benefitVerifier(std::move(benefitVerifier)), _log(log) {
        setDebugName("ConvertDepth2SpaceToTransposedConv");
    }

    mlir::LogicalResult matchAndRewrite(IE::DepthToSpaceOp, mlir::PatternRewriter&) const final;

private:
    std::unique_ptr<IE::D2SToTransposedConvVerifierBase> _benefitVerifier;
    Logger _log;
};

//
// createDepthFirstWeights
//

SmallVector<uint8_t> createDepthFirstWeights(ShapeRef filterShape) {
    auto inputChannels = filterShape[Dims4D::Filter::IC];
    auto outputChannels = filterShape[Dims4D::Filter::OC];

    auto filterWidth = filterShape[Dims4D::Filter::KX];
    auto filterHeight = filterShape[Dims4D::Filter::KY];

    auto filterSize = filterWidth * filterHeight;

    auto channelSize = filterShape.totalSize() / outputChannels;
    auto filterShapeSize = filterShape.totalSize();

    SmallVector<uint8_t> weights(filterShapeSize, 0);

    for (const auto& block : irange(filterSize)) {
        int64_t reverseBlockIndex = filterSize - 1 - block;

        for (const auto& channel : irange(outputChannels)) {
            int64_t channelIndex = channel * channelSize;

            int64_t y = channelIndex + reverseBlockIndex * inputChannels;
            int64_t x = channel * filterSize;

            weights[block + y + x] = 1;
        }
    }

    return weights;
}

mlir::Value createDepthFirstWeightsConst(mlir::MLIRContext* ctx, IE::DepthToSpaceOp d2sOp, ShapeRef filterShape,
                                         mlir::PatternRewriter& rewriter, [[maybe_unused]] Logger log) {
    auto weights = createDepthFirstWeights(filterShape);

    auto dataStorageTensorAttr = vpux::getTensorAttr(ctx, DimsOrder::OYXI, nullptr);
    auto dataStorageType = mlir::RankedTensorType::get(filterShape.raw(), getUInt8Type(ctx), dataStorageTensorAttr);

    return Const::createConst(rewriter, takeOpLoc(d2sOp, "depth_first"), dataStorageType, ArrayRef(weights),
                              [&](Const::ContentSetup& setup) {
                                  return setup.castElemType(mlir::Float16Type::get(ctx)).reorder(DimsOrder::OIYX);
                              });
}

//
// createBinaryFakeQuantize
//

IE::FakeQuantizeOp createBinaryFakeQuantize(mlir::MLIRContext* ctx, IE::DepthToSpaceOp d2sOp, mlir::Value inputOp,
                                            mlir::PatternRewriter& rewriter, [[maybe_unused]] Logger log) {
    auto elemType = mlir::cast<vpux::NDTypeInterface>(d2sOp.getInput().getType()).getElementType();

    const auto fqArgType = mlir::RankedTensorType::get({1, 1, 1, 1}, elemType);

    auto fqLow = Const::createFloatConst(rewriter, d2sOp.getLoc(), fqArgType, 0.0f);
    auto fqHigh = Const::createFloatConst(rewriter, d2sOp.getLoc(), fqArgType, 1.0f);

    auto levels = getIntAttr(ctx, 2);
    auto broadcast = vpux::IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);

    // lowFpType is ignored (nullptr), only levels are given
    return rewriter.create<IE::FakeQuantizeOp>(takeOpLoc(d2sOp, "fq_in"), inputOp, fqLow, fqHigh, fqLow, fqHigh, levels,
                                               nullptr, broadcast);
}

//
// convertDepthFirstOp
//

mlir::LogicalResult convertDepthFirstOp(mlir::MLIRContext* ctx, IE::DepthToSpaceOp d2sOp,
                                        mlir::PatternRewriter& rewriter, Logger log) {
    auto input = d2sOp.getInput();

    auto inputShape = mlir::cast<vpux::NDTypeInterface>(input.getType()).getShape();
    auto inputElemType = mlir::cast<vpux::NDTypeInterface>(input.getType()).getElementType();

    if (inputElemType == getUInt8Type(ctx) || inputElemType == getInt8Type(ctx)) {
        // TODO: Workaround. Upsample does not support unquantized u8/i8 types: E#109658
        return matchFailed(rewriter, d2sOp, "E#109658 upsample does not support U8/I8 types");
    }

    auto blockSize = d2sOp.getBlockSize();
    VPUX_THROW_WHEN(blockSize <= 0, "Unsupported block size: {0}", blockSize);

    auto filterWidth = blockSize;
    auto filterHeight = blockSize;

    uint64_t strideX = blockSize;
    uint64_t strideY = blockSize;

    uint64_t dilateX = 1;
    uint64_t dilateY = 1;

    auto padTop = 0;
    auto padLeft = 0;
    auto padBottom = 0;
    auto padRight = 0;

    auto inputChannels = inputShape[Dims4D::Act::C];
    auto outputChannels = inputChannels / (filterWidth * filterHeight);

    auto filterShape = Shape{outputChannels, inputChannels, filterHeight, filterWidth};

    // Check if conv parameters supported on hardware
    if (VPU::NCEInvariant::verifyKernel(d2sOp, filterHeight, filterWidth, strideY, strideX, padTop, padBottom, padLeft,
                                        padRight, log)
                .failed()) {
        return mlir::failure();
    }

    // Generate weights and FakeQuantize
    auto weightsConst = createDepthFirstWeightsConst(ctx, d2sOp, filterShape, rewriter, log);
    auto fq = createBinaryFakeQuantize(ctx, d2sOp, weightsConst, rewriter, log);

    // Replace D2S with TransposedConvolution
    auto strides = getIntArrayAttr(ctx, ov::Strides{strideX, strideY});

    auto padsBegin = getIntArrayAttr(ctx, ov::CoordinateDiff{padTop, padLeft});
    auto padsEnd = getIntArrayAttr(ctx, ov::CoordinateDiff{padBottom, padRight});

    auto dilations = getIntArrayAttr(ctx, ov::Strides{dilateX, dilateY});
    auto outputPadding = getIntArrayAttr(ctx, ov::CoordinateDiff{0, 0});

    auto transConv = rewriter.replaceOpWithNewOp<IE::TransposedConvolutionOp>(d2sOp,
                                                                              /* feature = */ input,
                                                                              /* filter = */ fq,
                                                                              /* outputShape = */ nullptr,
                                                                              /* bias = */ nullptr,
                                                                              /* strides = */ strides,
                                                                              /* padsBegin = */ padsBegin,
                                                                              /* padsEnd = */ padsEnd,
                                                                              /* dilations = */ dilations,
                                                                              /* outputPadding = */ outputPadding,
                                                                              /* postOp = */ nullptr,
                                                                              /* clamp = */ nullptr,
                                                                              /* outputPadding = */ nullptr,
                                                                              /* inputPadding = */ nullptr);
    extendOpLoc(transConv, "as_transcov");
    log.trace("transposed conv: '{0}'", transConv);

    return mlir::success();
}

//
// convertBlocksFirstOp
//

mlir::LogicalResult convertBlocksFirstOp(mlir::MLIRContext*, IE::DepthToSpaceOp d2sOp, mlir::PatternRewriter& rewriter,
                                         Logger log) {
    // TODO: Implement blocks_first mode: E#110158
    return matchFailed(log, rewriter, d2sOp, "blocks first mode is not implemented: '{0}'", d2sOp);
}

//
// matchAndRewrite
//

mlir::LogicalResult ConvertDepth2SpaceToTransposedConv::matchAndRewrite(IE::DepthToSpaceOp d2sOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("found '{0}' at '{1}'", d2sOp->getName(), d2sOp->getLoc());
    _log.nest().debug("d2sOp = '{0}'", d2sOp);

    auto ctx = rewriter.getContext();

    auto input = d2sOp.getInput();
    auto inputShape = mlir::cast<vpux::NDTypeInterface>(input.getType()).getShape();

    auto mode = d2sOp.getMode();
    auto blockSize = d2sOp.getBlockSize();
    if (blockSize <= 0) {
        return matchFailed(_log, rewriter, d2sOp, "Unsupported block size: {0}", blockSize);
    }

    if (_benefitVerifier->isBeneficialConversion(_log, rewriter, d2sOp).failed()) {
        return matchFailed(_log, rewriter, d2sOp, "DepthToSpace is not beneficial as TransposedConv: '{0}'",
                           d2sOp->getLoc());
    }

    if (d2sOp.getPaddedChannels().has_value()) {  // TODO: Support padded channels: E#109929
        return matchFailed(_log, rewriter, d2sOp, "padded channels are not supported: '{0}'", d2sOp);
    }

    if (inputShape[Dims4D::Act::C] % (blockSize * blockSize) != 0) {
        return matchFailed(_log, rewriter, d2sOp,
                           "input channels ({0}) not divisible by {1} (blockSize * blockSize): '{2}'",
                           inputShape[Dims4D::Act::C], blockSize * blockSize, d2sOp);
    }

    if (mode == IE::DepthToSpaceMode::DEPTH_FIRST) {
        return convertDepthFirstOp(ctx, d2sOp, rewriter, _log);

    } else if (mode == IE::DepthToSpaceMode::BLOCKS_FIRST) {
        return convertBlocksFirstOp(ctx, d2sOp, rewriter, _log);
    }

    return matchFailed(_log, rewriter, d2sOp, "unsupported mode '{0}': '{1}'", mode, d2sOp);
}

//
// ConvertDepth2SpaceToTransposedConvPass
//

class ConvertDepth2SpaceToTransposedConvPass final :
        public IE::impl::ConvertDepth2SpaceToTransposedConvBase<ConvertDepth2SpaceToTransposedConvPass> {
public:
    explicit ConvertDepth2SpaceToTransposedConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() override;
};

//
// safeRunOnFunc
//

void ConvertDepth2SpaceToTransposedConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto benefitVerifier = IE::createD2SToTransposedConvVerifier(config::getArch(module));

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConvertDepth2SpaceToTransposedConv>(&ctx, std::move(benefitVerifier), _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertDepth2SpaceToTransposedConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertDepth2SpaceToTransposedConvPass(Logger log) {
    return std::make_unique<ConvertDepth2SpaceToTransposedConvPass>(log);
}
