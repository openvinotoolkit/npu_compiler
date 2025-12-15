//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/Support/LogicalResult.h>

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZEAVGPOOLWITHUNALIGNEDCHANNELS
#define GEN_PASS_DEF_OPTIMIZEAVGPOOLWITHUNALIGNEDCHANNELS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

constexpr size_t SUPPORTED_RANK = 4;

//
// AvgPoolToConv
//

class AvgPoolToConv final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AvgPoolToConv(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
        setDebugName("AvgPoolToConv");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isEligibleConvertAvgPoolToConv(IE::AvgPoolOp origOp) const;
    Logger _log;
};

bool AvgPoolToConv::isEligibleConvertAvgPoolToConv(IE::AvgPoolOp origOp) const {
    if (IE::isEltwisePooling(origOp)) {
        _log.nest().trace("[{0}] Skip eltwise AvgPool '{1}'", getDebugName(), origOp->getLoc());
        return false;
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    if (inputType.getRank() != SUPPORTED_RANK) {
        _log.nest().trace("[{0}] Unsupported rank '{1}'", getDebugName(), inputType.getRank());
        return false;
    }

    const auto inputDimsOrder = inputType.getDimsOrder();
    if (inputDimsOrder != DimsOrder::NHWC) {
        _log.nest().trace("[{0}] Unsupported dims order {1}", getDebugName(), inputDimsOrder);
        return false;
    }

    const auto inputElementType = inputType.getElementType();
    if (!inputElementType.isF16()) {
        _log.nest().trace("[{0}] Unsupported element type {1} : only FP16 is supported for now", getDebugName(),
                          inputElementType);
        return false;
    }

    const auto inputShape = inputType.getShape();
    const int64_t IC = inputShape[Dims4D::Act::C];
    auto channelIface = mlir::cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());
    const auto alignment = channelIface.getInputChannelAlignment();
    if (IC % alignment == 0) {
        _log.nest().trace("[{0}] Channel number is aligned already", getDebugName());
        return false;
    }

    const auto padBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto padEnd = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    auto isGreaterThanOne = [](int64_t size) {
        return size > 1;
    };
    auto hasPadding = llvm::any_of(padBegin, isGreaterThanOne) || llvm::any_of(padEnd, isGreaterThanOne);
    if (hasPadding) {
        _log.nest().trace("[{0}] Unsupported AvgPool layer with padding at '{1}'", getDebugName(), origOp->getLoc());
        return false;
    }

    auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.getKernelSize());
    const auto strides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    const auto KX = kernelSize[Dims4D::Kernel::X.ind()];
    const auto SX = strides[Dims4D::Strides::X.ind()];
    const auto SY = strides[Dims4D::Strides::Y.ind()];

    // Ensure strides of converted Convolution can be folded
    if (!IE::isEligibleToFoldStrideKernel(inputType, outputType, KX, SX, SY, alignment, alignment, _log)) {
        return false;
    }

    // Make sure new width can be reshaped to align channel numbers, otherwise conversion is not beneficial
    const auto newInputShape = IE::getNewShapeAfterStrideFolding(inputShape, SX);
    return newInputShape[Dims4D::Act::W] % alignment == 0;
}

mlir::LogicalResult AvgPoolToConv::matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got AvgPool layer at '{1}'", getDebugName(), origOp->getLoc());

    if (!isEligibleConvertAvgPoolToConv(origOp)) {
        return mlir::failure();
    }

    const auto ctx = rewriter.getContext();
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto inputShape = inputType.getShape();
    const int64_t IC = inputShape[Dims4D::Act::C];

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.getKernelSize());
    const auto KX = kernelSize[Dims4D::Kernel::X.ind()];
    const auto KY = kernelSize[Dims4D::Kernel::Y.ind()];
    const auto OC = IC;

    const Shape weightsShape = {OC, IC, KY, KX};
    std::vector<float> weights(weightsShape.totalSize(), .0f);

    // assign values
    const auto weightsSizePerOC = IC * KY * KX;
    for (int64_t ocIndex = 0; ocIndex < OC; ocIndex++) {
        auto currOffsetPerOC = ocIndex * weightsSizePerOC;
        auto innerOffset = ocIndex * KX * KY;
        for (int64_t i = 0; i < KX * KY; i++) {
            weights[currOffsetPerOC + innerOffset + i] = 1.0f / checked_cast<float>(KX * KY);
        }
    }

    const DimsOrder weightsOrder = DimsOrder::OIYX;
    const auto weightsType =
            mlir::RankedTensorType::get(weightsShape.raw(), mlir::cast<NDTypeInterface>(inputType).getElementType(),
                                        getTensorAttr(rewriter.getContext(), weightsOrder, nullptr));
    auto filter = Const::buildWeightsConst(rewriter, origOp.getLoc(), weightsType, ArrayRef(weights));

    const auto weightsTypeNCHW = mlir::cast<vpux::NDTypeInterface>(filter.getType());
    const auto reorderType = weightsTypeNCHW.changeDimsOrder(DimsOrder::NHWC);
    const auto orderMap = DimsOrder::NHWC.toAffineMap(ctx);
    auto reorderFilter = rewriter.createOrFold<IE::ReorderOp>(origOp->getLoc(), reorderType, filter, orderMap);

    const auto dilations = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    auto newConv = rewriter.create<IE::ConvolutionOp>(
            origOp.getLoc(), origOp.getOutput().getType(), origOp.getInput(), reorderFilter, nullptr,
            origOp.getStridesAttr(), origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), dilations,
            origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getStaticScaleAttr(), origOp.getOutputPaddingAttr(),
            origOp.getInputPaddingAttr());

    rewriter.replaceOp(origOp, newConv);

    _log.nest().trace("[{0}] Successfully convert AvgPool to Conv", getDebugName());

    return mlir::success();
}

//
// OptimizeAvgPoolWithUnalignedChannelsPass
//

class OptimizeAvgPoolWithUnalignedChannelsPass final :
        public IE::impl::OptimizeAvgPoolWithUnalignedChannelsBase<OptimizeAvgPoolWithUnalignedChannelsPass> {
public:
    explicit OptimizeAvgPoolWithUnalignedChannelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeAvgPoolWithUnalignedChannelsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AvgPoolToConv>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createOptimizeAvgPoolWithUnalignedChannelsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeAvgPoolWithUnalignedChannelsPass(Logger log) {
    return std::make_unique<OptimizeAvgPoolWithUnalignedChannelsPass>(log);
}
