//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/IRMapping.h>

#include <openvino/op/convolution.hpp>
#include <openvino/op/parameter.hpp>

namespace vpux::IE {
#define GEN_PASS_DECL_ADJUSTCONVOLUTIONINPUTSHAPE
#define GEN_PASS_DEF_ADJUSTCONVOLUTIONINPUTSHAPE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// TODO: needs find suitable input shape size threshold value. Ticket: E#124225
constexpr int64_t THRESHOLD_FOR_BENEFICIAL_CONVERSION = 3072;

std::tuple<int64_t, int64_t> calcShapeAligned(int64_t divider, int64_t dimValue) {
    if (dimValue == 1) {
        divider *= VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT;
        dimValue = VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT;
    }

    return std::make_tuple(divider, dimValue);
}

//
// ReshapeSingleConstDWConvInput
//

class ReshapeSingleConstDWConvInput final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    ReshapeSingleConstDWConvInput(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::Value getReshapedConst(Const::DeclareOp constOp, ShapeRef shape, mlir::PatternRewriter& rewriter) {
    auto constOutputType = mlir::cast<vpux::NDTypeInterface>(constOp.getOutput().getType());
    const auto offset = Shape(shape.size(), 0);
    constOutputType = constOutputType.changeShape(shape);
    auto contentAttr = constOp.transformContentAttr().subview(offset, shape).get();
    return rewriter.create<Const::DeclareOp>(constOp.getLoc(), constOutputType, std::move(contentAttr)).getOutput();
}

mlir::LogicalResult ReshapeSingleConstDWConvInput::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    // Adjust from C to H and W like [1, C, 1, 1] -> [1, C/16, 4, 4]

    const auto ctx = origOp->getContext();
    if (!IE::isEltwiseGroupConv(origOp)) {
        return matchFailed(rewriter, origOp, "Not a valid groupConv");
    }

    const auto inputShape = getShape(origOp.getInput());
    const auto filterShape = getShape(origOp.getFilter());
    const auto filterConst = mlir::cast<Const::DeclareOp>(origOp.getFilter().getDefiningOp());

    // Current logic only works with input and filter shape with 4 dimensions
    const int rank4D = 4;
    if (inputShape.size() != rank4D || filterShape.size() != rank4D) {
        return matchFailed(rewriter, origOp, "Input shape size is not 4");
    }

    if (!allowsChannelsReshape(origOp)) {
        return matchFailed(rewriter, origOp, "Cannot reshape channels of operation.");
    }

    int64_t divider = 1;
    int64_t widthReshaped, heightReshaped, channelReshaped;
    std::tie(divider, widthReshaped) = calcShapeAligned(divider, inputShape[Dims4D::Act::W]);
    std::tie(divider, heightReshaped) = calcShapeAligned(divider, inputShape[Dims4D::Act::H]);
    if (divider == 1) {
        return matchFailed(rewriter, origOp, "Don't need to align");
    }

    if (inputShape[Dims4D::Act::C] % divider != 0) {
        return matchFailed(rewriter, origOp, "Input shape could not be divided {0}", inputShape[Dims4D::Act::C]);
    }

    channelReshaped = inputShape[Dims4D::Act::C] / divider;
    auto alignIface = mlir::cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());
    if (channelReshaped % alignIface.getInputChannelAlignment() != 0) {
        return matchFailed(rewriter, origOp, "Remaining channel is not aligned.");
    }

    // Reshape Input
    const SmallVector<int64_t> newInShape = {inputShape[Dims4D::Act::N], channelReshaped, heightReshaped,
                                             widthReshaped};
    auto inShapeCast =
            rewriter.create<IE::ShapeCastOp>(origOp->getLoc(), origOp.getInput(), getIntArrayAttr(ctx, newInShape));

    // Reshape Filter
    Shape newFilterShape{channelReshaped, filterShape[Dims4D::Filter::IC], filterShape[Dims4D::Filter::KY],
                         filterShape[Dims4D::Filter::KX]};
    auto newConstFilter = getReshapedConst(filterConst, newFilterShape, rewriter);

    // Reshape Bias
    auto bias = origOp.getBias();
    if (bias != nullptr) {
        auto biasConst = mlir::cast<Const::DeclareOp>(bias.getDefiningOp());
        auto biasShape = getShape(bias);
        Shape newBiasShape{biasShape[Dims4D::Act::N], channelReshaped, biasShape[Dims4D::Act::H],
                           biasShape[Dims4D::Act::W]};
        bias = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
                getReshapedConst(biasConst, newBiasShape, rewriter));
    }

    auto newGroupAttr = getIntAttr(ctx, channelReshaped);
    auto newGroupConv = rewriter.create<IE::GroupConvolutionOp>(
            origOp->getLoc(), inShapeCast.getResult(), newConstFilter, bias, origOp.getStridesAttr(),
            origOp.getPadsBeginAttr(), origOp.getPadsEnd(), origOp.getDilationsAttr(), newGroupAttr,
            origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
    auto origOutputType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
    newGroupConv.getOutput().setType(
            mlir::cast<mlir::RankedTensorType>(origOutputType.changeShape(getShape(newGroupConv.getOutput()))));

    auto outShape = getShape(origOp.getOutput()).raw();
    auto outShapeCast = rewriter.create<IE::ShapeCastOp>(origOp->getLoc(), newGroupConv.getOutput(),
                                                         getIntArrayAttr(ctx, outShape));

    rewriter.replaceOp(origOp, outShapeCast.getResult());

    return mlir::success();
}

//
// ReshapeAddInput
//

class ReshapeAddInput final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    ReshapeAddInput(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// Adjust input shape of AddOp with same input, from C to H and W
//   [1, C, 1, 1]      -> [1, C/16, 4, 4]
//   [1, 108864, 2, 1] -> [1, 108864/4/7, 2*7, 4]
//   [1, 12096, 9, 2]  -> [1, 12096/2, 9, 2*2]
//   [1, 256, 2, 1]    -> [1, 256/4, 2, 1*4]
//
mlir::LogicalResult ReshapeAddInput::matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    if (VPU::NCEInvariant::isSupported(origOp).failed()) {
        return matchFailed(_log, rewriter, origOp, "Not a valid NCE addOp");
    }

    const auto inputShape = getShape(origOp.getInput1());
    const auto outputShape = getShape(origOp.getOutput());
    const int64_t rank4D = 4;
    if (inputShape.size() != rank4D) {
        return matchFailed(_log, rewriter, origOp, "Not a valid addOp with input shape size 4");
    }

    const auto isConstInput = [](mlir::Value input) {
        return mlir::isa_and_nonnull<Const::DeclareOp>(input.getDefiningOp());
    };

    bool isSameInput = origOp.getInput1() == origOp.getInput2();
    if (!isSameInput && !isConstInput(origOp.getInput1()) && !isConstInput(origOp.getInput2())) {
        // Adjust shape on H since eltwise_add just support clustering and SOH when multi-clustering
        // For parallel eltwise ops, to avoid being split to more ops to avoid idle regressions
        // since WLM is not good for NPU40XX and subsequent platforms
        // TODO(E#148228): to support cases for NPU40XX and subsequent platforms
        auto hasParallelAdds = [](mlir::Operation* op) -> bool {
            auto preSliceOp = op->getOperand(0).getDefiningOp<IE::SliceOp>();
            if (preSliceOp == nullptr) {
                return false;
            }

            auto root = preSliceOp->getOperand(0).getDefiningOp();
            if (root == nullptr) {
                return false;
            }

            for (auto userOp : root->getUsers()) {
                auto sliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(*userOp);
                if (sliceOp == nullptr || sliceOp == preSliceOp) {
                    continue;
                }

                auto sliceUser = *sliceOp->getUsers().begin();
                if (mlir::isa_and_nonnull<IE::ShapeCastOp>(sliceUser)) {
                    sliceUser = *sliceUser->getUsers().begin();
                }

                if (mlir::isa_and_nonnull<IE::AddOp>(sliceUser)) {
                    return true;
                }
            }
            return false;
        };

        if (inputShape == getShape(origOp.getInput2()) && hasParallelAdds(origOp) &&
            mlir::isa_and_nonnull<IE::ViewLikeOpInterface>(*origOp->getUsers().begin()) &&
            vpux::config::getArch(origOp) == vpux::config::ArchKind::NPU37XX) {
            isSameInput = false;
        } else {
            return matchFailed(_log, rewriter, origOp, "Not a valid addOp with same input");
        }
    }

    if (!allowsChannelsReshape(origOp)) {
        return matchFailed(rewriter, origOp, "Cannot reshape channels of operation.");
    }

    const auto needToAdjustChannel = inputShape[Dims4D::Act::C] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
    if ((inputShape[Dims4D::Act::H] >= 4 || inputShape[Dims4D::Act::W] >= 4) && !needToAdjustChannel) {
        return matchFailed(_log, rewriter, origOp, "Input shape H or W is greater than 4");
    }

    // Adjust the width/height/channel for alignment
    int64_t divider = 1;
    int64_t widthReshaped, heightReshaped;
    std::tie(divider, widthReshaped) = calcShapeAligned(divider, inputShape[Dims4D::Act::W]);
    std::tie(divider, heightReshaped) = calcShapeAligned(divider, inputShape[Dims4D::Act::H]);
    if (divider == 1 && !needToAdjustChannel) {
        return matchFailed(_log, rewriter, origOp, "Don't need to align");
    }

    if (inputShape[Dims4D::Act::C] % divider != 0) {
        return matchFailed(_log, rewriter, origOp, "Input shape could not be divided {0}", inputShape[Dims4D::Act::C]);
    }

    auto channelReshaped = inputShape[Dims4D::Act::C] / divider;
    auto alignIface = mlir::cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());
    if (channelReshaped % alignIface.getInputChannelAlignment() != 0) {
        return matchFailed(_log, rewriter, origOp, "Remaining channel is not aligned.");
    }

    // Adjust channel to avoid split, for example: tensor<1x15360x4x4> => tensor<1x7680x8x4>
    const auto getNewChannelShape = [](int64_t lengthOfAlignment, int64_t inputToDivide,
                                       int64_t inputToMultiply) -> SmallVector<int64_t> {
        const auto maxFactor = std::max(checked_cast<int64_t>(2), divUp(lengthOfAlignment, checked_cast<int64_t>(2)));
        for (const auto i : irange<int64_t>(2, maxFactor)) {
            if (lengthOfAlignment % i == 0) {
                const auto newShrink = inputToDivide / i;
                if (newShrink > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
                    continue;
                }
                const auto newExpand = inputToMultiply * i;
                if (newExpand > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
                    return {};
                }

                return {newShrink, newExpand};
            }
        }
        return {};
    };

    if (channelReshaped > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        const auto channelAlignment = channelReshaped / alignIface.getInputChannelAlignment();
        const auto expandDim = heightReshaped <= widthReshaped ? heightReshaped : widthReshaped;
        const auto newChannelShape = getNewChannelShape(channelAlignment, channelReshaped, expandDim);
        if (!newChannelShape.empty()) {
            channelReshaped = newChannelShape[0];
            if (heightReshaped <= widthReshaped) {
                heightReshaped = newChannelShape[1];
            } else {
                widthReshaped = newChannelShape[1];
            }
        }
    }

    if (channelReshaped == inputShape[Dims4D::Act::C]) {
        return matchFailed(_log, rewriter, origOp, "The final channel is not changed.");
    }

    // Reshape Input
    const auto ctx = origOp->getContext();
    const SmallVector<int64_t> newInShape = {inputShape[Dims4D::Act::N], channelReshaped, heightReshaped,
                                             widthReshaped};
    auto inShapeCast1 =
            rewriter.create<IE::ShapeCastOp>(origOp->getLoc(), origOp.getInput1(), getIntArrayAttr(ctx, newInShape))
                    .getResult();

    mlir::Value inShapeCast2 = inShapeCast1;
    if (!isSameInput) {
        inShapeCast2 =
                rewriter.create<IE::ShapeCastOp>(origOp->getLoc(), origOp.getInput2(), getIntArrayAttr(ctx, newInShape))
                        .getResult();
    }

    // Update addOp
    const auto origType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
    const auto newType = origType.changeShape(ShapeRef(newInShape));
    auto newAddOp = rewriter.create<IE::AddOp>(
            origOp->getLoc(), newType, inShapeCast1, inShapeCast2, origOp.getAutoBroadcastAttr(),
            origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    // Reshape Output
    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(origOp, newAddOp.getOutput(), getIntArrayAttr(ctx, outputShape));

    return mlir::success();
}

//
// ReshapeConvInput
//

template <typename ConcreteOp>
class ReshapeConvInput final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ReshapeConvInput(mlir::MLIRContext* ctx, bool favorLargerHeight, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _favorLargerHeight(favorLargerHeight), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _favorLargerHeight;
    Logger _log;
};

mlir::Operation* buildCustomOp(IE::ConvolutionOp op, mlir::PatternRewriter& rewriter, mlir::Value input,
                               vpux::NDTypeInterface outType) {
    return rewriter.create<IE::ConvolutionOp>(
            appendLoc(op->getLoc(), "aligned"), outType, input, op.getFilter(), op.getBias(), op.getStrides(),
            op.getPadsBegin(), op.getPadsEnd(), op.getDilations(), op.getPostOpAttr(), op.getClampAttr(),
            op.getStaticScaleAttr(), op.getOutputPaddingAttr(), op.getInputPaddingAttr());
}

mlir::Operation* buildCustomOp(IE::GroupConvolutionOp op, mlir::PatternRewriter& rewriter, mlir::Value input,
                               vpux::NDTypeInterface outType) {
    return rewriter.create<IE::GroupConvolutionOp>(
            appendLoc(op->getLoc(), "aligned"), outType, input, op.getFilter(), op.getBias(), op.getStrides(),
            op.getPadsBegin(), op.getPadsEnd(), op.getDilations(), op.getGroupsAttr(), op.getPostOpAttr(),
            op.getClampAttr(), op.getOutputPaddingAttr(), op.getInputPaddingAttr());
}

template <typename ConcreteOp>
ConcreteOp expandDimToReshape(ConcreteOp op, mlir::PatternRewriter& rewriter, Dim dimToExpand, Logger log) {
    // Expand H or W to support shape balancing for better DPU efficiency
    auto inputShape = getShape(op.getInput());
    auto outputShape = getShape(op.getOutput());
    auto newSizeDimToExpand = alignValUp(inputShape[dimToExpand], VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT);
    auto targetShape = Shape(inputShape);
    targetShape[dimToExpand] = newSizeDimToExpand;
    log.debug("Expanding shape for op {0} {1} from {2} to {3}", op->getName(), op->getLoc(), inputShape, targetShape);
    // Rewrite Conv from [N, C, H, 1] to [N, C, new_H, 1], or [N, C, 1, W] to [N, C, 1, new_W]
    // Expand -> Conv -> Slice
    auto padBegin = SmallVector<int64_t>(inputShape.size(), 0);
    auto padEnd = SmallVector<int64_t>(inputShape.size(), 0);
    padEnd[dimToExpand.ind()] = newSizeDimToExpand - inputShape[dimToExpand];
    auto inputExpandOp = rewriter.create<IE::ExpandOp>(appendLoc(op->getLoc(), "expand"), op.getInput(),
                                                       getIntArrayAttr(rewriter, ArrayRef(padBegin)),
                                                       getIntArrayAttr(rewriter, ArrayRef(padEnd)));

    auto targetOutType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
    auto targetOutShape = Shape(getShape(op.getOutput()));
    targetOutShape[dimToExpand] = targetShape[dimToExpand];
    targetOutType = targetOutType.changeShape(targetOutShape);
    auto newOp = mlir::cast<ConcreteOp>(buildCustomOp(op, rewriter, inputExpandOp, targetOutType));

    auto offsets = SmallVector<int64_t>(outputShape.size(), 0);
    auto sizes = SmallVector<int64_t>(outputShape.begin(), outputShape.end());
    auto outputSliceOp =
            rewriter.create<IE::SliceOp>(appendLoc(op->getLoc(), "slice"), newOp, getIntArrayAttr(rewriter, offsets),
                                         getIntArrayAttr(rewriter, sizes));
    rewriter.replaceOp(op, outputSliceOp);
    return newOp;
}

template <typename ConcreteOp>
mlir::LogicalResult ReshapeConvInput<ConcreteOp>::matchAndRewrite(ConcreteOp convOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    /*
        Convert 1x1 convolution from
            input          filter                    input               filter
        [1, C, H, 1]     [OC, C, 1, 1]             [1, C, H, 1]        [OC, C, 1, 1]
              \             /                =>        |                   |
                   Conv                            AffineReshape           |
               [1, OC, H, 1]                     [1, C, H/4, 4]            |
                                                       \                  /
                                                              Conv
                                                        [1, OC, H/4, 4]
                                                               |
                                                          AffineReshape
                                                          [1, OC, H, 1]

            input          filter                    input               filter
        [1, C, 1, W]     [OC, C, 1, 1]            [1, C, 1, W]        [OC, C, 1, 1]
              \             /                =>        |                   |
                   Conv                            AffineReshape           |
               [1, OC, 1, W]                     [1, C, 4, W/4]            |
                                                       \                  /
                                                              Conv
                                                        [1, OC, 4, W/4]
                                                               |
                                                          AffineReshape
                                                          [1, OC, 1, W]
    */
    auto nestedLog = _log.nest();
    auto ctx = convOp->getContext();
    auto inputShape = getShape(convOp.getInput());
    const auto filterShape = getShape(convOp.getFilter());

    // Current logic only works with input and filter shape with 4 dimensions
    if (inputShape.size() != 4 || filterShape.size() != 4) {
        return mlir::failure();
    }

    _log.debug("Got '{0}' at '{1}'", convOp->getName(), convOp->getLoc());

    // check suitable 1x1 convolution with input width/height = 1 , strides = [1, 1], kernel = [1, 1]
    if ((inputShape[Dims4D::Act::W] != 1 && inputShape[Dims4D::Act::H] != 1) || filterShape[Dims4D::Filter::KX] != 1 ||
        filterShape[Dims4D::Filter::KY] != 1) {
        return matchFailed(nestedLog, rewriter, convOp, "Cannot adjust conv input: in shape {0}, filter shape {1}",
                           inputShape, filterShape);
    }

    const auto strides = parseIntArrayAttr<int64_t>(convOp.getStrides());
    auto stridesEqualToOne = llvm::all_of(strides, [](const int64_t elem) {
        return elem == 1;
    });
    if (!stridesEqualToOne) {
        return matchFailed(nestedLog, rewriter, convOp, "Cannot adjust conv input: strides are not [1, 1]");
    }

    auto alignOnH = inputShape[Dims4D::Act::H] == 1;
    if (alignOnH) {
        if (inputShape[Dims4D::Act::W] == VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT ||
            inputShape.totalSize() < THRESHOLD_FOR_BENEFICIAL_CONVERSION) {
            return matchFailed(nestedLog, rewriter, convOp, "Adjusting conv inputs is not beneficial.");
        }
    }

    int64_t convolutionInputShapeAlignment = VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT;
    auto inputShapeToAlign = alignOnH ? inputShape[Dims4D::Act::W] : inputShape[Dims4D::Act::H];

    // if the input height/width is small, reshape input is not beneficial. e.g. shape [1, 320, 4, 1]
    if (inputShapeToAlign <= VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT ||
        inputShapeToAlign / VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT < 2) {
        return matchFailed(nestedLog, rewriter, convOp,
                           "Adjusting conv inputs is not beneficial: shape to align is too small.");
    }

    // Find another factor if input height is not divisible by 4
    if (inputShapeToAlign % VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT != 0) {
        const int64_t val = inputShapeToAlign;
        auto factor = static_cast<int64_t>(sqrt(static_cast<double>(val)));
        for (; factor > 1; factor--) {
            if (val % factor == 0) {
                convolutionInputShapeAlignment = factor;
                break;
            }
        }

        if (factor == 1) {
            const auto minSize =
                    std::min(getShape(convOp.getInput()).totalSize(), getShape(convOp.getOutput()).totalSize());
            if (minSize < THRESHOLD_FOR_BENEFICIAL_CONVERSION) {
                // The extra costs of expand and slice are bigger than the benefits of balancing shape
                return matchFailed(nestedLog, rewriter, convOp, "Could not find factor for shape adjustment.");
            }
            nestedLog.debug("Need to align H or W first for {0}", convOp->getLoc());
            auto alignedNewConv =
                    expandDimToReshape(convOp, rewriter, alignOnH ? Dims4D::Act::W : Dims4D::Act::H, nestedLog);
            VPUX_THROW_WHEN(alignedNewConv == nullptr, "Failed to expand and reshape convolution input");
            convOp = alignedNewConv;
        }
    }

    rewriter.setInsertionPoint(convOp);
    inputShape = getShape(convOp.getInput());
    nestedLog.debug("Adjust input shape for convolution at '{0}'", convOp->getLoc());

    // Try to move larger dim size to H as it is more likely to lead to beneficial tiling & multiclustering
    int64_t newHeight, newWidth;
    std::tie(newHeight, newWidth) = [&]() -> std::tuple<int64_t, int64_t> {
        if (!_favorLargerHeight) {
            return alignOnH ? std::make_tuple(convolutionInputShapeAlignment,
                                              inputShape[Dims4D::Act::W] / convolutionInputShapeAlignment)
                            : std::make_tuple(inputShape[Dims4D::Act::H] / convolutionInputShapeAlignment,
                                              convolutionInputShapeAlignment);
        }

        if (alignOnH) {
            auto height = std::max(convolutionInputShapeAlignment,
                                   inputShape[Dims4D::Act::W] / convolutionInputShapeAlignment);
            return std::make_tuple(height, inputShape[Dims4D::Act::W] / height);
        }

        auto height =
                std::max(convolutionInputShapeAlignment, inputShape[Dims4D::Act::H] / convolutionInputShapeAlignment);
        return std::make_tuple(height, inputShape[Dims4D::Act::H] / height);
    }();

    const SmallVector<int64_t> newInShape = {inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C], newHeight,
                                             newWidth};

    const auto inputShapeAttr = getIntArrayAttr(convOp->getContext(), newInShape);
    nestedLog.trace("New input shape {0}, aligning on H {1}", inputShapeAttr, alignOnH);

    SmallVector<SmallVector<int64_t>> inDimMappingOnW{{Dims4D::Act::N.ind()},
                                                      {Dims4D::Act::C.ind()},
                                                      {Dims4D::Act::H.ind(), Dims4D ::Act::W.ind()},
                                                      {Dims4D::Act::W.ind()}};
    SmallVector<SmallVector<int64_t>> inDimMappingOnH{{Dims4D::Act::N.ind()},
                                                      {Dims4D::Act::C.ind()},
                                                      {Dims4D::Act::H.ind()},
                                                      {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()}};
    auto inDimMapping = alignOnH ? inDimMappingOnH : inDimMappingOnW;
    auto newInput =
            rewriter.create<IE::AffineReshapeOp>(appendLoc(convOp->getLoc(), "_balance_reshape_in"), convOp.getInput(),
                                                 getIntArrayOfArray(ctx, inDimMapping), inputShapeAttr);
    mlir::IRMapping mapper;
    mapper.map(convOp.getInput(), newInput.getOutput());
    auto newConvOp = mlir::dyn_cast<ConcreteOp>(rewriter.clone(*convOp, mapper));

    auto outputShape = getShape(convOp.getOutput());
    auto newOutputShape =
            Shape(SmallVector<int64_t>{outputShape[Dims4D::Act::N], outputShape[Dims4D::Act::C], newHeight, newWidth});

    auto newOutputType = mlir::cast<vpux::NDTypeInterface>(newConvOp.getOutput().getType());
    newOutputType = newOutputType.changeShape(newOutputShape);
    newConvOp.getOutput().setType(mlir::cast<mlir::RankedTensorType>(newOutputType));
    const auto outShape = getShape(convOp.getOutput()).raw();
    const auto outShapeAttr = getIntArrayAttr(ctx, outShape);

    SmallVector<SmallVector<int64_t>> outDimMappingOnW{{Dims4D::Act::N.ind()},
                                                       {Dims4D::Act::C.ind()},
                                                       {Dims4D::Act::H.ind()},
                                                       {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()}};
    SmallVector<SmallVector<int64_t>> outDimMappingOnH{{Dims4D::Act::N.ind()},
                                                       {Dims4D::Act::C.ind()},
                                                       {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()},
                                                       {Dims4D::Act::W.ind()}};
    auto outDimMapping = alignOnH ? outDimMappingOnH : outDimMappingOnW;
    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(convOp, newConvOp.getOutput(),
                                                     getIntArrayOfArray(ctx, outDimMapping), outShapeAttr);

    _log.debug("Successfully adjusted Conv input shapes.");
    return mlir::success();
}

//
// ReshapeExpandDWConvInput
//

class ReshapeExpandDWConvInput final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    ReshapeExpandDWConvInput(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReshapeExpandDWConvInput::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    // Only support GroupConvolution with constant filter
    // Kernel size must be 1x1, and must be a depthwise convolution.
    auto kernelShape = getShape(origOp.getFilter());
    if (kernelShape[Dims4D::Filter::KX] != 1 || kernelShape[Dims4D::Filter::KX] != 1 ||
        kernelShape[Dims4D::Filter::OC] != origOp.getGroups().value()) {
        return mlir::failure();
    }
    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };
    if (!VPU::NCEDepthConvolutionOp::isSupported(origOp, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true)) {
        return mlir::failure();
    }

    if (!allowsChannelsReshape(origOp)) {
        return matchFailed(rewriter, origOp, "Cannot reshape channels of operation.");
    }

    // Check stride
    auto strides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    if (strides[Dims4D::Strides::X.ind()] > 1 || strides[Dims4D::Strides::Y.ind()] == 1) {
        return mlir::failure();
    }
    auto parentExpandOp = origOp.getInput().getDefiningOp<IE::ExpandOp>();
    if (parentExpandOp == nullptr) {
        return mlir::failure();
    }
    if (!parentExpandOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());
    if (iface == nullptr) {
        return mlir::failure();
    }
    const auto alignment = iface.getInputChannelAlignment();

    const auto unExpandedInput = parentExpandOp.getInput();
    const auto unExpandedType = mlir::cast<vpux::NDTypeInterface>(unExpandedInput.getType());
    auto unExpandedShape = Shape(unExpandedType.getShape().toValues());

    auto IN = unExpandedShape[Dims4D::Act::N];
    auto IC = unExpandedShape[Dims4D::Act::C];
    auto IH = unExpandedShape[Dims4D::Act::H];
    auto IW = unExpandedShape[Dims4D::Act::W];

    if (IC % alignment == 0) {
        _log.trace("Channel is already aligned");
        return mlir::failure();
    }
    // Check if can align
    if (IC * IW % alignment != 0) {
        _log.trace("Channel cannot be aligned");
        return mlir::failure();
    }

    auto constInput = origOp.getFilter().getDefiningOp<Const::DeclareOp>();
    auto realDataSizeResult = vpux::IE::getBaseContentNumElements(constInput);
    auto activationDataSize =
            std::accumulate(unExpandedShape.begin(), unExpandedShape.end(), int64_t(1), std::multiplies<int64_t>());
    if (mlir::failed(realDataSizeResult) ||
        (realDataSizeResult.value() != 1 && realDataSizeResult.value() != activationDataSize)) {
        _log.trace("Unsupported const input {0} at {1}", constInput->getName(), constInput->getLoc());
        return mlir::failure();
    }

    // Insert ShapeCast to align input shape
    // For example: 1x3x640x640 -> 1x48x640x40
    auto newIC = alignment * IC;
    auto newIW = IC * IW / newIC;
    auto alignedInShape = Shape({IN, newIC, IH, newIW});
    auto alignedInputShapeAttr = getIntArrayAttr(rewriter.getContext(), alignedInShape);
    const auto dstType = unExpandedType.changeShape(ShapeRef(alignedInShape));

    auto shapeCastInputOp =
            rewriter.create<IE::ShapeCastOp>(origOp->getLoc(), dstType, unExpandedInput, alignedInputShapeAttr);

    const auto& contentAttr = constInput.getContentAttr();
    const auto& baseContent = contentAttr.getBaseContent();
    auto dataShape = getShape(constInput.getOutput()).toValues();

    Shape realDataShape = baseContent.getShapedType().getShape();

    auto newConstOutputType = mlir::cast<vpux::NDTypeInterface>(constInput.getOutput().getType());
    auto newConstantShape = Shape(newConstOutputType.getShape().size(), int64_t(1));
    newConstantShape[Dims4D::Act::N] = alignedInShape[Dims4D::Act::C];
    newConstOutputType = newConstOutputType.changeShape(newConstantShape);
    auto newContentAttrSetup = Const::ContentSetup(baseContent.getType())
                                       .broadcast(Dims4D::Act::N, alignedInShape[Dims4D::Act::C])
                                       .reshape(newConstantShape);

    for (auto& attr : contentAttr.getTransformations()) {
        if (mlir::isa<vpux::Const::PadWithZeroAttr>(attr) || mlir::isa<vpux::Const::BroadcastAttr>(attr)) {
            // skip the attributes that the contentAttr already contains
            continue;
        }
        if (mlir::isa<vpux::Const::ReshapeAttr>(attr)) {
            // Only remain the reshape attribute when it's used for dimension expansion to 4D,
            // and for dimension shrink from 5D to 4D
            // e.g., from [1x512] to [1x1x1x512]
            auto reshapeAttr = mlir::cast<vpux::Const::ReshapeAttr>(attr);
            auto reshapeShape = Shape(parseIntArrayAttr<int64_t>(reshapeAttr.getShape()));
            if (vpux::IE::isNotDimExpansionReshape(realDataShape, reshapeShape) &&
                vpux::IE::isNotDimShrinkReshape(realDataShape, reshapeShape)) {
                continue;
            }
        }

        newContentAttrSetup = newContentAttrSetup.addTransformation(attr);
    }

    auto newConstInput = rewriter.create<Const::DeclareOp>(
            origOp->getLoc(), newConstOutputType, Const::ContentAttr::get(baseContent, std::move(newContentAttrSetup)));

    // Infer group conv output shape
    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(origOp.getDilations());
    auto convInShape = to_small_vector(alignedInShape.raw());
    convInShape[1] /= newIC;
    auto filterShape = to_small_vector(newConstOutputType.getShape().raw());

    const auto op =
            ov::op::v1::Convolution(std::make_shared<ov::op::v0::Parameter>(
                                            ov::element::i32, ov::Shape(convInShape.begin(), convInShape.end())),
                                    std::make_shared<ov::op::v0::Parameter>(
                                            ov::element::i32, ov::Shape(filterShape.begin(), filterShape.end())),
                                    ov::Strides(windowStrides.begin(), windowStrides.end()),
                                    ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
                                    ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
                                    ov::Strides(windowDilations.begin(), windowDilations.end()));

    const auto& outputShape = op.get_output_partial_shape(0);
    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    const auto origOutType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType()).getElementType();
    const auto convOutType = unExpandedType.changeShapeElemType(ShapeRef(shapeI64), origOutType);
    auto groupsAttr = getIntAttr(rewriter, newIC);
    auto grpConv = rewriter.create<IE::GroupConvolutionOp>(
            origOp->getLoc(), convOutType, shapeCastInputOp, newConstInput, origOp.getBias(), origOp.getStridesAttr(),
            origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilationsAttr(), groupsAttr, origOp.getPostOpAttr(),
            origOp.getClampAttr(), origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    // Insert ShapeCast to reshape the output to original outShape
    auto unExpandedOutShape = Shape({IN, IC, convOutType.getShape()[Dims4D::Act::H], IW});
    auto shapeCastOutputAttr = getIntArrayAttr(rewriter.getContext(), unExpandedOutShape);
    auto shapeCastOutputOp = rewriter.create<IE::ShapeCastOp>(
            origOp->getLoc(), convOutType.changeShape(unExpandedOutShape), grpConv, shapeCastOutputAttr);

    auto newOutputExpandOp = rewriter.create<IE::ExpandOp>(
            origOp->getLoc(), shapeCastOutputOp, parentExpandOp.getPadsBeginAttr(), parentExpandOp.getPadsEndAttr());

    // Replace with new sub graph
    rewriter.replaceOp(origOp, newOutputExpandOp->getResult(0));

    return mlir::success();
}

//
// AdjustConvolutionInputShape
//

class AdjustConvolutionInputShapePass final :
        public IE::impl::AdjustConvolutionInputShapeBase<AdjustConvolutionInputShapePass> {
public:
    explicit AdjustConvolutionInputShapePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustConvolutionInputShapePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);

    // Adjust between H and W like:
    // [1, C, 1, W]  -> [1, C, H*4, W/4]
    // [1, C, H, 1]  -> [1, C, H/4, W*4]
    // if favorLargeHeight == true => newHeight = max(4, dim/4)
    patterns.add<ReshapeConvInput<IE::ConvolutionOp>>(&ctx, /*favorLargeHeight*/ true, _log);
    // Disabling reshape with large height for GroupConv
    // While it improves inference time for some models, it does bring perf regressions to several others due to being
    // able to assign SOH instead of SOK. To track: E147489
    patterns.add<ReshapeConvInput<IE::GroupConvolutionOp>>(&ctx, /*favorLargeHeight*/ false, _log);

    // Adjust from C to H and W like [1, C, 1, 1] -> [1, C/16, 4, 4]
    patterns.add<ReshapeSingleConstDWConvInput>(&ctx, _log);
    patterns.add<ReshapeAddInput>(&ctx, _log);

    // Adust between C and H/W like [1, C, H, W] -> [1, C*4, H, W/4]
    // Also need stride[H] > 1
    patterns.add<ReshapeExpandDWConvInput>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}
}  // namespace

//
// createAdjustConvolutionInputShapePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustConvolutionInputShapePass(Logger log) {
    return std::make_unique<AdjustConvolutionInputShapePass>(log);
}
