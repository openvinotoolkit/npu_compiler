//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/shape_utils.hpp"
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
    ReshapeConvInput(mlir::MLIRContext* ctx, bool favorLargerHeight, int64_t preferredSpatialAlignment, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx),
              _favorLargerHeight(favorLargerHeight),
              _preferredSpatialAlignment(preferredSpatialAlignment),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _favorLargerHeight;
    int64_t _preferredSpatialAlignment;
    Logger _log;
};

mlir::Operation* buildCustomOp(IE::ConvolutionOp op, mlir::PatternRewriter& rewriter, mlir::Value input,
                               vpux::NDTypeInterface outType) {
    auto newConv = cloneConvolutionOp(rewriter, op, outType, input, op.getFilter(), appendLoc(op->getLoc(), "aligned"));
    return newConv;
}

mlir::Operation* buildCustomOp(IE::GroupConvolutionOp op, mlir::PatternRewriter& rewriter, mlir::Value input,
                               vpux::NDTypeInterface outType) {
    return rewriter.create<IE::GroupConvolutionOp>(
            appendLoc(op->getLoc(), "aligned"), outType, input, op.getFilter(), op.getBias(), op.getStrides(),
            op.getPadsBegin(), op.getPadsEnd(), op.getDilations(), op.getGroupsAttr(), op.getPostOpAttr(),
            op.getClampAttr(), op.getOutputPaddingAttr(), op.getInputPaddingAttr());
}

template <typename ConcreteOp>
ConcreteOp expandDimToReshape(ConcreteOp op, mlir::PatternRewriter& rewriter, Dim dimToExpand, int64_t divide,
                              Logger log) {
    // Expand H or W to support shape balancing for better DPU efficiency
    auto inputShape = getShape(op.getInput());
    auto outputShape = getShape(op.getOutput());
    auto newSizeDimToExpand = alignValUp(inputShape[dimToExpand], divide);
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

    // Use the platform-preferred spatial alignment when the dimension to split is divisible by it
    // and the resulting orthogonal dimension is also large enough for efficient tiling.
    // Skip this override when convOp is connected to a SoftMaxOp, as the reshape breaks or
    // degrades vertical fusion efficiency (E#210093).
    if (_preferredSpatialAlignment > VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT) {
        const auto arch = vpux::config::getArch(convOp);
        const auto isConnectedToSoftmaxUser = [arch](mlir::Operation* op) {
            // Determines whether a user op should be walked through when searching for a
            // downstream SoftMaxOp.
            const auto isTransparentForSoftmaxSearch = [arch](mlir::Operation* user) {
                if (IE::isPureViewOp(user) || mlir::isa<IE::AddOp>(user)) {
                    return true;
                }
                (void)arch;
                return false;
            };

            llvm::SmallPtrSet<mlir::Operation*, 16> visited;
            SmallVector<mlir::Operation*> worklist(op->getUsers().begin(), op->getUsers().end());
            while (!worklist.empty()) {
                auto* user = worklist.pop_back_val();
                if (!visited.insert(user).second) {
                    continue;
                }
                if (mlir::isa<IE::SoftMaxOp>(user)) {
                    return true;
                }
                if (isTransparentForSoftmaxSearch(user)) {
                    worklist.append(user->getUsers().begin(), user->getUsers().end());
                }
            }
            return false;
        };
        const bool hasSoftmaxUser = isConnectedToSoftmaxUser(convOp);
        const auto getProducerThroughViewLikeOps = [](mlir::Value val) -> mlir::Operation* {
            auto* defOp = val.getDefiningOp();
            while (defOp != nullptr && IE::isPureViewOp(defOp) && defOp->getNumOperands() == 1) {
                defOp = defOp->getOperand(0).getDefiningOp();
            }
            return defOp;
        };
        const bool hasSoftmaxProducer =
                mlir::isa_and_nonnull<IE::SoftMaxOp>(getProducerThroughViewLikeOps(convOp.getInput()));
        const bool isConnectedToSoftmax = hasSoftmaxUser || hasSoftmaxProducer;
        if (!isConnectedToSoftmax && inputShapeToAlign % _preferredSpatialAlignment == 0 &&
            inputShapeToAlign / _preferredSpatialAlignment >= _preferredSpatialAlignment) {
            convolutionInputShapeAlignment = _preferredSpatialAlignment;
        }
    }

    if (inputShapeToAlign > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        convolutionInputShapeAlignment = inputShapeToAlign / VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
        convolutionInputShapeAlignment =
                ((convolutionInputShapeAlignment + VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT - 1) /
                 VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT) *
                VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT;
        if (convolutionInputShapeAlignment > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
            convolutionInputShapeAlignment = VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
        }
    }

    // Find another factor if input height is not divisible by 4
    if (inputShapeToAlign % convolutionInputShapeAlignment != 0) {
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
            auto alignedNewConv = expandDimToReshape(convOp, rewriter, alignOnH ? Dims4D::Act::W : Dims4D::Act::H,
                                                     convolutionInputShapeAlignment, nestedLog);
            VPUX_THROW_WHEN(alignedNewConv == nullptr, "Failed to expand and reshape convolution input");
            convOp = alignedNewConv;
        }
    }

    rewriter.setInsertionPoint(convOp);
    inputShape = getShape(convOp.getInput());
    nestedLog.debug("Adjust input shape for convolution at '{0}'", convOp->getLoc());

    SmallVector<SmallVector<int64_t>> outDimMappingOnW{{Dims4D::Act::N.ind()},
                                                       {Dims4D::Act::C.ind()},
                                                       {Dims4D::Act::H.ind()},
                                                       {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()}};
    SmallVector<SmallVector<int64_t>> outDimMappingOnH{{Dims4D::Act::N.ind()},
                                                       {Dims4D::Act::C.ind()},
                                                       {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()},
                                                       {Dims4D::Act::W.ind()}};
    auto outDimMapping = alignOnH ? outDimMappingOnH : outDimMappingOnW;

    auto maybeReshapeOp = convOp.getInput().getDefiningOp();
    if (mlir::isa_and_present<IE::ShapeCastOp>(maybeReshapeOp) && maybeReshapeOp->hasOneUse()) {
        auto reshapeInputOp = maybeReshapeOp->getOperand(0).getDefiningOp();
        auto isProducerLegalOp =
                reshapeInputOp != nullptr && reshapeInputOp->hasOneUse() && !IE::isPureViewOp(reshapeInputOp);

        auto userOp = *convOp->getUsers().begin();
        auto isUserLegalOp = !convOp->hasOneUse() || userOp->template hasTrait<mlir::OpTrait::IsTerminator>() ||
                             IE::isPureViewOp(userOp);

        if (isProducerLegalOp && isUserLegalOp) {
            auto reshapeInputShape = getShape(maybeReshapeOp->getOperand(0));
            // Match op1->reshapeOp->op2->[viewOp] to op1->op2->reshapeOp->[viewOp]
            // to reuse the reshape input for better tiling and fusion, and avoid extra copy
            if (reshapeInputShape.size() == 4 &&
                reshapeInputShape[Dims4D::Act::H] > VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT &&
                reshapeInputShape[Dims4D::Act::W] > VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT &&
                reshapeInputShape[Dims4D::Act::C] == inputShape[Dims4D::Act::C] &&
                reshapeInputShape[Dims4D::Act::N] == inputShape[Dims4D::Act::N]) {
                nestedLog.debug("Found Reshape input with H={0}, W={1}. Using it directly.",
                                reshapeInputShape[Dims4D::Act::H], reshapeInputShape[Dims4D::Act::W]);

                mlir::IRMapping mapper;
                mapper.map(convOp.getInput(), maybeReshapeOp->getOperand(0));
                auto newConvOp = rewriter.clone(*convOp, mapper);
                vpux::inferReturnTypes(newConvOp, vpux::InferShapedTypeMode::SHAPE);

                const auto outShapeAttr = getIntArrayAttr(ctx, getShape(convOp.getOutput()));
                rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(convOp, newConvOp->getResult(0),
                                                                 getIntArrayOfArray(ctx, outDimMapping), outShapeAttr);
                return mlir::success();
            }
        }
    }

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
            rewriter.create<IE::AffineReshapeOp>(appendLoc(convOp->getLoc(), "balance_reshape_in"), convOp.getInput(),
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

    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(convOp, newConvOp.getOutput(),
                                                     getIntArrayOfArray(ctx, outDimMapping), outShapeAttr);

    _log.debug("Successfully adjusted Conv input shapes.");
    return mlir::success();
}

//
// ReshapeConv1DInputWithHalo
//
// Rewrite a 1D convolution [1,C,1,L] / [OC,C,1,K] (or the H-major variant) into a 2D
// conv [1,C,numRows,L_row+K-1] with stride 1, pads = 0, by slicing numRows overlapping
// length-(L_row+K-1) windows and concatenating them along H. The K-1 row-boundary halo
// keeps the rewrite bit-exact while the kernel still slides along W.
//
// Motivation: the DPU MPE grid is 16x16 with an NTHW 16/2 stencil. Activation H=1 leaves
// 15 of 16 MPE rows idle (~16x MAC underuse). Folding L into numRows rows recovers MPE
// utilization for vocoder-style 1D Conv chains.

template <typename ConcreteOp>
class ReshapeConv1DInputWithHalo final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ReshapeConv1DInputWithHalo(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("ReshapeConv1DInputWithHalo");
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

namespace {

// Pick the largest numRows in [mpeAlignment, numDPU * mpeAlignment] that divides outLen
// and keeps L_row >= max(kernel, mpeAlignment). Bias toward larger numRows so the
// multi-cluster tiler has more SOH headroom.
//
// Skip layers below ~500 MMACs or with small inC*outC: those are DMA-bound, and the
// added Concat of numRows overlapping windows costs more than the MPE-utilisation win.
// Caller passes the kernel's per-group IC and activation's full OC, so the MAC formula
// works for both plain Conv and GroupConv.
int64_t findRowCount(int64_t outLen, int64_t kernel, int64_t inC, int64_t outC, int64_t mpeAlignment, int64_t numDPU) {
    constexpr int64_t kMinMACsForRewrite = 500'000'000;
    constexpr int64_t kMinChannelProduct = 64 * 64;

    if (inC * outC * outLen * kernel < kMinMACsForRewrite) {
        return 1;
    }
    if (inC * outC < kMinChannelProduct) {
        return 1;
    }

    const int64_t step = VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT;
    const int64_t minNumRows = mpeAlignment;
    const int64_t maxNumRows = std::max<int64_t>(numDPU, 1) * mpeAlignment;

    const int64_t minLRow = std::max<int64_t>(kernel, mpeAlignment);
    if (outLen < minLRow) {
        return 1;
    }

    const int64_t maxByLRow = outLen / minLRow;
    int64_t numRowsUpper = std::min<int64_t>(maxByLRow, maxNumRows);
    numRowsUpper = (numRowsUpper / step) * step;

    for (int64_t numRows = numRowsUpper; numRows >= minNumRows; numRows -= step) {
        if (outLen % numRows == 0) {
            return numRows;
        }
    }
    return 1;
}

mlir::Value buildPadOp(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input, Dim axis,
                       int64_t padBegin, int64_t padEnd) {
    if (padBegin == 0 && padEnd == 0) {
        return input;
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());

    auto makeZeroConst = [&](int64_t padSize, StringRef nameSuffix) -> mlir::Value {
        SmallVector<int64_t> constShape = to_small_vector(inputType.getShape());
        constShape[axis.ind()] = padSize;
        // changeShape inherits DimsOrder/encoding/memSpace; building a fresh RankedTensorType
        // would drop them and trigger a layout mismatch in the following IE.Concat.
        auto constType = mlir::cast<mlir::RankedTensorType>(inputType.changeShape(ShapeRef(constShape)));
        return Const::createZerosConst(rewriter, appendLoc(loc, nameSuffix), constType);
    };

    SmallVector<mlir::Value> concatInputs;
    if (padBegin > 0) {
        concatInputs.push_back(makeZeroConst(padBegin, "pad_begin"));
    }
    concatInputs.push_back(input);
    if (padEnd > 0) {
        concatInputs.push_back(makeZeroConst(padEnd, "pad_end"));
    }
    return rewriter.create<IE::ConcatOp>(loc, concatInputs, axis.ind()).getOutput();
}

// AffineReshape dim mapping that flattens the [N,C,numRows,L_row] row form back to a 1D
// activation: intoW yields [N,C,1,numRows*L_row], otherwise [N,C,numRows*L_row,1].
SmallVector<SmallVector<int64_t>> getRowMergeMapping(bool intoW) {
    if (intoW) {
        return {{Dims4D::Act::N.ind()},
                {Dims4D::Act::C.ind()},
                {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()},
                {Dims4D::Act::W.ind()}};
    }
    return {{Dims4D::Act::N.ind()},
            {Dims4D::Act::C.ind()},
            {Dims4D::Act::H.ind()},
            {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()}};
}

// A 1D-shaped conv keeps exactly one non-singleton spatial axis and carries the kernel extent on that same axis.
struct Conv1DLayout {
    bool longAxisIsW;
    Dim longDim;
    int64_t kernelLong;
    int64_t inLen;
    int64_t outLen;
};

std::optional<Conv1DLayout> getConv1DLayout(ShapeRef inputShape, ShapeRef filterShape, ShapeRef outputShape) {
    const int64_t inH = inputShape[Dims4D::Act::H];
    const int64_t inW = inputShape[Dims4D::Act::W];
    const int64_t outH = outputShape[Dims4D::Act::H];
    const int64_t outW = outputShape[Dims4D::Act::W];
    const int64_t kY = filterShape[Dims4D::Filter::KY];
    const int64_t kX = filterShape[Dims4D::Filter::KX];

    // Two layouts in the wild:
    //   layoutW: input [1,C,1,L], kernel [OC,C,1,K]
    //   layoutH: input [1,C,L,1], kernel [OC,C,K,1]
    if (inH == 1 && inW > 1 && kY == 1 && kX > 1 && outH == 1 && outW > 1) {
        return Conv1DLayout{/*longAxisIsW=*/true, Dims4D::Act::W, kX, inW, outW};
    }
    if (inW == 1 && inH > 1 && kX == 1 && kY > 1 && outW == 1 && outH > 1) {
        return Conv1DLayout{/*longAxisIsW=*/false, Dims4D::Act::H, kY, inH, outH};
    }
    return std::nullopt;
}

// Preconditions shared by both halo rewrites: 4D operands, an element type the extra
// Concat/AffineReshape plumbing can carry through unchanged, and no explicit padding attrs.
template <typename ConcreteOp>
mlir::LogicalResult checkHaloOperands(ConcreteOp convOp, mlir::PatternRewriter& rewriter, Logger log) {
    if (getShape(convOp.getInput()).size() != 4 || getShape(convOp.getFilter()).size() != 4) {
        return matchFailed(log, rewriter, convOp, "Expected 4D input and filter");
    }

    const auto inElemType = mlir::cast<vpux::NDTypeInterface>(convOp.getInput().getType()).getElementType();
    if (inElemType.isBF16() || mlir::isa_and_present<mlir::quant::UniformQuantizedPerAxisType>(inElemType)) {
        return matchFailed(log, rewriter, convOp, "Element type {0} not supported", inElemType);
    }

    if (convOp.getOutputPaddingAttr() != nullptr || convOp.getInputPaddingAttr() != nullptr) {
        return matchFailed(log, rewriter, convOp, "output_padding / input_padding not supported");
    }
    return mlir::success();
}

// Slice numRows windows of windowLen elements, spaced step apart along the long axis, and
// concat them along H. layoutH windows are singleton-swapped from [N,C,windowLen,1] to
// [N,C,1,windowLen] first, so after the concat H always indexes rows and the kernel always
// slides along W.
mlir::Value buildRowWindowsConcat(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value paddedInput,
                                  const Conv1DLayout& layout, int64_t numRows, int64_t step, int64_t windowLen) {
    auto* ctx = rewriter.getContext();
    const auto paddedShape = getShape(paddedInput);
    const auto inN = paddedShape[Dims4D::Act::N];
    const auto inC = paddedShape[Dims4D::Act::C];

    const auto windowShapeAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{inN, inC, 1, windowLen});
    const auto windowMappingAttr = getIntArrayOfArray(ctx, getRowMergeMapping(/*intoW=*/true));

    SmallVector<mlir::Value> windows;
    windows.reserve(numRows);
    for (int64_t h = 0; h < numRows; ++h) {
        SmallVector<int64_t> offsets{0, 0, 0, 0};
        SmallVector<int64_t> sizes{inN, inC, 1, 1};
        offsets[layout.longDim.ind()] = h * step;
        sizes[layout.longDim.ind()] = windowLen;
        mlir::Value window = rewriter.create<IE::SliceOp>(appendLoc(loc, "halo_slice_{0}", h), paddedInput,
                                                          getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sizes))
                                     .getResult();
        if (!layout.longAxisIsW) {
            window = rewriter.createOrFold<IE::AffineReshapeOp>(appendLoc(loc, "halo_reshape_{0}", h), window,
                                                                windowMappingAttr, windowShapeAttr);
        }
        windows.push_back(window);
    }
    return rewriter.create<IE::ConcatOp>(appendLoc(loc, "halo_concat"), windows, Dims4D::Act::H.ind()).getOutput();
}

// layoutH only: singleton-swap the kernel from [OC,C,K,1] to [OC,C,1,K] so it matches the
// row layout produced by buildRowWindowsConcat.
mlir::Value reshapeFilterToRowLayout(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value filter) {
    auto* ctx = rewriter.getContext();
    const auto filterShape = getShape(filter);
    const SmallVector<int64_t> newShape{filterShape[Dims4D::Filter::OC], filterShape[Dims4D::Filter::IC], 1,
                                        filterShape[Dims4D::Filter::KY]};
    const SmallVector<SmallVector<int64_t>> dimMapping{{Dims4D::Filter::OC.ind()},
                                                       {Dims4D::Filter::IC.ind()},
                                                       {Dims4D::Filter::KY.ind(), Dims4D::Filter::KX.ind()},
                                                       {Dims4D::Filter::KX.ind()}};
    return rewriter.createOrFold<IE::AffineReshapeOp>(appendLoc(loc, "halo_filter_reshape"), filter,
                                                      getIntArrayOfArray(ctx, dimMapping),
                                                      getIntArrayAttr(ctx, newShape));
}

}  // namespace

template <typename ConcreteOp>
mlir::LogicalResult ReshapeConv1DInputWithHalo<ConcreteOp>::matchAndRewrite(ConcreteOp convOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    auto nestedLog = _log.nest();

    const auto inputShape = getShape(convOp.getInput());
    const auto filterShape = getShape(convOp.getFilter());
    const auto outputShape = getShape(convOp.getOutput());

    if (mlir::failed(checkHaloOperands(convOp, rewriter, nestedLog))) {
        return mlir::failure();
    }

    const auto layout = getConv1DLayout(inputShape, filterShape, outputShape);
    if (!layout.has_value()) {
        return matchFailed(nestedLog, rewriter, convOp,
                           "Not a 1D-shaped conv (need exactly one of H/W == 1 with K>1 along the other)");
    }
    const bool longAxisIsW = layout->longAxisIsW;
    const Dim longDim = layout->longDim;
    const int64_t kernelLong = layout->kernelLong;
    const int64_t inLen = layout->inLen;
    const int64_t outLen = layout->outLen;

    const auto strides = parseIntArrayAttr<int64_t>(convOp.getStrides());
    const auto dilations = parseIntArrayAttr<int64_t>(convOp.getDilations());
    if (strides.size() != 2 || strides[0] != 1 || strides[1] != 1) {
        return matchFailed(nestedLog, rewriter, convOp, "Require strides == [1, 1]");
    }
    if (dilations.size() != 2 || dilations[0] != 1 || dilations[1] != 1) {
        return matchFailed(nestedLog, rewriter, convOp, "Require dilations == [1, 1]");
    }

    const auto padsBegin = parseIntArrayAttr<int64_t>(convOp.getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(convOp.getPadsEnd());
    if (padsBegin.size() != 2 || padsEnd.size() != 2) {
        return matchFailed(nestedLog, rewriter, convOp, "Unexpected pad rank");
    }
    // IE pad attribute layout is [padsY, padsX].
    const int64_t padShortBegin = longAxisIsW ? padsBegin[0] : padsBegin[1];
    const int64_t padShortEnd = longAxisIsW ? padsEnd[0] : padsEnd[1];
    const int64_t padLongBegin = longAxisIsW ? padsBegin[1] : padsBegin[0];
    const int64_t padLongEnd = longAxisIsW ? padsEnd[1] : padsEnd[0];
    if (padShortBegin != 0 || padShortEnd != 0) {
        return matchFailed(nestedLog, rewriter, convOp, "Require pads on the singleton axis to be 0");
    }
    if (outLen != inLen + padLongBegin + padLongEnd - kernelLong + 1) {
        return matchFailed(nestedLog, rewriter, convOp, "Output length doesn't match explicit pad arithmetic");
    }

    const auto inElemType = mlir::cast<vpux::NDTypeInterface>(convOp.getInput().getType()).getElementType();
    const int64_t mpeAlignment = VPU::NCEInvariant::getAlignment(inElemType);
    const int64_t numDPU = config::getTotalNumOfEngines(convOp, config::ExecutorKind::DPU);
    const int64_t numRows = findRowCount(outLen, kernelLong, filterShape[Dims4D::Filter::IC],
                                         outputShape[Dims4D::Act::C], mpeAlignment, numDPU);
    if (numRows <= 1) {
        return matchFailed(nestedLog, rewriter, convOp, "No beneficial row count for out_len {0}, K {1}", outLen,
                           kernelLong);
    }
    const int64_t lRow = outLen / numRows;

    nestedLog.trace("Rewriting 1D conv {0} (long={1}) with K={2}, L_out={3}, numRows={4}, L_row={5}", convOp->getLoc(),
                    longAxisIsW ? "W" : "H", kernelLong, outLen, numRows, lRow);

    auto* ctx = convOp->getContext();
    const auto origLoc = convOp->getLoc();

    // Step 1: absorb the conv's pads into a Pad-via-Concat along the long axis.
    auto paddedInput =
            buildPadOp(rewriter, appendLoc(origLoc, "halo_pad"), convOp.getInput(), longDim, padLongBegin, padLongEnd);

    // Step 2: numRows overlapping length-(L_row+K-1) windows, concatenated into rows.
    auto rowsInput = buildRowWindowsConcat(rewriter, origLoc, paddedInput, *layout, numRows, /*step=*/lRow,
                                           /*windowLen=*/lRow + kernelLong - 1);
    mlir::Value newFilter =
            longAxisIsW ? convOp.getFilter() : reshapeFilterToRowLayout(rewriter, origLoc, convOp.getFilter());

    // Step 3: rebuild the conv with pads = 0; output is [N,OC,numRows,L_row].
    auto zeroPads = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    Shape newOutputShape{outputShape[Dims4D::Act::N], outputShape[Dims4D::Act::C], numRows, lRow};
    auto origOutType = mlir::cast<vpux::NDTypeInterface>(convOp.getOutput().getType());
    auto newOutType = origOutType.changeShape(newOutputShape);

    auto newConvOp = mlir::cast<ConcreteOp>(rewriter.clone(*convOp.getOperation()));
    rewriter.modifyOpInPlace(newConvOp, [&]() {
        newConvOp->setOperand(0, rowsInput);
        newConvOp->setOperand(1, newFilter);
        newConvOp->setAttr(convOp.getPadsBeginAttrName(), zeroPads);
        newConvOp->setAttr(convOp.getPadsEndAttrName(), zeroPads);
        newConvOp.getOutput().setType(mlir::cast<mlir::RankedTensorType>(newOutType));
        newConvOp->setLoc(appendLoc(origLoc, "halo_conv"));
    });

    // Step 4: collapse [N,OC,numRows,L_row] back to the original output shape.
    const auto outShapeAttr = getIntArrayAttr(ctx, outputShape.raw());
    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(
            convOp, newConvOp.getOutput(), getIntArrayOfArray(ctx, getRowMergeMapping(longAxisIsW)), outShapeAttr);

    _log.trace("Successfully rewrote 1D conv at {0} into 2D form with halo (numRows={1}, layout={2})", origLoc, numRows,
               longAxisIsW ? "W" : "H");
    return mlir::success();
}

//
// ReshapeTransposedConv1DInputWithHalo
//
// A forward conv only *reads* a halo, so its per-row outputs stay disjoint. A strided
// transposed conv *scatters* each input sample over a K-wide output window, so the roles
// flip: each row reads a wider input window and keeps only the output range it owns, which
// is still disjoint across rows (no overlap-add).
//
// Index derivation (long axis, K = q*S with S = stride, haloPad = q - 1, row length m;
// K % S == 0 is required so all row windows have the same size):
//   - Row h owns output [h*lRow, (h+1)*lRow), lRow = S*m, fed by the window
//     paddedInput[h*m : h*m + m + haloPad) of the haloPad-padded (both sides) input.
//   - That window's local output is lRow + 2*S*haloPad wide; row h keeps the middle
//     [S*haloPad, S*haloPad + lRow) slice.
//   - numRows must divide L_in + haloPad. L_out = (L_in - 1)*S + K rarely factors nicely
//     (L_in=40960 -> 4 * 40961, prime), so the input may be zero-extended first and the
//     resulting output tail sliced off.

namespace {

struct TransposedConvRowPlan {
    int64_t expandBy;
    int64_t numRows;
};

// Smallest zero-expansion of the input that lets the row plan divide evenly, preferring
// the largest row count for SOH headroom.
std::optional<TransposedConvRowPlan> findTransposedConvRowPlan(int64_t inLen, int64_t stride, int64_t kernel,
                                                               int64_t mpeAlignment, int64_t numDPU) {
    const int64_t haloPad = kernel / stride - 1;
    const int64_t minNumRows = mpeAlignment;
    const int64_t maxNumRows = std::max<int64_t>(numDPU, 1) * mpeAlignment;
    const int64_t minLRow = std::max<int64_t>(kernel, mpeAlignment);

    constexpr int64_t kMaxExpand = 4096;
    for (int64_t expandBy = 0; expandBy < kMaxExpand; ++expandBy) {
        const int64_t lTotal = inLen + expandBy + haloPad;
        for (int64_t numRows = maxNumRows; numRows >= minNumRows; --numRows) {
            const int64_t m = lTotal / numRows;
            if (lTotal % numRows == 0 && m * stride >= minLRow) {
                return TransposedConvRowPlan{expandBy, numRows};
            }
        }
    }
    return std::nullopt;
}

}  // namespace

class ReshapeTransposedConv1DInputWithHalo final : public mlir::OpRewritePattern<IE::TransposedConvolutionOp> {
public:
    ReshapeTransposedConv1DInputWithHalo(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::TransposedConvolutionOp>(ctx), _log(log) {
        setDebugName("ReshapeTransposedConv1DInputWithHalo");
    }

    mlir::LogicalResult matchAndRewrite(IE::TransposedConvolutionOp convOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReshapeTransposedConv1DInputWithHalo::matchAndRewrite(IE::TransposedConvolutionOp convOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    auto nestedLog = _log.nest();

    const auto inputShape = getShape(convOp.getInput());
    const auto filterShape = getShape(convOp.getFilter());
    const auto outputShape = getShape(convOp.getOutput());

    if (mlir::failed(checkHaloOperands(convOp, rewriter, nestedLog))) {
        return mlir::failure();
    }
    if (convOp.getOutputShape() != nullptr) {
        return matchFailed(nestedLog, rewriter, convOp, "Explicit output_shape operand is not supported");
    }

    const auto layout = getConv1DLayout(inputShape, filterShape, outputShape);
    if (!layout.has_value()) {
        return matchFailed(nestedLog, rewriter, convOp,
                           "Not a 1D-shaped transposed conv (need exactly one of H/W == 1 with K>1 along the other)");
    }
    const bool longAxisIsW = layout->longAxisIsW;
    const Dim longDim = layout->longDim;
    const int64_t kernelLong = layout->kernelLong;
    const int64_t inLen = layout->inLen;

    const auto strides = parseIntArrayAttr<int64_t>(convOp.getStrides());
    const auto dilations = parseIntArrayAttr<int64_t>(convOp.getDilations());
    if (strides.size() != 2 || dilations.size() != 2 || dilations[0] != 1 || dilations[1] != 1) {
        return matchFailed(nestedLog, rewriter, convOp, "Require rank-2 strides and dilations == [1, 1]");
    }
    const int64_t strideLong = longAxisIsW ? strides[Dims4D::Strides::X.ind()] : strides[Dims4D::Strides::Y.ind()];
    if (strideLong <= 0 || kernelLong % strideLong != 0) {
        return matchFailed(nestedLog, rewriter, convOp, "Require kernel to be a multiple of the long-axis stride");
    }

    const auto isZero = [](int64_t val) {
        return val == 0;
    };
    if (!llvm::all_of(parseIntArrayAttr<int64_t>(convOp.getPadsBegin()), isZero) ||
        !llvm::all_of(parseIntArrayAttr<int64_t>(convOp.getPadsEnd()), isZero) ||
        !llvm::all_of(parseIntArrayAttr<int64_t>(convOp.getSpatialOutputPadding()), isZero)) {
        return matchFailed(nestedLog, rewriter, convOp, "Non-zero padding is not supported");
    }

    const auto inElemType = mlir::cast<vpux::NDTypeInterface>(convOp.getInput().getType()).getElementType();
    const int64_t mpeAlignment = VPU::NCEInvariant::getAlignment(inElemType);
    const int64_t numDPU = config::getTotalNumOfEngines(convOp, config::ExecutorKind::DPU);
    const auto plan = findTransposedConvRowPlan(inLen, strideLong, kernelLong, mpeAlignment, numDPU);
    if (!plan.has_value()) {
        return matchFailed(nestedLog, rewriter, convOp, "No beneficial row count for in_len {0}, K {1}, stride {2}",
                           inLen, kernelLong, strideLong);
    }
    const int64_t expandBy = plan->expandBy;
    const int64_t numRows = plan->numRows;

    const int64_t haloPad = kernelLong / strideLong - 1;
    const int64_t inLenExp = inLen + expandBy;
    const int64_t lTotal = inLenExp + haloPad;
    const int64_t m = lTotal / numRows;
    const int64_t lRow = m * strideLong;
    const int64_t windowLongIn = m + haloPad;
    const int64_t localOutW = lRow + 2 * strideLong * haloPad;

    VPUX_THROW_UNLESS(numRows * lRow == (inLenExp - 1) * strideLong + kernelLong,
                      "Row-plan arithmetic mismatch for transposed conv halo rewrite");

    nestedLog.trace("Rewriting 1D transposed conv {0} (long={1}) with K={2}, S={3}, L_in={4}, expandBy={5}, "
                    "numRows={6}, L_row={7}",
                    convOp->getLoc(), longAxisIsW ? "W" : "H", kernelLong, strideLong, inLen, expandBy, numRows, lRow);

    auto* ctx = convOp->getContext();
    const auto origLoc = convOp->getLoc();

    // Step 1: one Pad-via-Concat providing both the symmetric read halo that keeps every
    // row window in-bounds and the `expandBy` zero tail (see the doc comment).
    auto paddedInput = buildPadOp(rewriter, appendLoc(origLoc, "halo_pad"), convOp.getInput(), longDim, haloPad,
                                  haloPad + expandBy);

    // Step 2: numRows overlapping length-windowLongIn windows, concatenated into rows.
    auto rowsInput = buildRowWindowsConcat(rewriter, origLoc, paddedInput, *layout, numRows, /*step=*/m,
                                           /*windowLen=*/windowLongIn);
    mlir::Value newFilter =
            longAxisIsW ? convOp.getFilter() : reshapeFilterToRowLayout(rewriter, origLoc, convOp.getFilter());

    // Step 3: rebuild the transposed conv with pads = 0. After the concat the batched-row
    // axis is always H (independent rows, stride 1) and the long axis is always W (real
    // stride), regardless of the original op's layout.
    auto newStrides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, strideLong});
    auto zeroPads = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});

    auto origOutType = mlir::cast<vpux::NDTypeInterface>(convOp.getOutput().getType());
    Shape localOutShape{outputShape[Dims4D::Act::N], outputShape[Dims4D::Act::C], numRows, localOutW};
    auto localOutType = origOutType.changeShape(localOutShape);

    auto newConvOp = mlir::cast<IE::TransposedConvolutionOp>(rewriter.clone(*convOp.getOperation()));
    rewriter.modifyOpInPlace(newConvOp, [&]() {
        newConvOp->setOperand(0, rowsInput);
        newConvOp->setOperand(1, newFilter);
        newConvOp->setAttr(convOp.getStridesAttrName(), newStrides);
        newConvOp->setAttr(convOp.getPadsBeginAttrName(), zeroPads);
        newConvOp->setAttr(convOp.getPadsEndAttrName(), zeroPads);
        newConvOp.getOutput().setType(mlir::cast<mlir::RankedTensorType>(localOutType));
        newConvOp->setLoc(appendLoc(origLoc, "halo_conv"));
    });

    // Step 4: keep the valid [S*haloPad, S*haloPad+L_row) sub-range of every row's local
    // output; the elements around it are partial sums owned by the neighbouring rows.
    SmallVector<int64_t> validOffsets{0, 0, 0, 0};
    SmallVector<int64_t> validSizes{outputShape[Dims4D::Act::N], outputShape[Dims4D::Act::C], numRows, lRow};
    validOffsets[Dims4D::Act::W.ind()] = strideLong * haloPad;
    auto validSliceOp =
            rewriter.create<IE::SliceOp>(appendLoc(origLoc, "halo_valid_slice"), newConvOp.getOutput(),
                                         getIntArrayAttr(ctx, validOffsets), getIntArrayAttr(ctx, validSizes));

    // Step 5: collapse [N,OC,numRows,L_row] into the flat (possibly expand-lengthened)
    // long-axis form.
    SmallVector<int64_t> flatOutShapeVec{outputShape[Dims4D::Act::N], outputShape[Dims4D::Act::C], 1, 1};
    flatOutShapeVec[longDim.ind()] = numRows * lRow;
    mlir::Value flatOut = rewriter.createOrFold<IE::AffineReshapeOp>(
            appendLoc(origLoc, "halo_flat"), validSliceOp.getResult(),
            getIntArrayOfArray(ctx, getRowMergeMapping(longAxisIsW)), getIntArrayAttr(ctx, flatOutShapeVec));

    // Step 6: drop the trailing S*expandBy elements produced by the zero tail of step 1 to
    // land exactly on the original output shape.
    mlir::Value finalOut = flatOut;
    if (expandBy > 0) {
        const SmallVector<int64_t> finalOffsets{0, 0, 0, 0};
        finalOut = rewriter.create<IE::SliceOp>(appendLoc(origLoc, "halo_final_slice"), flatOut,
                                                getIntArrayAttr(ctx, finalOffsets),
                                                getIntArrayAttr(ctx, outputShape.raw()))
                           .getResult();
    }

    rewriter.replaceOp(convOp, finalOut);

    nestedLog.trace("Successfully rewrote 1D transposed conv at {0} into 2D form with halo (numRows={1}, layout={2})",
                    origLoc, numRows, longAxisIsW ? "W" : "H");
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
    auto newContentAttrSetup = Const::ContentSetup(baseContent, baseContent.getType())
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
            if (vpux::isNotDimExpansionReshape(realDataShape, reshapeShape) &&
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

    const auto spatialAlignment = config::getPreferredSpatialAlignment(func);

    mlir::RewritePatternSet patterns(&ctx);

    // Adjust between H and W like:
    // [1, C, 1, W]  -> [1, C, H*4, W/4]
    // [1, C, H, 1]  -> [1, C, H/4, W*4]
    // if favorLargeHeight == true => newHeight = max(4, dim/4)
    patterns.add<ReshapeConvInput<IE::ConvolutionOp>>(&ctx, /*favorLargeHeight*/ true, spatialAlignment, _log);
    // Disabling reshape with large height for GroupConv
    // While it improves inference time for some models, it does bring perf regressions to several others due to being
    // able to assign SOH instead of SOK. To track: E147489
    patterns.add<ReshapeConvInput<IE::GroupConvolutionOp>>(&ctx, /*favorLargeHeight*/ false, spatialAlignment, _log);

    // Adjust from C to H and W like [1, C, 1, 1] -> [1, C/16, 4, 4]
    patterns.add<ReshapeSingleConstDWConvInput>(&ctx, _log);
    patterns.add<ReshapeAddInput>(&ctx, _log);

    // Adjust between C and H/W like [1, C, H, W] -> [1, C*4, H, W/4]
    // Also need stride[H] > 1
    patterns.add<ReshapeExpandDWConvInput>(&ctx, _log);

    // Convert true 1D conv (K>1, one spatial dim == 1) into 2D form with explicit row halo
    // so the DPU MPE grid is fully utilised. ReshapeConvInput handles the disjoint 1x1-kernel case,
    // so the two patterns never compete.
    patterns.add<ReshapeConv1DInputWithHalo<IE::ConvolutionOp>>(&ctx, _log);
    patterns.add<ReshapeConv1DInputWithHalo<IE::GroupConvolutionOp>>(&ctx, _log);
    // TransposedConvolution needs its own halo variant because a strided transposed conv
    // mirrors the read/write halo roles of a forward conv.
    patterns.add<ReshapeTransposedConv1DInputWithHalo>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}
}  // namespace

//
// createAdjustConvolutionInputShapePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustConvolutionInputShapePass(Logger log) {
    return std::make_unique<AdjustConvolutionInputShapePass>(log);
}
