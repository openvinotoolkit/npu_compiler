//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux {
namespace IE {

//
// Helper functions for GroupConv conversion decision
//

namespace {

// Check if depthwise GroupConv would be supported by NCEDepthConvolution after HandleLargePadsPass
// reduces the padding to kernel/2
bool canBecomeNCEDepthConvAfterHandleLargePads(IE::GroupConvolutionOp groupconv, ShapeRef filterShape,
                                               ShapeRef inputShape, LogCb logCb) {
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto pads = PadInfo(groupconv.getPadsBegin(), groupconv.getPadsEnd());

    const auto weights = groupconv.getFilter();
    auto weightsCst = weights.getDefiningOp<Const::DeclareOp>();
    auto weightsFQ = weights.getDefiningOp<IE::FakeQuantizeOp>();
    if (weightsFQ) {
        weightsCst = weightsFQ.getInput().getDefiningOp<Const::DeclareOp>();
    }

    // For 1x1 kernel, kernel/2 = 0, so HandleLargePadsPass would reduce all padding to 0.
    // In this case, GroupConvToSingleConvConverter can handle it more efficiently.
    if (KY == 1 && KX == 1 && weightsCst) {
        return false;
    }

    // Check if padding exceeds the NCE limit (padding <= kernel/2)
    const bool hasLargePadding = pads.top > KY / 2 || pads.bottom > KY / 2 || pads.left > KX / 2 || pads.right > KX / 2;
    if (!hasLargePadding) {
        return false;
    }

    // For single-pixel input (1x1 spatial), DepthwiseConvSinglePixelInputToMultiplyConverter
    // provides a better optimization, so don't preserve for NCEDepthConv
    const auto inputH = inputShape[Dims4D::Act::H];
    const auto inputW = inputShape[Dims4D::Act::W];
    if (inputH == 1 && inputW == 1) {
        return false;
    }

    // Simulate reduced padding as HandleLargePadsPass would do
    const auto reducedPadTop = std::min(static_cast<int64_t>(pads.top), KY / 2);
    const auto reducedPadBottom = std::min(static_cast<int64_t>(pads.bottom), KY / 2);
    const auto reducedPadLeft = std::min(static_cast<int64_t>(pads.left), KX / 2);
    const auto reducedPadRight = std::min(static_cast<int64_t>(pads.right), KX / 2);

    const auto kernelStrides = parseIntArrayAttr<int64_t>(groupconv.getStrides());
    const auto kernelStridesShape = Shape(kernelStrides);
    const auto SY = kernelStridesShape[Dims4D::Strides::Y];
    const auto SX = kernelStridesShape[Dims4D::Strides::X];

    // Check if NCE constraints would be satisfied with reduced padding
    if (!VPU::NCEInvariant::isAttrsSupported(groupconv, KY, KX, SY, SX, reducedPadTop, reducedPadBottom, reducedPadLeft,
                                             reducedPadRight, logCb)) {
        return false;
    }

    // Check channel alignment constraints
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(groupconv.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(groupconv.getOutput().getType());
    const auto inputAlignment = VPU::NCEInvariant::getAlignment(inputType.getElementType());
    const auto outputAlignment = VPU::NCEInvariant::getAlignment(outputType.getElementType());

    return VPU::NCEInvariant::isInputActTypeSupported(inputType, inputAlignment, false) &&
           VPU::NCEInvariant::isOutputActTypeSupported(groupconv.getOperation(), outputType, outputAlignment);
}

}  // namespace

//
// canConvertGroupConvToConv
//
// Decision logic for whether a GroupConv should be converted to regular Convolution(s).
//
// Decision Tree:
// +-----------------------------------------------------------------------------------------+
// | GroupConv                                                                               |
// +-----------------------------------------------------------------------------------------+
// | 1. Non-depthwise GroupConv                                                              |
// |    -> Allow conversion (handled by GroupConvTo*Converter)                               |
// |                                                                                         |
// | 2. Depthwise GroupConv:                                                                 |
// |    2.1 Directly supported by NCEDepthConvolution                                        |
// |        -> Keep as GroupConv (will become NCEDepthConvolution)                           |
// |                                                                                         |
// |    2.2 Not directly supported, but could become supported after HandleLargePadsPass     |
// |        2.2.1 1x1 kernel (HandleLargePads ineffective, reduces all padding to 0)         |
// |              -> Allow conversion (GroupConvToSingleConvConverter handles it)            |
// |        2.2.2 Single-pixel input (1x1 spatial)                                           |
// |              -> Allow conversion (DepthwiseConvSinglePixelInputToMultiplyConverter)     |
// |        2.2.3 Otherwise                                                                  |
// |              -> Keep as GroupConv (HandleLargePadsPass + NCEDepthConv is better)        |
// |                                                                                         |
// |    2.3 Not supported even after HandleLargePadsPass                                     |
// |        -> Allow conversion                                                              |
// +-----------------------------------------------------------------------------------------+
//
// Returns: success() = allow conversion, failure() = keep as GroupConv
//

mlir::LogicalResult canConvertGroupConvToConv(IE::GroupConvolutionOp groupconv, bool isAttrCheckEnabled,
                                              bool checkHandleLargePads) {
    LogCb logCb = globalLogCb;

    // Basic validation
    if (!groupconv.getGroups().has_value()) {
        logCb(formatv("Grouped convolution does not have groups attribute"));
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(groupconv.getInput().getType());
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(groupconv.getFilter().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(groupconv.getOutput().getType());

    if (inputType.getRank() != 4 || outputType.getRank() != 4 || filterType.getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return mlir::failure();
    }

    const auto dilation = parseIntArrayAttr<int64_t>(groupconv.getDilations());
    if (dilation.size() != 2 || dilation[0] != 1 || dilation[1] != 1) {
        logCb(formatv("Dilated convolution is not supported"));
        return mlir::failure();
    }

    const auto group = groupconv.getGroups().value();
    const auto filterShape = getShape(groupconv.getFilter());
    const auto inputShape = getShape(groupconv.getInput());

    // Check if this is a depthwise convolution
    const bool isDepthwise = filterShape[Dims4D::Filter::OC] == group && inputShape[Dims4D::Act::C] == group;

    if (isDepthwise) {
        // Case 2.1: Directly supported by NCEDepthConvolution
        if (VPU::NCEDepthConvolutionOp::isSupported(groupconv, logCb, /*checkLayout=*/false,
                                                    /*checkChannelAlignment=*/false)) {
            logCb(formatv("Depthwise GroupConv is directly supported by NCEDepthConvolution"));
            return mlir::failure();
        }

        // Case 2.2/2.3: Check if it could become supported after HandleLarge*Pass
        if (checkHandleLargePads &&
            canBecomeNCEDepthConvAfterHandleLargePads(groupconv, filterShape, inputShape, logCb)) {
            logCb(formatv("Depthwise GroupConv will be supported after HandleLarge*Pass"));
            return mlir::failure();
        }
    }

    // Additional attribute check for specific use cases
    if (isAttrCheckEnabled) {
        const auto KY = filterShape[Dims4D::Filter::KY];
        const auto KX = filterShape[Dims4D::Filter::KX];
        const auto kernelStrides = parseIntArrayAttr<int64_t>(groupconv.getStrides());
        const auto kernelStridesShape = Shape(kernelStrides);
        const auto SY = kernelStridesShape[Dims4D::Strides::Y];
        const auto SX = kernelStridesShape[Dims4D::Strides::X];
        const auto pads = PadInfo(groupconv.getPadsBegin(), groupconv.getPadsEnd());

        if (!VPU::NCEInvariant::isAttrsSupported(groupconv, KY, KX, SY, SX, pads.top, pads.bottom, pads.left,
                                                 pads.right, logCb)) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

bool isEltwiseGroupConv(IE::GroupConvolutionOp convOp, bool isConstFilter) {
    if (convOp == nullptr) {
        return false;
    }
    // check kernel size is 1x1
    auto filterShape = getShape(convOp.getFilter());
    if (filterShape[Dims4D::Filter::KX] != 1 || filterShape[Dims4D::Filter::KY] != 1 ||
        filterShape[Dims4D::Filter::OC] != convOp.getGroups().value()) {
        return false;
    }
    // if there is stride > 1, it can not consider to be an eltwise op
    const auto greaterThanOne = [](auto stride) {
        return stride > 1;
    };
    const auto stridesGreaterThanOne = llvm::any_of(parseIntArrayAttr<int64_t>(convOp.getStrides()), greaterThanOne);
    if (stridesGreaterThanOne) {
        return false;
    }

    const auto dilationsGreaterThanOne =
            llvm::any_of(parseIntArrayAttr<int64_t>(convOp.getDilations()), greaterThanOne);
    if (dilationsGreaterThanOne) {
        return false;
    }

    const auto padStart = parseIntArrayAttr<int64_t>(convOp.getPadsBegin());
    const auto padEnd = parseIntArrayAttr<int64_t>(convOp.getPadsEnd());
    const auto nonZeroPadStart = llvm::any_of(padStart, [](auto pad) {
        return pad > 0;
    });
    const auto nonZeroPadEnd = llvm::any_of(padEnd, [](auto pad) {
        return pad > 0;
    });
    if (nonZeroPadStart || nonZeroPadEnd) {
        return false;
    }

    if (isConstFilter) {
        // check input const is single data or not
        mlir::SmallVector<Const::DeclareOp> constInputOps;
        constInputOps.push_back(convOp.getFilter().getDefiningOp<Const::DeclareOp>());
        if (convOp.getBias()) {
            constInputOps.push_back(convOp.getBias().getDefiningOp<Const::DeclareOp>());
        }
        return llvm::all_of(constInputOps, [](Const::DeclareOp constOp) {
            return IE::isBaseContentSplat(constOp);
        });
    }

    auto isSizeOneOrBroadCastFromOne = [&](mlir::Value input) {
        if (getShape(input).totalSize() == 1) {
            return true;
        }

        auto parent = input.getDefiningOp();
        while (parent) {
            if (mlir::isa<Const::DeclareOp>(parent)) {
                return getShape(parent->getResult(0)).totalSize() == 1;
            }
            if (IE::isPureViewOp(parent) || mlir::isa<IE::ReorderOp>(parent)) {
                parent = parent->getOperand(0).getDefiningOp();
                continue;
            }

            if (mlir::isa<IE::BroadcastOp, IE::TileOp>(parent)) {
                return getShape(parent->getOperand(0)).totalSize() == 1;
            }
            return false;
        }

        return false;
    };

    auto bias = convOp.getBias();
    if (bias && !isSizeOneOrBroadCastFromOne(bias)) {
        return false;
    }
    return isSizeOneOrBroadCastFromOne(convOp.getFilter());
}

//
// FuseConvAndBias
//

mlir::LogicalResult FuseConvAndBias::matchAndRewrite(IE::ScaleShiftOp biasOp, mlir::PatternRewriter& rewriter) const {
    if (biasOp.getWeights() != nullptr) {
        return matchFailed(rewriter, biasOp, "ScaleShift has scales operand");
    }
    if (!biasOp.getInput().hasOneUse()) {
        return matchFailed(rewriter, biasOp, "ScaleShift is not the only user of its operand");
    }
    if (biasOp.getBiases() == nullptr || mlir::failed(IE::getConstParentOp(biasOp.getBiases()))) {
        return matchFailed(rewriter, biasOp, "ScaleShift has non constant biases");
    }

    auto* op = biasOp.getInput().getDefiningOp();
    constexpr auto maxRepeatFq = 2;
    for (auto _ : irange(maxRepeatFq)) {
        std::ignore = _;
        if (!mlir::isa_and_nonnull<IE::FakeQuantizeOp>(op)) {
            break;
        }
        if (!op->getOperand(0).hasOneUse()) {
            return matchFailed(rewriter, biasOp, "FakeQuantize is not the only user of its operand");
        }
        op = op->getOperand(0).getDefiningOp();
    }
    if (op == nullptr || !mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::TransposedConvolutionOp>(op)) {
        return matchFailed(rewriter, biasOp, "ScaleShift producer is not a Convolution layer");
    }

    // For those Convolutions/GroupConvolutions/TransposedConvolutions cannot convert to NCE task should not fuse
    // ScaleShift as Bias. Because SW kernel will not support any Post Ops.
    if (auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(op)) {
        if (VPU::NCEConvolutionOp::verifyKernel(convOp).failed()) {
            return matchFailed(rewriter, convOp, "Conv cannot convert to NCE, not fuse ScaleShift");
        }
    }
    if (auto grConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(op)) {
        if (VPU::NCEDepthConvolutionOp::verifyKernel(grConvOp).failed() &&
            mlir::failed(IE::canConvertGroupConvToConv(grConvOp))) {
            return matchFailed(rewriter, grConvOp, "GroupConv cannot convert to NCE, not fuse ScaleShift");
        }
    }
    if (auto transposedConv = mlir::dyn_cast<IE::TransposedConvolutionOp>(op)) {
        auto seOp = mlir::dyn_cast<IE::SEOpInterface>(transposedConv.getOperation());
        if (!seOp || !seOp.isSupported(emptyLogCb)) {
            return matchFailed(rewriter, transposedConv, "TransposedConv cannot convert to NCE, not fuse ScaleShift");
        }
    }

    const auto convOutShape = getShape(op->getOpResult(0));
    const auto biasShape = getShape(biasOp.getBiases());

    if (biasShape.size() != 4) {
        return matchFailed(rewriter, biasOp, "ScaleShift 'shift' operand shape doesn't match bias restrictions");
    }
    if (biasShape[Dims4D::Act::N] != 1) {
        return matchFailed(rewriter, biasOp, "ScaleShift 'shift' operand shape doesn't match bias restrictions");
    }
    if (biasShape[Dims4D::Act::C] != convOutShape[Dims4D::Act::C]) {
        return matchFailed(rewriter, biasOp, "ScaleShift 'shift' operand shape doesn't match bias restrictions");
    }
    if (biasShape[Dims4D::Act::H] != 1) {
        return matchFailed(rewriter, biasOp, "ScaleShift 'shift' operand shape doesn't match bias restrictions");
    }
    if (biasShape[Dims4D::Act::W] != 1) {
        return matchFailed(rewriter, biasOp, "ScaleShift 'shift' operand shape doesn't match bias restrictions");
    }

    // HW applied bias before scale, so need to do following transformation to get correct result
    //   conv * scale + bias ==> (conv + bias/scale) * scale
    auto biasConst = [&]() -> mlir::Value {
        auto convolutionOp = mlir::dyn_cast<IE::ConvolutionOp>(op);
        if (convolutionOp == nullptr || convolutionOp.getStaticScaleAttr() == nullptr) {
            return biasOp.getBiases();
        }

        auto staticScale = convolutionOp.getStaticScaleAttr().getValueAsDouble();
        if (isDoubleEqual(staticScale, 1)) {
            return biasOp.getBiases();
        }

        auto biasConst = IE::getConstParentOp(biasOp.getBiases()).value();
        auto contentAttr = biasConst.transformContentAttr().rescale(1 / staticScale).get();
        return rewriter.create<Const::DeclareOp>(takeOpLoc(biasConst, "rescaled"), biasConst.getType(), contentAttr)
                .getOutput();
    }();

    if (mlir::isa<IE::GroupConvolutionOp>(op)) {
        if (op->getNumOperands() != 2) {
            return matchFailed(rewriter, biasOp, "ScaleShift producer already has fused biases");
        }

        auto* newConv = rewriter.clone(*op);
        newConv->insertOperands(newConv->getNumOperands(), biasConst);
        rewriter.replaceOp(biasOp, newConv->getOpResults());
    } else if (auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(op)) {
        if (convOp.getBias() != nullptr) {
            return matchFailed(rewriter, biasOp, "ScaleShift producer already has fused biases");
        }
        auto newConv = cloneConvolutionOp(rewriter, convOp, convOp.getInput(), convOp.getFilter(), biasConst,
                                          convOp.getScale(), convOp.getZeroPoints());
        rewriter.replaceOp(biasOp, newConv->getOpResults());
    } else if (auto transposedConv = mlir::dyn_cast<IE::TransposedConvolutionOp>(op)) {
        if (transposedConv.getBias() != nullptr) {
            return matchFailed(rewriter, biasOp, "ScaleShift producer already has fused biases");
        }
        auto newTransposedConv = rewriter.create<IE::TransposedConvolutionOp>(
                transposedConv.getLoc(), transposedConv.getInput(), transposedConv.getFilter(),
                transposedConv.getOutputShape(), biasOp.getBiases(), transposedConv.getStrides(),
                transposedConv.getPadsBegin(), transposedConv.getPadsEnd(), transposedConv.getDilations(),
                transposedConv.getSpatialOutputPaddingAttr(), transposedConv.getPostOpAttr(),
                transposedConv.getClampAttr(), transposedConv.getOutputPaddingAttr(),
                transposedConv.getInputPaddingAttr());

        rewriter.replaceOp(biasOp, newTransposedConv->getOpResults());
    } else {
        return matchFailed(rewriter, op, "Unexpected operation");
    }

    return mlir::success();
}

}  // namespace IE
}  // namespace vpux
