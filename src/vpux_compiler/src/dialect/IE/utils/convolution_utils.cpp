//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux {
namespace IE {

mlir::LogicalResult canConvertGroupConvToConv(IE::GroupConvolutionOp groupconv, bool isAttrCheckEnabled) {
    LogCb logCb = globalLogCb;
    if (!groupconv.getGroups().has_value()) {
        logCb(formatv("Grouped convolution does not have groups attribute"));
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(groupconv.getInput().getType());
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(groupconv.getFilter().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(groupconv.getOutput().getType());
    if (inputType.getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return mlir::failure();
    }
    if (outputType.getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return mlir::failure();
    }
    if (filterType.getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return mlir::failure();
    }

    const auto dilation = parseIntArrayAttr<int64_t>(groupconv.getDilations());
    if (dilation.size() != 2) {
        logCb(formatv("Expected dilations size to be 2, got '{0}'", dilation.size()));
        return mlir::failure();
    }
    if (dilation[0] != 1 || dilation[1] != 1) {
        logCb(formatv("Dilated convolution is not supported"));
        return mlir::failure();
    }

    const auto group = groupconv.getGroups().value();
    const auto filterShape = getShape(groupconv.getFilter());
    const auto inputShape = getShape(groupconv.getInput());
    // If DWConv cannot be converted to NCEDepthConvolution, convert it to Convolution. Here is not need to check layout
    // due to this pass is ahead of adjust layout and channel alignment pipeline.
    if (filterShape[Dims4D::Filter::OC] == group && inputShape[Dims4D::Act::C] == group &&
        (VPU::NCEDepthConvolutionOp::isSupported(groupconv, logCb, /*checkLayout=*/false,
                                                 /*checkChannelAlignment=*/false))) {
        logCb(formatv("Conversion is not needed for dw conv"));
        return mlir::failure();
    }

    // Channel alignment is not checked here because experiments show that NCE is still able to provide better
    // performance than SHAVE even if channel expand is done.

    // GroupConv with large kernels, padding, or strides may benefit from being converted to Convolution
    // to efficiently handle these parameters
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
        if (!VPU::isSupportedSEPTransposedConv(transposedConv, emptyLogCb, /*checkLayout=*/false,
                                               /*checkChannelAlignment=*/false)) {
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

    if (mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp>(op)) {
        if (op->getNumOperands() != 2) {
            return matchFailed(rewriter, biasOp, "ScaleShift producer already has fused biases");
        }

        auto* newConv = rewriter.clone(*op);

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

        newConv->insertOperands(newConv->getNumOperands(), biasConst);

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
