//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/fuse_quantized_ops.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"

using namespace vpux;
using namespace IE;

bool FuseWithConv::isSupportedConvBasedOp(IE::ConvolutionOp conv, Logger log) const {
    return VPU::NCEConvolutionOp::verifyKernel(conv, log).succeeded();
}

IE::ConvolutionOp FuseWithConv::createNewConvBasedOp(IE::QuantizeOp quantizeOp, IE::ConvolutionOp conv,
                                                     mlir::Value newInput, mlir::Value newWeights,
                                                     mlir::PatternRewriter& rewriter) const {
    auto users = conv.getResult().getUsers();
    auto userSize = std::distance(users.begin(), users.end());
    auto newLoc = takeOpLoc(conv, "{0}", userSize);
    auto newConv = cloneConvolutionOp(rewriter, conv, quantizeOp.getType(), newInput, newWeights, newLoc);
    return newConv;
}

bool FuseWithGroupConv::isSupportedConvBasedOp(IE::GroupConvolutionOp grConvOp, Logger log) const {
    return VPU::NCEDepthConvolutionOp::verifyKernel(grConvOp, log).succeeded();
}

IE::GroupConvolutionOp FuseWithGroupConv::createNewConvBasedOp(IE::QuantizeOp quantizeOp,
                                                               IE::GroupConvolutionOp grConvOp, mlir::Value newInput,
                                                               mlir::Value newWeights,
                                                               mlir::PatternRewriter& rewriter) const {
    auto users = grConvOp.getResult().getUsers();
    auto userSize = std::distance(users.begin(), users.end());
    auto newLoc = takeOpLoc(grConvOp, "{0}", userSize);

    auto newGroupConv = rewriter.create<IE::GroupConvolutionOp>(
            newLoc, quantizeOp.getType(), newInput, newWeights, grConvOp.getBias(), grConvOp.getStrides(),
            grConvOp.getPadsBegin(), grConvOp.getPadsEnd(), grConvOp.getDilations(), grConvOp.getGroupsAttr(),
            grConvOp.getPostOpAttr(), grConvOp.getClampAttr(), grConvOp.getOutputPaddingAttr(),
            grConvOp.getInputPaddingAttr());

    return newGroupConv;
}

bool FuseWithTransposedConv::isSupportedConvBasedOp(IE::TransposedConvolutionOp transposedConvOp, Logger log) const {
    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };
    auto seOp = mlir::dyn_cast<IE::SEOpInterface>(transposedConvOp.getOperation());
    return seOp && seOp.isSupported(logCb);
}

IE::TransposedConvolutionOp FuseWithTransposedConv::createNewConvBasedOp(IE::QuantizeOp quantizeOp,
                                                                         IE::TransposedConvolutionOp transposedConvOp,
                                                                         mlir::Value newInput, mlir::Value newWeights,
                                                                         mlir::PatternRewriter& rewriter) const {
    auto users = transposedConvOp.getResult().getUsers();
    auto userSize = std::distance(users.begin(), users.end());
    auto newLoc = takeOpLoc(transposedConvOp, "{0}", userSize);

    auto newTransposedConv = rewriter.create<IE::TransposedConvolutionOp>(
            newLoc, quantizeOp.getType(), newInput, newWeights, transposedConvOp.getOutputShape(),
            transposedConvOp.getBias(), transposedConvOp.getStrides(), transposedConvOp.getPadsBegin(),
            transposedConvOp.getPadsEnd(), transposedConvOp.getDilations(),
            transposedConvOp.getSpatialOutputPaddingAttr(), transposedConvOp.getPostOpAttr(),
            transposedConvOp.getClampAttr(), transposedConvOp.getOutputPaddingAttr(),
            transposedConvOp.getInputPaddingAttr());

    return newTransposedConv;
}

mlir::LogicalResult FuseWithMaxPool::matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const {
    auto maxPoolOp = quantizeOp.getInput().getDefiningOp<IE::MaxPoolOp>();
    if (maxPoolOp == nullptr) {
        return mlir::failure();
    }

    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(maxPoolOp.getOperation());
    if (!quantizedLayerOp || !quantizedLayerOp.isInOutQuantizationCompatible(quantizeOp.getOperation())) {
        return mlir::failure();
    }

    if (VPU::NCEMaxPoolOp::verifyKernel(maxPoolOp, _log).failed()) {
        return mlir::failure();
    }

    if (!quantizedLayerOp.isOutputQuantizationFusable(IE::isPerAxisQuant(quantizeOp.getOutput()),
                                                      /*isFloatInput=*/false)) {
        return mlir::failure();
    }

    if (!quantizedLayerOp.isInputQuantizationFusable()) {
        return mlir::failure();
    }
    auto inputDequantizeOp = maxPoolOp.getInput().getDefiningOp<IE::DequantizeOp>();

    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(
                    quantizeOp, quantizeOp.getType(), inputDequantizeOp.getInput(), maxPoolOp.getScale(),
                    maxPoolOp.getKernelSize(), maxPoolOp.getStrides(), maxPoolOp.getPadsBegin(), maxPoolOp.getPadsEnd(),
                    maxPoolOp.getRoundingType(), maxPoolOp.getPostOpAttr(), maxPoolOp.getClampAttr(),
                    maxPoolOp.getStaticScaleAttr(), maxPoolOp.getOutputPaddingAttr(), maxPoolOp.getInputPaddingAttr())
            ->setLoc(maxPoolOp->getLoc());

    return mlir::success();
}

mlir::LogicalResult FuseWithAveragePool::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto avgPoolOp = quantizeOp.getInput().getDefiningOp<IE::AvgPoolOp>();
    if (avgPoolOp == nullptr) {
        return mlir::failure();
    }

    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(avgPoolOp.getOperation());
    if (!quantizedLayerOp || !quantizedLayerOp.isInOutQuantizationCompatible(quantizeOp.getOperation())) {
        return mlir::failure();
    }

    if (VPU::NCEAveragePoolOp::verifyKernel(avgPoolOp, _log).failed()) {
        return mlir::failure();
    }

    if (!quantizedLayerOp.isOutputQuantizationFusable(IE::isPerAxisQuant(quantizeOp.getOutput()),
                                                      /*isFloatInput=*/false)) {
        return mlir::failure();
    }

    if (!quantizedLayerOp.isInputQuantizationFusable()) {
        return mlir::failure();
    }
    auto inputDequantizeOp = avgPoolOp.getInput().getDefiningOp<IE::DequantizeOp>();

    auto users = avgPoolOp.getResult().getUsers();
    auto userSize = std::distance(users.begin(), users.end());
    auto newLoc = takeOpLoc(avgPoolOp, "{0}", userSize);
    rewriter.replaceOpWithNewOp<IE::AvgPoolOp>(
                    quantizeOp, quantizeOp.getType(), inputDequantizeOp.getInput(), avgPoolOp.getScale(),
                    avgPoolOp.getKernelSize(), avgPoolOp.getStrides(), avgPoolOp.getPadsBegin(), avgPoolOp.getPadsEnd(),
                    avgPoolOp.getRoundingTypeAttr(), avgPoolOp.getExcludePadsAttr(), avgPoolOp.getPostOpAttr(),
                    avgPoolOp.getClampAttr(), avgPoolOp.getStaticScaleAttr(), avgPoolOp.getOutputPaddingAttr(),
                    avgPoolOp.getInputPaddingAttr())
            ->setLoc(newLoc);

    return mlir::success();
}

bool isLegalFuseOp(mlir::Operation* concreteOp, IE::QuantizeOp quantizeOp) {
    if (!areAllUsersQuantized(concreteOp)) {
        return false;
    }

    auto inputDequantizeOp = concreteOp->getOperand(0).getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return false;
    }

    auto origOutput = quantizeOp.getOutput();
    auto origInput = inputDequantizeOp.getInput();
    auto tileOpInputElementType = mlir::cast<vpux::NDTypeInterface>(origInput.getType()).getElementType();
    auto tileOpOutputElementType = mlir::cast<vpux::NDTypeInterface>(origOutput.getType()).getElementType();

    return tileOpInputElementType == tileOpOutputElementType;
}

mlir::LogicalResult FuseWithSlice::matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const {
    if (isPerAxisQuant(quantizeOp.getOutput())) {
        return mlir::failure();
    }

    auto sliceOp = quantizeOp.getInput().getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        return mlir::failure();
    }

    if (!isQuantizationSupported(quantizeOp, sliceOp, IE::TypeComparisonMode::STRICT_EQUAL)) {
        return mlir::failure();
    }

    if (!isLegalFuseOp(sliceOp, quantizeOp)) {
        return matchFailed(rewriter, sliceOp, "Quantize op cannot fuse into op {0} at {1}", sliceOp->getName(),
                           sliceOp->getLoc());
    }

    auto inputDequantizeOp = sliceOp.getSource().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    if (isPerAxisQuant(inputDequantizeOp.getInput())) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::SliceOp>(quantizeOp, quantizeOp.getType(),
                                             sliceOp.getSource().getDefiningOp<IE::DequantizeOp>().getInput(),
                                             sliceOp.getStaticOffsetsAttr(), sliceOp.getStaticSizesAttr())
            ->setLoc(sliceOp->getLoc());

    return mlir::success();
}

mlir::LogicalResult FuseWithTile::matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const {
    if (isPerAxisQuant(quantizeOp.getOutput())) {
        return mlir::failure();
    }

    auto tileOp = quantizeOp.getInput().getDefiningOp<IE::TileOp>();
    if (tileOp == nullptr) {
        return mlir::failure();
    }

    if (!isQuantizationSupported(quantizeOp, tileOp, IE::TypeComparisonMode::STRICT_EQUAL)) {
        return mlir::failure();
    }

    if (!isLegalFuseOp(tileOp, quantizeOp)) {
        return matchFailed(rewriter, tileOp, "Quantize op cannot fuse into op {0} at {1}", tileOp->getName(),
                           tileOp->getLoc());
    }

    auto inputDequantizeOp = tileOp.getInput().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    if (isPerAxisQuant(inputDequantizeOp.getInput())) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::TileOp>(quantizeOp, quantizeOp.getType(),
                                            tileOp.getInput().getDefiningOp<IE::DequantizeOp>().getInput(),
                                            tileOp.getRepeatsValuesAttr());

    return mlir::success();
}

mlir::LogicalResult FuseWithConcat::matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const {
    if (isPerAxisQuant(quantizeOp.getOutput())) {
        return mlir::failure();
    }

    auto concatOp = quantizeOp.getInput().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(concatOp)) {
        return mlir::failure();
    }

    if (!isQuantizationSupported(quantizeOp, concatOp, IE::TypeComparisonMode::STRICT_EQUAL)) {
        return mlir::failure();
    }

    SmallVector<mlir::Value> newConcatInputs;
    newConcatInputs.reserve(concatOp.getInputs().size());

    auto dequantizeOp = concatOp.getInputs().front().getDefiningOp<IE::DequantizeOp>();
    if (dequantizeOp == nullptr) {
        return mlir::failure();
    }

    for (auto in : concatOp.getInputs()) {
        auto inputDequantizeOp = in.getDefiningOp<IE::DequantizeOp>();
        if (inputDequantizeOp == nullptr) {
            return mlir::failure();
        }

        if (isPerAxisQuant(inputDequantizeOp.getInput())) {
            return mlir::failure();
        }

        if (!newConcatInputs.empty()) {
            const auto prevElemType =
                    mlir::cast<vpux::NDTypeInterface>(newConcatInputs.back().getType()).getElementType();
            const auto curElemType =
                    mlir::cast<vpux::NDTypeInterface>(inputDequantizeOp.getInput().getType()).getElementType();

            if (const auto prevPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(prevElemType)) {
                if (const auto curPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(curElemType)) {
                    if (!canBeMerged(prevPerAxisType, curPerAxisType)) {
                        return mlir::failure();
                    }
                } else {
                    return mlir::failure();
                }
            } else if (prevElemType != curElemType) {
                return mlir::failure();
            }
        }

        newConcatInputs.push_back(inputDequantizeOp.getInput());
    }

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(quantizeOp, newConcatInputs, concatOp.getPerAxisAttr(),
                                              concatOp.getStaticOffsetsAttr())
            ->setLoc(concatOp->getLoc());

    return mlir::success();
}

mlir::LogicalResult FuseWithInterpolate::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto interpOp = quantizeOp.getInput().getDefiningOp<IE::InterpolateOp>();
    if (interpOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(interpOp)) {
        return mlir::failure();
    }

    if (!isQuantizationSupported(quantizeOp, interpOp, IE::TypeComparisonMode::ALLOW_DIFFERENT_QUANT)) {
        return mlir::failure();
    }

    auto isNCESupported = VPU::NCEInvariant::isSupported(interpOp.getOperation(), _log);
    if (isNCESupported.failed()) {
        return mlir::failure();
    }

    auto inputDequantizeOp = interpOp.getInput().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    if (isPerAxisQuant(inputDequantizeOp.getInput())) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::InterpolateOp>(
                    quantizeOp, quantizeOp.getType(), inputDequantizeOp.getInput(), nullptr, nullptr, nullptr,
                    interpOp.getSizesAttr().value_or(nullptr), interpOp.getScalesAttr().value_or(nullptr),
                    interpOp.getAxesAttr().value_or(nullptr), interpOp.getTileOffsetAttrAttr(),
                    interpOp.getInitialInputDimsAttrAttr(), interpOp.getInitialOutputDimsAttrAttr(), interpOp.getAttr(),
                    interpOp.getOutputPaddingAttr(), interpOp.getInputPaddingAttr())
            ->setLoc(interpOp->getLoc());

    return mlir::success();
}

mlir::LogicalResult FuseWithMatMul::matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const {
    if (isPerAxisQuant(quantizeOp.getOutput())) {
        return mlir::failure();
    }
    auto matMulOp = quantizeOp.getInput().getDefiningOp<IE::MatMulOp>();
    if (matMulOp == nullptr) {
        return mlir::failure();
    }

    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(matMulOp.getOperation());
    if (!quantizedLayerOp || !quantizedLayerOp.isInOutQuantizationCompatible(quantizeOp.getOperation())) {
        return mlir::failure();
    }

    if (!quantizedLayerOp.isOutputQuantizationFusable(IE::isPerAxisQuant(quantizeOp.getOutput()),
                                                      /*isFloatInput=*/false)) {
        return mlir::failure();
    }

    if (!quantizedLayerOp.isInputQuantizationFusable()) {
        return mlir::failure();
    }
    auto input1DequantizeOp = matMulOp.getInput1().getDefiningOp<IE::DequantizeOp>();
    auto input2DequantizeOp = matMulOp.getInput2().getDefiningOp<IE::DequantizeOp>();

    rewriter.replaceOp(quantizeOp, cloneMatMulOp(rewriter, matMulOp, quantizeOp.getType(),
                                                 input1DequantizeOp.getInput(), input2DequantizeOp.getInput()));

    return mlir::success();
}
