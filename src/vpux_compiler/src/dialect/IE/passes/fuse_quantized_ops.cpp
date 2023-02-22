//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

bool areAllUsersQuantized(mlir::Operation* op) {
    for (auto user : op->getUsers()) {
        if (mlir::dyn_cast<IE::QuantizeOp>(user) == nullptr) {
            return false;
        }
    }
    return true;
}

//
// FuseWithConv
//

//
//       [input]
//          |
//     (dequantize)
//          |
//        (conv) --- (dequantize) -- [filter]
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithConv final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithConv(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithConv");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithConv::matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const {
    auto convOp = quantizeOp.input().getDefiningOp<IE::ConvolutionOp>();
    if (convOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(convOp)) {
        return mlir::failure();
    }

    if (VPUIP::NCEInvariant::verifyKernel(convOp, _log).failed()) {
        return mlir::failure();
    }

    auto inputDequantizeOp = convOp.input().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    auto filterDequantizeOp = convOp.filter().getDefiningOp<IE::DequantizeOp>();
    if (filterDequantizeOp == nullptr) {
        return mlir::failure();
    }

    // On VPUX37XX the prelu alpha multiplier used for integer input is unsigned, on floating input it is
    // signed. If input is floating, output is integer, quantize output need to be per tensor, this will check in
    // mix-precision pass
    const auto arch = VPU::getArch(quantizeOp->getParentOfType<mlir::ModuleOp>());
    if (arch == VPU::ArchKind::VPUX37XX) {
        if (convOp.post_opAttr() != nullptr &&
            convOp.post_opAttr().name().getValue() == IE::LeakyReluOp::getOperationName()) {
            IE::LeakyReluOp::Adaptor leakyRelu(None, convOp.post_opAttr().attrs());
            if (leakyRelu.verify(convOp->getLoc()).succeeded()) {
                const auto alpha = leakyRelu.negative_slope().convertToDouble();
                if (alpha < 0.0) {
                    return mlir::failure();
                }
            }
        }
    }

    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(quantizeOp, quantizeOp.getType(), inputDequantizeOp.input(),
                                                   filterDequantizeOp.input(), convOp.bias(), convOp.strides(),
                                                   convOp.pads_begin(), convOp.pads_end(), convOp.dilations(),
                                                   convOp.post_opAttr())
            ->setLoc(convOp->getLoc());

    return mlir::success();
}

//
// FuseWithGroupConv
//

//
//       [input]
//          |
//     (dequantize)
//          |
//        (conv) --- (dequantize) -- [filter]
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithGroupConv final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithGroupConv(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithGroupConv");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithGroupConv::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                       mlir::PatternRewriter& rewriter) const {
    auto grConvOp = quantizeOp.input().getDefiningOp<IE::GroupConvolutionOp>();
    if (grConvOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(grConvOp)) {
        return mlir::failure();
    }

    if (VPUIP::NCEInvariant::verifyKernel(grConvOp, _log).failed()) {
        return mlir::failure();
    }

    auto inputDequantizeOp = grConvOp.input().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    auto filterDequantizeOp = grConvOp.filter().getDefiningOp<IE::DequantizeOp>();
    if (filterDequantizeOp == nullptr) {
        return mlir::failure();
    }

    // On VPUX37XX the prelu alpha multiplier used for integer input is unsigned, on floating input it is
    // signed. If input is floating, output is integer, quantize output need to be per tensor, this will check in
    // mix-precision pass
    const auto arch = VPU::getArch(quantizeOp->getParentOfType<mlir::ModuleOp>());
    if (arch == VPU::ArchKind::VPUX37XX) {
        if (grConvOp.post_opAttr() != nullptr &&
            grConvOp.post_opAttr().name().getValue() == IE::LeakyReluOp::getOperationName()) {
            IE::LeakyReluOp::Adaptor leakyRelu(None, grConvOp.post_opAttr().attrs());
            if (leakyRelu.verify(grConvOp->getLoc()).succeeded()) {
                const auto alpha = leakyRelu.negative_slope().convertToDouble();
                if (alpha < 0.0) {
                    return mlir::failure();
                }
            }
        }
    }

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
                    quantizeOp, quantizeOp.getType(), inputDequantizeOp.input(), filterDequantizeOp.input(),
                    grConvOp.bias(), grConvOp.strides(), grConvOp.pads_begin(), grConvOp.pads_end(),
                    grConvOp.dilations(), grConvOp.groupsAttr(), grConvOp.post_opAttr())
            ->setLoc(grConvOp->getLoc());

    return mlir::success();
}

//
// FuseWithMaxPool
//

//
//       [input]
//          |
//     (dequantize)
//          |
//        (pool)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithMaxPool final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithMaxPool(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithMaxPool");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithMaxPool::matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const {
    auto maxPoolOp = quantizeOp.input().getDefiningOp<IE::MaxPoolOp>();
    if (maxPoolOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(maxPoolOp)) {
        return mlir::failure();
    }

    if (VPUIP::NCEInvariant::verifyKernel(maxPoolOp, _log).failed()) {
        return mlir::failure();
    }

    // MaxPool IDU does not support zero-point subtraction, so it compensates by ignoring output zero-point as well.
    // Since we are not subtracting the input zero-point, the non-linear post-op will operate on improper data.
    // Only zero-centered values would be supported. Currently, quantized MaxPool is disabled for all post-ops.
    auto mainOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(maxPoolOp.getOperation());
    if (mainOp != nullptr) {
        if (mainOp.getPostOp().hasValue()) {
            return mlir::failure();
        }
    }

    auto inputDequantizeOp = maxPoolOp.input().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(quantizeOp, quantizeOp.getType(), inputDequantizeOp.input(),
                                               maxPoolOp.kernel_size(), maxPoolOp.strides(), maxPoolOp.pads_begin(),
                                               maxPoolOp.pads_end(), maxPoolOp.rounding_type(), maxPoolOp.post_opAttr())
            ->setLoc(maxPoolOp->getLoc());

    return mlir::success();
}

//
// FuseWithAveragePool
//

//
//       [input]
//          |
//     (dequantize)
//          |
//        (pool)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithAveragePool final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithAveragePool(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithAveragePool");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithAveragePool::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto avgPoolOp = quantizeOp.input().getDefiningOp<IE::AvgPoolOp>();
    if (avgPoolOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(avgPoolOp)) {
        return mlir::failure();
    }

    if (VPUIP::NCEInvariant::verifyKernel(avgPoolOp, _log).failed()) {
        return mlir::failure();
    }

    // AveragePool IDU does not support zero-point subtraction, so it compensates by ignoring output zero-point as well.
    // Since we are not subtracting the input zero-point, the non-linear post-op will operate on improper data.
    // Only zero-centered values would be supported. Currently, quantized AveragePool is disabled for all post-ops.
    auto mainOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(avgPoolOp.getOperation());
    if (mainOp != nullptr) {
        if (mainOp.getPostOp().hasValue()) {
            return mlir::failure();
        }
    }

    auto inputDequantizeOp = avgPoolOp.input().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::AvgPoolOp>(quantizeOp, quantizeOp.getType(), inputDequantizeOp.input(),
                                               avgPoolOp.kernel_size(), avgPoolOp.strides(), avgPoolOp.pads_begin(),
                                               avgPoolOp.pads_end(), avgPoolOp.rounding_typeAttr(),
                                               avgPoolOp.exclude_padsAttr(), avgPoolOp.post_opAttr())
            ->setLoc(avgPoolOp->getLoc());

    return mlir::success();
}

//
// FuseWithSlice
//

//
//       [input]
//          |
//     (dequantize)
//          |
//       (slice)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithSlice final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithSlice");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithSlice::matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const {
    auto sliceOp = quantizeOp.input().getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(sliceOp)) {
        return mlir::failure();
    }

    auto inputDequantizeOp = sliceOp.source().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    auto origOutput = quantizeOp.output();
    auto origInput = inputDequantizeOp.input();
    auto sliceOpInputElementType = origInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    auto sliceOpOutputElementType = origOutput.getType().cast<vpux::NDTypeInterface>().getElementType();

    if (sliceOpInputElementType != sliceOpOutputElementType) {
        return matchFailed(_log, rewriter, sliceOp,
                           "Slice input ElementType:'{0}' and output ElementType:'{1}' mismatch.",
                           sliceOpInputElementType, sliceOpOutputElementType);
    }

    rewriter.replaceOpWithNewOp<IE::SliceOp>(quantizeOp, quantizeOp.getType(), inputDequantizeOp.input(),
                                             sliceOp.static_offsetsAttr(), sliceOp.static_sizesAttr())
            ->setLoc(sliceOp->getLoc());

    return mlir::success();
}

//
// FuseWithConcat
//

//
//       [input]
//          |
//     (dequantize)
//          |
//       (concat)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithConcat final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithConcat::matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const {
    auto concatOp = quantizeOp.input().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(concatOp)) {
        return mlir::failure();
    }

    SmallVector<mlir::Value> newConcatInputs;
    newConcatInputs.reserve(concatOp.inputs().size());

    auto dequantizeOp = concatOp.inputs().front().getDefiningOp<IE::DequantizeOp>();
    if (dequantizeOp == nullptr) {
        return mlir::failure();
    }

    for (auto in : concatOp.inputs()) {
        auto inputDequantizeOp = in.getDefiningOp<IE::DequantizeOp>();
        if (inputDequantizeOp == nullptr) {
            return mlir::failure();
        }

        if (!newConcatInputs.empty()) {
            const auto prevElemType = newConcatInputs.back().getType().cast<vpux::NDTypeInterface>().getElementType();
            const auto curElemType = inputDequantizeOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();

            if (const auto prevPerAxisType = prevElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
                if (const auto curPerAxisType = curElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
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

        newConcatInputs.push_back(inputDequantizeOp.input());
    }

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(quantizeOp, newConcatInputs, concatOp.per_axisAttr(),
                                              concatOp.static_offsetsAttr())
            ->setLoc(concatOp->getLoc());

    return mlir::success();
}

//
// FuseWithSplit
//

//
//       [input]
//          |
//     (dequantize)
//          |
//       (split)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithSplit final : public mlir::OpRewritePattern<IE::SplitOp> {
public:
    FuseWithSplit(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SplitOp>(ctx), _log(log) {
        setDebugName("FuseWithSplit");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SplitOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithSplit::matchAndRewrite(IE::SplitOp splitOp, mlir::PatternRewriter& rewriter) const {
    auto dequantizeOp = splitOp.input().getDefiningOp<IE::DequantizeOp>();
    if (dequantizeOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(splitOp)) {
        return mlir::failure();
    }

    SmallVector<mlir::Type> newSplitOutputsType;
    newSplitOutputsType.reserve(splitOp.outputs().size());

    SmallVector<IE::QuantizeOp> oldSplitOutputs;
    oldSplitOutputs.reserve(splitOp.outputs().size());

    for (auto outVal : splitOp.outputs()) {
        auto quantizeAfterSplitOp = outVal.hasOneUse() ? mlir::dyn_cast<IE::QuantizeOp>(*outVal.user_begin()) : nullptr;
        if (quantizeAfterSplitOp == nullptr) {
            return mlir::failure();
        }

        newSplitOutputsType.push_back(quantizeAfterSplitOp.getType());
        oldSplitOutputs.push_back(quantizeAfterSplitOp);
    }

    ArrayRef<mlir::Type> typesArray(newSplitOutputsType);

    auto newSplitOp = rewriter.create<IE::SplitOp>(splitOp.getLoc(), mlir::TypeRange(typesArray), dequantizeOp.input(),
                                                   splitOp.axis(), splitOp.num_splits(), splitOp.axis_valueAttr());

    for (auto ind : irange(oldSplitOutputs.size())) {
        auto oldResReorderOp = oldSplitOutputs[ind];
        auto newResVal = newSplitOp->getResult(checked_cast<uint32_t>(ind));
        rewriter.replaceOp(oldResReorderOp, newResVal);
    }

    return mlir::success();
}

//
// FuseWithEltwiseConverter
//

//
//      [input 1]     [input 2]
//          |             |
//     (dequantize)  (dequantize)
//          |             |
//           -(EltwiseOp)-
//                 |
//             [output]
//                 |
//            (quantize)
//

template <class ConcreteOp>
class FuseWithEltwiseConverter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithEltwiseConverter(mlir::MLIRContext* ctx,
                             FuncRef<mlir::LogicalResult(mlir::Type, mlir::Type)> checkInputTypes, Logger log)
            : mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _checkInputTypes(checkInputTypes), _log(log) {
        this->setDebugName("FuseWithEltwiseConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    FuncRef<mlir::LogicalResult(mlir::Type, mlir::Type)> _checkInputTypes;
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult FuseWithEltwiseConverter<ConcreteOp>::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    const auto quantOutType = quantizeOp.output().getType();
    auto quantElemOutType = quantOutType.cast<vpux::NDTypeInterface>().getElementType();
    if (quantElemOutType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        return mlir::failure();
    }

    auto eltwiseOp = quantizeOp.input().getDefiningOp<ConcreteOp>();
    if (eltwiseOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(eltwiseOp)) {
        return mlir::failure();
    }

    if (eltwiseOp.input1().getType().template cast<vpux::NDTypeInterface>().getShape() !=
        eltwiseOp.input2().getType().template cast<vpux::NDTypeInterface>().getShape()) {
        return mlir::failure();
    }

    const auto checkDequantizeOp = [](IE::DequantizeOp dequantOp) {
        if (dequantOp == nullptr) {
            return mlir::failure();
        }

        const auto dequantInType = dequantOp.input().getType();
        auto dequantElemInType = dequantInType.template cast<vpux::NDTypeInterface>().getElementType();
        if (!dequantElemInType.template isa<mlir::quant::UniformQuantizedType>()) {
            return mlir::failure();
        }

        return mlir::success();
    };

    auto input1DequantizeOp = eltwiseOp.input1().template getDefiningOp<IE::DequantizeOp>();
    if (mlir::failed(checkDequantizeOp(input1DequantizeOp))) {
        return mlir::failure();
    }

    auto input2DequantizeOp = eltwiseOp.input2().template getDefiningOp<IE::DequantizeOp>();
    if (mlir::failed(checkDequantizeOp(input2DequantizeOp))) {
        return mlir::failure();
    }

    const auto input1Type =
            input1DequantizeOp.input().getType().template cast<vpux::NDTypeInterface>().getElementType();
    const auto input2Type =
            input2DequantizeOp.input().getType().template cast<vpux::NDTypeInterface>().getElementType();
    if (mlir::failed(_checkInputTypes(input1Type, input2Type))) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<ConcreteOp>(quantizeOp, quantizeOp.getType(), input1DequantizeOp.input(),
                                            input2DequantizeOp.input(), eltwiseOp.auto_broadcastAttr(),
                                            eltwiseOp.post_opAttr())
            ->setLoc(eltwiseOp->getLoc());

    return mlir::success();
}

//
// FuseWithDepth2Space
//

//
//       [input]
//          |
//     (dequantize)
//          |
//        (DepthToSpace)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithDepth2Space final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithDepth2Space(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithDepth2Space");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithDepth2Space::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto depth2SpaceOp = quantizeOp.input().getDefiningOp<IE::DepthToSpaceOp>();
    if (depth2SpaceOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(depth2SpaceOp)) {
        return mlir::failure();
    }

    auto inputDequantizeOp = depth2SpaceOp.input().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::DepthToSpaceOp>(quantizeOp, quantizeOp.getType(), inputDequantizeOp.input(),
                                                    depth2SpaceOp.block_sizeAttr(), depth2SpaceOp.modeAttr())
            ->setLoc(depth2SpaceOp->getLoc());

    return mlir::success();
}

//
// FuseQuantizedOpsPass
//

class FuseQuantizedOpsPass final : public IE::FuseQuantizedOpsBase<FuseQuantizedOpsPass> {
public:
    explicit FuseQuantizedOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseQuantizedOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    const auto arch = VPU::getArch(func->getParentOfType<mlir::ModuleOp>());

    const auto checkAddInputTypes = [&](mlir::Type input1Type, mlir::Type input2Type) -> mlir::LogicalResult {
        auto dequantElemIn1Type = input1Type.cast<mlir::quant::UniformQuantizedType>();
        auto dequantElemIn2Type = input2Type.cast<mlir::quant::UniformQuantizedType>();

        // Perform check for input types. AddOp supports quantization with different zp, but not different scales.
        if (dequantElemIn1Type.getExpressedType() != dequantElemIn2Type.getExpressedType() ||
            dequantElemIn1Type.getStorageType() != dequantElemIn2Type.getStorageType() ||
            dequantElemIn1Type.isSigned() != dequantElemIn2Type.isSigned()) {
            return mlir::failure();
        }

        if (!supportsPerInputEltwiseScale(arch) && !isFloatEqual(static_cast<float>(dequantElemIn1Type.getScale()),
                                                                 static_cast<float>(dequantElemIn2Type.getScale()))) {
            return mlir::failure();
        }

        return mlir::success();
    };

    const auto checkMulInputTypes = [](mlir::Type input1Type, mlir::Type input2Type) -> mlir::LogicalResult {
        auto dequantElemIn1Type = input1Type.cast<mlir::quant::UniformQuantizedType>();
        auto dequantElemIn2Type = input2Type.cast<mlir::quant::UniformQuantizedType>();

        // Perform check for input types. MultiplyOp supports quantization with different scales and zp.
        if (dequantElemIn1Type.getExpressedType() != dequantElemIn2Type.getExpressedType() ||
            dequantElemIn1Type.getStorageType() != dequantElemIn2Type.getStorageType() ||
            dequantElemIn1Type.isSigned() != dequantElemIn2Type.isSigned()) {
            return mlir::failure();
        }

        return mlir::success();
    };

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseWithConv>(&ctx, _log);
    patterns.add<FuseWithGroupConv>(&ctx, _log);
    patterns.add<FuseWithEltwiseConverter<IE::AddOp>>(&ctx, checkAddInputTypes, _log);
    patterns.add<FuseWithSlice>(&ctx, _log);
    patterns.add<FuseWithMaxPool>(&ctx, _log);
    if (arch == VPU::ArchKind::VPUX37XX) {
        patterns.add<FuseWithAveragePool>(&ctx, _log);
    }
    patterns.add<FuseWithConcat>(&ctx, _log);
    patterns.add<FuseWithSplit>(&ctx, _log);
    // VPUX37XX does not support element-wise multiplication, skip the fusion
    if (arch == VPU::ArchKind::VPUX30XX || arch == VPU::ArchKind::VPUX311X) {
        patterns.add<FuseWithEltwiseConverter<IE::MultiplyOp>>(&ctx, checkMulInputTypes, _log);
    } else {
        patterns.add<FuseWithDepth2Space>(&ctx, _log);
    }

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFuseQuantizedOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseQuantizedOpsPass(Logger log) {
    return std::make_unique<FuseQuantizedOpsPass>(log);
}
