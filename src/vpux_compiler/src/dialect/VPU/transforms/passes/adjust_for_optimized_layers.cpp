//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_matmul_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_ADJUSTFOROPTIMIZEDLAYERS
#define GEN_PASS_DEF_ADJUSTFOROPTIMIZEDLAYERS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// Common Utils
//

Dim getLowestDim(ShapeRef shape, const DimsOrder& order) {
    const auto rank = order.numDims();
    auto lowestDim = order.dimAt(rank - 1);
    for (auto idx : irange(rank)) {
        auto dim = order.dimAt(idx);
        if (shape[dim] > 1) {
            lowestDim = dim;
        }
    }
    return lowestDim;
}

int64_t getTotalSizeBeforeDim(ShapeRef shape, const DimsOrder& order, const Dim& dim) {
    int64_t totalSize = 1;
    for (auto idx : irange(order.dimPos(dim))) {
        totalSize *= shape[order.dimAt(idx)];
    }
    return totalSize;
}

// Adjusts shape of Softmax to leverage the optimized softmax kernel implementation
// for axis 0 (the last dim in compiler scope)
// Examples:
//   - Softmax(shape=[1, 16, 24, 1], axisInd=2, layout=NCHW) is adjusted to
//     Softmax(shape=[1, 16, 1, 24], axisInd=3, layout=NCHW)
//   - Softmax(shape=[1, 1, 24, 16], axisInd=3, layout=NHWC) is adjusted to
//     Softmax(shape=[1, 16, 24, 1], axisInd=1, layout=NHWC)
// Note that these adjustments should not change the real data in memory, so this pattern
// will only be applied when axis dim is the lowest dim in memory
mlir::LogicalResult adjustForAxisZeroOpt(Shape& shape, int64_t& axisInd, const DimsOrder& order) {
    const auto axisDim = Dim(axisInd);
    const auto lowestDim = getLowestDim(shape, order);
    const auto lastDimInMem = order.dimAt(shape.size() - 1);

    if (axisDim != lowestDim || axisDim == lastDimInMem) {
        return mlir::failure();
    }

    // swap lowest dim with the last memdim
    shape[lastDimInMem] = shape[lowestDim];
    shape[lowestDim] = 1;
    // axis becomes the last memdim
    axisInd = lastDimInMem.ind();

    return mlir::success();
}

//
// AdjustShapeForSoftmax
//
// This rewritter adjusts shape of softmax for optimized kernel implementations
// Supported Optimizations:
//   - Kernel optimization for softmax with axis=0 (last memdim in compiler scope)
//   - Gather dimensions on the tile dim for multishave optimizations
class AdjustShapeForSoftmax final : public mlir::OpRewritePattern<VPU::SoftMaxOp> {
public:
    AdjustShapeForSoftmax(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::SoftMaxOp>(ctx), _log(log) {
        this->setDebugName("AdjustShapeForSoftmax");
    }

private:
    mlir::LogicalResult adjustForMultiShaveOpt(Shape& shape, int64_t& axisInd, const DimsOrder& order,
                                               const int64_t numActShaves) const;
    mlir::LogicalResult matchAndRewrite(VPU::SoftMaxOp softmaxOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Adjusts the shape of Softmax to leverage as much shave engines as possible by gather
// dimensions on tile dimension
// Or fuses batch dim to tile dim, as SplitOverBatch is not supported yet and non-outermost dim tiling will introduce
// strided copy
// Examples: (Assume 4 shave engines)
// * Leverage all shaves
//   - Softmax(shape=[1, 2, 16, 24], axisInd=3, layout=NCHW) is adjusted to
//     Softmax(shape=[1, 4, 8, 24], axisInd=3, layout=NCHW)
//   - Softmax(shape=[1, 24, 2, 16], axisInd=1, layout=NHWC) is adjusted to
//     Softmax(shape=[1, 24, 4, 8], axisInd=1, layout=NHWC)
// * Fuse batch dim
//   - Softmax(shape=[16, 4, 16, 24], axisInd=3, layout=NCHW) is adjusted to
//     Softmax(shape=[1, 64, 16, 24], axisInd=3, layout=NCHW)
// Note that these adjustments should not change the real data in memory, and the axis dim
// should not be the tile dim
mlir::LogicalResult AdjustShapeForSoftmax::adjustForMultiShaveOpt(Shape& shape, int64_t& axisInd,
                                                                  const DimsOrder& order,
                                                                  const int64_t numActShaves) const {
    const auto axisDim = Dim(axisInd);

    // only support NCHW and NHWC layout
    if (order != DimsOrder::NCHW && order != DimsOrder::NHWC) {
        return mlir::failure();
    }

    // NCHW tile at C, NHWC tile at H
    const auto tileDim = order.dimAt(1);

    // Fuse batch dim to tile dim
    const auto batchDim = order.dimAt(0);

    // the axis dim on or before the tile dim is not supported
    if (order.dimPos(tileDim) >= order.dimPos(axisDim)) {
        return mlir::failure();
    }

    // no need to adjust if the tile dim is large enough or
    // equal to the max possible dim shape
    const auto maxPossibleDimShape = getTotalSizeBeforeDim(shape, order, axisDim);
    if ((shape[tileDim] >= numActShaves && shape[batchDim] == 1) || shape[tileDim] == maxPossibleDimShape) {
        return mlir::failure();
    }

    const auto nextDim = order.dimAt(2);
    const auto totalSizeBeforeNextDim = getTotalSizeBeforeDim(shape, order, nextDim);
    // gather shape on the tile dim
    for (auto idx : irange(order.dimPos(nextDim))) {
        auto dim = order.dimAt(idx);
        shape[dim] = dim == tileDim ? totalSizeBeforeNextDim : 1;
    }

    if (shape[tileDim] >= numActShaves) {
        return mlir::success();
    }

    // Find the smallest factor which can satisfy multi-shave requirement
    if (nextDim != axisDim) {
        int64_t tileDimShape = shape[tileDim];
        int64_t nextDimShape = shape[nextDim];
        for (auto factor = 2; factor < nextDimShape; factor++) {
            if ((nextDimShape % factor == 0) && (tileDimShape * factor >= numActShaves)) {
                shape[nextDim] = nextDimShape / factor;
                shape[tileDim] = tileDimShape * factor;
                return mlir::success();
            }
        }
    }

    return mlir::failure();
}

mlir::LogicalResult AdjustShapeForSoftmax::matchAndRewrite(VPU::SoftMaxOp softmaxOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at loc '{1}'", softmaxOp->getName(), softmaxOp->getLoc());

    const auto ctx = getContext();

    const auto inType = mlir::cast<vpux::NDTypeInterface>(softmaxOp.getInput().getType());
    const auto inOrder = inType.getDimsOrder();
    const auto inShape = inType.getShape();

    auto shape = inShape.toValues();
    auto axisInd = softmaxOp.getAxisInd();

    const auto axisZeroOpt = adjustForAxisZeroOpt(shape, axisInd, inOrder);
    if (mlir::succeeded(axisZeroOpt)) {
        _log.nest(1).trace("Adjusted shape to {0} and axisInd to {1} for AxisZeroOpt", shape, axisInd);
    }

    const auto numActShaves = config::getTotalNumOfEngines(softmaxOp, VPU::ExecutorKind::SHAVE_ACT);
    const auto multiShaveOpt = adjustForMultiShaveOpt(shape, axisInd, inOrder, numActShaves);
    if (mlir::succeeded(multiShaveOpt)) {
        _log.nest(1).trace("Adjusted shape to {0} and axisInd to {1} for MultiShaveOpt", shape, axisInd);
    }

    if (mlir::failed(axisZeroOpt) && mlir::failed(multiShaveOpt)) {
        return mlir::failure();
    }

    auto reshapeInOp = rewriter.create<VPU::ShapeCastOp>(softmaxOp.getLoc(), inType.changeShape(shape),
                                                         softmaxOp.getInput(), getIntArrayAttr(ctx, shape));
    auto newSoftmaxOp = rewriter.create<VPU::SoftMaxOp>(softmaxOp.getLoc(), reshapeInOp.getResult(),
                                                        getIntAttr(ctx, axisInd), softmaxOp.getPadSizeAttr(), nullptr);
    auto reshapeOutOp = rewriter.create<VPU::ShapeCastOp>(softmaxOp.getLoc(), inType, newSoftmaxOp.getOutput(),
                                                          getIntArrayAttr(ctx, inShape));

    softmaxOp.replaceAllUsesWith(reshapeOutOp.getResult());
    rewriter.eraseOp(softmaxOp);

    return mlir::success();
}

mlir::LogicalResult adjustForMultiShaveOptGeneric(Shape& shape, const DimsOrder& order, const int64_t numActShaves) {
    // only support NCHW and NHWC layout
    if (order != DimsOrder::NCHW && order != DimsOrder::NHWC) {
        return mlir::failure();
    }

    const auto origTileDim = getHighestNonTrivialDim(shape, order);
    // impossible to adjust for shape 1x1x1x1
    if (!origTileDim.has_value()) {
        return mlir::failure();
    }

    // NCHW tile at C, NHWC tile at H
    const auto tileDim = order.dimAt(1);
    const auto dimN = order.dimAt(0);

    // always adjust shape when dim N is not 1 to prevent Clustering strategy
    // no need to adjust if the original tile dim is large enough or equal to the max possible dim shape
    const auto maxPossibleDimShape = shape.totalSize();
    const auto shapeAtTileDim = shape[origTileDim.value()];
    if (shape[dimN] == 1 && (shapeAtTileDim >= numActShaves || shapeAtTileDim == maxPossibleDimShape)) {
        return mlir::failure();
    }

    // gather shape on the tile dim
    for (size_t idx = 0; idx < shape.size(); idx++) {
        auto dim = order.dimAt(idx);
        shape[dim] = dim == tileDim ? maxPossibleDimShape : 1;
    }

    return mlir::success();
}

//
// AdjustShapeForGelu
//
// This rewritter adjusts shape of Gelu by gathering dimensions on the tiling dim for multi-Clusters and multi-SHAVEs
// optimization:
// 1. Shape is adjusted when SW layer has batch, otherwise Clustering strategy would be assigned.
// 2. Shape is adjusted to ensure the dim size of the highest dimension is enough for SHAVEs engines.
class AdjustShapeForGelu final : public mlir::OpRewritePattern<VPU::GeluOp> {
public:
    AdjustShapeForGelu(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::GeluOp>(ctx), _log(log) {
        this->setDebugName("AdjustShapeForGelu");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::GeluOp geluOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult adjustForMultiShaveOpt(Shape& shape, const DimsOrder& order, const int64_t numActShaves) const;

private:
    Logger _log;
};

mlir::LogicalResult AdjustShapeForGelu::adjustForMultiShaveOpt(Shape& shape, const DimsOrder& order,
                                                               const int64_t numActShaves) const {
    return adjustForMultiShaveOptGeneric(shape, order, numActShaves);
}

mlir::LogicalResult AdjustShapeForGelu::matchAndRewrite(VPU::GeluOp geluOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at loc '{1}'", geluOp->getName(), geluOp->getLoc());

    const auto ctx = getContext();

    const auto origIOType = mlir::cast<vpux::NDTypeInterface>(geluOp.getOutput().getType());
    const auto origIOOrder = origIOType.getDimsOrder();
    const auto origIOShape = origIOType.getShape();

    auto shape = origIOShape.toValues();

    // [Tracking number E#178570] skip dynamic shapes as ShapeCastOp cannot handle them yet
    if (shape.isDynamic()) {
        _log.trace("GeluOp has dynamic shape at {0}, no adjustment is required", geluOp->getLoc());
        return mlir::failure();
    }

    const auto numActShaves = config::getTotalNumOfEngines(geluOp, VPU::ExecutorKind::SHAVE_ACT);
    const auto multiShaveOpt = adjustForMultiShaveOpt(shape, origIOOrder, numActShaves);
    if (mlir::failed(multiShaveOpt)) {
        _log.trace("MultiShaveOpt is not required at {0}", geluOp->getLoc());
        return mlir::failure();
    }

    auto dimN = origIOOrder.dimAt(0);
    auto nonNDims = irange(static_cast<size_t>(1), origIOOrder.numDims());
    auto anyNonNDimIsGreaterThanShaveNum = llvm::any_of(nonNDims, [&](int64_t idx) {
        return origIOShape[origIOOrder.dimAt(idx)] >= numActShaves;
    });
    if (origIOShape[dimN] == 1 && anyNonNDimIsGreaterThanShaveNum) {
        _log.nest(1).trace("MultiShaveOpt is not required since some dim can support multi shave tiling at {1}",
                           geluOp->getLoc());
        return mlir::failure();
    }

    _log.nest(1).trace("Adjusted shape to {0} for MultiShaveOpt at {1}", shape, geluOp->getLoc());

    auto reshapeInOp = rewriter.create<VPU::ShapeCastOp>(geluOp->getLoc(), origIOType.changeShape(shape),
                                                         geluOp.getInput(), getIntArrayAttr(ctx, shape));

    auto newGeluOp = rewriter.create<VPU::GeluOp>(geluOp->getLoc(), reshapeInOp.getResult(), nullptr);

    auto reshapeOutOp = rewriter.create<VPU::ShapeCastOp>(geluOp->getLoc(), origIOType, newGeluOp.getOutput(),
                                                          getIntArrayAttr(ctx, origIOShape));

    geluOp.replaceAllUsesWith(reshapeOutOp.getResult());
    rewriter.eraseOp(geluOp);

    return mlir::success();
}

//
// AdjustShapeForMultiply
//
// This rewritter adjusts shape of Multiply by gathering dimensions on the tiling dim for multi-cluster and multi-SHAVEs
// optimization
// 1. Shape is adjusted when SW layer has batch, otherwise Clustering strategy would be assigned.
// 2. Shape is adjusted to ensure the dim size of the highest dimension is enough for SHAVEs engines.
class AdjustShapeForMultiply final : public mlir::OpRewritePattern<VPU::MultiplyOp> {
public:
    AdjustShapeForMultiply(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::MultiplyOp>(ctx), _log(log) {
        this->setDebugName("AdjustShapeForMultiply");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::MultiplyOp multiplyOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult adjustForMultiShaveOpt(Shape& shape, const DimsOrder& order, const int64_t numActShaves) const;

private:
    Logger _log;
};

mlir::LogicalResult AdjustShapeForMultiply::adjustForMultiShaveOpt(Shape& shape, const DimsOrder& order,
                                                                   const int64_t numActShaves) const {
    return adjustForMultiShaveOptGeneric(shape, order, numActShaves);
}

mlir::LogicalResult AdjustShapeForMultiply::matchAndRewrite(VPU::MultiplyOp multiplyOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at loc '{1}'", multiplyOp->getName(), multiplyOp->getLoc());

    if (multiplyOp.getInput1().getType() != multiplyOp.getInput2().getType()) {
        return mlir::failure();
    }

    auto isConstInput = [](mlir::Value input) {
        return input.getDefiningOp<Const::DeclareOp>() != nullptr;
    };
    if (isConstInput(multiplyOp.getInput1()) || isConstInput(multiplyOp.getInput2())) {
        return mlir::failure();
    }

    const auto ctx = getContext();

    const auto origIOType = mlir::cast<vpux::NDTypeInterface>(multiplyOp.getOutput().getType());
    const auto origIOOrder = origIOType.getDimsOrder();
    const auto origIOShape = origIOType.getShape();

    auto shape = origIOShape.toValues();

    const auto numActShaves = config::getTotalNumOfEngines(multiplyOp, VPU::ExecutorKind::SHAVE_ACT);
    const auto multiShaveOpt = adjustForMultiShaveOpt(shape, origIOOrder, numActShaves);
    if (mlir::failed(multiShaveOpt)) {
        return mlir::failure();
    }

    _log.nest(1).trace("Adjusted shape to {0} for MultiShaveOpt at {1}", shape, multiplyOp->getLoc());

    auto reshapeIn1Op = rewriter.create<VPU::ShapeCastOp>(multiplyOp->getLoc(), origIOType.changeShape(shape),
                                                          multiplyOp.getInput1(), getIntArrayAttr(ctx, shape));

    auto reshapeIn2Op = rewriter.create<VPU::ShapeCastOp>(multiplyOp->getLoc(), origIOType.changeShape(shape),
                                                          multiplyOp.getInput2(), getIntArrayAttr(ctx, shape));

    auto newMultiplyOp =
            rewriter.create<VPU::MultiplyOp>(multiplyOp->getLoc(), reshapeIn1Op.getResult(), reshapeIn2Op.getResult(),
                                             multiplyOp.getAutoBroadcastAttr(), multiplyOp.getPostOpAttr(), nullptr);

    auto reshapeOutOp = rewriter.create<VPU::ShapeCastOp>(multiplyOp->getLoc(), origIOType, newMultiplyOp.getOutput(),
                                                          getIntArrayAttr(ctx, origIOShape));

    multiplyOp.replaceAllUsesWith(reshapeOutOp.getResult());
    rewriter.eraseOp(multiplyOp);

    return mlir::success();
}

//
// AdjustShapeForMVN
//

// This rewritter adjusts shape of MVN with batch size larger than one, otherwise Clustering strategy would be assigned
class AdjustShapeForMVN final : public mlir::OpRewritePattern<VPU::MVNOp> {
public:
    AdjustShapeForMVN(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::MVNOp>(ctx), _log(log) {
        this->setDebugName("AdjustShapeForMVN");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::MVNOp mvnOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult adjustForMultiShaveOpt(Shape& shape, bool& isAcrossChannels, const DimsOrder order) const;

private:
    Logger _log;
};

mlir::LogicalResult AdjustShapeForMVN::adjustForMultiShaveOpt(Shape& shape, bool& isAcrossChannels,
                                                              const DimsOrder order) const {
    const auto N = shape[Dims4D::Act::N];
    if (order != DimsOrder::NCHW || N == 1) {
        return mlir::failure();
    }

    if (isAcrossChannels) {
        shape[Dims4D::Act::H] = shape.totalSize() / N;
        shape[Dims4D::Act::W] = 1;
        shape[Dims4D::Act::C] = N;
        isAcrossChannels = false;
    } else {
        shape[Dims4D::Act::C] = N * shape[Dims4D::Act::C];
    }
    shape[Dims4D::Act::N] = 1;

    return mlir::success();
}

mlir::LogicalResult AdjustShapeForMVN::matchAndRewrite(VPU::MVNOp mvnOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at loc '{1}'", mvnOp->getName(), mvnOp->getLoc());

    const auto ctx = getContext();

    const auto origIOType = mlir::cast<vpux::NDTypeInterface>(mvnOp.getOutput().getType());

    const auto origIOOrder = origIOType.getDimsOrder();
    const auto origIOShape = origIOType.getShape();

    auto shape = origIOShape.toValues();

    auto isAcrossChannels = mvnOp.getAcrossChannels();
    const auto multiShaveOpt = adjustForMultiShaveOpt(shape, isAcrossChannels, origIOOrder);
    if (mlir::failed(multiShaveOpt)) {
        return mlir::failure();
    }

    _log.nest(1).trace("Adjusted shape {0} to {1} for MultiShaveOpt at {1}", origIOShape, shape, mvnOp->getLoc());

    auto reshapeInOp = rewriter.create<VPU::ShapeCastOp>(mvnOp->getLoc(), origIOType.changeShape(shape),
                                                         mvnOp.getInput(), getIntArrayAttr(ctx, shape));

    auto newMVNOp = rewriter.create<VPU::MVNOp>(mvnOp->getLoc(), reshapeInOp.getResult(),
                                                mlir::BoolAttr::get(ctx, isAcrossChannels),
                                                mvnOp.getNormalizeVarianceAttr(), mvnOp.getEpsAttr());

    auto reshapeOutOp = rewriter.create<VPU::ShapeCastOp>(mvnOp->getLoc(), origIOType, newMVNOp.getOutput(),
                                                          getIntArrayAttr(ctx, origIOShape));

    mvnOp.replaceAllUsesWith(reshapeOutOp.getResult());
    rewriter.eraseOp(mvnOp);

    return mlir::success();
}

//
// AdjustShapeForReduce
//

template <class ReduceOp>
class AdjustShapeForReduce final : public mlir::OpRewritePattern<ReduceOp> {
public:
    AdjustShapeForReduce(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ReduceOp>(ctx), _log(log) {
        this->setDebugName("AdjustShapeForReduce");
    }

private:
    mlir::LogicalResult matchAndRewrite(ReduceOp reduceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ReduceOp>
mlir::LogicalResult AdjustShapeForReduce<ReduceOp>::matchAndRewrite(ReduceOp reduceOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at loc '{1}'", reduceOp->getName(), reduceOp->getLoc());

    const auto inType = mlir::cast<vpux::NDTypeInterface>(reduceOp.getInput().getType());
    const auto inOrder = inType.getDimsOrder();
    const auto inShape = inType.getShape();

    // For the Reduce Op like something below
    //     VPU.ReduceMin{axes_value = [1], keep_dims} : tensor<1x245760x1x1xf16> -> tensor<1x1x1x1xf16>
    // this op can't be tiled, or split for MC or MS so far. So put non-1 shape on memory inner dimension
    // to make Shave happy.
    //     VPU.ReduceMin{axes_value = [3], keep_dims} : tensor<1x1x1x245760xf16> -> tensor<1x1x1x1xf16>
    //
    const auto hasSingleNonOneDim = llvm::count_if(inShape, [](const auto dim) {
                                        return dim > 1;
                                    }) == 1;
    if (!hasSingleNonOneDim) {
        return mlir::failure();
    }

    const auto axesValue = parseIntArrayAttr<int64_t>(reduceOp.getAxesValue());
    if (axesValue.size() != 1) {
        return mlir::failure();
    }
    auto axisInd = axesValue[0];

    if (inShape[Dim(axisInd)] == 1) {
        _log.nest(1).trace("axis {0} is not matched to non-1 dimension", axisInd);
        return mlir::failure();
    }

    auto targetShape = inShape.toValues();
    const auto axisZeroOpt = adjustForAxisZeroOpt(targetShape, axisInd, inOrder);
    if (mlir::failed(axisZeroOpt)) {
        _log.nest(1).trace("Failed to do AxisZeroOpt with shape {0} and axisInd {1}", targetShape, axisInd);
        return mlir::failure();
    }

    _log.nest(1).trace("Adjusted shape to {0} for zero axis at {1}", targetShape, reduceOp->getLoc());

    const auto outType = mlir::cast<vpux::NDTypeInterface>(reduceOp.getOutput().getType());
    const auto ctx = rewriter.getContext();
    auto reshapeInOp = rewriter.create<VPU::ShapeCastOp>(reduceOp.getLoc(), inType.changeShape(targetShape),
                                                         reduceOp.getInput(), getIntArrayAttr(ctx, targetShape));
    auto newReduceOp =
            rewriter.create<ReduceOp>(reduceOp.getLoc(), reshapeInOp.getResult(),
                                      getIntArrayAttr(ctx, SmallVector<int64_t>{axisInd}), reduceOp.getKeepDimsAttr());
    auto reshapeOutOp = rewriter.create<VPU::ShapeCastOp>(reduceOp.getLoc(), outType, newReduceOp.getOutput(),
                                                          getIntArrayAttr(ctx, outType.getShape()));

    rewriter.replaceOp(reduceOp, reshapeOutOp.getResult());
    return mlir::success();
}

//
// AdjustShapeForNCEPermute
//

class AdjustShapeForNCEPermute final : public mlir::OpRewritePattern<VPU::NCEPermuteOp> {
public:
    AdjustShapeForNCEPermute(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::NCEPermuteOp>(ctx), _log(log) {
        this->setDebugName("AdjustShapeForNCEPermute");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::NCEPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AdjustShapeForNCEPermute::matchAndRewrite(VPU::NCEPermuteOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto ctx = origOp->getContext();
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    auto inOrder = inputType.getDimsOrder();
    auto outOrder = outputType.getDimsOrder();
    if (inOrder != DimsOrder::NCHW || outOrder != DimsOrder::NHWC) {
        return mlir::failure();
    }

    const int64_t MIN_HEIGHT_SIZE_PER_CLUSTER = 4;
    const int64_t MIN_DIM_SIZE_FOR_TILING =
            config::getTileExecutor(origOp.getOperation()->getParentOfType<mlir::ModuleOp>()).getCount() *
            MIN_HEIGHT_SIZE_PER_CLUSTER;
    auto inShape = getShape(origOp.getInput());
    if (inShape[Dims4D::Act::H] >= MIN_DIM_SIZE_FOR_TILING) {
        _log.trace("No need to adjust shape for NCEPermute at {0}", origOp->getLoc());
        return mlir::failure();
    }

    const auto outElemType = outputType.getElementType();
    const auto alignment = VPU::NCEInvariant::getAlignment(outElemType);
    if (inShape[Dims4D::Act::H] * inShape[Dims4D::Act::W] % alignment != 0) {
        _log.trace("Spatial shape size is not divisible by {0}", alignment);
        return mlir::failure();
    }

    const auto newHeight = inShape[Dims4D::Act::H] * inShape[Dims4D::Act::W] / alignment;
    if (newHeight < MIN_DIM_SIZE_FOR_TILING) {
        _log.trace("New dim size on height is not enough for split");
        return mlir::failure();
    }

    auto targetShape = Shape({inShape[Dims4D::Act::N], inShape[Dims4D::Act::C], newHeight, alignment});
    _log.trace("Create NCEPermute with target shape {0}", targetShape);
    // Reshape input
    auto inShapeCast =
            rewriter.create<VPU::ShapeCastOp>(origOp->getLoc(), origOp.getInput(), getIntArrayAttr(ctx, targetShape));

    // Create new NCEPermute
    auto origOutputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    targetShape[Dims4D::Act::C] = origOp.getExpandedChannels();
    auto newOutputType = origOutputType.changeShape(targetShape);
    auto newNCEPermuteOp = rewriter.create<VPU::NCEPermuteOp>(
            origOp->getLoc(), newOutputType, inShapeCast.getResult(), origOp.getExpandedChannelsAttr(),
            origOp.getDstElemTypeAttr(), origOp.getDstOrderAttr(), origOp.getPpeAttr(),
            origOp.getMultiClusterStrategyAttr());

    // Reshape output
    auto origOutputShape = outputType.getShape();
    auto outShapeCast = rewriter.create<VPU::ShapeCastOp>(origOp->getLoc(), newNCEPermuteOp.getOutput(),
                                                          getIntArrayAttr(ctx, origOutputShape));

    rewriter.replaceOp(origOp, outShapeCast.getResult());

    return mlir::success();
}

//
// AdjustShapeForNCEMatMul
//

class AdjustShapeForNCEMatMul final : public mlir::OpRewritePattern<VPU::NCEMatMulOp> {
public:
    AdjustShapeForNCEMatMul(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::NCEMatMulOp>(ctx), _log(log) {
        this->setDebugName("AdjustShapeForNCEMatMul");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::NCEMatMulOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

std::tuple<int64_t, int64_t> getBalanceHW(const int64_t inSize) {
    int64_t newSizeW = 1;
    if (inSize % VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT == 0) {
        newSizeW = VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT;
    } else {
        const auto sqrtVal = static_cast<int64_t>(std::sqrt(inSize));
        for (int64_t factor = sqrtVal; factor > 1; --factor) {
            if (inSize % factor == 0) {
                newSizeW = factor;
                break;
            }
        }
    }

    return std::make_tuple(inSize / newSizeW, newSizeW);
}

//  Convert 1x1 NCE MatMul from:
//
//      input               filter
//  [G, 1, C, H, 1]     [G, OC, IC, 1, 1]
//            \             /
//               NCE MatMul
//            [G, 1, OC, H, 1]
//
//  To:
//           input                  filter
//       [G, 1, C, H, 1]       [G, OC, IC, 1, 1]
//             |                       |
//        AffineReshape                |
//  [G, 1, C, H/factor, factor]        |
//                     \              /
//                         NCE MatMul
//                 [G, 1, OC, H/factor, factor]
//                             |
//                      AffineReshape
//                     [G, 1, OC, H, 1]
//
mlir::LogicalResult AdjustShapeForNCEMatMul::matchAndRewrite(VPU::NCEMatMulOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto ctx = origOp->getContext();

    // NCE MatMul should always have a 5D shape with a kernel size of 1, padding of 0, and stride of 1
    // Here, only the shape rank is checked to ensure correctness using DimsGroups5D::Act
    const auto inputShape = getShape(origOp.getInput());
    if (inputShape.size() != DimsGroups5D::Act::numDims) {
        return matchFailed(rewriter, origOp, "Only NCE MatMul with a 5D shape is supported");
    }

    const auto origH = inputShape[DimsGroups5D::Act::H];
    const auto origW = inputShape[DimsGroups5D::Act::W];
    if (origH == 1 || origW != 1) {
        return matchFailed(rewriter, origOp, "NCE MatMul does not need balancing");
    }

    const auto [newH, newW] = getBalanceHW(origH);
    if (newW == 1) {
        return matchFailed(rewriter, origOp, "NCE MatMul cannot balance H and W");
    }

    _log.trace("Adjust input shape for NCE MatMul at '{0}'", origOp.getLoc());

    auto newInShape = inputShape.toValues();
    newInShape[DimsGroups5D::Act::H] = newH;
    newInShape[DimsGroups5D::Act::W] = newW;
    SmallVector<SmallVector<int64_t>> inDimMapping{{DimsGroups5D::Act::G.ind()},
                                                   {DimsGroups5D::Act::N.ind()},
                                                   {DimsGroups5D::Act::C.ind()},
                                                   {DimsGroups5D::Act::H.ind(), DimsGroups5D ::Act::W.ind()},
                                                   {DimsGroups5D::Act::W.ind()}};

    auto newInput = rewriter.create<VPU::AffineReshapeOp>(origOp.getLoc(), origOp.getInput(),
                                                          getIntArrayOfArray(ctx, inDimMapping),
                                                          getIntArrayAttr(ctx, newInShape));

    auto input1Type = mlir::cast<vpux::NDTypeInterface>(newInput.getOutput().getType());
    auto input2Type = mlir::cast<vpux::NDTypeInterface>(origOp.getWeights().getType());
    auto origOutputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto outputType = VPU::inferNCEMatmulOutputType(input1Type, input2Type, origOutputType);

    auto newMatMulOp = rewriter.create<VPU::NCEMatMulOp>(
            origOp.getLoc(), outputType, newInput.getOutput(), origOp.getWeights(), origOp.getWeightsTable(),
            origOp.getWeightTableDataPtr(), origOp.getWeightTableSpPtr(), origOp.getWeightTableScale(),
            origOp.getWeightTableBias(), origOp.getWeightZeroPoints(), origOp.getStridesAttr(), origOp.getPadAttr(),
            origOp.getPpeAttr(), origOp.getMpeEngineAttr(), origOp.getRawFilterShapeAttr(),
            /* multiClusterStrategyAttr = */ nullptr);

    const auto outShape = getShape(origOp.getOutput()).toValues();
    SmallVector<SmallVector<int64_t>> outDimMapping{{DimsGroups5D::Act::G.ind()},
                                                    {DimsGroups5D::Act::N.ind()},
                                                    {DimsGroups5D::Act::C.ind()},
                                                    {DimsGroups5D::Act::H.ind()},
                                                    {DimsGroups5D::Act::H.ind(), DimsGroups5D::Act::W.ind()}};

    rewriter.replaceOpWithNewOp<VPU::AffineReshapeOp>(
            origOp, newMatMulOp.getOutput(), getIntArrayOfArray(ctx, outDimMapping), getIntArrayAttr(ctx, outShape));

    return mlir::success();
}

//
// AdjustShapeForNCEAvgPool
//

class AdjustShapeForNCEAvgPool final : public mlir::OpRewritePattern<VPU::NCEAveragePoolOp> {
public:
    AdjustShapeForNCEAvgPool(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::NCEAveragePoolOp>(ctx), _log(log) {
        this->setDebugName("AdjustShapeForNCEAvgPool");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::NCEAveragePoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isEltwisePooling(VPU::NCEAveragePoolOp poolingOp) {
    const auto inputType = mlir::cast<NDTypeInterface>(poolingOp.getInput().getType());
    const auto outputType = mlir::cast<NDTypeInterface>(poolingOp.getOutput().getType());
    if (inputType.getDimsOrder() != outputType.getDimsOrder()) {
        return false;
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(poolingOp.getKernelSize());
    const auto strides = parseIntArrayAttr<int64_t>(poolingOp.getStrides());

    const auto paddingAttr = poolingOp.getPad();
    const auto padTop = paddingAttr.getTop().getValue().getSExtValue();
    const auto padBottom = paddingAttr.getBottom().getValue().getSExtValue();
    const auto padLeft = paddingAttr.getLeft().getValue().getSExtValue();
    const auto padRight = paddingAttr.getRight().getValue().getSExtValue();
    auto noPadding = (padTop == 0 && padBottom == 0 && padLeft == 0 && padRight == 0);

    const auto isOne = [](const int64_t val) -> bool {
        return val == 1;
    };

    return llvm::all_of(kernelSize, isOne) && llvm::all_of(strides, isOne) && noPadding;
}

mlir::LogicalResult AdjustShapeForNCEAvgPool::matchAndRewrite(VPU::NCEAveragePoolOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    if (!isEltwisePooling(origOp)) {
        return mlir::failure();
    }

    if (IE::isPerAxisQuant(origOp.getInput()) || IE::isPerAxisQuant(origOp.getInput())) {
        return mlir::failure();
    }

    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto ctx = origOp->getContext();

    const auto inputShape = getShape(origOp.getInput());
    if (inputShape.size() != 4) {
        return matchFailed(rewriter, origOp, "Only NCE AveragePoolOp with a 4D shape is supported");
    }

    auto needToAdjustShape = inputShape[Dims4D::Act::N] > 1;
    if (!needToAdjustShape) {
        return matchFailed(rewriter, origOp, "No need for shape adjustment for NCE AveragePoolOp");
    }

    auto totalShapeSize = inputShape.totalSize();
    if (totalShapeSize % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT != 0) {
        return mlir::failure();
    }

    const auto remainingShapeSize = totalShapeSize / VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
    const auto factorsList = vpux::getFactorsList(remainingShapeSize);
    if (factorsList.empty()) {
        return matchFailed(rewriter, origOp, "Failed to get factors list for {0}", remainingShapeSize);
    }

    auto newInShape = inputShape.toValues();
    newInShape[Dims4D::Act::N] = 1;
    newInShape[Dims4D::Act::C] = VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
    newInShape[Dims4D::Act::H] = factorsList.back().first;
    newInShape[Dims4D::Act::W] = factorsList.back().second;

    _log.debug("Adjust shape for {0} to shape {1}", origOp, newInShape);

    // Reshape input
    auto inShapeCast =
            rewriter.create<VPU::ShapeCastOp>(origOp->getLoc(), origOp.getInput(), getIntArrayAttr(ctx, newInShape));

    // Create new pooling
    auto origOutputType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
    auto newOutputType = origOutputType.changeShape(newInShape);
    auto newPoolOp = rewriter.create<VPU::NCEAveragePoolOp>(
            origOp->getLoc(), newOutputType, inShapeCast.getResult(), origOp.getKernelSizeAttr(),
            origOp.getStridesAttr(), origOp.getPadAttr(), origOp.getPpeAttr(), origOp.getMultiClusterStrategyAttr(),
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    // Reshape output
    auto origOutputShape = getShape(origOp.getOutput());
    auto outShapeCast = rewriter.create<VPU::ShapeCastOp>(origOp->getLoc(), newPoolOp.getOutput(),
                                                          getIntArrayAttr(ctx, origOutputShape));

    rewriter.replaceOp(origOp, outShapeCast.getResult());

    return mlir::success();
}

//
// AdjustForOptimizedLayersPass
//

// Currently, this pass inserts ShapeCast ops to adjust tensor shape for SW layers to fully utilize SHAVEs
// It would be better to adjust tensor shape with considering sub-graph optimization as well
// See E#119868 for details.
class AdjustForOptimizedLayersPass final :
        public VPU::impl::AdjustForOptimizedLayersBase<AdjustForOptimizedLayersPass> {
public:
    explicit AdjustForOptimizedLayersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustForOptimizedLayersPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AdjustShapeForSoftmax>(&ctx, _log);
    patterns.add<AdjustShapeForGelu>(&ctx, _log);
    patterns.add<AdjustShapeForMultiply>(&ctx, _log);
    patterns.add<AdjustShapeForMVN>(&ctx, _log);

    patterns.add<AdjustShapeForReduce<VPU::ReduceMinOp>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceMaxOp>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceMeanOp>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceSumOp>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceProdOp>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceL1Op>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceL2Op>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceLogicalAndOp>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceLogicalOrOp>>(&ctx, _log);

    patterns.add<AdjustShapeForNCEPermute>(&ctx, _log);
    patterns.add<AdjustShapeForNCEMatMul>(&ctx, _log);
    patterns.add<AdjustShapeForNCEAvgPool>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createAdjustForOptimizedLayersPass(Logger log) {
    return std::make_unique<AdjustForOptimizedLayersPass>(log);
}
