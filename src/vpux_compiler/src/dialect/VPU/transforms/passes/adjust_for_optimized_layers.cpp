//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_matmul_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

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
    const auto outType = mlir::cast<vpux::NDTypeInterface>(softmaxOp.getOutput().getType());
    const auto inOrder = inType.getDimsOrder();
    const auto inShape = inType.getShape();

    auto shape = inShape.toValues();
    auto axisInd = softmaxOp.getAxisInd();

    const auto axisZeroOpt = adjustForAxisZeroOpt(shape, axisInd, inOrder);
    if (mlir::succeeded(axisZeroOpt)) {
        _log.nest(1).trace("Adjusted shape to {0} and axisInd to {1} for AxisZeroOpt", shape, axisInd);
    }

    const auto numActShaves = config::getTotalNumOfEngines(softmaxOp, config::ExecutorKind::SHAVE_ACT);
    const auto multiShaveOpt = adjustForMultiShaveOpt(shape, axisInd, inOrder, numActShaves);
    if (mlir::succeeded(multiShaveOpt)) {
        _log.nest(1).trace("Adjusted shape to {0} and axisInd to {1} for MultiShaveOpt", shape, axisInd);
    }

    if (mlir::failed(axisZeroOpt) && mlir::failed(multiShaveOpt)) {
        return mlir::failure();
    }

    auto reshapeInOp = rewriter.create<VPU::ShapeCastOp>(softmaxOp.getLoc(), inType.changeShape(shape),
                                                         softmaxOp.getInput(), getIntArrayAttr(ctx, shape));

    mlir::Value newMax;
    if (const auto origMax = softmaxOp.getMax()) {
        // max shape = adjusted input shape with the new axis dim set to 1
        auto maxShape = std::move(shape);
        maxShape[Dim(axisInd)] = 1;
        const auto maxType = mlir::cast<vpux::NDTypeInterface>(origMax.getType());
        auto reshapeMaxOp = rewriter.create<VPU::ShapeCastOp>(softmaxOp.getLoc(), maxType.changeShape(maxShape),
                                                              origMax, getIntArrayAttr(ctx, maxShape));
        newMax = reshapeMaxOp.getResult();
    }

    auto newSoftmaxOp = rewriter.create<VPU::SoftMaxOp>(softmaxOp.getLoc(), reshapeInOp.getResult(), newMax,
                                                        getIntAttr(ctx, axisInd), softmaxOp.getPadSizeAttr(),
                                                        softmaxOp.getDstElemTypeAttr(), softmaxOp.getMaskAwareAttr());
    auto reshapeOutOp = rewriter.create<VPU::ShapeCastOp>(softmaxOp.getLoc(), outType, newSoftmaxOp.getOutput(),
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

    if (shape.isDynamic()) {
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
// AdjustShapeForUnaryMultiShave
//
// Shared rewriter for unary elementwise SW ops (e.g. Gelu, Convert, Swish) that gathers dimensions on the
// tiling dim for multi-Cluster and multi-SHAVE optimization:
// 1. Shape is adjusted when the SW layer has a batch dimension, otherwise Clustering strategy would
//    be assigned.
// 2. Shape is adjusted to ensure the tile-dim size is large enough for all SHAVE engines.
// The adjusted op is built via rewriter.clone to avoid per-op builder overloads.
template <class UnaryOp>
class AdjustShapeForUnaryMultiShave final : public mlir::OpRewritePattern<UnaryOp> {
public:
    AdjustShapeForUnaryMultiShave(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<UnaryOp>(ctx), _log(log) {
        this->setDebugName("AdjustShapeForUnaryMultiShave");
    }

private:
    mlir::LogicalResult matchAndRewrite(UnaryOp op, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class UnaryOp>
mlir::LogicalResult AdjustShapeForUnaryMultiShave<UnaryOp>::matchAndRewrite(UnaryOp op,
                                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at loc '{1}'", op->getName(), op->getLoc());

    // Skip SwishOp with a tensor beta operand: reshaping both input and beta would require
    // per-operand shape analysis. Only the scalar beta_value attribute path is supported.
    if constexpr (std::is_same_v<UnaryOp, VPU::SwishOp>) {
        if (op.getBeta() != nullptr) {
            _log.trace("SwishOp has a beta tensor operand at {0}, skipping adjustment", op->getLoc());
            return mlir::failure();
        }
    }

    const auto ctx = rewriter.getContext();

    const auto origInputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto origOutputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
    const auto origIOOrder = origOutputType.getDimsOrder();
    const auto origIOShape = origOutputType.getShape();

    auto shape = origIOShape.toValues();

    // [Tracking number E#178570] skip dynamic shapes as ShapeCastOp cannot handle them yet
    if (shape.isDynamic()) {
        _log.trace("Op has dynamic shape at {0}, no adjustment is required", op->getLoc());
        return mlir::failure();
    }

    const auto numActShaves = config::getTotalNumOfEngines(op, config::ExecutorKind::SHAVE_ACT);
    if (mlir::failed(adjustForMultiShaveOptGeneric(shape, origIOOrder, numActShaves))) {
        _log.trace("MultiShaveOpt is not required at {0}", op->getLoc());
        return mlir::failure();
    }

    const auto dimN = origIOOrder.dimAt(0);
    const auto nonNDims = irange(static_cast<size_t>(1), origIOOrder.numDims());
    const auto anyNonNDimIsGreaterThanShaveNum = llvm::any_of(nonNDims, [&](int64_t idx) {
        return origIOShape[origIOOrder.dimAt(idx)] >= numActShaves;
    });
    if (origIOShape[dimN] == 1 && anyNonNDimIsGreaterThanShaveNum) {
        _log.nest(1).trace("MultiShaveOpt is not required since some dim can support multi shave tiling at {0}",
                           op->getLoc());
        return mlir::failure();
    }

    _log.nest(1).trace("Adjusted shape to {0} for MultiShaveOpt at {1}", shape, op->getLoc());

    auto reshapeInOp = rewriter.create<VPU::ShapeCastOp>(op->getLoc(), origInputType.changeShape(shape), op.getInput(),
                                                         getIntArrayAttr(ctx, shape));

    mlir::IRMapping mapping;
    mapping.map(op.getInput(), reshapeInOp.getResult());
    auto* clonedOp = rewriter.clone(*op, mapping);
    clonedOp->getResult(0).setType(origOutputType.changeShape(shape));

    rewriter.replaceOpWithNewOp<VPU::ShapeCastOp>(op, origOutputType, clonedOp->getResult(0),
                                                  getIntArrayAttr(ctx, origIOShape));

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

    const auto origInType = mlir::cast<vpux::NDTypeInterface>(multiplyOp.getInput1().getType());
    const auto origIOType = mlir::cast<vpux::NDTypeInterface>(multiplyOp.getOutput().getType());
    const auto origIOOrder = origIOType.getDimsOrder();
    const auto origIOShape = origIOType.getShape();

    auto shape = origIOShape.toValues();

    const auto numActShaves = config::getTotalNumOfEngines(multiplyOp, config::ExecutorKind::SHAVE_ACT);
    const auto multiShaveOpt = adjustForMultiShaveOpt(shape, origIOOrder, numActShaves);
    if (mlir::failed(multiShaveOpt)) {
        return mlir::failure();
    }

    _log.nest(1).trace("Adjusted shape to {0} for MultiShaveOpt at {1}", shape, multiplyOp->getLoc());

    auto reshapeIn1Op = rewriter.create<VPU::ShapeCastOp>(multiplyOp->getLoc(), origInType.changeShape(shape),
                                                          multiplyOp.getInput1(), getIntArrayAttr(ctx, shape));

    auto reshapeIn2Op = rewriter.create<VPU::ShapeCastOp>(multiplyOp->getLoc(), origInType.changeShape(shape),
                                                          multiplyOp.getInput2(), getIntArrayAttr(ctx, shape));

    auto newMultiplyOp = rewriter.create<VPU::MultiplyOp>(
            multiplyOp->getLoc(), origIOType.changeShape(shape), reshapeIn1Op.getResult(), reshapeIn2Op.getResult(),
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
    mlir::LogicalResult adjustForMultiShaveOpt(Shape& shape, bool& isAcrossChannels, const DimsOrder& order) const;

private:
    Logger _log;
};

mlir::LogicalResult AdjustShapeForMVN::adjustForMultiShaveOpt(Shape& shape, bool& isAcrossChannels,
                                                              const DimsOrder& order) const {
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
    auto inShape = getBoundedShape(origOp.getInput());
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
            origOp.getDstElemTypeAttr(), origOp.getDstOrderAttr(), origOp.getPpeAttr(), origOp.getMpeEngineAttr(),
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
    SmallVector<mlir::Type> extraReduceTypes;
    const auto resultSegmentSizes = origOp.getProperties().getResultSegmentSizes();
    // Fall back to the original op's reduce types; override with re-inferred types when available.
    mlir::Type reduceXyMaxType = origOp.getReduceXyMax() ? origOp.getReduceXyMax().getType() : nullptr;
    mlir::Type reduceXyMinType = origOp.getReduceXyMin() ? origOp.getReduceXyMin().getType() : nullptr;
    mlir::Type reduceTensorMinMaxType =
            origOp.getReduceTensorMinMax() ? origOp.getReduceTensorMinMax().getType() : nullptr;
    if (mlir::succeeded(VPU::inferReduceExtraNCETypes(origOp.getLoc(), outputType, origOp.getAxesValueAttr(),
                                                      resultSegmentSizes, extraReduceTypes)) &&
        !extraReduceTypes.empty()) {
        // resultSegmentSizes layout: [output, reduceXyMax, reduceXyMin, reduceTensorMinMax].
        // extraReduceTypes is densely packed — one entry per active reduce slot (indices 1-3).
        size_t reduceIdx = 0;
        reduceXyMaxType = resultSegmentSizes[1] ? extraReduceTypes[reduceIdx++] : nullptr;
        reduceXyMinType = resultSegmentSizes[2] ? extraReduceTypes[reduceIdx++] : nullptr;
        reduceTensorMinMaxType = resultSegmentSizes[3] ? extraReduceTypes[reduceIdx++] : nullptr;
    }

    auto newMatMulOp = rewriter.create<VPU::NCEMatMulOp>(
            origOp.getLoc(), outputType, reduceXyMaxType, reduceXyMinType, reduceTensorMinMaxType, newInput.getOutput(),
            origOp.getWeights(), origOp.getWeightsTable(), origOp.getWeightTableScale(), origOp.getWeightTableBias(),
            origOp.getWeightZeroPoints(), origOp.getStridesAttr(), origOp.getPadAttr(), origOp.getPpeAttr(),
            origOp.getMpeEngineAttr(),
            /*rawFilterShape=*/origOp.getRawFilterShape(), origOp.getStaticRawFilterShape(),
            /* multiClusterStrategyAttr = */ nullptr, origOp.getAxesValueAttr());

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
            origOp->getLoc(), newOutputType, inShapeCast.getResult(), origOp.getWeightTableScale(),
            origOp.getWeightTableBias(), origOp.getKernelSizeAttr(), origOp.getStridesAttr(), origOp.getPadAttr(),
            origOp.getPpeAttr(), origOp.getMpeEngineAttr(), origOp.getMultiClusterStrategyAttr(),
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    // Reshape output
    auto origOutputShape = getShape(origOp.getOutput());
    auto outShapeCast = rewriter.create<VPU::ShapeCastOp>(origOp->getLoc(), newPoolOp.getOutput(),
                                                          getIntArrayAttr(ctx, origOutputShape));

    rewriter.replaceOp(origOp, outShapeCast.getResult());

    return mlir::success();
}

//
// ConvertTileWithEltwiseToNCEPool
//

class ConvertTileWithEltwiseToNCEPool final : public mlir::OpRewritePattern<VPU::TileOp> {
public:
    ConvertTileWithEltwiseToNCEPool(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::TileOp>(ctx), _log(log) {
        this->setDebugName("AdjustForOptimizedLayersPass::ConvertTileWithEltwiseToNCEPool");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::TileOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertTileWithEltwiseToNCEPool::matchAndRewrite(VPU::TileOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    // NOTE: hasOneUse is not an essential condition. If all users consume origOp in the
    // same way as a single eltwise, this rewrite would still be valid. The generalization
    // is not implemented for now; revisit it if real cases appear.
    if (!origOp->hasOneUse()) {
        _log.trace("TileOp has multiple users at '{0}'", origOp->getLoc());
        return mlir::failure();
    }

    auto userOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(*origOp.getOutput().getUsers().begin());
    if (userOp == nullptr) {
        _log.trace("TileOp has a non-eltwise user at '{0}'", origOp->getLoc());
        return mlir::failure();
    }

    if (userOp.getWeightTableScale() != nullptr || userOp.getWeightTableBias() != nullptr) {
        // Existing table operands would need to be composed with the table produced from Tile:
        // for ADD, newBias = oldBias + oldScale * tileValue; for MULTIPLY,
        // newScale = oldScale * tileValue. The current rewrite does not materialize these
        // extra table computations, so keep such cases unchanged.
        _log.trace("Eltwise user already has weight-table scale/bias operands at '{0}'", userOp->getLoc());
        return mlir::failure();
    }

    const auto eltType = userOp.getOpType();
    if (eltType != VPU::EltwiseType::ADD && eltType != VPU::EltwiseType::MULTIPLY) {
        // The NCE pool rewrite models TileOp as either a per-channel bias (ADD) or scale
        // (MULTIPLY) table. SUBTRACT depends on operand order/sign, and the remaining eltwise
        // kinds are not linear scale/bias transforms, so they are intentionally left untouched.
        _log.trace("Eltwise type '{0}' is not supported for TileOp at '{1}'", eltType, origOp->getLoc());
        return mlir::failure();
    }

    auto ctx = origOp.getContext();
    const auto& ppeConfig = VPU::getPpeConfig(ctx);
    const auto scaleBiasAdapter = ppeConfig.getFactoryAs<VPU::IPpeAdapterScaleBias*>();
    if (scaleBiasAdapter == nullptr) {
        _log.trace("PPE scale/bias adapter is not available at '{0}'", origOp->getLoc());
        return mlir::failure();
    }

    auto input = origOp.getInput();
    const auto inputShape = getShape(input);
    if (inputShape.isDynamic()) {
        _log.trace("TileOp input has dynamic shape at '{0}'", origOp->getLoc());
        return mlir::failure();
    }

    if (inputShape.size() != 4) {
        _log.trace("TileOp input is not 4D at '{0}'", origOp->getLoc());
        return mlir::failure();
    }

    if (inputShape[Dims4D::Act::N] != 1 || inputShape[Dims4D::Act::H] != 1 || inputShape[Dims4D::Act::W] != 1) {
        _log.trace("TileOp input is not flattened to [1, C, 1, 1] at '{0}'", origOp->getLoc());
        return mlir::failure();
    }

    const auto repeatedValues = parseIntArrayAttr<int64_t>(origOp.getRepeatsValues());
    if (repeatedValues.size() != 4) {
        _log.trace("TileOp repeats are not 4D at '{0}'", origOp->getLoc());
        return mlir::failure();
    }
    if (Shape(repeatedValues)[Dims4D::Act::N] != 1 || Shape(repeatedValues)[Dims4D::Act::C] != 1) {
        _log.trace("TileOp repeats are not supported at '{0}'", origOp->getLoc());
        return mlir::failure();
    }

    const auto eltwiseInput1 = userOp.getInput1();
    const auto eltwiseInput2 = userOp.getInput2();
    auto newInput = eltwiseInput1 == origOp.getOutput() ? eltwiseInput2 : eltwiseInput1;
    if (getShape(newInput).isDynamic() || getShape(userOp.getOutput()).isDynamic()) {
        _log.trace("Eltwise input or output has dynamic shape at '{0}'", userOp->getLoc());
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto inputElemType = inputType.getElementType();

    auto fp16Type = mlir::Float16Type::get(ctx);
    auto fp32Type = mlir::Float32Type::get(ctx);
    if (inputElemType != fp16Type && inputElemType != fp32Type) {
        // The TileOp value is materialized as an FP32 weight-table scale or bias tensor. Quantized
        // and other element types would require zero-point and quantization-scale handling before
        // they can be safely reinterpreted as weight-table data.
        _log.trace("TileOp input element type is not supported at '{0}'", origOp->getLoc());
        return mlir::failure();
    }

    auto userPpe = userOp.getPpe();
    const auto scale = scaleBiasAdapter->getScale(userPpe);
    const auto bias = scaleBiasAdapter->getBias(userPpe);
    // The earlier check rejected eltwise users carrying per-channel weight-table scale/bias
    // operands, and quantized inputs were excluded above, so any scale/bias here is stored
    // per-tensor in the PPE and is expected to be present.
    VPUX_THROW_UNLESS(bias.has_value() && scale.has_value(),
                      "Eltwise user at '{0}' has no per-tensor scale/bias in its PPE", userOp->getLoc());
    // Non-neutral PPE scale/bias would need to be composed with the generated per-channel
    // weight-table scale/bias, for example by emitting a follow-up NCE pool op or by consolidating
    // compatible tables. That combination is not validated here, so this rewrite only handles
    // the neutral PPE case.
    if (!llvm::all_of(scale.value(), [](double scaleValue) {
            return scaleValue == 1.0;
        })) {
        _log.trace("Eltwise user has non-neutral scale at '{0}'", userOp->getLoc());
        return mlir::failure();
    }
    if (bias.value() != 0.0) {
        _log.trace("Eltwise user has non-neutral bias at '{0}'", userOp->getLoc());
        return mlir::failure();
    }

    const auto createWeightsTable = [&](llvm::StringRef suffix, float value) {
        const auto tableShape = Shape({inputShape[Dims4D::Act::C], 1, 1, 1});
        const auto tableType = mlir::RankedTensorType::get(tableShape.raw(), fp32Type);
        const SmallVector<float> tableValues = {value};
        return Const::createConst<float>(rewriter, appendLoc(origOp.getLoc(), suffix), tableType, ArrayRef(tableValues),
                                         [](Const::ContentSetup& contentSetup) -> Const::ContentSetup {
                                             return contentSetup.reorder(DimsOrder::NCHW);
                                         });
    };

    const SmallVector<int64_t> poolStrides = {1, 1};
    const SmallVector<int64_t> poolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    const auto padAttr = VPU::getPaddingAttr(ctx, PadInfo(getIntArrayAttr(ctx, pads), getIntArrayAttr(ctx, pads)));

    auto materializeWeightTableInput = [&](mlir::Value value) -> mlir::FailureOr<mlir::Value> {
        auto tableInput = value;
        auto tableInputType = mlir::cast<vpux::NDTypeInterface>(tableInput.getType());

        if (tableInputType.getElementType() == fp16Type) {
            const auto fp32OutputType = tableInputType.changeElemType(fp32Type);
            if (auto avgPoolOp = tableInput.getDefiningOp<VPU::NCEAveragePoolOp>()) {
                // NCE pool ops can produce FP32 directly, so avoid materializing an extra ConvertOp
                // when the per-channel table comes from another NCE pool.
                if (avgPoolOp->hasOneUse()) {
                    rewriter.modifyOpInPlace(avgPoolOp, [&]() {
                        avgPoolOp.getOutput().setType(fp32OutputType);
                    });
                    tableInput = avgPoolOp.getOutput();
                } else {
                    auto* clonedAvgPoolOp = rewriter.clone(*avgPoolOp);
                    rewriter.modifyOpInPlace(clonedAvgPoolOp, [&]() {
                        clonedAvgPoolOp->getResult(0).setType(fp32OutputType);
                    });
                    tableInput = clonedAvgPoolOp->getResult(0);
                }
            } else if (auto maxPoolOp = tableInput.getDefiningOp<VPU::NCEMaxPoolOp>()) {
                // NCE pool ops can produce FP32 directly, so avoid materializing an extra ConvertOp
                // when the per-channel table comes from another NCE pool.
                if (maxPoolOp->hasOneUse()) {
                    rewriter.modifyOpInPlace(maxPoolOp, [&]() {
                        maxPoolOp.getOutput().setType(fp32OutputType);
                    });
                    tableInput = maxPoolOp.getOutput();
                } else {
                    auto* clonedMaxPoolOp = rewriter.clone(*maxPoolOp);
                    rewriter.modifyOpInPlace(clonedMaxPoolOp, [&]() {
                        clonedMaxPoolOp->getResult(0).setType(fp32OutputType);
                    });
                    tableInput = clonedMaxPoolOp->getResult(0);
                }
            } else {
                auto materializedMaxPoolOp = rewriter.create<VPU::NCEMaxPoolOp>(
                        appendLoc(origOp.getLoc(), "weight_table_maxpool"), fp32OutputType,
                        /*reduceXyMax=*/nullptr, /*reduceXyMin=*/nullptr,
                        /*reduceTensorMinMax=*/nullptr, tableInput, /*weightsTable=*/nullptr,
                        /*weight_table_scale=*/nullptr, /*weight_table_bias=*/nullptr,
                        getIntArrayAttr(ctx, poolKernels), getIntArrayAttr(ctx, poolStrides), padAttr,
                        VPU::PPEStubAttr::get(ctx),
                        /*mpe_engineAttr=*/nullptr, /*multi_cluster_strategyAttr=*/nullptr,
                        /*output_paddingAttr=*/nullptr, /*input_paddingAttr=*/nullptr,
                        /*s2dd2s_configAttr=*/nullptr, /*axes_valueAttr=*/nullptr);
                // A stub PPE leaks into later stages: passes such as MultiClusterStrategyAssignment
                // query the architecture PPE factory, which only accepts the concrete PPEFpAttr and
                // throws on PPEStub. Assign the NOOP fp PPE produced by the active factory for this
                // freshly created conversion MaxPool.
                rewriter.modifyOpInPlace(materializedMaxPoolOp, [&]() {
                    materializedMaxPoolOp.setPpeAttr(ppeConfig.retrievePPEAttribute(materializedMaxPoolOp));
                });
                tableInput = materializedMaxPoolOp.getOutput();
            }
            tableInputType = mlir::cast<vpux::NDTypeInterface>(tableInput.getType());
        }

        if (tableInputType.getDimsOrder() != DimsOrder::NCHW) {
            const auto memPerm =
                    tryToFindPermutationForPermuteCast(tableInputType, DimsOrder::NCHW, tableInputType.getShape(), ctx);
            if (!memPerm.has_value()) {
                return mlir::failure();
            }

            auto permuteCastOp =
                    rewriter.create<VPU::PermuteCastOp>(appendLoc(origOp.getLoc(), "weight_table_permute"), tableInput,
                                                        DimsOrder::NCHW.toAffineMap(ctx), memPerm.value());
            tableInput = permuteCastOp.getOutput();
        }

        return tableInput;
    };

    rewriter.setInsertionPoint(userOp);

    auto fp32Input = materializeWeightTableInput(input);
    if (mlir::failed(fp32Input)) {
        _log.trace("Cannot materialize weight-table input for TileOp at '{0}'", origOp->getLoc());
        return mlir::failure();
    }

    auto shapeCastOp = rewriter.create<VPU::AffineReshapeOp>(
            appendLoc(origOp.getLoc(), "weight_table_reshape"), *fp32Input,
            getIntArrayOfArray(ctx, SmallVector<SmallVector<int64_t>>{{0}, {0}, {1}, {2, 3}}),
            getIntArrayAttr(ctx, SmallVector<int64_t>{inputShape[Dims4D::Act::C], 1, 1, 1}));

    mlir::Value scaleTable = nullptr;
    mlir::Value biasTable = nullptr;
    if (eltType == VPU::EltwiseType::ADD) {
        biasTable = shapeCastOp.getOutput();
        scaleTable = createWeightsTable("scale_table", 1.0f);
    } else if (eltType == VPU::EltwiseType::MULTIPLY) {
        scaleTable = shapeCastOp.getOutput();
        biasTable = createWeightsTable("bias_table", 0.0f);
    } else {
        VPUX_THROW("Unexpected eltwise type: {0}", eltType);
    }

    auto newPpe = scaleBiasAdapter->discardScaleBias(userPpe);
    rewriter.replaceOpWithNewOp<VPU::NCEMaxPoolOp>(
            userOp, userOp.getResult(0).getType(),
            /*reduceXyMax=*/nullptr, /*reduceXyMin=*/nullptr, /*reduceTensorMinMax=*/nullptr, newInput,
            /*weightsTable=*/nullptr, scaleTable, biasTable, getIntArrayAttr(ctx, poolKernels),
            getIntArrayAttr(ctx, poolStrides), padAttr, newPpe,
            /*mpe_engineAttr=*/nullptr, /*multi_cluster_strategyAttr=*/nullptr, userOp.getOutputPaddingAttr(),
            userOp.getInputPaddingAttr(), /*s2dd2s_configAttr=*/nullptr, /*axes_valueAttr=*/nullptr);

    rewriter.eraseOp(origOp);
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
    patterns.add<AdjustShapeForUnaryMultiShave<VPU::GeluOp>>(&ctx, _log);
    patterns.add<AdjustShapeForUnaryMultiShave<VPU::ConvertOp>>(&ctx, _log);
    patterns.add<AdjustShapeForUnaryMultiShave<VPU::SwishOp>>(&ctx, _log);
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

    const auto& strategyFactory = VPU::getVPUStrategyFactory(&ctx);
    if (strategyFactory != nullptr && strategyFactory->isConvertTileWithEltwiseToNCEPoolSupported()) {
        patterns.add<ConvertTileWithEltwiseToNCEPool>(&ctx, _log);
    }

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createAdjustForOptimizedLayersPass(Logger log) {
    return std::make_unique<AdjustForOptimizedLayersPass>(log);
}
