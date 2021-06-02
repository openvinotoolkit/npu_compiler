//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Transforms/Bufferize.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// BufferizeIEPass
//

class BufferizeIEPass final : public BufferizeIEBase<BufferizeIEPass> {
public:
    explicit BufferizeIEPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ConstantRewrite;
    class GenericReshapeRewrite;
    class SplitRewrite;
    class ConcatRewrite;
    class SubtensorRewrite;
    class LayerRewrite;

public:
    static SmallVector<mlir::Value> allocateResults(mlir::Location loc, mlir::OpBuilder& builder,
                                                    mlir::TypeConverter& typeConverter, mlir::ValueRange origResults);

private:
    void safeRunOnFunc() final;
};

//
// allocateResults
//

SmallVector<mlir::Value> BufferizeIEPass::allocateResults(mlir::Location loc, mlir::OpBuilder& builder,
                                                          mlir::TypeConverter& typeConverter,
                                                          mlir::ValueRange origResults) {
    return to_small_vector(origResults | transformed([&](mlir::Value origVal) -> mlir::Value {
                               auto origType = origVal.getType();
                               auto memRefType = typeConverter.convertType(origType);
                               auto allocOp =
                                       builder.create<mlir::memref::AllocOp>(loc, memRefType.cast<mlir::MemRefType>());
                               return allocOp.memref();
                           }));
}

//
// ConstantRewrite
//

class BufferizeIEPass::ConstantRewrite final : public mlir::OpConversionPattern<IE::ConstantOp> {
public:
    ConstantRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::ConstantOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConstantOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult BufferizeIEPass::ConstantRewrite::matchAndRewrite(IE::ConstantOp origOp, ArrayRef<mlir::Value>,
                                                                      mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Constant Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newType = typeConverter->convertType(origOp.getType());

    rewriter.replaceOpWithNewOp<IERT::ConstantOp>(origOp, newType, origOp.value());
    return mlir::success();
}

//
// GenericReshapeRewrite
//

class BufferizeIEPass::GenericReshapeRewrite final :
        public mlir::OpInterfaceConversionPattern<mlir::ViewLikeOpInterface> {
public:
    GenericReshapeRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceConversionPattern<mlir::ViewLikeOpInterface>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::ViewLikeOpInterface origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult BufferizeIEPass::GenericReshapeRewrite::matchAndRewrite(
        mlir::ViewLikeOpInterface origOp, ArrayRef<mlir::Value> newOperands,
        mlir::ConversionPatternRewriter& rewriter) const {
    if (!mlir::isa<IE::ReshapeOp, IE::SqueezeOp, IE::UnsqueezeOp>(origOp)) {
        return mlir::failure();
    }

    _log.trace("Found GenericReshape Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto outType = origOp->getResult(0).getType().cast<mlir::ShapedType>();

    if (!outType.hasStaticShape()) {
        return matchFailed(rewriter, origOp, "GenericReshape with dynamic shape is not supported yet");
    }

    VPUX_THROW_UNLESS(!newOperands.empty(), "Got wrong newOperands size : '{0}'", newOperands.size());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(outType);

    rewriter.replaceOpWithNewOp<IERT::GenericReshapeOp>(origOp, newOutType, newOperands[0]);
    return mlir::success();
}

//
// SplitRewrite
//

class BufferizeIEPass::SplitRewrite final : public mlir::OpConversionPattern<IE::SplitOp> {
public:
    SplitRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::SplitOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SplitOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult BufferizeIEPass::SplitRewrite::matchAndRewrite(IE::SplitOp origOp,
                                                                   ArrayRef<mlir::Value> newOperands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Split Operation '{0}'", origOp->getLoc());

    VPUX_THROW_UNLESS(newOperands.size() == origOp->getNumOperands(),
                      "Got wrong newOperands size : '{0}', expected '{1}'", newOperands.size(),
                      origOp->getNumOperands());

    if (!origOp.axis_value().hasValue()) {
        return matchFailed(rewriter, origOp, "Got non constant axis");
    }

    const auto inputType = newOperands[0].getType().cast<mlir::ShapedType>();
    const auto inputShape = getShape(inputType);

    const auto axis = origOp.getAxis();

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    auto allocatedBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOp.getResults());

    // Prepare strides array for subview. We have dense array, so all strides have to be equal 1
    SmallVector<int64_t> svStrides(inputShape.size(), 1);
    SmallVector<int64_t> svOffsets(inputShape.size(), 0);
    SmallVector<mlir::Value> results;

    const auto offsetStep = inputShape[axis] / origOp.num_splits();

    for (auto i : irange(origOp->getNumResults())) {
        const auto origOutputType = origOp.getResult(i).getType().cast<mlir::ShapedType>();
        const auto svSizes = origOutputType.getShape();

        _log.trace("Create SubView for output #'{0}'", i);
        auto subView = rewriter.create<mlir::memref::SubViewOp>(origOp.getLoc(), newOperands[0], svOffsets, svSizes,
                                                                svStrides);

        _log.trace("Copy SubView result to output buffer");

        auto copyOp = rewriter.create<IERT::CopyOp>(origOp->getLoc(), subView, allocatedBufs[i]);
        results.push_back(copyOp.output());

        svOffsets[axis.ind()] += offsetStep;
    }

    rewriter.replaceOp(origOp, results);

    return mlir::success();
}

//
// ConcatRewrite
//

class BufferizeIEPass::ConcatRewrite final : public mlir::OpConversionPattern<IE::ConcatOp> {
public:
    ConcatRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::ConcatOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult BufferizeIEPass::ConcatRewrite::matchAndRewrite(IE::ConcatOp origOp,
                                                                    ArrayRef<mlir::Value> newOperands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Layer Operation '{0}'", origOp->getLoc());

    VPUX_THROW_UNLESS(newOperands.size() == origOp->getNumOperands(),
                      "Got wrong newOperands size : '{0}', expected '{1}'", newOperands.size(),
                      origOp->getNumOperands());

    const auto axis = origOp.getAxis();

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    _log.trace("Add Alloc Operations for results");
    auto allocatedBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, {origOp.getResult()});

    const auto outputRank = origOp.getType().getRank();

    // Prepare strides array for subview. We have dense array, so all strides have to be equal 1
    SmallVector<int64_t> svStrides(outputRank, 1);
    SmallVector<int64_t> svOffsets(outputRank, 0);
    SmallVector<mlir::Value> results;

    for (auto i : irange(origOp->getNumOperands())) {
        const auto newInputType = newOperands[i].getType().cast<mlir::ShapedType>();
        const auto svSizes = newInputType.getShape();

        _log.trace("Create SubView for input #'{0}'", i);
        auto subView = rewriter.create<mlir::memref::SubViewOp>(origOp->getLoc(), allocatedBufs[0], svOffsets, svSizes,
                                                                svStrides);

        _log.trace("Copy new operand to SubView");
        auto copyOp = rewriter.create<IERT::CopyOp>(origOp->getLoc(), newOperands[i], subView);
        results.push_back(copyOp.output());

        svOffsets[axis.ind()] += svSizes[axis.ind()];
    }

    rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(origOp, results, allocatedBufs[0]);
    return mlir::success();
}

//
// StdSubtensorRewrite
//

class BufferizeIEPass::SubtensorRewrite final : public mlir::OpConversionPattern<mlir::SubTensorOp> {
public:
    SubtensorRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<mlir::SubTensorOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::SubTensorOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult BufferizeIEPass::SubtensorRewrite::matchAndRewrite(
        mlir::SubTensorOp origOp, ArrayRef<mlir::Value> newOperands, mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Layer Operation '{0}'", origOp->getLoc());

    VPUX_THROW_UNLESS(newOperands.size() == origOp->getNumOperands(),
                      "Got wrong newOperands size : '{0}', expected '{1}'", newOperands.size(),
                      origOp->getNumOperands());

    auto subView = rewriter.create<mlir::memref::SubViewOp>(
            origOp->getLoc(), newOperands[0], parseIntArrayAttr(origOp.static_offsets()),
            parseIntArrayAttr(origOp.static_sizes()), parseIntArrayAttr(origOp.static_strides()));

    auto allocatedBuf = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOp.getResult());

    auto copyOp = rewriter.create<IERT::CopyOp>(origOp->getLoc(), subView, allocatedBuf[0]);

    rewriter.replaceOp(origOp, {copyOp});
    return mlir::success();
}

//
// LayerRewrite
//

mlir::Operation* createRTLayer(mlir::quant::QuantizeCastOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::QuantizeOp::Adaptor newOp(allBufs);
    return b.create<IERT::QuantizeOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(mlir::quant::DequantizeCastOp origOp, ArrayRef<mlir::Value> allBufs,
                               mlir::OpBuilder& b) {
    IERT::DequantizeOp::Adaptor newOp(allBufs);
    return b.create<IERT::DequantizeOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::ConvertOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ConvertOp::Adaptor newOp(allBufs);
    return b.create<IERT::ConvertOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::ReLUOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ReLUOp::Adaptor newOp(allBufs);
    return b.create<IERT::ReLUOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::SigmoidOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SigmoidOp::Adaptor newOp(allBufs);
    return b.create<IERT::SigmoidOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::HSwishOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::HSwishOp::Adaptor newOp(allBufs);
    return b.create<IERT::HSwishOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::TanhOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::TanhOp::Adaptor newOp(allBufs);
    return b.create<IERT::TanhOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::NegativeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::NegativeOp::Adaptor newOp(allBufs);
    return b.create<IERT::NegativeOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::PReluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::PReluOp::Adaptor newOp(allBufs);
    return b.create<IERT::PReluOp>(origOp.getLoc(), newOp.input(), newOp.negative_slope(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::AddOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::AddOp::Adaptor newOp(allBufs);
    return b.create<IERT::AddOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::MultiplyOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::MultiplyOp::Adaptor newOp(allBufs);
    return b.create<IERT::MultiplyOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::DivideOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::DivideOp::Adaptor newOp(allBufs);
    return b.create<IERT::DivideOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::SquaredDifferenceOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SquaredDifferenceOp::Adaptor newOp(allBufs);
    return b.create<IERT::SquaredDifferenceOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::PowerOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::PowerOp::Adaptor newOp(allBufs);
    return b.create<IERT::PowerOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::FloorModOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::FloorModOp::Adaptor newOp(allBufs);
    return b.create<IERT::FloorModOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::MinimumOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::MinimumOp::Adaptor newOp(allBufs);
    return b.create<IERT::MinimumOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::MaximumOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::MaximumOp::Adaptor newOp(allBufs);
    return b.create<IERT::MaximumOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::TileOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::TileOp::Adaptor newOp(allBufs);
    return b.create<IERT::TileOp>(origOp.getLoc(), newOp.input(), newOp.repeats(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::SoftMaxOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SoftMaxOp::Adaptor newOp(allBufs);
    return b.create<IERT::SoftMaxOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.axisIndAttr());
}

mlir::Operation* createRTLayer(IE::AvgPoolOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::AvgPoolOp::Adaptor newOp(allBufs);
    return b.create<IERT::AvgPoolOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.kernel_sizeAttr(),
                                     origOp.stridesAttr(), origOp.pads_beginAttr(), origOp.pads_endAttr());
}

mlir::Operation* createRTLayer(IE::MaxPoolOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::MaxPoolOp::Adaptor newOp(allBufs);
    return b.create<IERT::MaxPoolOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.kernel_sizeAttr(),
                                     origOp.stridesAttr(), origOp.pads_beginAttr(), origOp.pads_endAttr());
}

mlir::Operation* createRTLayer(IE::ClampOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ClampOp::Adaptor newOp(allBufs);
    return b.create<IERT::ClampOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.minAttr(),
                                   origOp.maxAttr());
}

mlir::Operation* createRTLayer(IE::EluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::EluOp::Adaptor newOp(allBufs);
    return b.create<IERT::EluOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.xAttr());
}

mlir::Operation* createRTLayer(IE::LeakyReluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::LeakyReluOp::Adaptor newOp(allBufs);
    return b.create<IERT::LeakyReluOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(),
                                       origOp.negative_slopeAttr());
}

mlir::Operation* createRTLayer(IE::GRNOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::GRNOp::Adaptor newOp(allBufs);
    return b.create<IERT::GRNOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.biasAttr());
}

mlir::Operation* createRTLayer(IE::PerAxisTileOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::PerAxisTileOp::Adaptor newOp(allBufs);
    return b.create<IERT::PerAxisTileOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.axisAttr(),
                                         origOp.tilesAttr());
}

mlir::Operation* createRTLayer(IE::FakeQuantizeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::FakeQuantizeOp::Adaptor newOp(allBufs);
    return b.create<IERT::FakeQuantizeOp>(origOp.getLoc(), newOp.input(), newOp.input_low(), newOp.input_high(),
                                          newOp.output_low(), newOp.output_high(), newOp.output_buff(),
                                          origOp.levelsAttr());
}

mlir::Operation* createRTLayer(IE::ROIPoolingOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ROIPoolingOp::Adaptor newOp(allBufs);
    return b.create<IERT::ROIPoolingOp>(origOp.getLoc(), newOp.input(), newOp.coords(), newOp.output_buff(),
                                        origOp.output_sizeAttr(), origOp.spatial_scaleAttr(), origOp.methodAttr());
}

mlir::Operation* createRTLayer(IE::ConvolutionOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ConvolutionOp::Adaptor newOp(allBufs);
    return b.create<IERT::ConvolutionOp>(origOp.getLoc(), newOp.input(), newOp.filter(), newOp.bias(),
                                         newOp.output_buff(), origOp.stridesAttr(), origOp.pads_beginAttr(),
                                         origOp.pads_endAttr(), origOp.dilationsAttr());
}

mlir::Operation* createRTLayer(IE::GroupConvolutionOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::GroupConvolutionOp::Adaptor newOp(allBufs);
    return b.create<IERT::GroupConvolutionOp>(origOp.getLoc(), newOp.input(), newOp.filter(), newOp.bias(),
                                              newOp.output_buff(), origOp.stridesAttr(), origOp.pads_beginAttr(),
                                              origOp.pads_endAttr(), origOp.dilationsAttr(), origOp.groupsAttr());
}

mlir::Operation* createRTLayer(IE::SwishOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SwishOp::Adaptor newOp(allBufs);
    return b.create<IERT::SwishOp>(origOp.getLoc(), newOp.input(), newOp.beta(), newOp.output_buff(),
                                   origOp.beta_valueAttr());
}

mlir::Operation* createRTLayer(IE::FullyConnectedOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::FullyConnectedOp::Adaptor newOp(allBufs);
    return b.create<IERT::FullyConnectedOp>(origOp.getLoc(), newOp.input(), newOp.weights(), newOp.bias(),
                                            newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::DetectionOutputOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    const auto newPreds = origOp.in_additional_preds() != nullptr ? allBufs[3] : nullptr;
    const auto newProposals = origOp.in_additional_proposals() != nullptr ? allBufs[4] : nullptr;
    return b.create<IERT::DetectionOutputOp>(origOp->getLoc(), allBufs[0], allBufs[1], allBufs[2], newPreds,
                                             newProposals, allBufs.back(), origOp.attr());
}

mlir::Operation* createRTLayer(IE::ScaleShiftOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    mlir::Value newWeights;
    mlir::Value newBiases;
    if (origOp.weights() != nullptr && origOp.biases() != nullptr) {
        newWeights = allBufs[1];
        newBiases = allBufs[2];
    } else if (origOp.weights() != nullptr) {
        newWeights = allBufs[1];
    } else if (origOp.biases() != nullptr) {
        newBiases = allBufs[1];
    } else {
        VPUX_THROW("ScaleShift must have weights or biases");
    }
    return b.create<IERT::ScaleShiftOp>(origOp->getLoc(), allBufs[0], newWeights, newBiases, allBufs.back());
}

mlir::Operation* createRTLayer(IE::TransposeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::TransposeOp::Adaptor newOp(allBufs);
    return b.create<IERT::TransposeOp>(origOp.getLoc(), newOp.input(), newOp.order(), newOp.output_buff(),
                                       origOp.order_valueAttr());
}

mlir::Operation* createRTLayer(IE::CTCGreedyDecoderOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::CTCGreedyDecoderOp::Adaptor newOp(allBufs);
    return b.create<IERT::CTCGreedyDecoderOp>(origOp.getLoc(), newOp.input(), newOp.sequenceLengths(),
                                              newOp.output_buff(), origOp.mergeRepeatedAttr());
}

mlir::Operation* createRTLayer(IE::CTCGreedyDecoderSeqLenOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::CTCGreedyDecoderSeqLenOp::Adaptor newOp(allBufs);
    return b.create<IERT::CTCGreedyDecoderSeqLenOp>(origOp.getLoc(), newOp.input(), newOp.sequenceLength(),
                                                    newOp.blankIndex(), newOp.output_buff(), newOp.outputLength_buff(),
                                                    origOp.mergeRepeatedAttr());
}

mlir::Operation* createRTLayer(IE::PadOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::PadOp::Adaptor newOp(allBufs);
    if (!origOp.pads_begin_attr().hasValue() || !origOp.pads_end_attr().hasValue()) {
        VPUX_THROW("PadOp must use attributes for `pads_begin` and `pads_end` params");
    }

    return b.create<IERT::PadOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(),
                                 origOp.pads_begin_attr().getValue(), origOp.pads_end_attr().getValue(),
                                 origOp.pad_value_attrAttr(), origOp.mode());
}

mlir::Operation* createRTLayer(IE::ExpOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ExpOp::Adaptor newOp(allBufs);
    return b.create<IERT::ExpOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

class BufferizeIEPass::LayerRewrite final : public mlir::ConversionPattern {
public:
    LayerRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::ConversionPattern(typeConverter, mlir::Pattern::MatchAnyOpTypeTag{}, benefitLow, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    template <class InLayerOp>
    static mlir::Operation* dispatch(mlir::Operation* origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b);

private:
    Logger _log;
};

template <class InLayerOp>
mlir::Operation* BufferizeIEPass::LayerRewrite::dispatch(mlir::Operation* origOp, ArrayRef<mlir::Value> allBufs,
                                                         mlir::OpBuilder& b) {
    return createRTLayer(mlir::cast<InLayerOp>(origOp), allBufs, b);
}

mlir::LogicalResult BufferizeIEPass::LayerRewrite::matchAndRewrite(mlir::Operation* origOp,
                                                                   ArrayRef<mlir::Value> newOperands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
    using CreateFunc =
            mlir::Operation* (*)(mlir::Operation * origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder & b);

#define CASE(_OP_)                                   \
    .Case<_OP_>([](mlir::Operation*) -> CreateFunc { \
        return dispatch<_OP_>;                       \
    })

    const CreateFunc createFunc =
            llvm::TypeSwitch<mlir::Operation*, CreateFunc>(origOp) CASE(mlir::quant::QuantizeCastOp)
    CASE(mlir::quant::DequantizeCastOp)
    CASE(IE::ConvertOp)
    CASE(IE::SoftMaxOp)
    CASE(IE::AvgPoolOp)
    CASE(IE::MaxPoolOp)
    CASE(IE::ConvolutionOp)
    CASE(IE::GroupConvolutionOp)
    CASE(IE::ReLUOp)
    CASE(IE::SigmoidOp)
    CASE(IE::ClampOp)
    CASE(IE::EluOp)
    CASE(IE::HSwishOp)
    CASE(IE::TanhOp)
    CASE(IE::FakeQuantizeOp)
    CASE(IE::PReluOp)
    CASE(IE::LeakyReluOp)
    CASE(IE::AddOp)
    CASE(IE::MultiplyOp)
    CASE(IE::DivideOp)
    CASE(IE::SquaredDifferenceOp)
    CASE(IE::PowerOp)
    CASE(IE::FloorModOp)
    CASE(IE::MinimumOp)
    CASE(IE::MaximumOp)
    CASE(IE::SwishOp)
    CASE(IE::GRNOp)
    CASE(IE::TileOp)
    CASE(IE::PerAxisTileOp)
    CASE(IE::NegativeOp)
    CASE(IE::ROIPoolingOp)
    CASE(IE::FullyConnectedOp)
    CASE(IE::DetectionOutputOp)
    CASE(IE::ScaleShiftOp)
    CASE(IE::TransposeOp)
    CASE(IE::CTCGreedyDecoderOp)
    CASE(IE::CTCGreedyDecoderSeqLenOp)
    CASE(IE::PadOp)
    CASE(IE::ExpOp)
    .Default([](mlir::Operation*) {
        return nullptr;
    });

#undef CASE

    if (createFunc == nullptr) {
        return mlir::failure();
    }

    _log.trace("Found Layer Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    VPUX_THROW_UNLESS(newOperands.size() == origOp->getNumOperands(), "Got wrong newOperands size : '{0}'",
                      newOperands.size());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto resultBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    SmallVector<mlir::Value> allBufs;
    allBufs.reserve(newOperands.size() + resultBufs.size());
    allBufs.append(newOperands.begin(), newOperands.end());
    allBufs.append(resultBufs.begin(), resultBufs.end());

    const auto newOp = createFunc(origOp, allBufs, rewriter);
    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

//
// safeRunOnFunc
//

void BufferizeIEPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::BufferizeTypeConverter typeConverter;

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<IERT::IERTDialect>();
    target.addIllegalDialect<IE::IEDialect>();
    target.addIllegalDialect<mlir::quant::QuantizationDialect>();
    target.addLegalOp<IE::CNNNetworkOp, IE::DataInfoOp>();
    target.addLegalOp<mlir::memref::AllocOp>();
    target.addLegalOp<mlir::memref::SubViewOp>();
    mlir::populateBufferizeMaterializationLegality(target);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConstantRewrite>(typeConverter, &ctx, _log);
    patterns.insert<GenericReshapeRewrite>(typeConverter, &ctx, _log);
    patterns.insert<SplitRewrite>(typeConverter, &ctx, _log);
    patterns.insert<ConcatRewrite>(typeConverter, &ctx, _log);
    patterns.insert<LayerRewrite>(typeConverter, &ctx, _log);
    patterns.insert<SubtensorRewrite>(typeConverter, &ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createBufferizeIEPass
//

std::unique_ptr<mlir::Pass> vpux::createBufferizeIEPass(Logger log) {
    return std::make_unique<BufferizeIEPass>(log);
}
