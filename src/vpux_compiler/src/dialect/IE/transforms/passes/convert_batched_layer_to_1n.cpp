//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

#include <mlir/IR/IRMapping.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTBATCHEDLAYERTO1N
#define GEN_PASS_DEF_CONVERTBATCHEDLAYERTO1N
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;
using namespace IE;

namespace {

bool isEqualToOne(mlir::Value value, const Dim& dim) {
    const auto shape = getShape(value);
    VPUX_THROW_UNLESS(shape.size() > checked_cast<size_t>(dim.ind()), "Invalid Dim {0} for shape {1}", dim, shape);
    return shape[dim] == 1;
}

std::optional<Dim> getDimEqualsToOne(ArrayRef<mlir::Value> values) {
    const SmallVector<Dim> candidates = {Dims4D::Act::H, Dims4D::Act::W};
    for (const auto& dim : candidates) {
        auto dimEqualsToOne = llvm::all_of(values, [&](const auto& value) {
            return isEqualToOne(value, dim);
        });
        if (dimEqualsToOne) {
            return dim;
        }
    }
    return std::nullopt;
}

bool isShapeRankEq4(mlir::Value val) {
    const auto inputShape = getShape(val);
    return inputShape.size() == 4;
}

bool isLegalConvToGroupConv(IE::ConvolutionOp op) {
    auto inputShape = getShape(op.getInput());
    auto filterShape = getShape(op.getFilter());

    // Only convert 4D tensors with IC=1, OC=1, FC=1, and N>1
    if (inputShape.size() != 4 || filterShape.size() != 4) {
        return true;
    }

    const auto N = inputShape[Dims4D::Act::N];
    const auto IC = inputShape[Dims4D::Act::C];
    const auto H = inputShape[Dims4D::Act::H];
    const auto W = inputShape[Dims4D::Act::W];
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto FC = filterShape[Dims4D::Filter::IC];

    return !(IC == 1 && OC == 1 && FC == 1 && N > 1 && H > 1 && W > 1);
}

IE::TransposeOp createTransposeForLayerInput(mlir::PatternRewriter& rewriter, mlir::Value input, const Dim& dim,
                                             mlir::Location loc) {
    auto originShape = getShape(input);
    auto dimOrder = DimsOrder::fromValue(input);
    SmallVector<unsigned> transPerm(originShape.size());
    std::iota(transPerm.begin(), transPerm.end(), 0);

    transPerm[dimOrder.dimPos(Dims4D::Act::N)] = checked_cast<unsigned>(dimOrder.dimPos(dim));
    transPerm[dimOrder.dimPos(dim)] = checked_cast<unsigned>(dimOrder.dimPos(Dims4D::Act::N));

    const auto orderAttr =
            mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(transPerm, rewriter.getContext()));
    auto newLoc = appendLoc(loc, "_ConvertBatchedLayer_inTranspose");
    return rewriter.create<IE::TransposeOp>(newLoc, input, nullptr, orderAttr);
}

//
// BaseLayerConverter
//

template <class ConcreteOp>
class BaseLayerConverter : public mlir::OpRewritePattern<ConcreteOp> {
public:
    BaseLayerConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("BaseLayerConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

protected:
    virtual mlir::LogicalResult layerSpecificRewriter(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const = 0;

    mlir::IRMapping mapOperands(ConcreteOp origOp, mlir::AffineMapAttr& orderAttr,
                                mlir::PatternRewriter& rewriter) const;

protected:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult BaseLayerConverter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Got layer at '{0}'", origOp->getLoc());

    if (mlir::succeeded(layerSpecificRewriter(origOp, rewriter))) {
        return mlir::success();
    }

    mlir::AffineMapAttr transPermAttr = nullptr;
    auto mapper = mapOperands(origOp, transPermAttr, rewriter);
    VPUX_THROW_WHEN(transPermAttr == nullptr, "Can not get order value from input tranpose");
    auto* newOp = rewriter.clone(*origOp.getOperation(), mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE | vpux::InferShapedTypeMode::LAYOUT);

    auto output = newOp->getResult(0);
    _log.trace("Insert new layer without batch: {0}", output);

    auto outTranspose = rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, output, nullptr, transPermAttr);
    outTranspose->setLoc(appendLoc(origOp->getLoc(), "_transpose_output"));
    _log.trace("Insert transpose {0} for output", outTranspose);

    return mlir::success();
}

template <class ConcreteOp>
mlir::IRMapping BaseLayerConverter<ConcreteOp>::mapOperands(ConcreteOp origOp, mlir::AffineMapAttr& orderAttr,
                                                            mlir::PatternRewriter& rewriter) const {
    mlir::IRMapping mapper;
    SmallVector<mlir::Value> inputVals;
    mlir::Operation* op = origOp;
    for (auto input : origOp.getInputs() | indexed) {
        if (!op->hasTrait<IE::EltwiseOp>() && input.index() > 0) {
            continue;
        }
        inputVals.push_back(input.value());
    }
    auto dim = getDimEqualsToOne(inputVals).value();
    for (auto input : inputVals) {
        auto inTranspose = createTransposeForLayerInput(rewriter, input, dim, origOp->getLoc());
        orderAttr = orderAttr == nullptr ? inTranspose.getOrderValueAttr() : orderAttr;
        _log.trace("Insert transpose for input: {0}", inTranspose);
        mapper.map(input, inTranspose.getOutput());
    }
    return mapper;
}

//
// EltwiseLayerConverter
//

template <class ConcreteOp>
class EltwiseLayerConverter final : public BaseLayerConverter<ConcreteOp> {
    using BaseLayerConverter<ConcreteOp>::_log;

public:
    EltwiseLayerConverter(mlir::MLIRContext* ctx, Logger log): BaseLayerConverter<ConcreteOp>(ctx, log) {
        this->setDebugName("EltwiseLayerConverter");
    }

protected:
    mlir::LogicalResult layerSpecificRewriter(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const override;
    void reshapeForEltwiseOp(ConcreteOp eltwiseOp, mlir::PatternRewriter& rewriter) const;
    void createNewEltwiseOp(ConcreteOp eltwiseOp, mlir::PatternRewriter& rewriter, mlir::RankedTensorType resultType,
                            mlir::Value input1, mlir::Value input2, const Shape& outputShape) const;
};

template <class ConcreteOp>
mlir::LogicalResult EltwiseLayerConverter<ConcreteOp>::layerSpecificRewriter(ConcreteOp origOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    if (!getDimEqualsToOne({origOp.getInput1(), origOp.getInput2()}).has_value()) {
        reshapeForEltwiseOp(origOp, rewriter);
        _log.trace("Reshape {0} directly for not suitable transpose cases", origOp->getName());
        return mlir::success();
    }

    return mlir::failure();
}

// Specialization for PowerOp - use the same logic as other eltwise ops but handle the attributes properly
template <>
mlir::LogicalResult EltwiseLayerConverter<IE::PowerOp>::layerSpecificRewriter(IE::PowerOp origOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    if (!getDimEqualsToOne({origOp.getInput1(), origOp.getInput2()}).has_value()) {
        // No suitable dimension for transpose, use ShapeCast path
        reshapeForEltwiseOp(origOp, rewriter);
        _log.trace("Reshape PowerOp using ShapeCast path for not suitable transpose cases, {0}", origOp->getName());
        return mlir::success();
    }

    return mlir::failure();
}

template <class ConcreteOp>
void EltwiseLayerConverter<ConcreteOp>::reshapeForEltwiseOp(ConcreteOp eltwiseOp,
                                                            mlir::PatternRewriter& rewriter) const {
    const auto ctx = eltwiseOp->getContext();
    auto inputShape1 = getShape(eltwiseOp.getInput1());
    auto inputShape2 = getShape(eltwiseOp.getInput2());
    auto outputShape = getShape(eltwiseOp.getOutput());

    Shape newInShape1 = Shape(inputShape1);
    newInShape1[Dims4D::Act::N] = 1;
    newInShape1[Dims4D::Act::C] = inputShape1[Dims4D::Act::C] * inputShape1[Dims4D::Act::N];
    Shape newInShape2 = Shape(inputShape2);
    newInShape2[Dims4D::Act::N] = 1;
    newInShape2[Dims4D::Act::C] = inputShape2[Dims4D::Act::C] * inputShape2[Dims4D::Act::N];
    Shape newOutputShape = Shape(outputShape);
    newOutputShape[Dims4D::Act::N] = 1;
    newOutputShape[Dims4D::Act::C] = outputShape[Dims4D::Act::C] * outputShape[Dims4D::Act::N];
    const auto resultType =
            mlir::RankedTensorType::get(newOutputShape.raw(), eltwiseOp.getOutput().getType().getElementType());

    auto inShapeCast1 = rewriter.create<IE::ShapeCastOp>(eltwiseOp->getLoc(), eltwiseOp.getInput1(),
                                                         getIntArrayAttr(ctx, newInShape1));
    auto inShapeCast2 = rewriter.create<IE::ShapeCastOp>(eltwiseOp->getLoc(), eltwiseOp.getInput2(),
                                                         getIntArrayAttr(ctx, newInShape2));

    createNewEltwiseOp(eltwiseOp, rewriter, resultType, inShapeCast1.getResult(), inShapeCast2.getResult(),
                       Shape(outputShape));
}

// Helper function to create new eltwise operations with proper attributes
template <class ConcreteOp>
void EltwiseLayerConverter<ConcreteOp>::createNewEltwiseOp(ConcreteOp eltwiseOp, mlir::PatternRewriter& rewriter,
                                                           mlir::RankedTensorType resultType, mlir::Value input1,
                                                           mlir::Value input2, const Shape& outputShape) const {
    const auto ctx = eltwiseOp->getContext();

    auto newEltwiseOp = rewriter.create<ConcreteOp>(eltwiseOp->getLoc(), resultType, input1, input2,
                                                    eltwiseOp.getAutoBroadcastAttr(), eltwiseOp.getPostOpAttr(),
                                                    eltwiseOp.getClampAttr(), eltwiseOp.getOutputPaddingAttr(),
                                                    eltwiseOp.getInputPaddingAttr());

    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(eltwiseOp, newEltwiseOp.getOutput(),
                                                 getIntArrayAttr(ctx, outputShape));
}

// Specialization for PowerOp which has different attributes
template <>
void EltwiseLayerConverter<IE::PowerOp>::createNewEltwiseOp(IE::PowerOp eltwiseOp, mlir::PatternRewriter& rewriter,
                                                            mlir::RankedTensorType resultType, mlir::Value input1,
                                                            mlir::Value input2, const Shape& outputShape) const {
    const auto ctx = eltwiseOp->getContext();

    auto newPowerOp = rewriter.create<IE::PowerOp>(eltwiseOp->getLoc(), resultType, input1, input2,
                                                   eltwiseOp.getAutoBroadcastAttr());

    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(eltwiseOp, newPowerOp.getOutput(), getIntArrayAttr(ctx, outputShape));
}

//
// PoolLayerConverter
//

template <class ConcreteOp>
class PoolLayerConverter final : public BaseLayerConverter<ConcreteOp> {
    using BaseLayerConverter<ConcreteOp>::_log;

public:
    PoolLayerConverter(mlir::MLIRContext* ctx, Logger log): BaseLayerConverter<ConcreteOp>(ctx, log) {
        this->setDebugName("PoolLayerConverter");
    }

protected:
    mlir::LogicalResult layerSpecificRewriter(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const override;
};

template <class ConcreteOp>
mlir::LogicalResult PoolLayerConverter<ConcreteOp>::layerSpecificRewriter(ConcreteOp poolOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    if (!isIdentityPooling(poolOp)) {
        return mlir::failure();
    }

    auto inShape = getShape(poolOp.getInput());
    auto outShape = getShape(poolOp.getOutput());

    auto ctx = poolOp.getContext();
    auto newInShape = SmallVector<int64_t>{1, inShape[Dims4D::Act::N], inShape[Dims4D::Act::C],
                                           inShape[Dims4D::Act::H] * inShape[Dims4D::Act::W]};
    const auto inputShapeAttr = getIntArrayAttr(poolOp->getContext(), newInShape);
    SmallVector<SmallVector<int64_t>> inDimMapping{{Dims4D::Act::N.ind(), Dims4D::Act::C.ind()},
                                                   {Dims4D::Act::H.ind()},
                                                   {Dims4D::Act::W.ind()},
                                                   {Dims4D::Act::W.ind()}};
    auto inAffineReshape = rewriter.create<IE::AffineReshapeOp>(poolOp->getLoc(), poolOp.getInput(),
                                                                getIntArrayOfArray(ctx, inDimMapping), inputShapeAttr);

    mlir::IRMapping mapper;
    mapper.map(poolOp.getOperand(), inAffineReshape.getOutput());
    auto* newPool = rewriter.clone(*poolOp.getOperation(), mapper);
    vpux::inferReturnTypes(newPool, vpux::InferShapedTypeMode::SHAPE | vpux::InferShapedTypeMode::LAYOUT);
    auto output = newPool->getResult(0);

    const auto outputShapeAttr = getIntArrayAttr(poolOp->getContext(), outShape);
    SmallVector<SmallVector<int64_t>> outDimMapping{{Dims4D::Act::N.ind()},
                                                    {Dims4D::Act::N.ind()},
                                                    {Dims4D::Act::C.ind()},
                                                    {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()}};

    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(poolOp, output, getIntArrayOfArray(ctx, outDimMapping),
                                                     outputShapeAttr);

    return mlir::success();
}

//
// ConvLayerConverter
//

class ConvLayerConverter final : public BaseLayerConverter<IE::ConvolutionOp> {
public:
    ConvLayerConverter(mlir::MLIRContext* ctx, Logger log): BaseLayerConverter<IE::ConvolutionOp>(ctx, log) {
        this->setDebugName("ConvLayerConverter");
    }

protected:
    mlir::LogicalResult layerSpecificRewriter(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvLayerConverter::layerSpecificRewriter(IE::ConvolutionOp convOp,
                                                              mlir::PatternRewriter& rewriter) const {
    if (convOp.getInput() != convOp.getFilter() && isLegalConvToGroupConv(convOp)) {
        return mlir::failure();
    }

    if (convOp.getInput() == convOp.getFilter()) {
        auto dim = getDimEqualsToOne({convOp.getInput()}).value();
        auto inTranspose = createTransposeForLayerInput(rewriter, convOp.getInput(), dim, convOp->getLoc());
        auto transPermAttr = inTranspose.getOrderValueAttr();
        VPUX_THROW_WHEN(transPermAttr == nullptr, "Can not get order value from input transpose");
        auto output = rewriter.create<IE::ConvolutionOp>(convOp->getLoc(), inTranspose.getOutput(), convOp.getFilter(),
                                                         convOp.getBias(), convOp.getStridesAttr(),
                                                         convOp.getPadsBeginAttr(), convOp.getPadsEndAttr(),
                                                         convOp.getDilationsAttr(), convOp.getPostOpAttr(),
                                                         convOp.getClampAttr(), convOp.getStaticScaleAttr(),
                                                         convOp.getOutputPaddingAttr(), convOp.getInputPaddingAttr())
                              .getOutput();

        auto parentOpOutputType = mlir::cast<NDTypeInterface>(output.getType());
        auto outElemType = mlir::cast<NDTypeInterface>(convOp.getOutput().getType()).getElementType();
        output.getDefiningOp()->getResult(0).setType(parentOpOutputType.changeElemType(outElemType));
        _log.trace("Insert new layer without batch: {0}", output);
        auto outTranspose = rewriter.replaceOpWithNewOp<IE::TransposeOp>(convOp, output, nullptr, transPermAttr);
        outTranspose->setLoc(appendLoc(convOp->getLoc(), "_transpose_output"));
        _log.trace("Insert transpose {0} for output", outTranspose);

        return mlir::success();
    }

    auto inputShape = getShape(convOp.getInput());
    auto filterShape = getShape(convOp.getFilter());
    auto outputShape = getShape(convOp.getOutput());

    _log.trace("Converting Convolution to GroupConvolution: input {0}, filter {1}, output {2}", inputShape, filterShape,
               outputShape);

    const auto ctx = convOp->getContext();

    // 1. Broadcast filter: [1, 1, KH, KW] -> [N, 1, KH, KW]
    const auto N = inputShape[Dims4D::Act::N];
    Shape newFilterShape = Shape(filterShape);
    newFilterShape[Dims4D::Filter::OC] = N;
    auto filterBroadcast =
            IE::createBroadcast(rewriter, takeOpLoc(convOp, "filter"), convOp.getFilter(), newFilterShape);

    // 2. Reshape input: [N, 1, H, W] -> [1, N, H, W]
    Shape newInputShape = Shape(inputShape);
    newInputShape[Dims4D::Act::N] = 1;
    newInputShape[Dims4D::Act::C] = N;
    auto inputReshape =
            rewriter.create<IE::ShapeCastOp>(convOp->getLoc(), convOp.getInput(), getIntArrayAttr(ctx, newInputShape));

    // 3. Create GroupConvolution with groups = N
    Shape newOutputShape = Shape(outputShape);
    newOutputShape[Dims4D::Act::N] = 1;
    newOutputShape[Dims4D::Act::C] = N;
    const auto resultType =
            mlir::RankedTensorType::get(newOutputShape.raw(), convOp.getOutput().getType().getElementType());

    auto groupConvOp = rewriter.create<IE::GroupConvolutionOp>(
            convOp->getLoc(), resultType, inputReshape.getResult(), filterBroadcast, convOp.getBias(),
            convOp.getStridesAttr(), convOp.getPadsBeginAttr(), convOp.getPadsEndAttr(), convOp.getDilationsAttr(),
            getIntAttr(ctx, N), convOp.getPostOpAttr(), convOp.getClampAttr(), convOp.getOutputPaddingAttr(),
            convOp.getInputPaddingAttr());

    // 4. Reshape output back: [1, N, H', W'] -> [N, 1, H', W']
    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(convOp, groupConvOp.getOutput(), getIntArrayAttr(ctx, outputShape));

    _log.trace("Successfully converted Convolution to GroupConvolution with {0} groups", N);
    return mlir::success();
}

//
// GroupConvLayerConverter
//

class GroupConvLayerConverter final : public BaseLayerConverter<IE::GroupConvolutionOp> {
public:
    GroupConvLayerConverter(mlir::MLIRContext* ctx, Logger log): BaseLayerConverter<IE::GroupConvolutionOp>(ctx, log) {
        this->setDebugName("GroupConvLayerConverter");
    }

protected:
    mlir::LogicalResult layerSpecificRewriter(IE::GroupConvolutionOp origOp,
                                              mlir::PatternRewriter& rewriter) const final;
    void reshapeForGroupConv(IE::GroupConvolutionOp groupConvOp, mlir::PatternRewriter& rewriter) const;
};

mlir::LogicalResult GroupConvLayerConverter::layerSpecificRewriter(IE::GroupConvolutionOp groupConvOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    if (isEqualToOne(groupConvOp.getInput(), Dims4D::Act::C)) {
        reshapeForGroupConv(groupConvOp, rewriter);
        _log.trace("Reshape GroupConvOp directly for not suitable transpose cases");
        return mlir::success();
    }

    if (groupConvOp.getInput() != groupConvOp.getFilter()) {
        return mlir::failure();
    }

    auto dim = getDimEqualsToOne({groupConvOp.getInput()}).value();
    auto inTranspose = createTransposeForLayerInput(rewriter, groupConvOp.getInput(), dim, groupConvOp->getLoc());
    auto transPermAttr = inTranspose.getOrderValueAttr();
    VPUX_THROW_WHEN(transPermAttr == nullptr, "Can not get order value from input tranpose");
    auto output = rewriter.create<IE::GroupConvolutionOp>(
                                  groupConvOp->getLoc(), inTranspose.getOutput(), groupConvOp.getFilter(),
                                  groupConvOp.getBias(), groupConvOp.getStridesAttr(), groupConvOp.getPadsBeginAttr(),
                                  groupConvOp.getPadsEndAttr(), groupConvOp.getDilationsAttr(),
                                  groupConvOp.getGroupsAttr(), groupConvOp.getPostOpAttr(), groupConvOp.getClampAttr(),
                                  groupConvOp.getOutputPaddingAttr(), groupConvOp.getInputPaddingAttr())
                          .getOutput();

    _log.trace("Insert new layer without batch: {0}", output);
    auto outTranspose = rewriter.replaceOpWithNewOp<IE::TransposeOp>(groupConvOp, output, nullptr, transPermAttr);
    outTranspose->setLoc(appendLoc(groupConvOp->getLoc(), "_transpose_output"));
    _log.trace("Insert transpose {0} for output", outTranspose);

    return mlir::success();
}

void GroupConvLayerConverter::reshapeForGroupConv(IE::GroupConvolutionOp groupConvOp,
                                                  mlir::PatternRewriter& rewriter) const {
    const auto ctx = groupConvOp->getContext();

    Shape newInputShape = Shape(getShape(groupConvOp.getInput()));
    newInputShape[Dims4D::Act::C] = newInputShape[Dims4D::Act::N];
    newInputShape[Dims4D::Act::N] = 1;
    Shape newOutputShape = Shape(getShape(groupConvOp.getOutput()));
    newOutputShape[Dims4D::Act::C] = newOutputShape[Dims4D::Act::N];
    newOutputShape[Dims4D::Act::N] = 1;
    auto group = newInputShape[Dims4D::Act::C];

    auto inShapeCast = rewriter.create<IE::ShapeCastOp>(groupConvOp->getLoc(), groupConvOp.getInput(),
                                                        getIntArrayAttr(ctx, newInputShape));

    const auto filter = groupConvOp.getFilter();
    auto targetConstShape = Shape(getShape(filter));
    targetConstShape[Dims4D::Filter::OC] = group;
    auto newFilter = IE::createBroadcast(rewriter, takeOpLoc(groupConvOp, "filter"), filter, targetConstShape);

    auto newGroupConvOp = rewriter.create<IE::GroupConvolutionOp>(
            groupConvOp->getLoc(), inShapeCast.getResult(), newFilter, groupConvOp.getBias(),
            groupConvOp.getStridesAttr(), groupConvOp.getPadsBeginAttr(), groupConvOp.getPadsEndAttr(),
            groupConvOp.getDilationsAttr(), getIntAttr(ctx, group), groupConvOp.getPostOpAttr(),
            groupConvOp.getClampAttr(), groupConvOp.getOutputPaddingAttr(), groupConvOp.getInputPaddingAttr());

    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(groupConvOp, newGroupConvOp.getOutput(),
                                                 getIntArrayAttr(ctx, getShape(groupConvOp.getOutput())));
}

//
// isLegalOp
//

template <class ConcreteOp>
bool isLegalGeneric(ConcreteOp op) {
    auto hasPerAxisQuantization = isPerAxisQuant(op.getInput()) || isPerAxisQuant(op.getOutput());
    if (!isShapeRankEq4(op.getInput()) || isEqualToOne(op.getInput(), Dims4D::Act::N) || hasPerAxisQuantization) {
        return true;
    }
    return false;
}

bool isLegalConvOp(IE::ConvolutionOp op) {
    if (isLegalGeneric(op)) {
        return true;
    }
    auto transposedDim = getDimEqualsToOne({op.getInput()});
    return !transposedDim.has_value() || !isEqualToOne(op.getFilter(), transposedDim.value());
}

bool isLegalGroupConvOp(IE::GroupConvolutionOp op) {
    if (isLegalGeneric(op)) {
        return true;
    }
    auto transposedDim = getDimEqualsToOne({op.getInput()});
    return (!transposedDim.has_value() || !isEqualToOne(op.getFilter(), transposedDim.value())) &&
           !isEqualToOne(op.getInput(), Dims4D::Act::C);
}

template <class ConcreteOp>
bool isLegalPoolOp(ConcreteOp op) {
    if (isLegalGeneric(op)) {
        return true;
    }

    if (isIdentityPooling(op)) {
        return false;
    }

    auto transposedDim = getDimEqualsToOne({op.getInput()});
    if (!transposedDim.has_value()) {
        return true;
    }
    const auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSize());
    if (transposedDim.value() == Dims4D::Act::H) {
        return kernelSize[Dims4D::Kernel::Y.ind()] != 1;
    } else if (transposedDim.value() == Dims4D::Act::W) {
        return kernelSize[Dims4D::Kernel::X.ind()] != 1;
    }
    return true;
}

bool isLegalImplicitOp(mlir::Operation* op) {
    auto hasPerAxisQuantization = isPerAxisQuant(op->getOperand(0)) || isPerAxisQuant(op->getResult(0));
    auto inShape = getShape(op->getOperand(0));

    return !isShapeRankEq4(op->getOperand(0)) || isEqualToOne(op->getOperand(0), Dims4D::Act::N) ||
           hasPerAxisQuantization || inShape.isDynamic() || op->getNumOperands() != 1 || op->getNumResults() != 1;
}

template <class ConcreteOp>
bool isLegalEltwiseOp(ConcreteOp op) {
    if (op->getNumOperands() < 2) {
        return true;
    }
    auto hasPerAxisQuantization =
            isPerAxisQuant(op->getOperand(0)) || isPerAxisQuant(op->getOperand(1)) || isPerAxisQuant(op->getResult(0));
    auto inShape1 = getShape(op->getOperand(0));
    auto inShape2 = getShape(op->getOperand(1));

    // Multiply is converted to N=1 to prepare for potential conversion to ScaleShift and reduce unrolling
    // When cannot be convert to ScaleShift (H/W > 1), we only allow conversion when N>=C to reduce excessive unrolling
    if (auto mulOp = mlir::dyn_cast<IE::MultiplyOp>(*op)) {
        const auto lhsType = mlir::cast<mlir::ShapedType>(mulOp.getInput1().getType());
        const auto outShapeRes = mlir::cast<mlir::ShapedType>(mulOp.getOutput().getType());
        bool lhsIsActivation = (lhsType == outShapeRes);
        auto weightsInput = lhsIsActivation ? mulOp.getInput2() : mulOp.getInput1();
        auto weightsShape = getShape(weightsInput);
        if ((weightsShape[Dims4D::Act::H] != 1 || weightsShape[Dims4D::Act::W] != 1) &&
            weightsShape[Dims4D::Act::N] < weightsShape[Dims4D::Act::C]) {
            return true;
        }
    }

    return static_cast<bool>(!isShapeRankEq4(op->getOperand(1)) || isEqualToOne(op->getOperand(0), Dims4D::Act::N) ||
                             !isShapeRankEq4(op->getOperand(0)) ||
                             !isBroadcastable(inShape1[Dims4D::Act::N] * inShape1[Dims4D::Act::C],
                                              inShape2[Dims4D::Act::N] * inShape2[Dims4D::Act::C]) ||
                             hasPerAxisQuantization);
};

//
// ImplicitLayerConverter
//

template <class ConcreteOp>
class ImplicitLayerConverter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ImplicitLayerConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

IE::AffineReshapeOp buildAffineReshape(mlir::Location loc, mlir::Value input, ArrayRef<int64_t> targetShape,
                                       mlir::PatternRewriter& rewriter, mlir::Operation* replacedOp) {
    const auto ctx = rewriter.getContext();
    const auto srcType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto dstType = srcType.changeShape(ShapeRef(targetShape));

    SmallVector<SmallVector<int64_t>> inDimMapping{{Dims4D::Act::N.ind(), Dims4D::Act::C.ind()},
                                                   {Dims4D::Act::C.ind()},
                                                   {Dims4D::Act::H.ind()},
                                                   {Dims4D::Act::W.ind()}};
    SmallVector<SmallVector<int64_t>> outDimMapping{{Dims4D::Act::N.ind()},
                                                    {Dims4D::Act::N.ind(), Dims4D::Act::C.ind()},
                                                    {Dims4D::Act::H.ind()},
                                                    {Dims4D::Act::W.ind()}};
    auto dimMapping = replacedOp == nullptr ? inDimMapping : outDimMapping;

    auto reshapeOp = replacedOp == nullptr ? rewriter.create<IE::AffineReshapeOp>(loc, dstType, input,
                                                                                  getIntArrayOfArray(ctx, dimMapping),
                                                                                  getIntArrayAttr(ctx, targetShape))
                                           : rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(
                                                     replacedOp, dstType, input, getIntArrayOfArray(ctx, dimMapping),
                                                     getIntArrayAttr(ctx, targetShape));

    return reshapeOp;
}

IE::AffineReshapeOp affineReshapeInput(mlir::Location loc, mlir::Value input, ShapeRef origShape,
                                       mlir::PatternRewriter& rewriter) {
    Shape targetShape(origShape.raw());

    targetShape[Dims4D::Act::C] *= targetShape[Dims4D::Act::N];
    targetShape[Dims4D::Act::N] = 1;

    const auto reshapedLoc = appendLoc(loc, "reshape input for implicit layer");
    return buildAffineReshape(reshapedLoc, input, ArrayRef(targetShape.raw()), rewriter, nullptr);
}

IE::AffineReshapeOp affineReshapeOutput(mlir::Operation* origOp, mlir::Value output, mlir::PatternRewriter& rewriter) {
    const Shape origShape = getShape(origOp->getResult(0)).toValues();
    const SmallVector<int64_t> targetShape = origShape.raw();

    const auto reshapedLoc = appendLoc(origOp->getLoc(), "reshape output for implicit layer");
    return buildAffineReshape(reshapedLoc, output, ArrayRef(targetShape), rewriter, origOp);
}

template <class ConcreteOp>
mlir::LogicalResult ImplicitLayerConverter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    // Create affineReshapeOp for input
    const auto inShape = getShape(origOp.getOperand());
    auto reshapeIn = affineReshapeInput(origOp.getLoc(), origOp.getOperand(), inShape, rewriter);

    // Create new concreteOp
    mlir::IRMapping mapper;
    mapper.map(origOp.getOperand(), reshapeIn);
    auto* newOp = rewriter.clone(*origOp.getOperation(), mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);

    // Create affineReshapeOp for output
    affineReshapeOutput(origOp, newOp->getResult(0), rewriter);

    return mlir::success();
}

//
// ConvertBatchedLayerTo1NPass
//

class ConvertBatchedLayerTo1NPass final : public IE::impl::ConvertBatchedLayerTo1NBase<ConvertBatchedLayerTo1NPass> {
public:
    explicit ConvertBatchedLayerTo1NPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertBatchedLayerTo1NPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([](IE::ConvolutionOp op) {
        return isLegalConvOp(op) && isLegalConvToGroupConv(op);
    });

    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>(isLegalGroupConvOp);
    target.addDynamicallyLegalOp<IE::MaxPoolOp>(isLegalPoolOp<IE::MaxPoolOp>);
    target.addDynamicallyLegalOp<IE::AvgPoolOp>(isLegalPoolOp<IE::AvgPoolOp>);
    target.addDynamicallyLegalOp<IE::AddOp>(isLegalEltwiseOp<IE::AddOp>);
    target.addDynamicallyLegalOp<IE::MultiplyOp>(isLegalEltwiseOp<IE::MultiplyOp>);
    target.addDynamicallyLegalOp<IE::SubtractOp>(isLegalEltwiseOp<IE::SubtractOp>);
    target.addDynamicallyLegalOp<IE::PowerOp>(isLegalEltwiseOp<IE::PowerOp>);
    target.addDynamicallyLegalOp<IE::SigmoidOp>(isLegalImplicitOp);
    target.addDynamicallyLegalOp<IE::ConvertOp>(isLegalImplicitOp);

    target.addLegalOp<IE::TransposeOp>();
    target.addLegalOp<IE::ShapeCastOp>();
    target.addLegalOp<IE::BroadcastOp>();
    target.addLegalOp<IE::AffineReshapeOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvLayerConverter>(&ctx, _log);
    patterns.add<GroupConvLayerConverter>(&ctx, _log);
    patterns.add<PoolLayerConverter<IE::MaxPoolOp>>(&ctx, _log);
    patterns.add<PoolLayerConverter<IE::AvgPoolOp>>(&ctx, _log);
    patterns.add<EltwiseLayerConverter<IE::AddOp>>(&ctx, _log);
    patterns.add<EltwiseLayerConverter<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<EltwiseLayerConverter<IE::SubtractOp>>(&ctx, _log);
    patterns.add<EltwiseLayerConverter<IE::PowerOp>>(&ctx, _log);
    patterns.add<ImplicitLayerConverter<IE::SigmoidOp>>(&ctx, _log);
    patterns.add<ImplicitLayerConverter<IE::ConvertOp>>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertBatchedLayerTo1NPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertBatchedLayerTo1NPass(Logger log) {
    return std::make_unique<ConvertBatchedLayerTo1NPass>(log);
}
