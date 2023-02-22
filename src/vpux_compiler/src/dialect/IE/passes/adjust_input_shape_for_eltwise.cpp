//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <numeric>

using namespace vpux;
using namespace IE;
namespace {

//
// AdjustInputShapeForEltwisePass
//
class AdjustInputShapeForEltwisePass final : public AdjustInputShapeForEltwiseBase<AdjustInputShapeForEltwisePass> {
public:
    explicit AdjustInputShapeForEltwisePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }
    mlir::LogicalResult adjustInputShape(mlir::Operation* origOp);

private:
    void safeRunOnFunc() final;
};

//
// ExpandEltwisePattern
//

class ExpandEltwisePattern {
public:
    ExpandEltwisePattern(mlir::Operation* eltwiseOp, Logger log): _eltwiseOp(eltwiseOp), _log(log) {
    }

    bool init();

private:
    mlir::Operation* _eltwiseOp;
    mlir::DenseSet<IE::ExpandOp> _expandInputs{};
    mlir::DenseSet<Const::DeclareOp> _constInputs{};
    mlir::DenseSet<mlir::Operation*> _nonExpandInputs{};
    mlir::DenseSet<IE::SliceOp> _sliceOutputs{};
    mlir::DenseSet<mlir::Operation*> _nonSliceOutputs{};
    Shape _unExpandedShape;
    Shape _newExpandedShape;
    Logger _log;

    void checkAndCorrectGroupConv();

public:
    bool opCostReduced();
    mlir::LogicalResult rewrite();
};

mlir::FailureOr<int64_t> getRealDataSize(Const::DeclareOp constOp) {
    if (constOp == nullptr) {
        return mlir::failure();
    }
    auto contentAttr = constOp.contentAttr();
    if (contentAttr == nullptr) {
        return mlir::failure();
    }
    auto baseContent = contentAttr.getBaseContent();

    if (auto denseBaseAttr = baseContent.dyn_cast<mlir::DenseElementsAttr>()) {
        return denseBaseAttr.getType().getNumElements();
    } else if (auto opaqueBaseAttr = baseContent.dyn_cast<mlir::OpaqueElementsAttr>()) {
        return opaqueBaseAttr.getType().getNumElements();
    } else {
        VPUX_THROW("Got unsupported 'baseContent' in 'ContentAttr'");
    }
}

bool isDimExpansionReshape(ShapeRef origShape, ShapeRef reshapeShape) {
    auto getNonOneDims = [](ShapeRef shape) {
        Shape resultShape;
        llvm::copy_if(shape, std::back_inserter(resultShape), [](int64_t elem) {
            return elem != 1;
        });
        return resultShape;
    };
    return getNonOneDims(origShape) == getNonOneDims(reshapeShape);
}

bool hasSingleUserSliceOp(mlir::Operation* eltwiseOp) {
    if (eltwiseOp->hasOneUse()) {
        auto user = *(eltwiseOp->getUsers().begin());
        auto sliceOp = mlir::dyn_cast<IE::SliceOp>(user);
        return sliceOp != nullptr;
    }
    return false;
}

void ExpandEltwisePattern::checkAndCorrectGroupConv() {
    auto groupConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(_eltwiseOp);
    if (groupConvOp == nullptr) {
        return;
    }
    auto groupSize = groupConvOp.groupsAttr().getInt();
    if (groupSize == _newExpandedShape[Dims4D::Act::C]) {
        return;
    }
    mlir::OpBuilder builder(_eltwiseOp);
    auto ctx = builder.getContext();
    auto newGroupAttr = getIntAttr(ctx, _newExpandedShape[Dims4D::Act::C]);
    auto newGroupConvOp = builder.create<IE::GroupConvolutionOp>(
            groupConvOp->getLoc(), groupConvOp.input(), groupConvOp.filter(), groupConvOp.bias(),
            groupConvOp.stridesAttr(), groupConvOp.pads_beginAttr(), groupConvOp.pads_end(),
            groupConvOp.dilationsAttr(), newGroupAttr, groupConvOp.post_opAttr());
    groupConvOp->replaceAllUsesWith(newGroupConvOp);
    auto origOutputType = groupConvOp.getType().cast<vpux::NDTypeInterface>();
    newGroupConvOp.output().setType(origOutputType.changeShape(_newExpandedShape));
    _eltwiseOp = newGroupConvOp.getOperation();
    return;
}

/* Try to match the Expand-Eltwise patterns
    Expand     Expand
      |          |
       \        /
         Eltwise
            |
      Slice (optional)

or:

   Expand     Expand
     |          |
QuantizeCast  QuantizeCast
       \        /
         Eltwise
            |
      Slice (optional)
*/
bool ExpandEltwisePattern::init() {
    auto log = _log.nest();
    // only support eltwise ops with same input and output layouts
    auto eltwiseOutputType = _eltwiseOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto uniformQElemType = eltwiseOutputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
    auto eltwiseOutputLayout = eltwiseOutputType.getDimsOrder();
    for (auto operand : _eltwiseOp->getOperands()) {
        if (operand.getDefiningOp() && mlir::isa<Const::DeclareOp>(operand.getDefiningOp())) {
            continue;
        }

        auto inputType = operand.getType().cast<vpux::NDTypeInterface>();
        auto isF16 = inputType.getElementType().isF16();
        if (isF16 && uniformQElemType && !hasSingleUserSliceOp(_eltwiseOp)) {
            return false;
        }

        auto inputLayout = inputType.getDimsOrder();
        if (inputLayout != eltwiseOutputLayout) {
            _log.trace("Unsupported eltwise input and output layout");
            return false;
        }
    }
    // match input expands and non-expands
    for (auto operand : _eltwiseOp->getOperands()) {
        if (auto expand = operand.getDefiningOp<IE::ExpandOp>()) {
            _expandInputs.insert(expand);
        } else if (auto quantCast = operand.getDefiningOp<IE::QuantizeCastOp>()) {
            auto prevExpand = quantCast.input().getDefiningOp<IE::ExpandOp>();
            if (prevExpand) {
                _expandInputs.insert(prevExpand);
            } else {
                _nonExpandInputs.insert(operand.getDefiningOp());
            }
        } else if (auto constDeclare = operand.getDefiningOp<Const::DeclareOp>()) {
            _constInputs.insert(constDeclare);
        } else {
            _nonExpandInputs.insert(operand.getDefiningOp());
        }
    }
    log.trace("{0} Expand input(s) and {1} Const with Expand input(s) found", _expandInputs.size(),
              _constInputs.size());
    if (_expandInputs.empty()) {
        log.trace("Cannot find any input ExpandOp");
        return false;
    }

    // match output slices or non-slices
    for (auto user : _eltwiseOp->getResult(0).getUsers()) {
        if (auto slice = mlir::dyn_cast<IE::SliceOp>(user)) {
            _sliceOutputs.insert(slice);
        } else {
            _nonSliceOutputs.insert(user);
        }
    }
    log.trace("{0} Slice output(s) found", _sliceOutputs.size());

    // save the original shape and generate new shape
    auto expandInputOp = *_expandInputs.begin();
    _unExpandedShape = expandInputOp.input().getType().cast<vpux::NDTypeInterface>().getShape().toValues();
    for (auto expandInput : llvm::drop_begin(_expandInputs)) {
        auto otherExpandInput = expandInput.input().getType().cast<vpux::NDTypeInterface>().getShape().toValues();
        if (otherExpandInput != _unExpandedShape) {
            log.trace("The ExpandOp's input shapes are not equal, {0} and {1} separately, not supported",
                      otherExpandInput, _unExpandedShape);
            return false;
        }
    }

    auto activationDataSize =
            std::accumulate(_unExpandedShape.begin(), _unExpandedShape.end(), int64_t(1), std::multiplies<int64_t>());
    for (auto constDeclare : _constInputs) {
        auto realDataSizeResult = getRealDataSize(constDeclare);
        if (mlir::failed(realDataSizeResult) ||
            (realDataSizeResult.getValue() != 1 && realDataSizeResult.getValue() != activationDataSize)) {
            log.trace("Unsupported const input {0} at {1}", constDeclare->getName(), constDeclare->getLoc());
            return false;
        }
    }

    auto newExpandedShapeResult = getShapeCastExpandedShape(_eltwiseOp, getShape(_eltwiseOp->getOperand(0)).toValues(),
                                                            _unExpandedShape, _log.nest());
    if (mlir::failed(newExpandedShapeResult)) {
        return false;
    }
    _newExpandedShape = newExpandedShapeResult.getValue();
    return true;
}

bool ExpandEltwisePattern::opCostReduced() {
    // check 1: all inputs are ExpandOp
    if (_nonExpandInputs.size() > 0) {
        _log.trace("{0} input op(s) are not ExpandOp", _nonExpandInputs.size());
        return false;
    }

    // check 2: when any of the expands to reduce is u8, the newly added expand cannot be fp16
    auto quantInputExpandExist = llvm::any_of(_expandInputs, [&](IE::ExpandOp expand) {
        auto outputType = expand.output().getType().cast<vpux::NDTypeInterface>();
        return outputType.getElementType().isUnsignedInteger(8);
    });
    auto floatOutputExpandToAdd = llvm::any_of(_nonSliceOutputs, [&](mlir::Operation* op) {
        auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        return inputType.getElementType().isa<mlir::FloatType>();
    });
    if (quantInputExpandExist && floatOutputExpandToAdd) {
        _log.trace("U8 Expand to reduce but float Expand to add. Expand cost will increase");
        return false;
    }
    return true;
}

/* Rewrite the pattern from:
                                        Const filter (Const bias)
   Expand      Expand          Expand    (1 elem)    (1 elem)
      |          |                  |       |         |
       \        /                    \      |       /
         Eltwise        or               GroupConv
            |                               |
      Slice (optional)               Slice (optional)

    to:                                  Const filter (Const bias)
  ShapeCast    ShapeCast        ShapeCast (broadcast) (broadcast)
      |          |                   |        |        |
       \        /                     \       |       /
         Eltwise                          GroupConv
            |                                 |
        ShapeCast                         ShapeCast
            |                                 |
          Expand                           Expand
            |                                 |
      Slice (optional)                 Slice (optional)
 */
mlir::LogicalResult ExpandEltwisePattern::rewrite() {
    mlir::OpBuilder builder(_eltwiseOp);
    auto ctx = builder.getContext();

    _log.trace("Converting unexpanded shape {0} to new aligned shape {1}", _unExpandedShape, _newExpandedShape);
    // Replace input Expands with ShapeCasts
    for (auto expand : _expandInputs) {
        auto inputValue = expand.input();
        auto inputType = inputValue.getType().cast<vpux::NDTypeInterface>();
        builder.setInsertionPointAfter(expand);
        auto inputShapeCastOp =
                builder.create<IE::ShapeCastOp>(_eltwiseOp->getLoc(), inputType.changeShape(_newExpandedShape),
                                                inputValue, getIntArrayAttr(ctx, _newExpandedShape.raw()));
        auto getOwnerIgnoreQuantizeCast = [&](mlir::OpOperand& opOperand) -> mlir::Operation* {
            auto ownerOp = opOperand.getOwner();
            while (auto quantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(ownerOp)) {
                auto quantizeUsers = quantizeCastOp.output().getUsers();
                if (quantizeUsers.empty()) {
                    return ownerOp;
                }
                ownerOp = *quantizeUsers.begin();
            }
            return ownerOp;
        };
        expand.output().replaceUsesWithIf(inputShapeCastOp.result(), [&](mlir::OpOperand& opOperand) {
            // replace only current user uses
            return getOwnerIgnoreQuantizeCast(opOperand) == _eltwiseOp;
        });
        // propagate the shape if QuantCasts exit
        auto innerOp = *inputShapeCastOp.result().getUsers().begin();
        while (innerOp != _eltwiseOp) {
            auto innerOpResult = innerOp->getResult(0);
            auto innerOutputType = innerOpResult.getType().cast<vpux::NDTypeInterface>();
            innerOp->getResult(0).setType(innerOutputType.changeShape(_newExpandedShape));
            if (innerOp->getResult(0).getUsers().empty()) {
                break;
            }
            innerOp = *innerOp->getResult(0).getUsers().begin();
        }
    }

    for (auto constDeclare : _constInputs) {
        auto contentAttr = constDeclare.contentAttr();
        auto baseContent = contentAttr.getBaseContent();
        auto dataShape = getShape(constDeclare.output()).toValues();

        Const::ContentAttr newContentAttr;
        Shape realDataShape;
        if (auto denseBaseAttr = baseContent.dyn_cast<mlir::DenseElementsAttr>()) {
            newContentAttr = Const::ContentAttr::get(denseBaseAttr);
            realDataShape = denseBaseAttr.getType().getShape();
        } else if (auto opaqueBaseAttr = baseContent.dyn_cast<mlir::OpaqueElementsAttr>()) {
            newContentAttr = Const::ContentAttr::get(opaqueBaseAttr);
            realDataShape = opaqueBaseAttr.getType().getShape();
        } else {
            VPUX_THROW("Got unsupported 'baseContent' in 'ContentAttr'");
        }

        auto newConstOutputType = constDeclare.output().getType().cast<vpux::NDTypeInterface>();
        const auto realDataSizeResult = getRealDataSize(constDeclare);
        if (mlir::failed(realDataSizeResult)) {
            return mlir::failure();
        }
        const auto singleValueData = realDataSizeResult.getValue() == 1;
        if (singleValueData) {
            for (auto dim : enumerate(dataShape)) {
                if (dim.value() > 1) {
                    newContentAttr = newContentAttr.broadcast(Dim(dim.index()), _newExpandedShape[Dims4D::Act::C]);
                    auto newConstantShape = Shape(newConstOutputType.getShape().size(), int64_t(1));
                    newConstantShape[Dim(dim.index())] = _newExpandedShape[Dims4D::Act::C];
                    newConstOutputType = newConstOutputType.changeShape(newConstantShape);
                    newContentAttr = newContentAttr.reshape(newConstantShape);
                }
            }
        }
        for (auto attr : contentAttr.getTransformations()) {
            if (attr.isa<Const::PadWithZeroAttr>() || attr.isa<Const::BroadcastAttr>()) {
                // Ignore
                continue;
            }
            if (attr.isa<Const::ReshapeAttr>()) {
                // Only remain the reshape attribute when it's used for dimension expansion to 4D
                // e.g., from [1x512] to [1x1x1x512]
                auto reshapeAttr = attr.cast<Const::ReshapeAttr>();
                auto reshapeShape = Shape(parseIntArrayAttr<int64_t>(reshapeAttr.getShape()));
                if (singleValueData || !isDimExpansionReshape(realDataShape, reshapeShape)) {
                    continue;
                }
            }
            newContentAttr = Const::ContentAttr::addTransformation(newContentAttr, attr);
        }
        if (!singleValueData) {
            newContentAttr = newContentAttr.reshape(_newExpandedShape);
            newConstOutputType = newConstOutputType.changeShape(_newExpandedShape);
        }
        builder.setInsertionPoint(_eltwiseOp);
        auto newConstDeclare =
                builder.create<Const::DeclareOp>(constDeclare.getLoc(), newConstOutputType, newContentAttr);
        constDeclare.output().replaceUsesWithIf(newConstDeclare.output(), [&](mlir::OpOperand& opOperand) {
            return opOperand.getOwner() == _eltwiseOp;
        });
    }

    // Replace the eltwise GroupConv with correct attributes
    checkAndCorrectGroupConv();

    // Insert ShapeCasts and Expands after eltwise ops
    auto outputType = _eltwiseOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    _eltwiseOp->getResult(0).setType(outputType.changeShape(_newExpandedShape));
    builder.setInsertionPointAfter(_eltwiseOp);
    auto outputShapeCastOp =
            builder.create<IE::ShapeCastOp>(_eltwiseOp->getLoc(), outputType.changeShape(_unExpandedShape),
                                            _eltwiseOp->getResult(0), getIntArrayAttr(ctx, _unExpandedShape.raw()));
    auto inputExpandOp = *_expandInputs.begin();
    auto newOutputExpandOp = builder.create<IE::ExpandOp>(_eltwiseOp->getLoc(), outputShapeCastOp.result(),
                                                          inputExpandOp.pads_beginAttr(), inputExpandOp.pads_endAttr());
    _eltwiseOp->getResult(0).replaceAllUsesExcept(newOutputExpandOp.output(), outputShapeCastOp);
    return mlir::success();
}

//
// ExpandEltwiseRewriter
//

template <class EltwiseOp>
class ExpandEltwiseRewriter final : public mlir::OpRewritePattern<EltwiseOp> {
public:
    ExpandEltwiseRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<EltwiseOp>(ctx), _log(log) {
        this->setDebugName("ExpandEltwiseRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(EltwiseOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class EltwiseOp>
mlir::LogicalResult ExpandEltwiseRewriter<EltwiseOp>::matchAndRewrite(EltwiseOp layerOp,
                                                                      mlir::PatternRewriter& /*rewriter*/) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), layerOp->getName(), layerOp->getLoc());
    auto pattern = ExpandEltwisePattern(layerOp.getOperation(), _log);
    if (!pattern.init()) {
        return mlir::failure();
    }
    if (pattern.opCostReduced()) {
        return pattern.rewrite();
    }
    return mlir::failure();
}

//
// ExpandGroupConvRewriter
//

class ExpandGroupConvRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    ExpandGroupConvRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
        setDebugName("ExpandGroupConvRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ExpandGroupConvRewriter::matchAndRewrite(IE::GroupConvolutionOp layerOp,
                                                             mlir::PatternRewriter&) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), layerOp->getName(), layerOp->getLoc());
    mlir::SmallVector<Const::DeclareOp> constInputOps;
    constInputOps.push_back(layerOp.filter().getDefiningOp<Const::DeclareOp>());
    if (layerOp.bias()) {
        constInputOps.push_back(layerOp.bias().getDefiningOp<Const::DeclareOp>());
    }
    // Only support GroupConvolution with constant filter
    // if the GroupConvolution has bias, the bias has to be constant as well
    // the total size of filter constant is 1, as well as the bias, i.e., denseElem.getType().getNumElements() = 1
    // in that case, the GroupConvolution can be considered as an Eltwise
    const auto supportedGroupConvConst = llvm::all_of(constInputOps, [](Const::DeclareOp constOp) {
        auto realDataSizeResult = getRealDataSize(constOp);
        return mlir::succeeded(realDataSizeResult) && realDataSizeResult.getValue() == 1;
    });
    if (!supportedGroupConvConst) {
        return mlir::failure();
    }

    auto pattern = ExpandEltwisePattern(layerOp.getOperation(), _log);
    if (!pattern.init()) {
        return mlir::failure();
    }
    if (pattern.opCostReduced()) {
        return pattern.rewrite();
    }
    return mlir::failure();
}

void AdjustInputShapeForEltwisePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ExpandEltwiseRewriter<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<ExpandEltwiseRewriter<IE::SubtractOp>>(&ctx, _log);
    patterns.add<ExpandEltwiseRewriter<IE::AddOp>>(&ctx, _log);
    patterns.add<ExpandGroupConvRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
        return;
    }
}
}  // namespace

//
// createAdjustInputShapeForEltwisePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustInputShapeForEltwisePass(Logger log) {
    return std::make_unique<AdjustInputShapeForEltwisePass>(log);
}
