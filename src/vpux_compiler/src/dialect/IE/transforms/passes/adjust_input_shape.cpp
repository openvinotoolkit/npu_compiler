//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/expand_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_quantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Operation.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_ADJUSTINPUTSHAPE
#define GEN_PASS_DEF_ADJUSTINPUTSHAPE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;
using namespace IE;

namespace {

const uint32_t levelCount = 2;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

bool checkValidPermuteQuantizePads(IE::PermuteQuantizeOp op) {
    const auto padStart = parseIntArrayAttr<int64_t>(op.getPadsBegin());
    const auto padEnd = parseIntArrayAttr<int64_t>(op.getPadsEnd());

    const auto nonZeroPadStart = llvm::any_of(padStart, [](auto pad) {
        return pad > 0;
    });

    const auto nonZeroPadEnd = llvm::any_of(padEnd, [](auto pad) {
        return pad > 0;
    });

    return !(nonZeroPadStart || nonZeroPadEnd);
}

//
// AdjustInputShapePass
//
class AdjustInputShapePass final : public IE::impl::AdjustInputShapeBase<AdjustInputShapePass> {
public:
    explicit AdjustInputShapePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

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

    Logger getLogger() {
        return _log;
    }

    mlir::Operation* getEltwiseOperation() {
        return _eltwiseOp;
    }

    void addExpandInput(IE::ExpandOp expand) {
        _expandInputs.insert(expand);
    }

    mlir::DenseSet<IE::ExpandOp> getExpandInputs() {
        return _expandInputs;
    }

    size_t getExpandInputsNum() {
        return _expandInputs.size();
    }

    void addSliceOutput(IE::SliceOp slice) {
        _sliceOutputs.insert(slice);
    }

    void addNonSliceOutput(mlir::Operation* op) {
        _nonSliceOutputs.insert(op);
    }

    size_t getSliceOutputsNum() {
        return _sliceOutputs.size();
    }

    void setUnExpandedShape(Shape shape) {
        _unExpandedShape = std::move(shape);
    }

    void setNewExpandedShape(Shape shape) {
        _newExpandedShape = std::move(shape);
    }

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

    void checkAndCorrectGroupConv(mlir::PatternRewriter& rewriter);

public:
    bool opCostReduced();
    mlir::LogicalResult rewrite(mlir::PatternRewriter& rewriter);
};

void ExpandEltwisePattern::checkAndCorrectGroupConv(mlir::PatternRewriter& rewriter) {
    auto groupConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(_eltwiseOp);
    if (groupConvOp == nullptr) {
        return;
    }
    auto groupSize = groupConvOp.getGroupsAttr().getInt();
    if (groupSize == _newExpandedShape[Dims4D::Act::C]) {
        return;
    }

    auto ctx = rewriter.getContext();
    auto newGroupAttr = getIntAttr(ctx, _newExpandedShape[Dims4D::Act::C]);
    auto newGroupConvOp = rewriter.create<IE::GroupConvolutionOp>(
            groupConvOp->getLoc(), groupConvOp.getInput(), groupConvOp.getFilter(), groupConvOp.getBias(),
            groupConvOp.getStridesAttr(), groupConvOp.getPadsBeginAttr(), groupConvOp.getPadsEnd(),
            groupConvOp.getDilationsAttr(), newGroupAttr, groupConvOp.getPostOpAttr(), groupConvOp.getClampAttr(),
            /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);
    groupConvOp->replaceAllUsesWith(newGroupConvOp);
    auto origOutputType = mlir::cast<vpux::NDTypeInterface>(groupConvOp.getType());
    newGroupConvOp.getOutput().setType(
            mlir::cast<mlir::RankedTensorType>(origOutputType.changeShape(_newExpandedShape)));
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
    auto eltwiseOutputLayout = mlir::cast<vpux::NDTypeInterface>(_eltwiseOp->getResult(0).getType()).getDimsOrder();
    for (auto operand : _eltwiseOp->getOperands()) {
        if (mlir::isa_and_nonnull<Const::DeclareOp>(operand.getDefiningOp())) {
            continue;
        }

        auto inputLayout = mlir::cast<vpux::NDTypeInterface>(operand.getType()).getDimsOrder();
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
            auto prevExpand = quantCast.getInput().getDefiningOp<IE::ExpandOp>();

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
    log.trace("{0} Expand setInput(s) and {1} Const with Expand input(s) found", _expandInputs.size(),
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
    log.trace("{0} Slice setOutput(s) found", _sliceOutputs.size());

    /* If user is AlignedChannelsOpInterface, the op always need channel alignment, so expand is needed after reshape
       For example, if the pattern like:
        Expand     channel_aligned_input
           |         |
           \         /
             Eltwise
            /       \
           Op1      Op2
      to:
        ShapeCast   channel_aligned_input
           |          |
           |       Slice
           \         /
             Eltwise
               |
             Expand
             /    \
            Op1   Op2
       Here will introduce an extra slice
    */
    if (!_expandInputs.empty() && !_nonExpandInputs.empty() && _sliceOutputs.empty()) {
        // If there is no slice output(s), that means no need expand after shapecast. The exception is if users are
        // eltwise, like expand -> add -> add, continue the optimization.
        for (auto user : _eltwiseOp->getResult(0).getUsers()) {
            auto groupConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(user);
            if (!mlir::isa<IE::MultiplyOp, IE::SubtractOp, IE::AddOp>(user) && !isEltwiseGroupConv(groupConvOp)) {
                return false;
            }
        }
    }

    // save the original shape and generate new shape
    auto expandInputOp = *_expandInputs.begin();
    _unExpandedShape = getBoundedShape(expandInputOp.getInput().getType()).toValues();

    for (auto expandInput : llvm::drop_begin(_expandInputs)) {
        auto otherExpandInput =
                mlir::cast<vpux::NDTypeInterface>(expandInput.getInput().getType()).getShape().toValues();
        if (otherExpandInput != _unExpandedShape) {
            log.trace("The ExpandOp's input shapes are not equal, {0} and {1} separately, not supported",
                      otherExpandInput, _unExpandedShape);
            return false;
        }
    }

    // Only IE::MultiplyOp, IE::SubtractOp, IE::AddOp and IE::GroupConvolutionOp has constant input
    for (auto constDeclare : _constInputs) {
        auto baseContentNum = IE::getBaseContentNumElements(constDeclare);
        if (mlir::failed(baseContentNum)) {
            log.trace("Unsupported const of {0} at {1}", _eltwiseOp->getName(), _eltwiseOp->getLoc());
            return false;
        }

        // Only support two kinds of constant input for IE::MultiplyOp, IE::SubtractOp, IE::AddOp
        // 1. Constant input baseContentNum == 1
        //  - It can be `broadcast` or `reshape` to any shape size
        //  For example: input 1: "tensor<1x3x32x32xf16>", input 2: "dense<1.0> : tensor<1x1x1x1xf16>"
        // 2. Constant input without last padWithZero == unExpand activation
        if (mlir::isa<IE::MultiplyOp, IE::SubtractOp, IE::AddOp>(_eltwiseOp) && baseContentNum.value() != 1) {
            const auto& contentAttr = constDeclare.getContentAttr();
            if (contentAttr.getTransformations().empty()) {
                return false;
            }
            auto lastAttr = contentAttr.getTransformations().back();
            auto padWithZeroAttr = mlir::dyn_cast_or_null<vpux::Const::PadWithZeroAttr>(lastAttr);
            if (padWithZeroAttr == nullptr) {
                return false;
            }
            auto expand = *_expandInputs.begin();
            const auto expandPadsBegin = parseIntArrayAttr<int64_t>(expand.getPadsBegin());
            const auto expandPadsEnd = parseIntArrayAttr<int64_t>(expand.getPadsEnd());
            const auto padZeroAttrPadsBegin = parseIntArrayAttr<int64_t>(padWithZeroAttr.getPadBefore());
            const auto padZeroAttrPadsEnd = parseIntArrayAttr<int64_t>(padWithZeroAttr.getPadAfter());
            if (expandPadsBegin != padZeroAttrPadsBegin || expandPadsEnd != padZeroAttrPadsEnd) {
                return false;
            }
        }

        // Only support two kinds of constant input for IE::GroupConvolutionOp
        // 1. Constant Weights/Bias baseContentNum == 1
        //  - It can be `broadcast` or `reshape` to any shape size
        //  For example: Activation: "tensor<1x3x32x32xf16>", Weights: "dense<1.0> : tensor<1x1x1x1xf16>"
        // 2. Constant Weights/Bias baseContentNum > 1, but all the element has the same value
        //  - It can be considered as the first case.
        //    After slice with single baseContentNum it can be `broadcast` or `reshape` to any shape size
        //  For example: Activation: "tensor<1x3x32x32xf16>", Weights: "dense<1.0> : tensor<3x1x1x1xf16>"
        if (mlir::isa<IE::GroupConvolutionOp>(_eltwiseOp) && !IE::isBaseContentSplat(constDeclare)) {
            log.trace("Unsupported {0} at {1} with input constant isn't single value", _eltwiseOp->getName(),
                      _eltwiseOp->getLoc());
            return false;
        }
    }
    auto expandedShape = getShape(_eltwiseOp->getOperand(0));
    auto onlyExpandOnDimC = llvm::all_of(irange(expandedShape.size()), [&](auto idx) {
        if (Dim(idx) == Dims4D::Act::C) {
            return true;
        }
        return expandedShape[Dim(idx)] == _unExpandedShape[Dim(idx)];
    });

    auto hasEfficientWorkload = [&]() {
        auto moduleOp = _eltwiseOp->getParentOfType<mlir::ModuleOp>();
        const auto numCluster = config::getTileExecutor(moduleOp).getCount();
        VPUX_THROW_WHEN(numCluster <= 0, "Number of clusters should be a positive integer, while it is {0}",
                        numCluster);

        const auto inputType = mlir::cast<vpux::NDTypeInterface>(_eltwiseOp->getOperand(0).getType());
        const auto sizeToAlign = VPU::NCEInvariant::getAlignment(inputType.getElementType());
        auto dimSizeForWC = _unExpandedShape[Dims4D::Act::W] * _unExpandedShape[Dims4D::Act::C];
        if (dimSizeForWC % sizeToAlign != 0) {
            return false;
        }

        return _unExpandedShape[Dims4D::Act::H] % VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT == 0 &&
               (_unExpandedShape[Dims4D::Act::H] / VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT) >= numCluster &&
               (dimSizeForWC / sizeToAlign) % VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT == 0 &&
               sizeToAlign <= _unExpandedShape[Dims4D::Act::H] * dimSizeForWC / sizeToAlign;
    };

    auto isLargeOpNeedTiling = [&]() {
        auto module = _eltwiseOp->getParentOfType<mlir::ModuleOp>();
        const auto numClusters = config::getTileExecutor(module).getCount();
        const auto availableCMXSizePerCluster = vpux::VPU::getTotalCMXSize(_eltwiseOp).count();
        const auto totalAvailableCMXSize = availableCMXSizePerCluster * numClusters;
        SmallVector<Byte> buffSizes;
        for (auto operand : _eltwiseOp->getOperands()) {
            auto inType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
            buffSizes.push_back(_unExpandedShape.totalSize() * inType.getElemTypeSize());
        }

        for (auto output : _eltwiseOp->getResults()) {
            auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());
            buffSizes.push_back(_unExpandedShape.totalSize() * outputType.getElemTypeSize());
        }

        const auto arch = config::getArch(_eltwiseOp);
        auto requiredCMXSize = vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffSizes).count();
        return requiredCMXSize > totalAvailableCMXSize;
    };

    auto newExpandedShapeResult =
            getShapeCastExpandedShape(_eltwiseOp, getShape(_eltwiseOp->getOperand(0)), _unExpandedShape, _log.nest());

    if (onlyExpandOnDimC && hasEfficientWorkload() && isLargeOpNeedTiling()) {
        auto _newExpandedShapeWithMinimalDimChange =
                getShapeCastExpandedShapeWithMinimalDimChange(_eltwiseOp, _unExpandedShape, _log.nest());
        if (mlir::succeeded(_newExpandedShapeWithMinimalDimChange)) {
            newExpandedShapeResult = _newExpandedShapeWithMinimalDimChange.value();
        }
    }

    if (mlir::failed(newExpandedShapeResult)) {
        return false;
    }
    _newExpandedShape = newExpandedShapeResult.value();

    // If it is legal to adjust the eltwise shape to avoid expansion
    // there is a chance to avoid spilling by keeping the same shape as the producer
    // A typical pattern: AlignedChannelsOp -> ViewLikeOp -> ExpandOp (channel) -> EltwiseOp
    for (auto input : _expandInputs) {
        auto producerOp = input.getInput().getDefiningOp();
        while (mlir::isa_and_nonnull<IE::ViewLikeOpInterface>(producerOp) && producerOp->getResult(0).hasOneUse()) {
            producerOp = producerOp->getOperand(0).getDefiningOp();
        }

        if (mlir::isa_and_nonnull<IE::AlignedChannelsOpInterface>(producerOp)) {
            auto outType = mlir::cast<NDTypeInterface>(producerOp->getResult(0).getType());
            auto outShape = outType.getShape();

            auto alignIface = mlir::cast<IE::AlignedChannelsOpInterface>(_eltwiseOp);
            const auto sizeToAlignChannel = alignIface.getInputChannelAlignment();
            auto module = producerOp->getParentOfType<mlir::ModuleOp>();
            const auto numCluster = config::getTileExecutor(module).getCount();
            VPUX_THROW_WHEN(numCluster <= 0, "Number of clusters should be a positive integer, while it is {0}",
                            numCluster);
            // There are two restrictions to ensure this updated shape is always the best solution:
            // 1. The shape meets functional requirements:
            //    Since operations with AlignedChannelsOpInterface are not always NCE tasks and vary with the platform
            //    the shape should be checked with a 4D tensor where N equals 1 and channels are aligned
            // 2. Workload efficiency:
            //    - Ensure H is evenly divisible by "number of clusters * VPU_SPATIAL_ALIGNMENT" to avoid uneven splits
            //    - Ensure W is evenly divisible by "VPU_SPATIAL_ALIGNMENT" to avoid inefficient workloads
            //    - Ensure the spatial size (H * W) is larger than the channel size
            bool isShapeFunctional = outShape.size() == 4 && outShape[Dims4D::Act::N] == 1 &&
                                     outShape[Dims4D::Act::C] % sizeToAlignChannel == 0;
            bool isWorkloadEfficient =
                    outShape[Dims4D::Act::H] % (numCluster * VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT) == 0 &&
                    outShape[Dims4D::Act::W] % VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT == 0 &&
                    outShape[Dims4D::Act::C] <= outShape[Dims4D::Act::H] * outShape[Dims4D::Act::W];

            if (isShapeFunctional && isWorkloadEfficient) {
                _newExpandedShape = Shape(outShape.raw());
                break;
            }
        }
    }

    return true;
}

bool ExpandEltwisePattern::opCostReduced() {
    // check 1: all inputs are ExpandOp
    const auto isTwoInputsOp = mlir::isa<IE::MultiplyOp, IE::SubtractOp, IE::AddOp>(_eltwiseOp);
    int64_t numNonExpandInputs = isTwoInputsOp ? 1 : 0;

    if (_nonExpandInputs.size() > numNonExpandInputs) {
        _log.trace("{0} input op(s) are not ExpandOp", _nonExpandInputs.size());
        return false;
    }

    // check 2: when any of the expands to reduce is u8, the newly added expand cannot be fp16
    auto quantInputExpandExist = llvm::any_of(_expandInputs, [&](IE::ExpandOp expand) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(expand.getOutput().getType());
        return outputType.getElementType().isUnsignedInteger(8);
    });
    auto floatOutputExpandToAdd = llvm::any_of(_nonSliceOutputs, [&](mlir::Operation* op) {
        auto inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
        return mlir::isa<mlir::FloatType>(inputType.getElementType());
    });
    if (quantInputExpandExist && floatOutputExpandToAdd) {
        _log.trace("U8 Expand to reduce but float Expand to add. Expand cost will increase");
        return false;
    }
    return true;
}

/* Rewrite the pattern from:
                                            Const filter (Const bias)
   Expand      Expand (optional)    Expand    (1 elem)    (1 elem)
      |          |                      |       |         |
       \        /                        \      |        /
         Eltwise                 or         GroupConv
            |                                   |
      Slice (optional)                  Slice (optional)

    to:
               Slice (optional)
                 |                          Const filter (Const bias)
  ShapeCast    ShapeCast            ShapeCast (broadcast) (broadcast)
      |          |                      |        |        |
       \        /                        \       |       /
         Eltwise                             GroupConv
            |                                    |
        ShapeCast                            ShapeCast
            |                                    |
          Expand                               Expand
            |                                     |
      Slice (optional)                      Slice (optional)
 */
mlir::LogicalResult ExpandEltwisePattern::rewrite(mlir::PatternRewriter& rewriter) {
    auto ctx = rewriter.getContext();

    _log.trace("Converting unexpanded shape {0} to new aligned shape {1}", _unExpandedShape, _newExpandedShape);

    if (isPerAxisQuant(_eltwiseOp->getResult(0))) {
        _log.trace("Per axis quantization is not supported for replace by ShapeCast");
        return mlir::failure();
    }

    auto getOwnerIgnoreQuantizeCast = [&](mlir::OpOperand& opOperand) -> mlir::Operation* {
        auto ownerOp = opOperand.getOwner();
        while (auto quantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(ownerOp)) {
            auto quantizeUsers = quantizeCastOp.getOutput().getUsers();

            if (quantizeUsers.empty()) {
                return ownerOp;
            }
            ownerOp = *quantizeUsers.begin();
        }
        return ownerOp;
    };

    // Insert slice for non Expand input
    const auto expandInputType = mlir::cast<vpux::NDTypeInterface>((*_expandInputs.begin()).getInput().getType());
    const auto sliceOffset = parseIntArrayAttr<int64_t>((*_expandInputs.begin()).getPadsBeginAttr());
    for (auto nonExpand : _nonExpandInputs) {
        if (nonExpand == nullptr) {
            return mlir::failure();
        }

        rewriter.setInsertionPointAfter(nonExpand);
        auto inputSliceOp = rewriter.create<IE::SliceOp>(_eltwiseOp->getLoc(), nonExpand->getResult(0),
                                                         getIntArrayAttr(ctx, sliceOffset),
                                                         getIntArrayAttr(ctx, expandInputType.getShape().raw()));
        auto inputShapeCastOp = rewriter.create<IE::ShapeCastOp>(_eltwiseOp->getLoc(), inputSliceOp.getResult(),
                                                                 getIntArrayAttr(ctx, _newExpandedShape.raw()));
        nonExpand->getResult(0).replaceUsesWithIf(inputShapeCastOp.getResult(), [&](mlir::OpOperand& opOperand) {
            return getOwnerIgnoreQuantizeCast(opOperand) == _eltwiseOp;
        });
    }

    // Replace input Expands with ShapeCasts
    for (auto expand : _expandInputs) {
        auto inputValue = expand.getInput();
        auto inputType = mlir::cast<vpux::NDTypeInterface>(inputValue.getType());
        rewriter.setInsertionPointAfter(expand);
        auto inputShapeCastOp =
                rewriter.create<IE::ShapeCastOp>(_eltwiseOp->getLoc(), inputType.changeShape(_newExpandedShape),
                                                 inputValue, getIntArrayAttr(ctx, _newExpandedShape.raw()));

        expand.getOutput().replaceUsesWithIf(inputShapeCastOp.getResult(), [&](mlir::OpOperand& opOperand) {
            // replace only current user uses
            return getOwnerIgnoreQuantizeCast(opOperand) == _eltwiseOp;
        });
        // propagate the shape if QuantCasts exit
        auto innerOp = *inputShapeCastOp.getResult().getUsers().begin();
        while (innerOp != _eltwiseOp) {
            auto innerOpResult = innerOp->getResult(0);
            auto innerOutputType = mlir::cast<vpux::NDTypeInterface>(innerOpResult.getType());

            innerOp->getResult(0).setType(innerOutputType.changeShape(_newExpandedShape));
            if (innerOp->getResult(0).getUsers().empty()) {
                break;
            }
            innerOp = *innerOp->getResult(0).getUsers().begin();
        }
    }

    // Only support IE::MultiplyOp, IE::SubtractOp, IE::AddOp and IE::GroupConvolutionOp has constant input
    const auto opsCanHaveConstInput =
            mlir::isa<IE::MultiplyOp, IE::SubtractOp, IE::AddOp, IE::GroupConvolutionOp>(_eltwiseOp);
    VPUX_THROW_WHEN(!_constInputs.empty() && !opsCanHaveConstInput,
                    "Unexpect Op {0} at {1} has constant input. Cannot ensure it has right reshape logic.",
                    _eltwiseOp->getName(), _eltwiseOp->getLoc());
    for (auto constDeclare : _constInputs) {
        const auto& contentAttr = constDeclare.getContentAttr();
        Const::ContentAttr newContentAttr;

        auto newConstOutputType = mlir::cast<vpux::NDTypeInterface>(constDeclare.getOutput().getType());
        // For IE::MultiplyOp, IE::SubtractOp, IE::AddOp, we just undo expand by adding subview and then reshape
        if (mlir::isa<IE::MultiplyOp, IE::SubtractOp, IE::AddOp>(_eltwiseOp)) {
            const auto subOffset = Shape(_unExpandedShape.size(), int64_t(0));
            newContentAttr =
                    contentAttr.transform().subview(subOffset, _unExpandedShape).reshape(_newExpandedShape).get();
            newConstOutputType = newConstOutputType.changeShape(_newExpandedShape);
        }
        // Only support two kinds of constant input for IE::GroupConvolutionOp
        // 1. Constant Weights/Bias baseContentNum == 1
        //  - First broadcast, then reshape to target shape size
        //  For example: Activation: "tensor<1x3x32x32xf16>", Weights: "dense<1.0> : tensor<1x1x1x1xf16>"
        //  New Weights Attr: [#const.Broadcast<1 : i64, 16 : i64>, #const.Reshape<[1, 16, 1, 1]>]
        // 2. Constant Weights/Bias baseContentNum > 1, but all the element has the same value
        //  - First slice with single baseContentNum, then it can be `broadcast` or `reshape` as first case
        //  For example: Activation: "tensor<1x3x32x32xf16>", Weights: "dense<1.0> : tensor<3x1x1x1xf16>"
        //  New Weights Attr: [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 1]>, #const.Broadcast<0 : i64, 16 : i64>,
        //                     #const.Reshape<[16, 1, 1, 1]>]
        if (mlir::isa<IE::GroupConvolutionOp>(_eltwiseOp)) {
            // "const.pad" and "const.broadcast" should be removed. It will update with the new rule.
            // The remaining attribution, such as "const.Reorder", should keep the same.
            // To avoid conflict between "const.Reshape" and "const.broadcast", set the "const.Reshape" with ones
            // and update shape with "const.broadcast".
            const auto& baseContent = contentAttr.getBaseContent();
            auto baseType = baseContent.getType();
            Const::ContentSetup newContentAttrSetup(baseType);
            auto numDims = newConstOutputType.getShape().size();
            auto newConstantShape = Shape(numDims, int64_t(1));
            auto constType = mlir::cast<NDTypeInterface>(baseType);
            for (auto attr : contentAttr.getTransformations()) {
                constType = attr.inferOutputType(constType);
                if (!mlir::isa<vpux::Const::PadWithZeroAttr, vpux::Const::BroadcastAttr, vpux::Const::ReshapeAttr,
                               vpux::Const::AffineReshapeAttr>(attr)) {
                    newContentAttrSetup = newContentAttrSetup.addTransformation(attr);
                }
                if (mlir::isa<vpux::Const::ReshapeAttr, vpux::Const::AffineReshapeAttr>(attr)) {
                    newContentAttrSetup = newContentAttrSetup.reshape(newConstantShape);
                }
            }

            const Shape baseContentShape = baseContent.getShapedType().getShape();
            auto baseContentNum = IE::getBaseContentNumElements(constDeclare);
            VPUX_THROW_WHEN(mlir::failed(baseContentNum), "Cannot get baseContentNum");

            if (baseContentNum.value() > 1) {
                const auto subOffset = Shape(newConstOutputType.getShape().size(), int64_t(0));
                const auto subShape = Shape(newConstOutputType.getShape().size(), int64_t(1));
                newContentAttrSetup = newContentAttrSetup.subview(subOffset, subShape);
            }

            auto constOutShape = getShape(constDeclare.getOutput()).toValues();
            const auto isLargerThanOne = [](const int64_t dimSize) -> bool {
                return dimSize > 1;
            };
            VPUX_THROW_UNLESS(std::count_if(constOutShape.begin(), constOutShape.end(), isLargerThanOne) == 1 &&
                                      (constOutShape[Dims4D::Act::N] > 1 || constOutShape[Dims4D::Act::C] > 1),
                              "Unexpect constant for GroupConvOp");

            // Weights should only output channel (Dims4D::Act::N) larger than one. e.g. 16x1x1x1xfp16
            // Bias should only channel (Dims4D::Act::C) larger than one. e.g. 1x16x1x1xfp16
            const auto broadcastDim = constOutShape[Dims4D::Act::N] > 1 ? Dims4D::Act::N : Dims4D::Act::C;
            newConstantShape[broadcastDim] = _newExpandedShape[Dims4D::Act::C];
            newConstOutputType = newConstOutputType.changeShape(newConstantShape);
            newContentAttr = Const::ContentAttr::get(
                    baseContent, newContentAttrSetup.broadcast(broadcastDim, _newExpandedShape[Dims4D::Act::C])
                                         .reshape(newConstantShape));
        }
        rewriter.setInsertionPoint(_eltwiseOp);
        auto newConstDeclare =
                rewriter.create<Const::DeclareOp>(constDeclare.getLoc(), newConstOutputType, std::move(newContentAttr));
        constDeclare.getOutput().replaceUsesWithIf(newConstDeclare.getOutput(), [&](mlir::OpOperand& opOperand) {
            return opOperand.getOwner() == _eltwiseOp;
        });
    }

    // Replace the eltwise GroupConv with correct attributes
    checkAndCorrectGroupConv(rewriter);

    // Insert ShapeCasts and Expands after eltwise ops
    auto outputType = mlir::cast<vpux::NDTypeInterface>(_eltwiseOp->getResult(0).getType());

    if (_eltwiseOp->hasAttr(VPU::INPUT_PADDING_ATTR_NAME)) {
        rewriter.modifyOpInPlace(_eltwiseOp, [&] {
            _eltwiseOp->removeAttr(VPU::INPUT_PADDING_ATTR_NAME);
        });
    }
    if (_eltwiseOp->hasAttr(VPU::OUTPUT_PADDING_ATTR_NAME)) {
        rewriter.modifyOpInPlace(_eltwiseOp, [&] {
            _eltwiseOp->removeAttr(VPU::OUTPUT_PADDING_ATTR_NAME);
        });
    }

    _eltwiseOp->getResult(0).setType(outputType.changeShape(_newExpandedShape));
    rewriter.setInsertionPointAfter(_eltwiseOp);
    auto outputShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(_eltwiseOp->getLoc(), outputType.changeShape(_unExpandedShape),
                                             _eltwiseOp->getResult(0), getIntArrayAttr(ctx, _unExpandedShape.raw()));

    auto inputExpandOp = *_expandInputs.begin();
    auto newOutputExpandOp =
            rewriter.create<IE::ExpandOp>(_eltwiseOp->getLoc(), outputShapeCastOp.getResult(),
                                          inputExpandOp.getPadsBeginAttr(), inputExpandOp.getPadsEndAttr());
    _eltwiseOp->getResult(0).replaceAllUsesExcept(newOutputExpandOp.getOutput(), outputShapeCastOp);

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
                                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), layerOp->getName(), layerOp->getLoc());
    auto pattern = ExpandEltwisePattern(layerOp.getOperation(), _log);
    if (!pattern.init()) {
        return mlir::failure();
    }
    if (pattern.opCostReduced()) {
        return pattern.rewrite(rewriter);
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
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), layerOp->getName(), layerOp->getLoc());
    // Only support GroupConvolution with constant filter
    // if the GroupConvolution has bias, the bias has to be constant as well
    // the filter constant must be with single same value, as well as the bias.
    // even if the BaseContent total size larger than 1.
    // Kernel size and Stride size must be 1x1, and must be a depthwise convolution.
    // in that case, the GroupConvolution can be considered as an Eltwise
    if (!isEltwiseGroupConv(layerOp)) {
        return mlir::failure();
    }

    auto pattern = ExpandEltwisePattern(layerOp.getOperation(), _log);
    if (!pattern.init()) {
        return mlir::failure();
    }
    if (pattern.opCostReduced()) {
        return pattern.rewrite(rewriter);
    }
    return mlir::failure();
}

//
// ExpandPoolingPattern
//

class ExpandPoolingPattern : public ExpandEltwisePattern {
public:
    ExpandPoolingPattern(mlir::Operation* pooling, Logger log): ExpandEltwisePattern(pooling, log) {
    }

    // Overwrite ExpandEltwisePattern::init()
    bool init();
};

/* Try to match the Expand-pooling patterns
         Expand
            |
        pooling
            |
      Slice (optional)
*/

bool ExpandPoolingPattern::init() {
    auto log = getLogger().nest();
    auto op = getEltwiseOperation();

    // match input expand
    auto operand = op->getOperand(0);
    if (auto expand = operand.getDefiningOp<IE::ExpandOp>()) {
        addExpandInput(expand);
        log.trace("{0} Expand setInput(s) found", getExpandInputsNum());
    } else {
        log.trace("Cannot find any input ExpandOp");
        return false;
    }

    // match output slices or non-slices
    for (auto user : op->getResult(0).getUsers()) {
        if (auto slice = mlir::dyn_cast<IE::SliceOp>(user)) {
            addSliceOutput(slice);
        } else {
            addNonSliceOutput(user);
        }
    }
    log.trace("{0} Slice setOutput(s) found", getSliceOutputsNum());

    // save the original shape and generate new shape
    auto expandInputOp = *getExpandInputs().begin();
    auto unExpandedShape = mlir::cast<vpux::NDTypeInterface>(expandInputOp.getInput().getType()).getShape().toValues();
    setUnExpandedShape(unExpandedShape);

    mlir::FailureOr<Shape> newExpandedShapeResult =
            getShapeCastExpandedShape(op, getShape(op->getOperand(0)), unExpandedShape, log);
    if (mlir::failed(newExpandedShapeResult)) {
        return false;
    }

    auto newExpandedShape = newExpandedShapeResult.value();
    setNewExpandedShape(std::move(newExpandedShape));
    return true;
}

//
// ExpandPoolingRewriter
//

template <class PoolingOp>
class ExpandPoolingRewriter final : public mlir::OpRewritePattern<PoolingOp> {
public:
    ExpandPoolingRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<PoolingOp>(ctx, benefit), _log(log) {
        this->setDebugName("ExpandPoolingOpRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(PoolingOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class PoolingOp>
mlir::LogicalResult ExpandPoolingRewriter<PoolingOp>::matchAndRewrite(PoolingOp layerOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    const auto supportedPooling = [](PoolingOp layerOp) {
        const auto kernels = parseIntArrayAttr<int64_t>(layerOp.getKernelSize());
        const auto padStart = parseIntArrayAttr<int64_t>(layerOp.getPadsBegin());
        const auto padEnd = parseIntArrayAttr<int64_t>(layerOp.getPadsEnd());
        const auto strides = parseIntArrayAttr<int64_t>(layerOp.getStrides());

        mlir::Value input = layerOp.getInput();
        mlir::Value output = layerOp.getOutput();
        const auto inputLayout = mlir::cast<vpux::NDTypeInterface>(input.getType()).getDimsOrder();
        const auto outputLayout = mlir::cast<vpux::NDTypeInterface>(output.getType()).getDimsOrder();
        // input and output layer need to be same
        if (inputLayout != outputLayout) {
            return false;
        }

        auto hasValidKernels = llvm::all_of(kernels, [&](const auto& kernel) {
            return kernel == 1;
        });
        auto hasValidPadStart = llvm::all_of(padStart, [&](const auto& pad) {
            return pad == 0;
        });
        auto hasValidPadEnd = llvm::all_of(padEnd, [&](const auto& pad) {
            return pad == 0;
        });
        auto hasValidStrides = llvm::all_of(strides, [&](const auto& stride) {
            return stride == 1;
        });

        return hasValidKernels && hasValidPadStart && hasValidPadEnd && hasValidStrides;
    };
    if (!supportedPooling(layerOp)) {
        return mlir::failure();
    }

    _log.debug("[{0}] Got '{1}' at '{2}'", this->getDebugName(), layerOp->getName(), layerOp->getLoc());
    auto nestedLog = _log.nest();
    auto pattern = ExpandPoolingPattern(layerOp.getOperation(), nestedLog);
    if (!pattern.init()) {
        return matchFailed(nestedLog, rewriter, layerOp, "Unsupported Expand - Pooling pattern.");
    }
    if (pattern.opCostReduced()) {
        nestedLog.debug("Pattern rewrite is beneficial. Applying it.");
        return pattern.rewrite(rewriter);
    }

    return matchFailed(nestedLog, rewriter, layerOp,
                       "New pattern has higher cost than original one; will not be applied.");
}

class ExpandSingleChannelPoolingPattern final : public ExpandPoolingPattern {
public:
    ExpandSingleChannelPoolingPattern(mlir::Operation* pooling, Logger log): ExpandPoolingPattern(pooling, log) {
    }

    bool init();
    mlir::LogicalResult rewrite(mlir::PatternRewriter& rewriter);
};

template <class PoolingOp>
bool isSuitableSingleChannelPooling(PoolingOp layerOp) {
    const auto kernels = parseIntArrayAttr<int64_t>(layerOp.getKernelSize());
    const auto padStart = parseIntArrayAttr<int64_t>(layerOp.getPadsBegin());
    const auto padEnd = parseIntArrayAttr<int64_t>(layerOp.getPadsEnd());
    const auto strides = parseIntArrayAttr<int64_t>(layerOp.getStrides());

    auto input = layerOp.getInput();
    auto output = layerOp.getOutput();
    auto inputLayout = mlir::cast<vpux::NDTypeInterface>(input.getType()).getDimsOrder();
    auto outputLayout = mlir::cast<vpux::NDTypeInterface>(output.getType()).getDimsOrder();
    // input and output layouts need to be the same
    if (inputLayout != outputLayout) {
        return false;
    }

    auto inShape = getShape(input);
    if (inShape.size() != 4) {
        return false;
    }
    if (inShape[Dims4D::Act::W] == 1) {
        return false;
    }

    auto hasValidKernels = kernels.back() == 1;
    auto hasValidPadStart = padStart.back() == 0;
    auto hasValidPadEnd = padEnd.back() == 0;
    auto hasValidStrides = strides.back() == 1;
    return hasValidKernels && hasValidPadStart && hasValidPadEnd && hasValidStrides;
}

/* Try to match the Expand-pooling patterns
         input
            |
         Expand
            |
         pooling
            |
          Slice[Optional]
*/

bool ExpandSingleChannelPoolingPattern::init() {
    auto log = getLogger().nest();
    auto op = getEltwiseOperation();

    // match input expand
    auto operand = op->getOperand(0);
    auto expand = operand.getDefiningOp<IE::ExpandOp>();
    if (expand == nullptr) {
        log.trace("Cannot find any input ExpandOp");
        return false;
    }
    if (!expand->hasOneUse()) {
        return false;
    }

    addExpandInput(expand);
    log.trace("{0} Expand setInput(s) found", getExpandInputsNum());

    // match output slices or non-slices
    for (auto user : op->getResult(0).getUsers()) {
        if (auto slice = mlir::dyn_cast<IE::SliceOp>(user)) {
            addSliceOutput(slice);
        } else {
            addNonSliceOutput(user);
        }
    }

    return true;
}

/*
        Input                             Input
     [N, 1, H, W]                     [N, 1, H, W]
         |                                  |
       Expand                           ShapeCast
     [N, 16, H, W]      =>            [N, W, H, 1]
         |                                  |
     Pool(SY=1, KY=1)                    Expand
     [N, 16, H', W]                   [N, W', H, 1]
         |                                  |
     Slice[Optional]                Pool(SY=1, KY=1)
     [N, 1, H', W]                   [N, W', H', 1]
                                            |
                                        ShapeCast
                                      [N, 1, H', W']
                                            |
                                          Slice
                                      [N, 1, H', W]
                                            |
                                         Expand
                                      [N, 16, H', W]
                                            |
                                          Slice[Optional]
                                      [N, 1, H', W]

   The purpose of this transformation is that if ExpandOp will be lowered into DMA op, then try to change the op from
   `[N, 1, H, W]->Expand->[N, 16, H, W]` to `[N, W, H, 1]->Expand->[N, W', H, 1]`, in which the new pattern is more
   performant, since the original one will only move 1 element per cycle, total N*W*H cycles, and the new one will try
   to move W elements per cycle, total N*H cycles.
*/

mlir::LogicalResult ExpandSingleChannelPoolingPattern::rewrite(mlir::PatternRewriter& rewriter) {
    auto log = getLogger().nest();

    auto expand = *getExpandInputs().begin();
    auto op = getEltwiseOperation();
    log.trace("Adjust input shape for pooling op at {0}", op->getLoc());

    auto ctx = rewriter.getContext();

    auto inShape = getShape(expand.getInput());
    auto newInShape = SmallVector<int64_t>{inShape[Dims4D::Act::N], inShape[Dims4D::Act::W], inShape[Dims4D::Act::H],
                                           inShape[Dims4D::Act::C]};

    auto inShapeCast =
            rewriter.create<IE::ShapeCastOp>(expand->getLoc(), expand.getInput(), getIntArrayAttr(ctx, newInShape));

    // input for the new pooling op
    mlir::Value input;
    auto channelsInfo = mlir::cast<IE::AlignedChannelsOpInterface>(op);
    const auto alignedInputC = alignValUp(newInShape[Dims4D::Act::C.ind()], channelsInfo.getInputChannelAlignment());
    auto needInsertExpandForNewInputShape = alignedInputC > newInShape[Dims4D::Act::C.ind()];
    if (needInsertExpandForNewInputShape) {
        // Adjust Channel size to meet alignment requirement
        auto padBegin = mlir::SmallVector<int64_t>(newInShape.size(), 0);
        auto padEnd = mlir::SmallVector<int64_t>(newInShape.size(), 0);
        padEnd[vpux::Dims4D::Act::C.ind()] = alignedInputC - newInShape[Dims4D::Act::C.ind()];
        auto newExpand =
                rewriter.create<IE::ExpandOp>(expand->getLoc(), inShapeCast, getIntArrayAttr(ctx, ArrayRef(padBegin)),
                                              getIntArrayAttr(ctx, ArrayRef(padEnd)));
        expand->replaceAllUsesWith(newExpand);
        input = newExpand.getResult();
    } else {
        expand->replaceAllUsesWith(inShapeCast);
        input = inShapeCast.getResult();
    }

    mlir::IRMapping mapper;
    mapper.map(op->getOperand(0), input);
    auto newPoolOp = rewriter.clone(*op, mapper);
    vpux::inferReturnTypes(newPoolOp, vpux::InferShapedTypeMode::SHAPE);

    // create ShapeCast for the output of the pooling op
    auto poolOutput = newPoolOp->getResult(0);
    auto outShape = getShape(poolOutput);
    auto newOutShape = SmallVector<int64_t>{outShape[Dims4D::Act::N], outShape[Dims4D::Act::W],
                                            outShape[Dims4D::Act::H], outShape[Dims4D::Act::C]};
    auto outShapeCast =
            rewriter.create<IE::ShapeCastOp>(poolOutput.getLoc(), poolOutput, getIntArrayAttr(ctx, newOutShape));

    mlir::Value newOutput = outShapeCast.getResult();
    if (needInsertExpandForNewInputShape) {
        // The original W has been expanded, need to slice for the new output
        SmallVector<int64_t> sliceOffset(outShape.size(), 0);
        newOutShape[Dims4D::Act::W.ind()] = inShape[Dims4D::Act::W];
        newOutput = rewriter.create<IE::SliceOp>(outShapeCast->getLoc(), outShapeCast,
                                                 getIntArrayAttr(ctx, sliceOffset), getIntArrayAttr(ctx, newOutShape));
    }

    if (newPoolOp->hasAttr(VPU::INPUT_PADDING_ATTR_NAME)) {
        rewriter.modifyOpInPlace(newPoolOp, [&] {
            newPoolOp->removeAttr(VPU::INPUT_PADDING_ATTR_NAME);
        });
    }
    if (newPoolOp->hasAttr(VPU::OUTPUT_PADDING_ATTR_NAME)) {
        rewriter.modifyOpInPlace(newPoolOp, [&] {
            newPoolOp->removeAttr(VPU::OUTPUT_PADDING_ATTR_NAME);
        });
    }

    auto newOutExpand = rewriter.create<IE::ExpandOp>(expand->getLoc(), newOutput, expand.getPadsBeginAttr(),
                                                      expand.getPadsEndAttr());

    op->replaceAllUsesWith(newOutExpand->getResults());
    return mlir::success();
}

//
// ExpandSingleChannelPoolingRewriter
//

template <class PoolingOp>
class ExpandSingleChannelPoolingRewriter final : public mlir::OpRewritePattern<PoolingOp> {
public:
    ExpandSingleChannelPoolingRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<PoolingOp>(ctx, benefit), _log(log) {
        this->setDebugName("ExpandSingleChannelPoolingOpRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(PoolingOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class PoolingOp>
mlir::LogicalResult ExpandSingleChannelPoolingRewriter<PoolingOp>::matchAndRewrite(
        PoolingOp layerOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), layerOp->getName(), layerOp->getLoc());
    auto pattern = ExpandSingleChannelPoolingPattern(layerOp.getOperation(), _log);
    if (!pattern.init()) {
        return mlir::failure();
    }

    // Check the pooling op has SY=1, KY=1
    if (!isSuitableSingleChannelPooling(layerOp)) {
        return mlir::failure();
    }

    // Check the expand op has C=1
    auto expandOp = *pattern.getExpandInputs().begin();

    if (IE::isEligibleConvertToConv(expandOp, _log, this->getDebugName())) {
        // Expand will be converted to Conv
        return mlir::failure();
    };

    auto unExpandedShape = getShape(expandOp.getInput());
    if (unExpandedShape[Dims4D::Act::C] != 1) {
        return mlir::failure();
    }
    auto paddingBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
    auto paddingBeginEqualToZero = llvm::all_of(paddingBegin, [](const auto& padVal) {
        return padVal == 0;
    });
    if (!paddingBeginEqualToZero) {
        return mlir::failure();
    }

    if (pattern.opCostReduced()) {
        return pattern.rewrite(rewriter);
    }
    return mlir::failure();
}

//
// ExpandPermuteQuantizePattern
//

class ExpandPermuteQuantizePattern final : public ExpandEltwisePattern {
public:
    ExpandPermuteQuantizePattern(mlir::Operation* permuteQuantize, Logger log)
            : ExpandEltwisePattern(permuteQuantize, log) {
    }

    // Overwrite ExpandEltwisePattern::init()
    bool init();

private:
    bool checkValidPermuteQuantizeOrders(IE::PermuteQuantizeOp op);
    mlir::FailureOr<Shape> getWidthAlignedExpandedShape(mlir::Operation* operation, ShapeRef unExpandedShape,
                                                        Logger log);
};

//
// For PermuteQuantize shape adjustment.
// e.g
// 2x2x4 tensor:
// a11 a12 a13 a14               b11 b12 b13 b14
// a21 a22 a23 a24               b21 b22 b23 b24
// Layout in memory NCHW: a11 a12 a13 a14 a21 a22 a23 a24 b11 b12 b13 b14 b21 b22 b23 b24
// Layout in memory NHWC: a11 b11 a12 b12 a13 b13 a14 b14 a21 b21 a22 b22 a23 b23 a24 b24
//
// 2x4x2 tensor:
// a11 a12                              b11 b12
// a13 a14                              b13 b14
// a21 a22                              b21 b22
// a23 a24                              b23 b24
// Layout in memory NCHW: a11 a12 a13 a14 a21 a22 a23 a24 b11 b12 b13 b14 b21 b22 b23 b24
// Layout in memory NHWC: a11 b11 a12 b12 a13 b13 a14 b14 a21 b21 a22 b22 a23 b23 a24 b24
//
bool ExpandPermuteQuantizePattern::checkValidPermuteQuantizeOrders(IE::PermuteQuantizeOp op) {
    auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto inputLayout = inType.getDimsOrder();
    auto outType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
    auto outputLayout = outType.getDimsOrder();

    const auto supportedPerm = vpux::DimsOrder::NHWC.toAffineMap(op->getContext());

    return inputLayout == DimsOrder::NCHW && outputLayout == DimsOrder::NHWC && op.getMemPerm() == supportedPerm;
}

mlir::FailureOr<Shape> ExpandPermuteQuantizePattern::getWidthAlignedExpandedShape(mlir::Operation* operation,
                                                                                  ShapeRef unExpandedShape,
                                                                                  Logger log) {
    auto permuteQuantize = mlir::dyn_cast_or_null<IE::PermuteQuantizeOp>(operation);
    if (permuteQuantize == nullptr || !checkValidPermuteQuantizeOrders(permuteQuantize) ||
        !checkValidPermuteQuantizePads(permuteQuantize)) {
        return mlir::failure();
    }

    if (unExpandedShape.size() != 4) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(operation->getOperand(0).getType());
    const auto alignment = VPU::NCEInvariant::getAlignment(inputType.getElementType());

    auto IH = unExpandedShape[Dims4D::Act::H];
    auto IW = unExpandedShape[Dims4D::Act::W];
    if (IH * IW % alignment != 0) {
        log.trace("Unable to adjust the input shape for op {0} at {1}, shape {2}", operation->getName(),
                  operation->getLoc(), unExpandedShape);
        return mlir::failure();
    }

    auto newExpandedShape = Shape(unExpandedShape.size(), 1);
    newExpandedShape[Dims4D::Act::N] = unExpandedShape[Dims4D::Act::N];
    newExpandedShape[Dims4D::Act::C] = unExpandedShape[Dims4D::Act::C];
    newExpandedShape[Dims4D::Act::H] = IH * IW / alignment;
    newExpandedShape[Dims4D::Act::W] = alignment;

    return newExpandedShape;
}

/* Try to match the Expand-PermuteQuantize patterns
         Expand
            |
     PermuteQuantize
            |
      Slice (optional)
*/

bool ExpandPermuteQuantizePattern::init() {
    auto log = getLogger().nest();
    auto op = getEltwiseOperation();
    auto permuteQuantize = mlir::dyn_cast<IE::PermuteQuantizeOp>(op);
    if (permuteQuantize == nullptr) {
        return false;
    }

    if (!checkValidPermuteQuantizeOrders(permuteQuantize)) {
        log.trace("Invalid PermuteQuantize layouts. '{0}' at '{1}'", permuteQuantize->getName(),
                  permuteQuantize->getLoc());
        return false;
    }

    // match input expand
    auto operand = op->getOperand(0);
    if (auto expand = operand.getDefiningOp<IE::ExpandOp>()) {
        const auto padsEnd = Shape(parseIntArrayAttr<int64_t>(expand.getPadsEnd()));
        if (padsEnd[Dims4D::Act::N] == 0 && padsEnd[Dims4D::Act::C] == 0 && padsEnd[Dims4D::Act::H] == 0) {
            // only width expanding should be handled
            addExpandInput(expand);
        }
    }

    log.trace("{0} Expand setInput(s) found", getExpandInputsNum());
    if (getExpandInputsNum() == 0) {
        log.trace("Cannot find any input ExpandOp");
        return false;
    }

    // match output slices or non-slices
    for (auto user : op->getResult(0).getUsers()) {
        if (auto slice = mlir::dyn_cast<IE::SliceOp>(user)) {
            addSliceOutput(slice);
        } else {
            addNonSliceOutput(user);
        }
    }
    log.trace("{0} Slice setOutput(s) found", getSliceOutputsNum());

    // save the original shape and generate new shape
    auto expandInputOp = *getExpandInputs().begin();
    auto unExpandedShape = mlir::cast<vpux::NDTypeInterface>(expandInputOp.getInput().getType()).getShape().toValues();
    setUnExpandedShape(unExpandedShape);

    mlir::FailureOr<Shape> newExpandedShapeResult = getWidthAlignedExpandedShape(op, unExpandedShape, log);

    if (mlir::failed(newExpandedShapeResult)) {
        return false;
    }

    setNewExpandedShape(newExpandedShapeResult.value());
    return true;
}

//
// ExpandPermuteQuantizeRewriter
//

class ExpandPermuteQuantizeRewriter final : public mlir::OpRewritePattern<IE::PermuteQuantizeOp> {
public:
    ExpandPermuteQuantizeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PermuteQuantizeOp>(ctx), _log(log) {
        setDebugName("ExpandPermuteQuantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::PermuteQuantizeOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ExpandPermuteQuantizeRewriter::matchAndRewrite(IE::PermuteQuantizeOp layerOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), layerOp->getName(), layerOp->getLoc());

    auto pattern = ExpandPermuteQuantizePattern(layerOp.getOperation(), _log);
    if (!pattern.init()) {
        return mlir::failure();
    }
    if (pattern.opCostReduced()) {
        return pattern.rewrite(rewriter);
    }
    return mlir::failure();
}

//
// AdjustPermuteQuantizeRewriter
//

class AdjustPermuteQuantizeRewriter final : public mlir::OpRewritePattern<IE::PermuteQuantizeOp> {
public:
    AdjustPermuteQuantizeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PermuteQuantizeOp>(ctx), _log(log) {
        setDebugName("AdjustPermuteQuantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::PermuteQuantizeOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AdjustPermuteQuantizeRewriter::matchAndRewrite(IE::PermuteQuantizeOp layerOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), layerOp->getName(), layerOp->getLoc());

    const auto logCb = [&](const formatv_object_base&) {};
    if (!VPU::NCEPermuteOp::isSupported(layerOp, logCb, /*checkLayout=*/true,
                                        /*checkChannelAlignment=*/true)) {
        return mlir::failure();
    }

    if (!checkValidPermuteQuantizePads(layerOp)) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(layerOp.getInput().getType());
    const auto inputShape = inputType.getShape();
    auto H = inputShape[Dims4D::Act::H];
    auto W = inputShape[Dims4D::Act::W];
    if ((VPU::NCEInvariant::VPU_DIMENSION_LIMIT >= W && VPU::NCEInvariant::VPU_DIMENSION_LIMIT >= H) ||
        (W > VPU::NCEInvariant::VPU_DIMENSION_LIMIT && H > VPU::NCEInvariant::VPU_DIMENSION_LIMIT)) {
        return mlir::failure();
    }

    auto adjustHW = IE::getAdjustHW(VPU::NCEInvariant::getAlignment(inputType), W, H);
    if (!adjustHW.has_value()) {
        return mlir::failure();
    }
    W = adjustHW.value().front();
    H = adjustHW.value().back();

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(layerOp.getOutput().getType());
    const auto newShape = Shape({inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C], H, W});
    const auto newOutputType = outputType.changeShape(newShape);
    auto inputShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(layerOp.getLoc(), inputType.changeShape(newShape), layerOp.getInput(),
                                             getIntArrayAttr(layerOp.getContext(), newShape));

    auto newOp = rewriter.create<IE::PermuteQuantizeOp>(layerOp.getLoc(), newOutputType, inputShapeCastOp.getResult(),
                                                        layerOp.getDstOrderAttr(), layerOp.getMemPermAttr(),
                                                        layerOp.getDstElemTypeAttr(), layerOp.getPadsBeginAttr(),
                                                        layerOp.getPadsEndAttr());

    auto outputShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(layerOp.getLoc(), outputType, newOp.getOutput(),
                                             getIntArrayAttr(layerOp.getContext(), outputType.getShape()));

    rewriter.replaceOp(layerOp, outputShapeCastOp.getResult());
    return mlir::success();
}

//
// EltwiseShapeRewriter
//

template <class EltwiseOp>
class EltwiseShapeRewriter final : public mlir::OpRewritePattern<EltwiseOp> {
public:
    EltwiseShapeRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<EltwiseOp>(ctx), _log(log) {
        this->setDebugName("EltwiseShapeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(EltwiseOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isOutputShapeAligned(mlir::Operation* op, ShapeRef newOutputShape) {
    auto alignIface = mlir::dyn_cast_or_null<IE::AlignedChannelsOpInterface>(op);
    if (alignIface == nullptr) {
        // If not AlignedChannelsOpInterface, the op does not have a channel alignment requirement
        // so output shape is always aligned
        return true;
    }
    return newOutputShape[Dims4D::Act::C] % alignIface.getOutputChannelAlignment() == 0;
}

enum class PatternType { NO_MATCH = 0, SHAPE_OP_BEFORE_NCE = 1, SHAPE_OP_AFTER_NCE = 2 };

template <class EltwiseOp>
PatternType getEltwiseShapePatternType(EltwiseOp layerOp, mlir::Operation*& origShapeOp, Logger log) {
    if (layerOp->getOperands().size() != 2) {
        return PatternType::NO_MATCH;
    }

    if (layerOp.getInputPaddingAttr() != nullptr || layerOp.getOutputPaddingAttr() != nullptr) {
        return PatternType::NO_MATCH;
    }

    const auto inputOp1 = layerOp->getOperand(0).getDefiningOp();
    const auto inputOp2 = layerOp->getOperand(1).getDefiningOp();
    const bool hasInputsFromSameOp = inputOp1 != nullptr && inputOp1 == inputOp2;
    // Match Eltwise(QuantizeCast) -> ShapeCast/AffineReshape -> NCE
    auto currentOp = layerOp.getOperation();
    if (mlir::isa_and_nonnull<IE::QuantizeCastOp>(*layerOp->getUsers().begin()) && !VPU::hasMultiBranches(currentOp)) {
        currentOp = *layerOp->getUsers().begin();
    }
    auto userMaybeShapeOp = *currentOp->getUsers().begin();
    if (mlir::isa_and_nonnull<IE::ShapeCastOp, IE::AffineReshapeOp>(userMaybeShapeOp)) {
        if (auto maybeNCEOp = *userMaybeShapeOp->getUsers().begin()) {
            const bool hasShapeOpBefore = mlir::isa_and_nonnull<IE::ShapeCastOp, IE::AffineReshapeOp>(inputOp1);
            if (mlir::succeeded(VPU::NCEInvariant::isSupported(maybeNCEOp, log)) &&
                (!hasInputsFromSameOp || hasShapeOpBefore) && !VPU::hasMultiBranches(userMaybeShapeOp) &&
                !VPU::hasMultiBranches(currentOp) &&
                isOutputShapeAligned(currentOp, getShape(userMaybeShapeOp->getResult(0)))) {
                origShapeOp = userMaybeShapeOp;
                return PatternType::SHAPE_OP_BEFORE_NCE;
            }
        }
    }
    // Match NCE -> (Activation) -> ShapeCast/AffineReshape => Eltwise(QuantizeCast)
    auto producerMaybeShapeOp = layerOp->getOperand(0).getDefiningOp();
    if (mlir::isa_and_nonnull<IE::ShapeCastOp, AffineReshapeOp>(producerMaybeShapeOp)) {
        if (mlir::Operation* maybeNCEOp = producerMaybeShapeOp->getOperand(0).getDefiningOp()) {
            if (maybeNCEOp->hasTrait<IE::EltwiseOp>() && maybeNCEOp->getNumOperands() == 1 &&
                !VPU::hasMultiBranches(maybeNCEOp)) {
                maybeNCEOp = maybeNCEOp->getOperand(0).getDefiningOp();
            }
            if (maybeNCEOp != nullptr && mlir::succeeded(VPU::NCEInvariant::isSupported(maybeNCEOp, log)) &&
                hasInputsFromSameOp && !VPU::hasMultiBranches(producerMaybeShapeOp) &&
                !VPU::hasMultiBranches(maybeNCEOp) &&
                isOutputShapeAligned(layerOp.getOperation(), getShape(producerMaybeShapeOp->getOperand(0)))) {
                origShapeOp = producerMaybeShapeOp;
                return PatternType::SHAPE_OP_AFTER_NCE;
            }
        }
    }
    return PatternType::NO_MATCH;
}

/**
 * @brief Propagate the ShapeCast/AffineReshape to make Eltwise and NCE adjacent
 *
 * @details For case Eltwise(QuantizeCast) -> ShapeCast/AffineReshape -> NCE
 *          1. if the Eltwise has two different inputs, it is likely to have spilling of at least one input.
 *          So move the ShapeCast before Eltwise to avoid the spilling of output.
 *          2. if the op before Eltwise is also ShapeOp, move the ShapeOp before Eltwise and make them fused.
 *
 *          For case NCE -> ShapeCast/AffineReshape -> Eltwise(QuantizeCast), if the Eltwise's two inputs are the same,
 *          Move the ShapeOp after Eltwise to reduce the spilling
 */
template <class EltwiseOp>
mlir::LogicalResult EltwiseShapeRewriter<EltwiseOp>::matchAndRewrite(EltwiseOp layerOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), layerOp->getName(), layerOp->getLoc());

    mlir::Operation* origShapeOp = nullptr;
    auto matchType = getEltwiseShapePatternType(layerOp, origShapeOp, _log);
    if (matchType == PatternType::NO_MATCH) {
        return mlir::failure();
    }

    _log.nest().trace("Matched pattern, rewriting");
    const auto origShapeInType = mlir::cast<vpux::NDTypeInterface>(origShapeOp->getOperand(0).getType());
    const auto origShapeOutType = mlir::cast<vpux::NDTypeInterface>(origShapeOp->getResult(0).getType());
    const auto origEltwiseOutputType = mlir::cast<vpux::NDTypeInterface>(layerOp->getResult(0).getType());
    auto quantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(*layerOp->getUsers().begin());

    if (matchType == PatternType::SHAPE_OP_BEFORE_NCE) {
        // For Eltwise(QuantizeCast)-ShapeCast/AffineReshape-NCE
        // create new ShapeCastOp/AffineReshapeOp for each input
        SmallVector<mlir::Value> newInputValues;
        for (const auto inputOperand : layerOp->getOperands()) {
            mlir::IRMapping mapper;
            mapper.map(origShapeOp->getOperand(0), inputOperand);
            auto newShapeOp = rewriter.clone(*origShapeOp, mapper);
            vpux::inferReturnTypes(newShapeOp, vpux::InferShapedTypeMode::ALL);
            newInputValues.push_back(newShapeOp->getResult(0));
        }
        // Update EltwiseOp
        auto newEltwiseShape = origShapeOutType.getShape();
        auto newEltwiseOutputType = origEltwiseOutputType.changeShape(newEltwiseShape);
        auto newEltwise = rewriter.template create<EltwiseOp>(
                layerOp->getLoc(), newEltwiseOutputType, newInputValues[0], newInputValues[1],
                layerOp.getAutoBroadcast(), layerOp.getPostOpAttr(), layerOp.getClampAttr(),
                layerOp.getOutputPaddingAttr(), layerOp.getInputPaddingAttr());
        mlir::Value newOutput = newEltwise->getResult(0);
        // Update QuantizeCastOp
        if (quantizeCastOp) {
            newOutput = rewriter.template create<IE::QuantizeCastOp>(quantizeCastOp->getLoc(), newOutput,
                                                                     quantizeCastOp.getDstElemTypeAttr())
                                .getResult();
        }
        origShapeOp->getResult(0).replaceAllUsesWith(newOutput);
    } else if (matchType == PatternType::SHAPE_OP_AFTER_NCE) {
        // For NCE-(Activation)-ShapeCast/AffineReshape-Eltwise(QuantizeCast)
        // Update EltwiseOp
        auto newEltwiseShape = origShapeInType.getShape();
        auto newEltwiseOutputType = origEltwiseOutputType.changeShape(newEltwiseShape);
        auto newEltwise = rewriter.template create<EltwiseOp>(
                layerOp->getLoc(), newEltwiseOutputType, origShapeOp->getOperand(0), origShapeOp->getOperand(0),
                layerOp.getAutoBroadcast(), layerOp.getPostOpAttr(), layerOp.getClampAttr(),
                layerOp.getOutputPaddingAttr(), layerOp.getInputPaddingAttr());
        mlir::Value origOutput = layerOp->getResult(0);
        mlir::Value newOutput = newEltwise->getResult(0);
        // Update QuantizeCastOp
        if (quantizeCastOp) {
            origOutput = quantizeCastOp->getResult(0);
            newOutput = rewriter.template create<IE::QuantizeCastOp>(quantizeCastOp->getLoc(), newOutput,
                                                                     quantizeCastOp.getDstElemTypeAttr())
                                .getResult();
        }
        origShapeOp->getResult(0).replaceAllUsesWith(newOutput);
        // create one shapeOp for output
        mlir::IRMapping mapper;
        mapper.map(origShapeOp->getOperand(0), newOutput);
        auto newShapeOp = rewriter.clone(*origShapeOp, mapper);

        vpux::inferReturnTypes(newShapeOp, vpux::InferShapedTypeMode::ALL);
        origOutput.replaceAllUsesWith(newShapeOp->getResult(0));
    } else {
        VPUX_THROW("Unknown PatternType");
    }

    return mlir::success();
}

// AdjustInputShapePass

void AdjustInputShapePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ExpandEltwiseRewriter<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<ExpandEltwiseRewriter<IE::SubtractOp>>(&ctx, _log);
    patterns.add<ExpandEltwiseRewriter<IE::AddOp>>(&ctx, _log);
    patterns.add<ExpandGroupConvRewriter>(&ctx, _log);
    patterns.add<ExpandPermuteQuantizeRewriter>(&ctx, _log);
    patterns.add<AdjustPermuteQuantizeRewriter>(&ctx, _log);
    patterns.add<ExpandPoolingRewriter<IE::AvgPoolOp>>(&ctx, benefitLevels[0], _log);
    patterns.add<ExpandPoolingRewriter<IE::MaxPoolOp>>(&ctx, benefitLevels[0], _log);
    patterns.add<ExpandSingleChannelPoolingRewriter<IE::AvgPoolOp>>(&ctx, benefitLevels[1], _log);
    patterns.add<ExpandSingleChannelPoolingRewriter<IE::MaxPoolOp>>(&ctx, benefitLevels[1], _log);
    patterns.add<EltwiseShapeRewriter<IE::AddOp>>(&ctx, _log);

    // There is case for `EltwiseShapeRewriter` that the iteration time larger than default value
    // TODO: E#126695 Refactor to avoid specific maxIterations
    auto greedyRewriteConfig = getDefaultGreedyRewriteConfig();
    greedyRewriteConfig.maxIterations *= 20;
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), greedyRewriteConfig))) {
        signalPassFailure();
        return;
    }
}
}  // namespace

//
// createAdjustInputShapePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustInputShapePass(Logger log) {
    return std::make_unique<AdjustInputShapePass>(log);
}
