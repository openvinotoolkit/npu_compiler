//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters/propagate_transpose_affine_reshape_common.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/transpose_op_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPOPERATIONS
#define GEN_PASS_DEF_SWAPOPERATIONS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

const int64_t SUPPORTED_RANK = 4;
const int8_t CHANNEL_ALIGNMENT = 16;

bool checkOrderCompatible(mlir::Operation* origOp, const DimsOrder& origOrder, const DimsOrder& parentOrder) {
    if (origOrder != parentOrder) {
        auto iface = mlir::dyn_cast<IE::LayoutInfoOpInterface>(origOp);
        if (iface == nullptr) {
            return false;
        }

        // Current logic (orderInfo.setInput) cannot set a new order with a different rank
        // e.g, 4D tensor -> AffineReshape -> 3D tensor -> 3D op  ===>  4D op -> 4D tensor -> AffineReshape -> 3D tensor
        // TODO: Fix E#79970 and remove the following conditional statement
        if (parentOrder.numDims() != origOrder.numDims()) {
            return false;
        }

        auto orderInfo = iface.getLayoutInfo();
        orderInfo.setInput(0, parentOrder);
        iface.inferLayoutInfo(orderInfo);
        if (orderInfo.getInput(0) != parentOrder) {
            return false;
        }
        if (orderInfo.getOutput(0) != parentOrder) {
            return false;
        }
    }

    return true;
}

void updateOutputOrder(mlir::Value output, const DimsOrder& origOrder, const DimsOrder& parentOrder) {
    if (origOrder != parentOrder) {
        const auto newAddOutputType = mlir::cast<vpux::NDTypeInterface>(output.getType());
        const auto newType = newAddOutputType.changeDimsOrder(parentOrder);
        output.setType(newType);
    }
}

mlir::Value alignConstant(mlir::PatternRewriter& rewriter, mlir::Operation* parent, mlir::Value constInput) {
    return llvm::TypeSwitch<mlir::Operation*, mlir::Value>(parent)
            .Case<IE::AffineReshapeOp, IE::ReshapeOp>([&](auto origOp) {
                const auto constInputShape = getShape(constInput);
                const auto parentInputShape = getShape(origOp.getInput());

                // For AffineReshapeOp, only swap if the bias non-trivial dim maps back to the C
                // dimension of the parent input; otherwise the swapped Add cannot be fused as a
                // per-channel bias into the preceding NCE op, which would cause AC regressions. A splat
                // bias has a single value and can be re-aligned to any channel, so this only restricts a
                // genuine per-channel Add bias.
                bool isSplatConst = false;
                if (auto declareOp = constInput.getDefiningOp<Const::DeclareOp>()) {
                    isSplatConst = declareOp.getContent().isSplat();
                }
                auto affineReshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(origOp.getOperation());
                if (affineReshapeOp != nullptr && !isSplatConst) {
                    int64_t constNonTrivialDim = Dims4D::Act::C.ind();
                    for (int64_t d = 0; d < static_cast<int64_t>(constInputShape.size()); ++d) {
                        if (constInputShape[Dim(d)] > 1) {
                            constNonTrivialDim = d;
                            break;
                        }
                    }

                    const auto nonTrivialSize = constInputShape[Dim(constNonTrivialDim)];
                    const auto outputShape = getShape(affineReshapeOp.getOutput());
                    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.getDimMapping());

                    // Backtrace the bias non-trivial output dim to its source input dim. A single input dim may
                    // be split into several output dims, so require that the whole mapped group carries the bias
                    // (input dim size equals the bias size and every sibling output dim is trivial). Abort if the
                    // mapping is ambiguous (more than one candidate) or no source dim is found.
                    SmallVector<int64_t> candidates;
                    for (size_t inDim = 0; inDim < dimMapping.size(); ++inDim) {
                        bool mapsToBias = false;
                        bool siblingsTrivial = true;
                        for (const auto outDim : dimMapping[inDim]) {
                            if (outDim == constNonTrivialDim) {
                                mapsToBias = true;
                            } else if (outputShape[Dim(outDim)] > 1) {
                                siblingsTrivial = false;
                            }
                        }
                        if (mapsToBias && siblingsTrivial && parentInputShape[Dim(inDim)] == nonTrivialSize) {
                            candidates.push_back(static_cast<int64_t>(inDim));
                        }
                    }

                    if (candidates.size() != 1 || candidates.front() != Dims4D::Act::C.ind()) {
                        return mlir::Value();
                    }
                }

                const auto parentInputDimC = parentInputShape[Dims4D::Act::C];
                if (constInputShape.totalSize() != parentInputDimC) {
                    auto declareOp = constInput.getDefiningOp<Const::DeclareOp>();
                    if (declareOp != nullptr) {
                        const auto content = declareOp.getContent();
                        const auto elemType = content.getType().getElementType();
                        if (content.isSplat() && mlir::isa_and_present<mlir::FloatType>(elemType)) {
                            auto splatAttr = rewriter.getFloatAttr(elemType, content.getSplatValue<double>());
                            SmallVector<int64_t> newShape(constInputShape.size(), 1);
                            newShape[Dims4D::Act::C.ind()] = parentInputDimC;

                            const auto newType = mlir::RankedTensorType::get(newShape, elemType);
                            const auto newContentAttr =
                                    Const::ContentAttr::get(mlir::DenseElementsAttr::get(newType, splatAttr));
                            return rewriter
                                    .create<Const::DeclareOp>(takeOpLoc(origOp, "splt_cst"), newType, newContentAttr)
                                    .getResult();
                        }
                    }

                    return mlir::Value();
                }

                SmallVector<int64_t> constShape(constInputShape.size(), 1);
                constShape[Dims4D::Act::C.ind()] = parentInputDimC;

                const auto constReshape = rewriter.createOrFold<IE::ReshapeOp>(
                        takeOpLoc(origOp, "reshape_cst"), constInput,
                        getIntArrayAttr(origOp->getContext(), ArrayRef(constShape)));

                const auto outOrder = DimsOrder::fromValue(constReshape);
                const auto inOrder = DimsOrder::fromValue(origOp.getInput());
                if (outOrder == inOrder) {
                    return constReshape;
                } else {
                    const auto newOrderMap = inOrder.toAffineMap(rewriter.getContext());
                    return rewriter.createOrFold<IE::ReorderOp>(takeOpLoc(origOp, "reorder_cst"), constReshape,
                                                                newOrderMap);
                }
            })
            .Case<IE::TransposeOp>([&](auto origOp) {
                const auto dstOrder = IE::deduceInverseOrder(origOp);
                const auto dstPerm = dstOrder.toAffineMap(origOp->getContext());
                const auto dstOrderAttr = mlir::AffineMapAttr::get(dstPerm);

                return rewriter.createOrFold<IE::TransposeOp>(takeOpLoc(origOp, "transpose_cst"), constInput, nullptr,
                                                              dstOrderAttr);
            })
            .Default([](mlir::Operation* op) -> mlir::Value {
                VPUX_THROW("Unsupported operation '{0}' at '{1}'", op->getName(), op->getLoc());
            });
}

bool isSingleValueBias(mlir::Value constInput) {
    auto declareOp = constInput.getDefiningOp<Const::DeclareOp>();
    if (declareOp == nullptr) {
        return false;
    }

    auto constShape = getShape(constInput).raw();
    auto hasNonTrivialDim = llvm::any_of(constShape, [](int64_t dim) {
        return dim != 1;
    });

    return !hasNonTrivialDim;
}

mlir::Value reshapeSingleValueConstant(mlir::PatternRewriter& rewriter, mlir::Location loc, int64_t numDims,
                                       mlir::Value constInput) {
    VPUX_THROW_UNLESS(isSingleValueBias(constInput), "Expext single value bias");
    auto ctx = rewriter.getContext();
    auto newConstShape = SmallVector<int64_t>(numDims, 1);
    auto reshapeConst = rewriter.create<IE::ReshapeOp>(loc, constInput, getIntArrayAttr(ctx, newConstShape));
    return reshapeConst;
}

//
// SwapWithBias
//

//
// SwapWithBias
//

namespace {

mlir::Value getBiasInput(IE::AddOp op) {
    bool lhsIsActivation = mlir::failed(IE::getConstParentOp(op.getInput1()));
    return lhsIsActivation ? op.getInput2() : op.getInput1();
}

mlir::Value getBiasInput(IE::ScaleShiftOp op) {
    return op.getBiases();
}

mlir::Value getActivationInput(IE::AddOp op) {
    bool lhsIsActivation = mlir::failed(IE::getConstParentOp(op.getInput1()));
    return lhsIsActivation ? op.getInput1() : op.getInput2();
}

mlir::Value getActivationInput(IE::ScaleShiftOp op) {
    return op.getInput();
}

bool isValidBiasOp(IE::AddOp) {
    return true;
}

bool isValidBiasOp(IE::ScaleShiftOp op) {
    return op.getWeights() == nullptr && op.getBiases() != nullptr;
}

IE::AddOp createNewBiasOp(mlir::PatternRewriter& rewriter, IE::AddOp origOp, mlir::Location loc, mlir::Value activation,
                          mlir::Value bias) {
    return rewriter.create<IE::AddOp>(loc, activation, bias, origOp.getAutoBroadcastAttr(), nullptr, nullptr, nullptr,
                                      nullptr);
}

IE::ScaleShiftOp createNewBiasOp(mlir::PatternRewriter& rewriter, IE::ScaleShiftOp, mlir::Location loc,
                                 mlir::Value activation, mlir::Value bias) {
    return rewriter.create<IE::ScaleShiftOp>(loc, activation, nullptr, bias);
}

}  // namespace

template <typename ConcreteOp>
class SwapWithBias final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    SwapWithBias(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<ConcreteOp>(ctx, benefit), _log(log) {
        this->setDebugName("SwapWithBias");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult SwapWithBias<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Found operation {1}", this->getDebugName(), origOp);

    if (!isValidBiasOp(origOp)) {
        return mlir::failure();
    }

    auto activationInput = getActivationInput(origOp);
    auto biasInput = getBiasInput(origOp);

    if (activationInput == nullptr || biasInput == nullptr) {
        return mlir::failure();
    }

    auto isEltwise = mlir::failed(IE::getConstParentOp(biasInput));
    if (isEltwise) {
        _log.trace("[{0}] Don't swap operations with Eltwise {1}", this->getDebugName(), origOp);
        return mlir::failure();
    }

    auto parentOp = activationInput.getDefiningOp();

    if (parentOp == nullptr) {
        return mlir::failure();
    }

    if (!mlir::isa<IE::ElemTypeInfoOpInterface>(parentOp)) {
        _log.trace("[{0}] Swapped operation {1} doesn't implement ElemTypeInfoOpInterface interface",
                   this->getDebugName(), *parentOp);
        return mlir::failure();
    }

    if (!parentOp->hasOneUse()) {
        _log.trace("[{0}] Swapped operation {1} has more than one use", this->getDebugName(), *parentOp);
        return mlir::failure();
    }

    auto parentInput = parentOp->getOperand(0);
    const auto origOrder = DimsOrder::fromValue(activationInput);
    const auto parentOrder = DimsOrder::fromValue(parentInput);

    auto singleValueBias = isSingleValueBias(biasInput);

    // Only the following situations are considered for Bias Swap:
    // From: NCE Task -> AffineReshapeOp/ReshapeOp/TransposeOp -> Add
    // To:   NCE Task -> Add -> AffineReshapeOp/ReshapeOp/TransposeOp
    // So that Add can as bias and fuse into NCE Task
    //
    // Single value bias is a special case:
    // 1.Can be swapped with ConcatOp
    // 2.Always be order compatible with parent op
    if (!mlir::isa<IE::AffineReshapeOp, IE::ReshapeOp, IE::TransposeOp>(parentOp)) {
        if (!(mlir::isa<IE::ConcatOp>(parentOp) && singleValueBias)) {
            _log.trace("[{0}] Only support AffineReshapeOp, ReshapeOp and TransposeOp, but got {1}",
                       this->getDebugName(), *parentOp);
            return mlir::failure();
        }
        _log.trace("[{0}] Swap single value bias with ConcatOp {1}", this->getDebugName(), parentOp->getLoc());
    }

    auto parentInputRank = mlir::cast<vpux::NDTypeInterface>(parentInput.getType()).getRank();
    if (mlir::isa<IE::ScaleShiftOp>(origOp) && parentInputRank != SUPPORTED_RANK) {
        _log.trace("[{0}] Swapped operation {1} doesn't have rank {2}", this->getDebugName(), *parentOp,
                   SUPPORTED_RANK);
        return mlir::failure();
    }

    if (!singleValueBias) {
        if (parentInputRank != SUPPORTED_RANK) {
            _log.trace("[{0}] Swapped operation doesn't have rank {1}", this->getDebugName(), SUPPORTED_RANK);
            return mlir::failure();
        }

        if (!checkOrderCompatible(origOp, origOrder, parentOrder)) {
            return mlir::failure();
        }
    }

    rewriter.setInsertionPointAfter(origOp);
    SmallVector<mlir::Value> newParentOpOperands;
    // Create new Add ops for each input of parent operation.
    for (auto& operand : parentOp->getOpOperands()) {
        mlir::Value newConstant;
        const size_t operandId = operand.getOperandNumber();
        if (singleValueBias) {
            auto oprandShape = getShape(operand.get()).raw();
            newConstant = reshapeSingleValueConstant(rewriter, takeOpLoc(origOp, "reshape_in_{0}", operandId),
                                                     oprandShape.size(), biasInput);
        } else {
            // TODO: E#68168 check the layout info as we did for Sigmod/Relu/Tanh
            newConstant = alignConstant(rewriter, parentOp, biasInput);
        }
        if (newConstant == nullptr) {
            _log.trace("[{0}] Swapped operation {1} fails to align constant", this->getDebugName(), *parentOp);
            return mlir::failure();
        }

        auto newBiasOp =
                createNewBiasOp(rewriter, origOp, takeOpLoc(origOp, "add_{0}", operandId), operand.get(), newConstant);

        // The new add must have the same output element type as the original one
        const auto origBiasOpOutputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
        auto newBiasOpOutputType = mlir::cast<vpux::NDTypeInterface>(newBiasOp->getResult(0).getType());
        newBiasOpOutputType = newBiasOpOutputType.changeElemType(origBiasOpOutputType.getElementType());
        rewriter.modifyOpInPlace(newBiasOp, [&] {
            newBiasOp->getResult(0).setType(newBiasOpOutputType);
            updateOutputOrder(newBiasOp->getResult(0), origOrder, parentOrder);
        });
        newParentOpOperands.push_back(newBiasOp->getResult(0));
    }

    // Update input of Operation. NewAddOp -> parent Op.
    mlir::IRMapping mapper;
    mapper.map(parentOp->getOperands(), newParentOpOperands);
    auto newParentOp = rewriter.clone(*parentOp, mapper);

    // The input and output element type must be the same for AffineReshape/Transpose/Reshape after swap
    const auto parentInputType = mlir::dyn_cast<vpux::NDTypeInterface>(newParentOp->getOpOperand(0).get().getType());
    const auto oldParentOpOutType = mlir::dyn_cast<vpux::NDTypeInterface>(newParentOp->getResult(0).getType());
    const auto newParentOpOutType = oldParentOpOutType.changeElemType(parentInputType.getElementType());
    newParentOp->getResult(0).setType(newParentOpOutType);

    // Remove old Add ops.
    rewriter.replaceOp(origOp, newParentOp);

    return mlir::success();
}

//
// SwapWithActivation
//

template <class Activation>
class SwapWithActivation final : public mlir::OpRewritePattern<Activation> {
public:
    SwapWithActivation(mlir::MLIRContext* ctx, Logger log, bool seOpsEnabled, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<Activation>(ctx, benefit), _log(log), _seOpsEnabled(seOpsEnabled) {
        this->setDebugName("SwapWithActivation");
    }

public:
    mlir::LogicalResult matchAndRewrite(Activation origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _seOpsEnabled;
};

template <class Activation>
mlir::LogicalResult SwapWithActivation<Activation>::matchAndRewrite(Activation origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Found activation function {1}", this->getDebugName(), origOp);

    auto parentOp = origOp.getInput().getDefiningOp();

    if (parentOp == nullptr) {
        return mlir::failure();
    }

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    if (!vpux::IE::isSupportedElemTypeInfoCase(parentOp, _seOpsEnabled, logCb)) {
        return mlir::failure();
    }

    if (!mlir::isa<IE::ElemTypeInfoOpInterface>(parentOp) || mlir::isa<IE::LayerWithPostOpInterface>(parentOp) ||
        mlir::isa<IE::SliceOp>(parentOp) || mlir::isa<Activation>(parentOp)) {
        _log.trace("[{0}] Swapped operation {1} doesn't implement ElemTypeInfoOpInterface interface {0} or it is an "
                   "activation",
                   this->getDebugName(), parentOp);
        return mlir::failure();
    }

    if (!parentOp->hasOneUse()) {
        _log.trace("[{0}] Swapped operation {1} has more than one use", this->getDebugName(), parentOp);
        return mlir::failure();
    }

    for (mlir::Value parentInput : parentOp->getOperands()) {
        if (mlir::cast<vpux::NDTypeInterface>(parentInput.getType()).getRank() != SUPPORTED_RANK) {
            _log.trace("[{0}] Swapped operation doesn't have rank {1}", this->getDebugName(), SUPPORTED_RANK);
            return mlir::failure();
        }
    }

    mlir::Value origOperand = origOp->getResult(0);
    const auto origOrder = mlir::cast<vpux::NDTypeInterface>(origOperand.getType()).getDimsOrder();
    mlir::Value parentInput = parentOp->getOperand(0);
    const auto parentOrder = mlir::cast<vpux::NDTypeInterface>(parentInput.getType()).getDimsOrder();

    if (!checkOrderCompatible(origOp, origOrder, parentOrder)) {
        return mlir::failure();
    }

    rewriter.startOpModification(parentOp);
    rewriter.setInsertionPoint(parentOp);

    auto origElemType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType()).getElementType();
    if (mlir::template dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(origElemType)) {
        return mlir::failure();
    }

    const auto parentOpInputs = parentOp->getOperands();
    for (auto i : irange<size_t>(0, parentOpInputs.size())) {
        auto newActivation = rewriter.clone(*origOp);
        extendOpLoc(newActivation, "act_{0}", i);
        newActivation->setOperand(0, parentOpInputs[i]);
        newActivation->getOpResult(0).setType(parentOpInputs[i].getType());
        if (mlir::isa<IE::LeakyReluOp>(origOp)) {
            auto origElemType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType()).getElementType();
            auto newType = mlir::cast<vpux::NDTypeInterface>(newActivation->getOpResult(0).getType());
            newActivation->getOpResult(0).setType(newType.changeElemType(origElemType));
        }
        parentOp->getOpOperand(static_cast<uint32_t>(i)).set(newActivation->getResult(0));
    }
    inferReturnTypes(parentOp, InferShapedTypeMode::ELEM_TYPE);
    rewriter.replaceOp(origOp, parentOp->getResults());

    rewriter.finalizeOpModification(parentOp);

    return mlir::success();
}

class SwapGeluExpand final : public mlir::OpRewritePattern<IE::GeluOp> {
public:
    SwapGeluExpand(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::GeluOp>(ctx, benefit), _log(log) {
        this->setDebugName("SwapOperationsPass::SwapGeluExpand");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::GeluOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapGeluExpand::matchAndRewrite(IE::GeluOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto expandOp = origOp.getInput().getDefiningOp<IE::ExpandOp>();
    if (expandOp == nullptr) {
        return mlir::failure();
    }

    if (!expandOp->hasOneUse()) {
        return mlir::failure();
    }

    auto newGelu = rewriter.create<IE::GeluOp>(origOp.getLoc(), expandOp.getInput());
    auto newExpand = rewriter.replaceOpWithNewOp<IE::ExpandOp>(origOp, newGelu.getOutput(), expandOp.getPadsBeginAttr(),
                                                               expandOp.getPadsEndAttr());
    extendOpLoc(newExpand, "swap");

    return mlir::success();
}

class SwapTanhSlice final : public mlir::OpRewritePattern<IE::TanhOp> {
public:
    SwapTanhSlice(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::TanhOp>(ctx, benefit), _log(log) {
        this->setDebugName("SwapOperationsPass::SwapTanhSlice");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::TanhOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapTanhSlice::matchAndRewrite(IE::TanhOp originOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", originOp->getName(), originOp->getLoc());
    auto sliceOp = originOp.getInput().getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        return mlir::failure();
    }

    auto oldSliceType = mlir::cast<vpux::NDTypeInterface>(sliceOp->getResult(0).getType());
    auto oldLayerType = mlir::cast<vpux::NDTypeInterface>(originOp->getResult(0).getType());
    auto newType =
            oldLayerType.changeShape(mlir::cast<vpux::NDTypeInterface>(sliceOp.getSource().getType()).getShape());

    const auto oldSliceShape = oldSliceType.getShape();
    const auto newLayerShape = newType.getShape();

    // Move tanH only when the slice is due to channel alignment X % 16 != 0
    if (oldSliceShape[Dims4D::Act::C] % CHANNEL_ALIGNMENT == 0) {
        return mlir::failure();
    }

    // In case when actual number of channels is less than 1/2 of the aligned channel value
    // Such cases avoid moving TanH as it would be computationally expensive operation and does not offer any gain
    // e.g. Actual channels: 3 Aligned Channels 16, we don't want to compute TanH with 16 Channels for such case
    if (oldSliceShape[Dims4D::Act::C] < newLayerShape[Dims4D::Act::C] / 2) {
        return mlir::failure();
    }

    auto newOp = rewriter.create<IE::TanhOp>(originOp.getLoc(), newType, sliceOp.getSource());
    auto newSlice = rewriter.replaceOpWithNewOp<IE::SliceOp>(
            originOp, newOp->getResult(0), sliceOp.getStaticOffsetsAttr(), sliceOp.getStaticSizesAttr());
    extendOpLoc(newSlice, "swap");
    newSlice->getResult(0).setType(oldSliceType);

    return mlir::success();
}

//
// SwapExpandQuantizeCast
//
// Move the QuantizeCast before Expand
// to support the possible Expand-Copy optimization in the following passes
//   Expand                         QuantizeCast
//      |                                 |
// QuantizeCast              ->        Expand

class SwapExpandQuantizeCast final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    SwapExpandQuantizeCast(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx, benefit), _log(log) {
        this->setDebugName("SwapExpandQuantizeCast");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp expandOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapExpandQuantizeCast::matchAndRewrite(IE::ExpandOp expandOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), expandOp->getName(), expandOp->getLoc());
    if (!expandOp->hasOneUse()) {
        return mlir::failure();
    }
    auto quantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(*expandOp.getOutput().getUsers().begin());
    if (quantizeCastOp == nullptr) {
        return mlir::failure();
    }
    auto quantizeCastOutputType = mlir::cast<vpux::NDTypeInterface>(quantizeCastOp.getOutput().getType());
    auto quantizeCastInputType = mlir::cast<vpux::NDTypeInterface>(quantizeCastOp.getInput().getType());
    auto isPerChannel = [](vpux::NDTypeInterface type) {
        return mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(type.getElementType());
    };
    if (isPerChannel(quantizeCastInputType) || isPerChannel(quantizeCastOutputType)) {
        return mlir::failure();
    }
    auto log = _log.nest();
    log.trace("Got Expand-QuantizeCast pattern: {0} -> {1}", expandOp->getLoc(), quantizeCastOp->getLoc());
    // Swap Expand-QuantizeCast to QuantizeCast-Expand
    auto expandInput = expandOp.getInput();
    auto quantizeCastOutputElemType = quantizeCastOutputType.getElementType();
    auto newQuantizeCastOp =
            rewriter.create<IE::QuantizeCastOp>(quantizeCastOp->getLoc(), expandInput, quantizeCastOutputElemType);
    expandOp.setOperand(newQuantizeCastOp);
    expandOp.getOutput().setType(mlir::cast<mlir::RankedTensorType>(quantizeCastOutputType));
    quantizeCastOp->replaceAllUsesWith(expandOp);
    log.trace("Swapped the Expand-QuantizeCast pattern: {0} -> {1}", newQuantizeCastOp->getLoc(), expandOp->getLoc());
    return mlir::success();
}

//
// SwapTanhShapeCast
//

class SwapTanhShapeCast final : public mlir::OpRewritePattern<IE::TanhOp> {
public:
    SwapTanhShapeCast(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::TanhOp>(ctx, benefit), _log(log) {
        this->setDebugName("SwapOperationsPass::SwapTanhShapeCast");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::TanhOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapTanhShapeCast::matchAndRewrite(IE::TanhOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    auto shapeCastOp = origOp.getInput().getDefiningOp<IE::ShapeCastOp>();
    if (shapeCastOp == nullptr) {
        return mlir::failure();
    }

    if (!shapeCastOp->hasOneUse()) {
        return mlir::failure();
    }

    auto newTanhOp =
            rewriter.create<IE::TanhOp>(origOp.getLoc(), shapeCastOp.getSource().getType(), shapeCastOp.getSource());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getResult().getType());
    auto castOp = rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(
            origOp, outputType, newTanhOp.getResult(), getIntArrayAttr(origOp.getContext(), outputType.getShape()));
    extendOpLoc(castOp, "swap");
    return mlir::success();
}

class SwapDequantMemPermute final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    SwapDequantMemPermute(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit), _log(log) {
        this->setDebugName("SwapDequantMemPermute");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapDequantMemPermute::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), memPermuteOp->getName(), memPermuteOp->getLoc());
    // const -> IE.Dequantize -> IE.MemPermute
    // const -> IE.MemPermute -> IE.Dequantize
    // const [IE.MemPermute] -> IE.Dequantize
    auto dequant = memPermuteOp.getInput().getDefiningOp<IE::DequantizeOp>();
    if (dequant == nullptr) {
        return mlir::failure();
    }

    if (!mlir::matchPattern(dequant.getInput(), mlir::m_Constant())) {
        return mlir::failure();
    }

    auto newPermute = rewriter.create<IE::MemPermuteOp>(memPermuteOp.getLoc(), dequant.getInput(),
                                                        memPermuteOp.getDstOrderAttr(), memPermuteOp.getMemPermAttr());

    auto newDequant =
            rewriter.create<IE::DequantizeOp>(dequant.getLoc(), newPermute.getOutput(), dequant.getDstElemType());

    rewriter.replaceOp(memPermuteOp, newDequant.getOutput());
    return mlir::success();
}

//
// SwapAffineReshapeFakeQuantize
//

class SwapAffineReshapeFakeQuantize final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    SwapAffineReshapeFakeQuantize(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx, benefit), _log(log) {
        setDebugName("SwapAffineReshapeFakeQuantize");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp fakeQuantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::FailureOr<ConcreteOp> hasSingleValueBiasUser(mlir::Operation* operation) {
    auto user = std::find_if(operation->user_begin(), operation->user_end(), [](mlir::Operation* user) {
        if (!mlir::isa<ConcreteOp>(user)) {
            return false;
        }
        auto concreteOp = mlir::cast<ConcreteOp>(user);
        bool lhsIsActivation = mlir::failed(IE::getConstParentOp(concreteOp.getInput1()));
        auto biasInput = lhsIsActivation ? concreteOp.getInput2() : concreteOp.getInput1();
        return isSingleValueBias(biasInput);
    });

    if (user != operation->user_end()) {
        return mlir::cast<ConcreteOp>(*user);
    } else {
        return mlir::failure();
    }
}

mlir::LogicalResult SwapAffineReshapeFakeQuantize::matchAndRewrite(IE::FakeQuantizeOp fakeQuantizeOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), fakeQuantizeOp->getName(), fakeQuantizeOp->getLoc());

    if (getShape(fakeQuantizeOp.getInput()) != getShape(fakeQuantizeOp.getOutput())) {
        return matchFailed(_log, rewriter, fakeQuantizeOp, "FakeQuantizeOp shape changed");
    }

    // IE::isPerTensorFQ returns false if any of arguments is Per Axis
    if (!IE::isPerTensorFQ({fakeQuantizeOp})) {
        return matchFailed(_log, rewriter, fakeQuantizeOp, "FakeQuantizeOp is per-axis");
    }

    auto affineReshapeOp = fakeQuantizeOp.getInput().getDefiningOp<IE::AffineReshapeOp>();
    if (affineReshapeOp == nullptr) {
        return matchFailed(_log, rewriter, fakeQuantizeOp, "AffineReshapeOp not found");
    }
    if (!affineReshapeOp->hasOneUse()) {
        return matchFailed(_log, rewriter, fakeQuantizeOp, "AffineReshapeOp has multiple uses");
    }

    // Swap with FQ-Gelu-FQ could result in worse performance
    // TODO(E#144643): confirm if it's possible to remove this constraint
    auto hasGeluUser = [fakeQuantizeOp]() {
        auto geluUser =
                std::find_if(fakeQuantizeOp->user_begin(), fakeQuantizeOp->user_end(), [](mlir::Operation* user) {
                    return mlir::isa<IE::GeluOp>(user);
                });
        return geluUser != fakeQuantizeOp->user_end();
    }();
    if (hasGeluUser) {
        return matchFailed(_log, rewriter, fakeQuantizeOp, "Do not swap with FQ when user has Gelu");
    }

    // Swap with FQ-Add-Mul could result in worse performance
    // TODO(E#144643): confirm if it's possible to remove this constraint
    auto hasSingleValueBiasAddMulUser = [fakeQuantizeOp]() {
        auto addOp = hasSingleValueBiasUser<IE::AddOp>(fakeQuantizeOp);
        if (mlir::failed(addOp)) {
            return false;
        };
        auto multiplyOp = hasSingleValueBiasUser<IE::MultiplyOp>(addOp.value());
        return mlir::succeeded(multiplyOp);
    }();
    if (hasSingleValueBiasAddMulUser) {
        return matchFailed(_log, rewriter, fakeQuantizeOp,
                           "Do not swap FQ when user has singleValueBias Add and Multiply");
    }

    if (IE::doesAffineReshapeChangeRank(affineReshapeOp)) {
        return matchFailed(_log, rewriter, fakeQuantizeOp, "AffineReshapeOp changes rank");
    }

    _log.trace("[{0}] Swap '{1}' at '{2}' with  '{3}' at '{4}'", getDebugName(), affineReshapeOp->getName(),
               affineReshapeOp->getLoc(), fakeQuantizeOp->getName(), fakeQuantizeOp->getLoc());

    auto newFakeQuantizeOp = rewriter.create<IE::FakeQuantizeOp>(
            fakeQuantizeOp->getLoc(), affineReshapeOp.getInput(), fakeQuantizeOp.getInputLow(),
            fakeQuantizeOp.getInputHigh(), fakeQuantizeOp.getOutputLow(), fakeQuantizeOp.getOutputHigh(),
            fakeQuantizeOp.getLevelsAttr(), fakeQuantizeOp.getLowFpTypeAttr(), fakeQuantizeOp.getAutoBroadcastAttr());

    auto newAffineReshapeOp = rewriter.create<IE::AffineReshapeOp>(
            affineReshapeOp.getLoc(), newFakeQuantizeOp.getOutput(), affineReshapeOp.getDimMappingAttr(),
            affineReshapeOp.getShapeValueAttr());
    fakeQuantizeOp.replaceAllUsesWith(newAffineReshapeOp.getOutput());

    return mlir::success();
}

//
// SwapBatchedMemPermuteSlice
//

class SwapBatchedMemPermuteSlice final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    SwapBatchedMemPermuteSlice(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit), _log(log) {
        this->setDebugName("SwapBatchedMemPermuteSlice");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapBatchedMemPermuteSlice::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), memPermuteOp->getName(), memPermuteOp->getLoc());

    auto inputShape = getShape(memPermuteOp.getInput());
    const auto batchSize = inputShape[Dims4D::Act::N];
    if (inputShape.size() < 4 || batchSize == 1) {
        return mlir::failure();
    }

    auto inOrder = mlir::cast<vpux::NDTypeInterface>(memPermuteOp.getInput().getType()).getDimsOrder();
    const auto srcPermutedOrder = applyPermutation(inOrder, DimsOrder::fromAffineMap(memPermuteOp.getMemPerm()));
    const auto fullPermAffineMap =
            applyPermutation(srcPermutedOrder, DimsOrder::fromAffineMap(memPermuteOp.getDstOrder()))
                    .DimsOrder::toAffineMap(rewriter.getContext());
    auto firstDimExpr = mlir::cast<mlir::AffineDimExpr>(fullPermAffineMap.getResults()[0]);
    // Check that the N dimension (batch) does not change position after MemPermute
    if (firstDimExpr.getPosition() != 0) {
        _log.trace("d0 dimension position changed from 0 to {0}", firstDimExpr.getPosition());
        return mlir::failure();
    }

    // check if all users are valid sliceOp
    auto hasRestrictedUsers = llvm::any_of(memPermuteOp->getUsers(), [&](auto user) {
        if (auto sliceOp = mlir::dyn_cast_if_present<IE::SliceOp>(user)) {
            auto sliceShape = getShape(sliceOp.getResult());
            if (sliceShape[Dims4D::Act::N] == 1 && sliceShape[Dims4D::Act::C] == inputShape[Dims4D::Act::C] &&
                sliceShape[Dims4D::Act::H] == inputShape[Dims4D::Act::H] &&
                sliceShape[Dims4D::Act::W] == inputShape[Dims4D::Act::W]) {
                return false;
            }
        }
        return true;
    });

    if (hasRestrictedUsers) {
        return mlir::failure();
    }

    llvm::SmallVector<mlir::Operation*> sliceUsers(memPermuteOp->getUsers().begin(), memPermuteOp->getUsers().end());
    llvm::sort(sliceUsers, [](mlir::Operation* a, mlir::Operation* b) {
        auto sliceA = mlir::cast<IE::SliceOp>(a);
        auto sliceB = mlir::cast<IE::SliceOp>(b);
        auto offsetsA = parseIntArrayAttr<int64_t>(sliceA.getStaticOffsetsAttr());
        auto offsetsB = parseIntArrayAttr<int64_t>(sliceB.getStaticOffsetsAttr());
        return offsetsA[0] < offsetsB[0];
    });

    rewriter.setInsertionPoint(memPermuteOp);

    SmallVector<mlir::Value> newResults;
    for (size_t i = 0; i < sliceUsers.size(); ++i) {
        auto sliceOp = mlir::cast<IE::SliceOp>(sliceUsers[i]);

        // Get the actual batch offset from the original slice
        auto batchOffset = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsetsAttr())[0];

        auto inputSlice = rewriter.create<IE::SliceOp>(takeOpLoc(memPermuteOp, "input_slice_{0}", batchOffset),
                                                       memPermuteOp.getInput(), sliceOp.getStaticOffsetsAttr(),
                                                       sliceOp.getStaticSizesAttr());

        auto newMemPermute = rewriter.create<IE::MemPermuteOp>(
                takeOpLoc(memPermuteOp, "batch_permute_{0}", batchOffset), inputSlice.getResult(),
                memPermuteOp.getDstOrderAttr(), memPermuteOp.getMemPermAttr());

        auto expectedOutputType = sliceOp.getResult().getType();
        newMemPermute.getResult().setType(expectedOutputType);

        newResults.push_back(newMemPermute.getResult());
    }

    for (size_t i = 0; i < sliceUsers.size(); ++i) {
        rewriter.replaceOp(sliceUsers[i], newResults[i]);
    }

    rewriter.eraseOp(memPermuteOp);

    return mlir::success();
}

bool isSafeToTraverse(mlir::Operation* op, Logger log, bool seOpsEnabled) {
    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };

    if (mlir::isa<IE::LayerWithPostOpInterface, IE::SliceOp, IE::MultiplyOp>(op)) {
        return false;
    }

    if (!op->hasOneUse()) {
        log.trace("Operation {0} has more than one use", *op);
        return false;
    }

    auto isTraversableCastOp = [](mlir::Operation* op) -> bool {
        if (mlir::isa<IE::QuantizeCastOp>(op)) {
            return false;
        }
        return mlir::isa<IE::ViewLikeOpInterface>(op);
    };

    if (!vpux::IE::isSupportedElemTypeInfoCase(op, seOpsEnabled, logCb) && !isTraversableCastOp(op)) {
        log.trace("Operation {0} does not implement the ElemTypeInfoOpInterface and is not a traversable "
                  "ViewLikeOpInterface.",
                  *op);
        return false;
    }

    for (mlir::Value parentInput : op->getOperands()) {
        if (mlir::cast<vpux::NDTypeInterface>(parentInput.getType()).getRank() != SUPPORTED_RANK) {
            log.trace("Operation {0} doesn't have rank {1}", *op, SUPPORTED_RANK);
            return false;
        }
    }

    return true;
}

bool isSafeToSwap(mlir::Operation* producerOp, mlir::Operation* firstUserOp, mlir::Operation* activationOp) {
    if (!producerOp) {
        return false;
    }

    if (firstUserOp == activationOp) {
        return false;
    }

    // TODO: E#194430 Replace Clamp rewriter with new implementation
    if (mlir::isa_and_present<IE::ClampOp>(activationOp)) {
        return false;
    }

    mlir::Value origOperand = activationOp->getResult(0);
    const auto origOrder = mlir::cast<vpux::NDTypeInterface>(origOperand.getType()).getDimsOrder();
    mlir::Value producerOperand = producerOp->getResult(0);
    const auto producerOrder = mlir::cast<vpux::NDTypeInterface>(producerOperand.getType()).getDimsOrder();

    if (!checkOrderCompatible(activationOp, origOrder, producerOrder)) {
        return false;
    }

    auto origElemType = mlir::cast<vpux::NDTypeInterface>(activationOp->getResult(0).getType()).getElementType();
    if (mlir::template dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(origElemType)) {
        return false;
    }

    return true;
}

bool isBeneficialToSwap(mlir::Operation* op, mlir::Operation* firstUserOp, mlir::Operation* candidateOp, Logger log) {
    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };

    auto producerOp = mlir::dyn_cast_if_present<IE::LayerWithPostOpInterface>(op);

    bool isSupportedPPE = mlir::isa_and_nonnull<IE::ClampOp>(candidateOp)
                                  ? producerOp.isSupportedClampOp(candidateOp, logCb)
                                  : producerOp.isSupportedPostOp(candidateOp, logCb);

    // TODO: E#194429 Remove special Add after accuracy regressions are solved
    bool isSpecialCase = mlir::isa_and_present<IE::AddOp>(producerOp) &&
                         mlir::isa_and_present<IE::SigmoidOp, IE::ReLUOp, IE::LeakyReluOp>(candidateOp);

    return (isSupportedPPE || isSpecialCase) && isSafeToSwap(op, firstUserOp, candidateOp);
}

void swapActivation(mlir::PatternRewriter& rewriter, mlir::Operation* firstUserOp, mlir::Operation* prevOp,
                    mlir::Operation* activationOp, mlir::Operation* concatOp, mlir::Operation* beforeConcatOp) {
    auto moveOperation = [&](mlir::Operation* moveBeforeOp, mlir::Operation* previousOp, mlir::Operation* opToBeMoved) {
        const auto inputValues = moveBeforeOp->getOperands();
        rewriter.startOpModification(moveBeforeOp);
        rewriter.setInsertionPoint(moveBeforeOp);

        for (auto i : irange<size_t>(0, inputValues.size())) {
            auto* newActivation = rewriter.clone(*opToBeMoved);
            extendOpLoc(newActivation, StringLiteral("act_{0}"), i);
            newActivation->setOperand(0, inputValues[i]);
            newActivation->getOpResult(0).setType(inputValues[i].getType());
            inferReturnTypes(newActivation, InferShapedTypeMode::ELEM_TYPE);

            if (mlir::isa<IE::LeakyReluOp>(opToBeMoved)) {
                auto origElemType =
                        mlir::cast<vpux::NDTypeInterface>(opToBeMoved->getResult(0).getType()).getElementType();
                auto newType = mlir::cast<vpux::NDTypeInterface>(newActivation->getOpResult(0).getType());
                newActivation->getOpResult(0).setType(newType.changeElemType(origElemType));
            }

            moveBeforeOp->getOpOperand(static_cast<uint32_t>(i)).set(newActivation->getResult(0));
        }

        // Update moveBeforeOp's output type first: its input operand was just changed to the new
        // activation (which may carry a different element type), so its declared output type is now
        // stale. Without this call, previousOp would see the old element type on moveBeforeOp's
        // output and infer the wrong type for its own output.
        inferReturnTypes(moveBeforeOp, InferShapedTypeMode::ELEM_TYPE);
        // Now propagate the updated element type through previousOp so that when opToBeMoved is
        // replaced by previousOp's result, consumers receive the correct output type.
        inferReturnTypes(previousOp, InferShapedTypeMode::ELEM_TYPE);
        rewriter.replaceOp(opToBeMoved, previousOp->getResults());
        rewriter.finalizeOpModification(moveBeforeOp);
    };

    // if we have a Concat op in between firstUserOp and the activation
    // we need to first move the activation before the concat
    // after that we can move it again before firstUserOp
    if (concatOp && !mlir::isa<IE::ConcatOp>(firstUserOp)) {
        moveOperation(concatOp, prevOp, activationOp);
        activationOp = *beforeConcatOp->getUsers().begin();
        prevOp = beforeConcatOp;
    }

    moveOperation(firstUserOp, prevOp, activationOp);
}

//
// FindChildActivationAndSwap
//

class FindChildActivationAndSwap final : public mlir::OpInterfaceRewritePattern<IE::LayerWithPostOpInterface> {
public:
    FindChildActivationAndSwap(mlir::MLIRContext* ctx, Logger log, bool seOpsEnabled, mlir::PatternBenefit benefit = 1)
            : mlir::OpInterfaceRewritePattern<IE::LayerWithPostOpInterface>(ctx, benefit),
              _log(log),
              _seOpsEnabled(seOpsEnabled) {
        this->setDebugName("FindChildActivationAndSwap");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayerWithPostOpInterface op, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _seOpsEnabled;
};

mlir::LogicalResult FindChildActivationAndSwap::matchAndRewrite(IE::LayerWithPostOpInterface op,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), op->getName(), op->getLoc());
    for (auto* firstUserOp : op->getUsers()) {
        mlir::Operation* prevOp = firstUserOp;
        mlir::Operation* candidateOp = firstUserOp;
        mlir::Operation* concatOp = nullptr;
        mlir::Operation* beforeConcatOp = nullptr;

        // we should check every user of the DPU op to see if it's an activation or
        // if we can move an activation before it when we actually find one
        while (candidateOp) {
            // check to see if we can move an activation before prevOp
            if (!isSafeToTraverse(prevOp, _log, _seOpsEnabled)) {
                break;
            }

            // if we find a Concat operation during our traversal we need to
            // take it into consideration when we actually move the activation
            if (concatOp == nullptr && mlir::isa<IE::ConcatOp>(candidateOp)) {
                concatOp = candidateOp;
                beforeConcatOp = prevOp;
            }

            auto nextOpToCheck = candidateOp;
            if (isBeneficialToSwap(op, firstUserOp, candidateOp, _log)) {
                // if we move the activation from it's place then
                // the next operation's user we need to check will be the prevOp
                // because after the move is done, it will have a new user
                nextOpToCheck = prevOp;
                swapActivation(rewriter, firstUserOp, prevOp, candidateOp, concatOp, beforeConcatOp);
            }

            auto listOfUsers = nextOpToCheck->getUsers();
            if (!listOfUsers.empty()) {
                prevOp = nextOpToCheck;
                candidateOp = *listOfUsers.begin();
            } else {
                break;
            }
        }
    }

    return mlir::failure();
}

//
// SwapLeakyReluTranspose
//

class SwapLeakyReluTranspose final : public mlir::OpRewritePattern<IE::LeakyReluOp> {
public:
    SwapLeakyReluTranspose(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::LeakyReluOp>(ctx, benefit), _log(log) {
        this->setDebugName("SwapLeakyReluTranspose");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::LeakyReluOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapLeakyReluTranspose::matchAndRewrite(IE::LeakyReluOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto transposeOp = origOp.getInput().getDefiningOp<IE::TransposeOp>();
    if (transposeOp == nullptr) {
        return mlir::failure();
    }

    auto convertOp = transposeOp.getInput().getDefiningOp<IE::ConvertOp>();
    if (convertOp == nullptr) {
        return mlir::failure();
    }

    if (!isSafeToSwap(convertOp, transposeOp, origOp)) {
        return mlir::failure();
    }

    auto newLeakyRelu =
            rewriter.create<IE::LeakyReluOp>(origOp.getLoc(), convertOp.getOutput(), origOp.getNegativeSlopeAttr());
    auto origElemType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType()).getElementType();
    auto newType = mlir::cast<vpux::NDTypeInterface>(newLeakyRelu->getOpResult(0).getType());
    newLeakyRelu->getOpResult(0).setType(newType.changeElemType(origElemType));

    auto newTranspose = rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, newLeakyRelu.getOutput(), nullptr,
                                                                     transposeOp.getOrderValueAttr());
    extendOpLoc(newTranspose, "swap");

    newLeakyRelu->moveAfter(convertOp);
    newTranspose->moveAfter(newLeakyRelu);

    return mlir::success();
}

/*
Gather              QuantizeCast
     |         =>        |
QuantizeCast          Gather
*/
class SwapGatherQuantizeCast final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    SwapGatherQuantizeCast(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::GatherOp>(ctx, benefit), _log(log) {
        setDebugName("SwapGatherQuantizeCast");
    }

    mlir::LogicalResult matchAndRewrite(IE::GatherOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapGatherQuantizeCast::matchAndRewrite(IE::GatherOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    if (!origOp->hasOneUse()) {
        return mlir::failure();
    }

    auto quantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(*origOp->getUsers().begin());
    if (quantizeCastOp == nullptr) {
        return mlir::failure();
    }

    if (IE::isPerAxisQuant(quantizeCastOp.getInput()) || IE::isPerAxisQuant(quantizeCastOp.getOutput())) {
        return matchFailed(_log, rewriter, origOp, "QuantizeCastOp doesn't have valid quantize type");
    }

    auto outputType = mlir::cast<vpux::NDTypeInterface>(quantizeCastOp.getOutput().getType());
    auto outElemType = outputType.getElementType();
    auto newQuantizeCastOp =
            rewriter.create<IE::QuantizeCastOp>(quantizeCastOp->getLoc(), origOp.getInput(), outElemType);
    auto newGatherOp = rewriter.create<IE::GatherOp>(origOp.getLoc(), newQuantizeCastOp.getOutput(),
                                                     origOp.getIndices(), origOp.getAxis(), origOp.getAxisValueAttr(),
                                                     origOp.getBatchDims(), origOp.getIndicesRankAttr());
    rewriter.replaceOp(quantizeCastOp, newGatherOp.getOutput());
    return mlir::success();
}

//
// SwapOperationsPass
//

class SwapOperationsPass final : public IE::impl::SwapOperationsBase<SwapOperationsPass> {
public:
    explicit SwapOperationsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SwapOperationsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto op = getModuleOp(func);
    mlir::RewritePatternSet patterns(&ctx);

    const auto seOpsEnabled = config::hasEnableSEPtrsOperations(op);
    const auto experimentalSeOpsEnabled = config::hasEnableExperimentalSEPtrsOperations(op);

    // TODO: E#194430 Replace Clamp rewriter with new implementation
    patterns.add<SwapWithActivation<IE::ClampOp>>(&ctx, _log.nest(), seOpsEnabled || experimentalSeOpsEnabled);
    patterns.add<SwapGeluExpand>(&ctx, _log.nest(), seOpsEnabled || experimentalSeOpsEnabled);
    patterns.add<SwapWithBias<IE::AddOp>>(&ctx, _log.nest());
    patterns.add<SwapWithBias<IE::ScaleShiftOp>>(&ctx, _log.nest());
    // TODO: E#18651 Support ElemTypeInfoOpInterface for Slice
    patterns.add<SwapTanhSlice>(&ctx, _log.nest());
    patterns.add<SwapTanhShapeCast>(&ctx, _log.nest());
    patterns.add<SwapExpandQuantizeCast>(&ctx, _log.nest());
    patterns.add<SwapGatherQuantizeCast>(&ctx, _log.nest());
    patterns.add<SwapDequantMemPermute>(&ctx, _log.nest());
    patterns.add<SwapBatchedMemPermuteSlice>(&ctx, _log.nest());
    patterns.add<SwapAffineReshapeFakeQuantize>(&ctx, _log.nest());
    IE::AffineReshapeOp::getCanonicalizationPatterns(patterns, &ctx);

    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    {
        // we look for DPU operations that we can later fuse activations into
        mlir::RewritePatternSet activationPatterns(&ctx);
        activationPatterns.add<FindChildActivationAndSwap>(&ctx, _log.nest(), seOpsEnabled || experimentalSeOpsEnabled);

        collectOpsAndApplyPatterns(func, std::move(activationPatterns));
    }

    {
        // TODO: E#195330 Remove this extra case
        // this extra walk is done do avoid performance regressions
        mlir::RewritePatternSet activationPatterns(&ctx);
        activationPatterns.add<SwapLeakyReluTranspose>(&ctx, _log.nest());

        collectOpsAndApplyPatterns(func, std::move(activationPatterns));
    }
}

}  // namespace

//
// createSwapOperationsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSwapOperationsPass(Logger log) {
    return std::make_unique<SwapOperationsPass>(log);
}

void vpux::IE::registerSwapOperationsRewriters(RewriterRegistry& registry, ArrayRef<mlir::PatternBenefit> benefitLevels,
                                               size_t index, bool seOpsEnabled, Logger log) {
    registry.registerRewriterSet("swap-operations-set", [&registry, benefitLevels, index, seOpsEnabled, log]() {
        registry.registerRewriter<SwapWithActivation<IE::ClampOp>>("swap-with-activation-clamp", log.nest(),
                                                                   seOpsEnabled, benefitLevels[index]);
        registry.registerRewriter<SwapGeluExpand>("swap-gelu-expand", log.nest(), benefitLevels[index]);
        registry.registerRewriter<SwapWithBias<IE::AddOp>>("swap-add-with-bias", log.nest(), benefitLevels[index]);
        registry.registerRewriter<SwapWithBias<IE::ScaleShiftOp>>("swap-scaleshift-with-bias", log.nest(),
                                                                  benefitLevels[index]);
        // TODO: E#18651 Support ElemTypeInfoOpInterface for Slice
        registry.registerRewriter<SwapTanhSlice>("swap-tanh-slice", log.nest(), benefitLevels[index]);
        registry.registerRewriter<SwapTanhShapeCast>("swap-tanh-shape-cast", log.nest(), benefitLevels[index]);
        registry.registerRewriter<SwapExpandQuantizeCast>("swap-expand-quantize-cast", log.nest(),
                                                          benefitLevels[index]);
        registry.registerRewriter<SwapGatherQuantizeCast>("swap-gather-quantize-cast", log.nest(),
                                                          benefitLevels[index]);
        registry.registerRewriter<SwapDequantMemPermute>("swap-dequant-mem-permute", log.nest(), benefitLevels[index]);
        registry.registerRewriter<SwapBatchedMemPermuteSlice>("swap-batched-mem-permute-slice", log.nest(),
                                                              benefitLevels[index]);
        registry.registerRewriter<SwapAffineReshapeFakeQuantize>("swap-affine-reshape-fake-quantize", log.nest(),
                                                                 benefitLevels[index]);
        // Manually invoking this rewriter despite canonicalizer handling it. This is required for dynamic rewriter
        // implementation
        IE::registerAffineReshapeOpRewriters(registry, benefitLevels, index);
        registry.registerRewriter<FindChildActivationAndSwap>("find-child-activation-and-swap", log.nest(),
                                                              seOpsEnabled, benefitLevels[index + 1]);
        registry.registerRewriter<SwapLeakyReluTranspose>("swap-leakyrelu-transpose", log.nest(),
                                                          benefitLevels[index + 2]);
    });
}
