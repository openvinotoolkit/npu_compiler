//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTSPLITCONCATTOTRANSPOSE
#define GEN_PASS_DEF_CONVERTSPLITCONCATTOTRANSPOSE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

template <class ConvolutionType = IE::ConvolutionOp>
mlir::FailureOr<mlir::Operation*> getConcatOpConsumer(mlir::Operation* op, bool requireAffineReshape,
                                                      bool requireConvolution) {
    if (op == nullptr || op->getUsers().empty()) {
        return mlir::failure();
    }

    mlir::Operation* concatOp = nullptr;

    for (auto user : op->getUsers()) {
        mlir::Operation* operation = user;

        if (requireAffineReshape) {
            if (!mlir::isa<IE::AffineReshapeOp>(operation) || operation->getUsers().empty() ||
                (!requireConvolution && !operation->hasOneUse())) {
                return mlir::failure();
            }
            operation = *(operation->getUsers().begin());
        }

        if (requireConvolution) {
            if (!mlir::isa<ConvolutionType>(operation) || operation->getUsers().empty() || !operation->hasOneUse()) {
                return mlir::failure();
            }

            auto convOp = mlir::dyn_cast<ConvolutionType>(operation);
            auto constFilter = mlir::dyn_cast<Const::DeclareOp>(convOp.getFilter().getDefiningOp());
            if (constFilter == nullptr) {
                return mlir::failure();
            }

            auto isConst = [](mlir::Value value) {
                return mlir::isa<Const::DeclareOp>(value.getDefiningOp());
            };

            if (!isConst(convOp.getFilter())) {
                return mlir::failure();
            }

            operation = *(operation->getUsers().begin());
        }

        if (!mlir::isa<IE::ConcatOp>(operation)) {
            return mlir::failure();
        }

        if (concatOp == nullptr) {
            concatOp = operation;
            continue;
        } else if (concatOp != operation) {
            return mlir::failure();
        }
    }

    return concatOp;
}

// Check the split dim size after splitOp is 1 to make it feasible to convert into TransposeOp
mlir::FailureOr<vpux::Dim> getSplitDimToShape1(IE::SplitOp splitOp) {
    const auto splitInputShape = getShape(splitOp.getInput());
    const auto splitDim = Dim(splitOp.getAxisValue().value());
    const auto splitNum = splitOp.getNumSplits();

    if (splitInputShape[splitDim] != splitNum) {
        return mlir::failure();
    }

    return splitDim;
}

// Check the concat dim input size is 1 to make it feasible to convert into TransposeOp
mlir::FailureOr<SmallVector<Dim>> getConcatDimWithShape1(IE::ConcatOp concatOp, bool supportAdjacentDims) {
    const auto concatStaticOffsets = concatOp.getStaticOffsets().value();
    if (concatStaticOffsets.size() != concatOp.getInputs().size()) {
        return mlir::failure();
    }

    const auto concatInputType = mlir::cast<vpux::NDTypeInterface>(concatOp.getInputs()[0].getType());
    const auto concatOutputType = mlir::cast<vpux::NDTypeInterface>(concatOp.getOutput().getType());
    const auto concatInShape = concatInputType.getShape();
    const auto concatOutShape = concatOutputType.getShape();
    if (concatInShape.size() != concatOutShape.size()) {
        return mlir::failure();
    }

    SmallVector<Dim> concatDims;
    for (const auto& idx : irange(concatInShape.size())) {
        if (concatInShape[Dim(idx)] != concatOutShape[Dim(idx)]) {
            concatDims.push_back(Dim(idx));
        }
    }

    if (concatDims.empty() || concatDims.size() > 1) {
        return mlir::failure();
    }

    for (const auto& input : concatOp.getInputs()) {
        const auto inputShape = getShape(input);
        if (supportAdjacentDims) {
            SmallVector<Dim> adjustDims;
            if (concatDims[0].ind() - 1 > 0) {
                adjustDims.push_back(Dim(concatDims[0].ind() - 1));
            }
            if (concatDims[0].ind() + 1 < checked_cast<int32_t>(concatInShape.size())) {
                adjustDims.push_back(Dim(concatDims[0].ind() + 1));
            }

            for (auto dim : adjustDims) {
                if (inputShape[dim] == 1) {
                    concatDims[0] = dim;
                    break;
                }
            }

            if (inputShape[concatDims[0]] != 1) {
                return mlir::failure();
            }
        }
    }

    return concatDims;
}

bool isSupportedAffineReshape(IE::SplitOp splitOp) {
    auto userOp = splitOp.getOutputs()[0].getUsers().begin();
    auto affineReshapeOp = mlir::cast<IE::AffineReshapeOp>(*userOp);
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.getDimMapping());

    const auto affineInShape = getShape(affineReshapeOp.getInput());
    const auto affineOutShape = getShape(affineReshapeOp.getOutput());

    bool isInChEquivalentToOutH = affineInShape[Dims4D::Act::C] == affineOutShape[Dims4D::Act::H] &&
                                  std::any_of(dimMapping[Dims4D::Act::C.ind()].begin(),
                                              dimMapping[Dims4D::Act::C.ind()].end(), [](int value) {
                                                  return value == Dims4D::Act::H.ind();
                                              });
    bool isInHEquivalentToOutWOrCh = (affineInShape[Dims4D::Act::H] == affineOutShape[Dims4D::Act::W] ||
                                      affineInShape[Dims4D::Act::H] == affineOutShape[Dims4D::Act::C]) &&
                                     std::any_of(dimMapping[Dims4D::Act::H.ind()].begin(),
                                                 dimMapping[Dims4D::Act::H.ind()].end(), [](int value) {
                                                     return value == Dims4D::Act::W.ind();
                                                 });

    return isInChEquivalentToOutH && isInHEquivalentToOutWOrCh;
}

bool checkAffineReshapeDimMapping(IE::SplitOp splitOp) {
    auto userOp = splitOp.getOutputs()[0].getUsers().begin();
    auto affineReshapeOp = mlir::cast<IE::AffineReshapeOp>(*userOp);
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.getDimMapping());

    for (const auto& res : splitOp.getOutputs()) {
        auto curAffineReshape = mlir::cast<IE::AffineReshapeOp>(*res.getUsers().begin());
        const auto curDimMapping = parseIntArrayOfArrayAttr<int64_t>(curAffineReshape.getDimMapping());
        if (curDimMapping != dimMapping) {
            return false;
        }
    }

    const auto affineInShape = getShape(affineReshapeOp.getInput());
    const auto affineOutShape = getShape(affineReshapeOp.getOutput());
    for (size_t inIdx = 0; inIdx < dimMapping.size(); inIdx++) {
        auto mappedDim = dimMapping[inIdx];
        for (auto outIdx : mappedDim) {
            // merge case: N x 1 -> N
            // merge case: 1 x N -> N
            if (inIdx > 0 && mappedDim == dimMapping[inIdx - 1]) {
                if (affineInShape[Dim(inIdx)] != affineOutShape[Dim(outIdx)] && affineInShape[Dim(inIdx)] != 1) {
                    return false;
                } else if (affineInShape[Dim(inIdx - 1)] != affineOutShape[Dim(outIdx)] &&
                           affineInShape[Dim(inIdx - 1)] != 1) {
                    return false;
                }
            }

            // split case: N -> N x 1
            // split case: N -> 1 x N
            if (mappedDim.size() > 1) {
                if (affineInShape[Dim(inIdx)] != affineOutShape[Dim(outIdx)] && affineOutShape[Dim(outIdx)] != 1) {
                    return false;
                }
            }
        }
    }

    return true;
}

//
// SplitAffineReshapeConcatRewriter
//

//
//               |
//            SplitOp
//          /         \                                |
//   AffineReshape  AffineReshape       ->         Transpose
//          \         /                                |
//            ConcatOp
//               |
//

class SplitAffineReshapeConcatRewriter final : public mlir::OpRewritePattern<IE::SplitOp> {
public:
    SplitAffineReshapeConcatRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SplitOp>(ctx), _log(log) {
        setDebugName("SplitAffineReshapeConcatRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::SplitOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitAffineReshapeConcatRewriter::matchAndRewrite(IE::SplitOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("Rewrite Split operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto getConsumerResult = getConcatOpConsumer(origOp, true, false);
    if (mlir::failed(getConsumerResult)) {
        return mlir::failure();
    }

    auto concatOp = mlir::dyn_cast_or_null<IE::ConcatOp>(getConsumerResult.value());
    VPUX_THROW_WHEN(concatOp == nullptr, "Not a Concat operation");

    if (origOp.getOutputs().size() != concatOp.getInputs().size()) {
        return mlir::failure();
    }

    const auto concatInputType = mlir::cast<vpux::NDTypeInterface>(concatOp.getInputs()[0].getType());
    const auto concatInShape = concatInputType.getShape();

    // Supported case for splitOp: split the dim to shape 1
    auto getSplitDim = getSplitDimToShape1(origOp);
    if (mlir::failed(getSplitDim)) {
        return mlir::failure();
    }

    const auto splitDim = getSplitDim.value();

    // Supported case for concatOp: concat the dim with shape 1
    auto getconcatDims = getConcatDimWithShape1(concatOp, false);
    if (mlir::failed(getconcatDims)) {
        return mlir::failure();
    }

    const auto concatDims = getconcatDims.value();

    // affineReshapeOp dim_mapping supported cases:
    // merge case: N x 1 -> N, 1 x N -> N
    // split case: N -> N x 1, N -> 1 x N
    if (!checkAffineReshapeDimMapping(origOp)) {
        return mlir::failure();
    }

    // Create new transposeOp
    SmallVector<unsigned> transPerm(getShape(origOp.getInput()).size(), 0);
    for (const auto& idx : irange(concatInShape.size())) {
        if (Dim(idx) == splitDim) {
            transPerm[idx] = concatDims[0].ind();
        } else if (Dim(idx) == concatDims[0]) {
            transPerm[idx] = splitDim.ind();
        } else {
            transPerm[idx] = idx;
        }
    }

    const auto orderAttr =
            mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(transPerm, rewriter.getContext()));
    auto newTransposeOp =
            rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_in"), origOp.getInput(), nullptr, orderAttr);
    concatOp.replaceAllUsesWith(newTransposeOp.getOutput());

    _log.trace("[{0}] Replaced with 'IE::TransposeOp'", getDebugName());

    return mlir::success();
}

mlir::Value composeNewFilter(IE::ConcatOp concatOp, mlir::PatternRewriter& rewriter) {
    auto convOp = concatOp.getInputs()[0].getDefiningOp();

    const auto filterShape = getShape(convOp->getOperand(1));
    auto concatSize = concatOp.getInputs().size();

    SmallVector<mlir::Value> newFilterConstVec;
    for (unsigned int idx = 0; idx < concatSize; idx++) {
        auto inConvOp = concatOp.getInputs()[idx].getDefiningOp();
        auto constFilter = mlir::cast<Const::DeclareOp>(inConvOp->getOperand(1).getDefiningOp());

        // The split dim size after SplitOp is 1, so use the index to count the padding offset.
        Shape cstPadBegin = {0, idx, 0, 0};
        Shape cstPadEnd = {0, static_cast<unsigned int>(concatSize) - idx - 1, 0, 0};
        auto newConstFilterAttr = constFilter.getContentAttr().transform().padWithZero(cstPadBegin, cstPadEnd).get();

        const auto newFilterShape = Shape{filterShape[Dims4D::Filter::OC], static_cast<unsigned int>(concatSize),
                                          filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]};
        auto filterExpressedType = mlir::RankedTensorType::get(
                newFilterShape.raw(),
                mlir::cast<vpux::NDTypeInterface>(concatOp.getOutput().getType()).getElementType());
        auto newLoc = appendLoc(constFilter.getLoc(), "_{0}", idx);
        auto newConstFilter =
                rewriter.create<Const::DeclareOp>(newLoc, filterExpressedType, std::move(newConstFilterAttr));

        newFilterConstVec.push_back(newConstFilter.getOutput());
    }

    auto newConcatFilterLoc = appendLoc(concatOp.getLoc(), "_composed_weights");
    auto newFilter =
            rewriter.create<IE::ConcatOp>(newConcatFilterLoc, newFilterConstVec, Dims4D::Act::N.ind()).getOutput();

    return newFilter;
}

mlir::Value composeNewBias(IE::ConcatOp concatOp, mlir::PatternRewriter& rewriter) {
    auto concatSize = concatOp.getInputs().size();

    // If all Convolutions without bias, skip construct dummy bias.
    const auto atLeastOneConvHasBias = llvm::any_of(concatOp.getInputs(), [](mlir::Value input) {
        auto conv = input.getDefiningOp();
        return conv->getNumOperands() > 2;
    });

    if (!atLeastOneConvHasBias) {
        return nullptr;
    }

    SmallVector<mlir::Value> newBiasVec;
    for (unsigned int idx = 0; idx < concatSize; idx++) {
        auto inConvOp = concatOp.getInputs()[idx].getDefiningOp();

        if (inConvOp->getNumOperands() > 2) {
            newBiasVec.push_back(inConvOp->getOperand(2));
        } else {
            auto input = inConvOp->getOperand(0);
            auto inputShape = getShape(input);
            auto oc = inputShape[Dims4D::Act::C];
            const Shape biasShape = {1, oc, 1, 1};
            std::vector<float> biasValue(oc, .0f);

            const DimsOrder biasOrder = DimsOrder::NCHW;
            const auto biasType = mlir::RankedTensorType::get(
                    biasShape.raw(), mlir::cast<NDTypeInterface>(input.getType()).getElementType(),
                    getTensorAttr(rewriter.getContext(), biasOrder, nullptr, nullptr));

            auto newLoc = appendLoc(inConvOp->getLoc(), "_{0}", idx);

            newBiasVec.push_back(Const::buildWeightsConst(rewriter, newLoc, biasType, ArrayRef(biasValue)));
        }
    }

    auto newConcatBiasLoc = appendLoc(concatOp.getLoc(), "_composed_weights");
    auto newBias = rewriter.create<IE::ConcatOp>(newConcatBiasLoc, newBiasVec, Dims4D::Act::C.ind()).getOutput();

    return newBias;
}

mlir::Operation* createConvolution(IE::ConcatOp origOp, mlir::Value weights, mlir::Value bias, mlir::Value activation,
                                   mlir::PatternRewriter& rewriter) {
    const auto weightsShape = getShape(weights);
    const auto outChannels = weightsShape[Dims4D::Filter::OC];
    const Shape convInShape = getShape(activation).toValues();
    const Shape convOutShape = {convInShape[Dims4D::Act::N], outChannels, convInShape[Dims4D::Act::H],
                                convInShape[Dims4D::Act::W]};

    auto newConcatLoc = appendLoc(origOp.getLoc(), "_new_merged_conv");
    auto convLikeOp = origOp.getInputs()[0].getDefiningOp();
    mlir::IRMapping mapper;
    if (bias != nullptr) {
        // if bias found, mapping conv with bias to new conv with composed bias
        for (auto input : origOp.getInputs()) {
            auto convOp = input.getDefiningOp();
            if (convOp->getNumOperands() > 2) {
                convLikeOp = convOp;
                mapper.map(convLikeOp->getOperand(2), bias);
                break;
            }
        }
    }
    mapper.map(convLikeOp->getOperand(0), activation);
    mapper.map(convLikeOp->getOperand(1), weights);
    auto newOp = rewriter.clone(*convLikeOp, mapper);
    newOp->setLoc(newConcatLoc);
    return newOp;
}

//
// SplitAffineReshapeConvConcatRewriter
//

//                 |
//              SplitOp                                    |
//              /      \                               Transpose
//  AffineReshape       AffineReshape       ->             |     cst_0  cst_1
//     |     cst_0  cst_1    |                             |        \    /
//      \     /       \     /                              |         cst
//   Convolution     Convolution                            \        /
//           \        /                                     Convolution
//            ConcatOp                                           |
//               |
//
//

class SplitAffineReshapeConvConcatRewriter final : public mlir::OpRewritePattern<IE::SplitOp> {
public:
    SplitAffineReshapeConvConcatRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SplitOp>(ctx), _log(log) {
        setDebugName("SplitAffineReshapeConvConcatRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::SplitOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitAffineReshapeConvConcatRewriter::matchAndRewrite(IE::SplitOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Rewrite Split-AffineReshape-Conv-Concat operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    auto getConsumerResult = getConcatOpConsumer<IE::ConvolutionOp>(origOp, true, true);
    if (mlir::failed(getConsumerResult)) {
        getConsumerResult = getConcatOpConsumer<IE::TransposedConvolutionOp>(origOp, true, true);
        if (mlir::failed(getConsumerResult)) {
            _log.nest().trace("Not Split-AffineReshape-Conv-Concat pattern");
            return mlir::failure();
        }
    }

    auto concatOp = mlir::dyn_cast_or_null<IE::ConcatOp>(getConsumerResult.value());
    VPUX_THROW_WHEN(concatOp == nullptr, "Not a Concat operation");

    if (origOp.getOutputs().size() != concatOp.getInputs().size()) {
        return mlir::failure();
    }

    // Supported case for splitOp: split the dim to shape 1
    auto getSplitDim = getSplitDimToShape1(origOp);
    if (mlir::failed(getSplitDim)) {
        return mlir::failure();
    }

    // Supported case for concatOp: concat the dim with shape 1
    auto getconcatDims = getConcatDimWithShape1(concatOp, false);
    if (mlir::failed(getconcatDims)) {
        return mlir::failure();
    }

    // Supported case for affineReshape. Currently only support split on H&W and reshape C to H.
    if (!isSupportedAffineReshape(origOp)) {
        return mlir::failure();
    }

    // Create new transposeOp
    SmallVector<unsigned> transPerm;
    if (origOp.getAxisValue().value() == 3) {
        transPerm = {0, 3, 1, 2};
    } else if (origOp.getAxisValue().value() == 2) {
        transPerm = {0, 2, 1, 3};
    }

    const auto orderAttr =
            mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(transPerm, rewriter.getContext()));
    auto newTransposeOp = rewriter.create<IE::TransposeOp>(takeOpLoc(concatOp, "transpose_in"), origOp.getInput(),
                                                           nullptr, orderAttr);

    // Create new ConvolutionOp / TransposedConvolutionOp
    auto newFilter = composeNewFilter(concatOp, rewriter);
    auto newBias = composeNewBias(concatOp, rewriter);
    auto convLikeOp = createConvolution(concatOp, newFilter, newBias, newTransposeOp.getOutput(), rewriter);
    vpux::inferReturnTypes(convLikeOp, vpux::InferShapedTypeMode::ALL);
    concatOp.replaceAllUsesWith(convLikeOp->getResult(0));

    return mlir::success();
}

//
// SplitConcatRewriter
//

//
//               |
//            SplitOp
//              | |                                    |
//            ConcatOp          ->              AffineReshapeOp
//               |                                     |

class SplitConcatRewriter final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    SplitConcatRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("SplitConcatRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitConcatRewriter::matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Rewrite ConcatOp operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto splitOp = origOp.getOperand(0).getDefiningOp<IE::SplitOp>();
    if (splitOp == nullptr) {
        return mlir::failure();
    }

    auto getConsumerResult = getConcatOpConsumer(splitOp, false, false);
    if (mlir::failed(getConsumerResult)) {
        return mlir::failure();
    }

    VPUX_THROW_WHEN(mlir::dyn_cast_or_null<IE::ConcatOp>(getConsumerResult.value()) == nullptr,
                    "Not a Concat operation");

    if (splitOp.getOutputs().size() != origOp.getInputs().size()) {
        return mlir::failure();
    }

    // Supported case for splitOp: split the dim to shape 1
    auto getSplitDim = getSplitDimToShape1(splitOp);
    if (mlir::failed(getSplitDim)) {
        return mlir::failure();
    }

    // Supported case for concatOp: axis dim or adjust dims of concat with shape 1
    auto getconcatDims = getConcatDimWithShape1(origOp, true);
    if (mlir::failed(getconcatDims)) {
        return mlir::failure();
    }
    const auto concatDims = getconcatDims.value();

    const auto origOutputShape = getShape(origOp.getOutput());
    const auto reassociationMap =
            vpux::IE::getReassociationMap(getShape(splitOp.getInput()).raw(), origOutputShape.raw());
    if (mlir::failed(reassociationMap)) {
        return mlir::failure();
    }

    auto affineReshape =
            rewriter.create<IE::AffineReshapeOp>(takeOpLoc(origOp, "reshape_in"), splitOp.getInput(),
                                                 getIntArrayOfArray(getContext(), reassociationMap.value()),
                                                 getIntArrayAttr(rewriter.getContext(), origOutputShape));
    rewriter.replaceOp(origOp, affineReshape.getOutput());

    _log.trace("[{0}] Replaced with 'IE::AffineReshapeOp'", getDebugName());

    return mlir::success();
}

//
// ConvertSplitConcatToTransposePass
//

class ConvertSplitConcatToTransposePass final :
        public IE::impl::ConvertSplitConcatToTransposeBase<ConvertSplitConcatToTransposePass> {
public:
    explicit ConvertSplitConcatToTransposePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertSplitConcatToTransposePass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<SplitConcatRewriter>(&ctx, _log);
    patterns.insert<SplitAffineReshapeConcatRewriter>(&ctx, _log);
    patterns.insert<SplitAffineReshapeConvConcatRewriter>(&ctx, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertSplitConcatToTransposePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertSplitConcatToTransposePass(Logger log) {
    return std::make_unique<ConvertSplitConcatToTransposePass>(log);
}
