//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/locations.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/IR/AffineMap.h>
#include <mlir/IR/PatternMatch.h>

namespace vpux::IE {
#define GEN_PASS_DECL_ADJUSTQKVTRANSPOSEFORUNROLLSDPA
#define GEN_PASS_DEF_ADJUSTQKVTRANSPOSEFORUNROLLSDPA
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

struct PatternOps {
    IE::TransposeOp transpose;
    IE::FakeQuantizeOp fq;
    IE::GatherOp gather;
};

struct PatternOpsWithRootOp {
    std::pair<IE::TransposeOp, Dim> transposeOp;
    std::pair<IE::AffineReshapeOp, Dim> affineReshape1;
    std::pair<IE::AddOp, Dim> rootAdd;
    std::pair<IE::AffineReshapeOp, Dim> affineReshape2;
    std::pair<IE::FullyConnectedOp, Dim> rootFC;

    SmallVector<PatternOps> patternOpsVec;
};

class AdjustQKVTransposeForUnrollSDPA final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    AdjustQKVTransposeForUnrollSDPA(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::TransposeOp>(ctx), _log(log) {
        setDebugName("AdjustQKVTransposeForUnrollSDPA");
    }

    mlir::LogicalResult matchAndRewrite(IE::TransposeOp transposeOp, mlir::PatternRewriter& rewriter) const final;

private:
    void rebuildSubgraph(mlir::PatternRewriter& rewriter, const PatternOpsWithRootOp& patternOpsWithRootOp) const;

private:
    Logger _log;
};

void AdjustQKVTransposeForUnrollSDPA::rebuildSubgraph(mlir::PatternRewriter& rewriter,
                                                      const PatternOpsWithRootOp& patternOpsWithRootOp) const {
    auto eraseDimFromShape = [](ShapeRef shape, Dim dimToErase) {
        SmallVector<int64_t> newShape;
        for (auto dim : irange(shape.size())) {
            if (checked_cast<int64_t>(dim) == dimToErase.ind()) {
                continue;
            }
            newShape.push_back(shape[Dim(checked_cast<int64_t>(dim))]);
        }
        return newShape;
    };

    auto eraseDimFromAffineReshape = [&](IE::AffineReshapeOp affineReshapeOp, Dim outputDimToErase) {
        const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.getDimMapping());
        SmallVector<SmallVector<int64_t>> newDimMapping;
        newDimMapping.reserve(dimMapping.size());

        for (const auto& outputDims : dimMapping) {
            SmallVector<int64_t> newOutputDims;
            for (const auto outputDim : outputDims) {
                if (outputDim == outputDimToErase.ind()) {
                    continue;
                }
                newOutputDims.push_back(outputDim > outputDimToErase.ind() ? outputDim - 1 : outputDim);
            }
            newDimMapping.push_back(std::move(newOutputDims));
        }

        auto newOutputShape = eraseDimFromShape(getShape(affineReshapeOp.getOutput()), outputDimToErase);
        return std::make_pair(getIntArrayOfArray(rewriter.getContext(), newDimMapping),
                              getIntArrayAttr(rewriter.getContext(), newOutputShape));
    };

    auto cloneFQ = [&](IE::FakeQuantizeOp fqOp, mlir::Value input, int64_t offset, int64_t size) -> mlir::Value {
        auto createConstParams = [&](mlir::Value parameter, mlir::Location loc) -> mlir::Value {
            auto constOp = parameter.getDefiningOp<Const::DeclareOp>();
            VPUX_THROW_UNLESS(constOp != nullptr, "Expected parameter to be a constant");
            if (getShape(parameter).totalSize() == 1) {
                const Shape targetShape({1});
                const auto newContentAttr = constOp.transformContentAttr().reshape(targetShape).get();
                return rewriter.create<Const::DeclareOp>(loc, newContentAttr.getType(), std::move(newContentAttr))
                        .getOutput();
            }

            auto paramShape = getShape(parameter);
            auto nonOneDims = vpux::getNonOneDim(paramShape);
            VPUX_THROW_UNLESS(nonOneDims.size() == 1, "Expected parameter to have only one non-one dimension, got {0}",
                              nonOneDims.size());
            auto sliceDim = nonOneDims[0];
            auto sliceOffset = SmallVector<int64_t>(paramShape.size(), 0);
            sliceOffset[sliceDim.ind()] = offset * size;
            auto sliceSize = SmallVector<int64_t>(paramShape.begin(), paramShape.end());
            sliceSize[sliceDim.ind()] = size;
            const auto newContentAttr =
                    constOp.transformContentAttr().subview(ShapeRef(sliceOffset), ShapeRef(sliceSize)).get();
            return rewriter.create<Const::DeclareOp>(loc, newContentAttr.getType(), std::move(newContentAttr))
                    .getOutput();
        };

        auto inputLow = createConstParams(fqOp.getInputLow(), appendLoc(fqOp.getLoc(), "input_low"));
        auto inputHigh = createConstParams(fqOp.getInputHigh(), appendLoc(fqOp.getLoc(), "input_high"));
        auto outputLow = createConstParams(fqOp.getOutputLow(), appendLoc(fqOp.getLoc(), "output_low"));
        auto outputHigh = createConstParams(fqOp.getOutputHigh(), appendLoc(fqOp.getLoc(), "output_high"));

        return rewriter
                .create<IE::FakeQuantizeOp>(appendLoc(fqOp.getLoc(), "_{}", offset), input, inputLow, inputHigh,
                                            outputLow, outputHigh, fqOp.getLevelsAttr(), fqOp.getLowFpTypeAttr(),
                                            fqOp.getAutoBroadcastAttr())
                .getOutput();
    };

    auto sharedTransposeOp = patternOpsWithRootOp.transposeOp.first;
    auto rootFC = patternOpsWithRootOp.rootFC.first;
    const auto fcOutputShape = getShape(rootFC.getOutput());
    VPUX_THROW_UNLESS(fcOutputShape.size() == 2, "Expected 2D FullyConnected output");
    const auto fcSliceDim = patternOpsWithRootOp.rootFC.second;
    const auto size = fcOutputShape[fcSliceDim] / 3;

    //                                         -> FQ -> Gather -> Q
    // Rebuild input -> RootOps -> TransposeOp -> FQ -> Gather -> K
    //                                         -> FQ -> Gather -> V
    rewriter.setInsertionPoint(sharedTransposeOp);
    auto ctx = rewriter.getContext();
    for (const auto& patternOps : patternOpsWithRootOp.patternOpsVec) {
        auto gatherOp = patternOps.gather;
        auto gatherIndicesConst = gatherOp.getIndices().getDefiningOp<Const::DeclareOp>();
        const auto offset = gatherIndicesConst.getContentAttr().fold().getSplatValue<int64_t>();
        auto fcWeightsShape = getShape(rootFC.getWeights());
        SmallVector<int64_t> fcSliceOffset(fcWeightsShape.size(), 0);
        fcSliceOffset[0] = offset * size;
        SmallVector<int64_t> fcSliceSize(fcWeightsShape.begin(), fcWeightsShape.end());
        fcSliceSize[0] = size;

        mlir::Value slicedWeights;
        if (auto fq = mlir::dyn_cast_if_present<IE::FakeQuantizeOp>(rootFC.getWeights().getDefiningOp())) {
            slicedWeights = rewriter.createOrFold<IE::SliceOp>(
                    appendLoc(rootFC.getLoc(), "qkv_slice_weights_{0}", offset), fq.getInput(),
                    getIntArrayAttr(ctx, fcSliceOffset), getIntArrayAttr(ctx, fcSliceSize));
            slicedWeights = cloneFQ(fq, slicedWeights, offset, size);
        } else {
            slicedWeights = rewriter.createOrFold<IE::SliceOp>(
                    appendLoc(rootFC.getLoc(), "qkv_slice_weights_{0}", offset), rootFC.getWeights(),
                    getIntArrayAttr(ctx, fcSliceOffset), getIntArrayAttr(ctx, fcSliceSize));
        }
        mlir::Value output = rewriter.create<IE::FullyConnectedOp>(appendLoc(rootFC.getLoc(), "qkv_{0}", offset),
                                                                   rootFC.getInput(), slicedWeights, rootFC.getBias())
                                     .getOutput();

        if (patternOpsWithRootOp.affineReshape2.first != nullptr) {
            auto affineReshapeOp = patternOpsWithRootOp.affineReshape2.first;
            auto newShapeValue = IE::computeShapeValueFromAffineReshape(affineReshapeOp, getShape(output));
            VPUX_THROW_UNLESS(newShapeValue.has_value(), "Failed to compute AffineReshape output shape");
            output = rewriter.create<IE::AffineReshapeOp>(appendLoc(affineReshapeOp.getLoc(), "qkv_{0}", offset),
                                                          output, affineReshapeOp.getDimMapping(),
                                                          getIntArrayAttr(rewriter.getContext(), newShapeValue.value()))
                             .getOutput();
        }

        if (patternOpsWithRootOp.rootAdd.first != nullptr) {
            auto addOp = patternOpsWithRootOp.rootAdd.first;
            const auto sliceDim = patternOpsWithRootOp.rootAdd.second;
            auto addInput2Shape = getShape(addOp.getInput2());
            mlir::Value newInput2 = addOp.getInput2();
            VPUX_THROW_UNLESS(addInput2Shape.size() > checked_cast<size_t>(sliceDim.ind()),
                              "Add input2 shape rank {0} is not greater than unrolled dim {1}", addInput2Shape.size(),
                              sliceDim.ind());
            if (addInput2Shape[sliceDim] != 1 && addInput2Shape.size() == checked_cast<size_t>(sliceDim.ind() + 1)) {
                auto addSliceOffset = SmallVector<int64_t>(addInput2Shape.size(), 0);
                addSliceOffset[sliceDim.ind()] = offset * size;
                auto addSliceSize = SmallVector<int64_t>(addInput2Shape.begin(), addInput2Shape.end());
                addSliceSize[sliceDim.ind()] = size;
                newInput2 = rewriter.createOrFold<IE::SliceOp>(
                        appendLoc(addOp.getLoc(), "qkv_slice_add_input2_{0}", offset), addOp.getInput2(),
                        getIntArrayAttr(ctx, addSliceOffset), getIntArrayAttr(ctx, addSliceSize));
            }

            mlir::IRMapping mapper;
            auto newOperands = SmallVector<mlir::Value>{output, newInput2};
            mapper.map(addOp->getOperands(), newOperands);
            auto* newAddOp = rewriter.clone(*addOp, mapper);
            inferReturnTypes(newAddOp, InferShapedTypeMode::SHAPE);
            newAddOp->setLoc(appendLoc(addOp.getLoc(), "qkv_{0}", offset));
            output = newAddOp->getResult(0);
        }

        const auto rootOutputGatherDim = patternOpsWithRootOp.transposeOp.second;
        if (auto affineReshapeOp = patternOpsWithRootOp.affineReshape1.first) {
            auto [newDimMappingAttr, newShapeValueAttr] =
                    eraseDimFromAffineReshape(affineReshapeOp, rootOutputGatherDim);
            output = rewriter.create<IE::AffineReshapeOp>(appendLoc(affineReshapeOp.getLoc(), "qkv_{0}", offset),
                                                          output, newDimMappingAttr, newShapeValueAttr)
                             .getOutput();
        }

        if (patternOps.fq != nullptr) {
            output = cloneFQ(patternOps.fq, output, offset, size);
        }

        const auto orderValue = sharedTransposeOp.getOrderValue();
        VPUX_THROW_UNLESS(orderValue.has_value(), "TransposeOp must have order_value");
        SmallVector<mlir::AffineExpr> newResults;
        for (auto resultInd : irange(orderValue.value().getNumResults())) {
            const auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(orderValue.value().getResult(resultInd));
            VPUX_THROW_WHEN(dimExpr == nullptr, "Only affine dim expressions are supported in TransposeOp order_value");

            const auto oldInputDim = checked_cast<int64_t>(dimExpr.getPosition());
            if (oldInputDim == rootOutputGatherDim.ind()) {
                continue;
            }

            const auto newInputDim = oldInputDim > rootOutputGatherDim.ind() ? oldInputDim - 1 : oldInputDim;
            newResults.push_back(mlir::getAffineDimExpr(newInputDim, rewriter.getContext()));
        }

        const auto newInputRank = checked_cast<unsigned>(orderValue.value().getNumDims() - 1);
        auto newOrderValue =
                mlir::AffineMapAttr::get(mlir::AffineMap::get(newInputRank, 0, newResults, rewriter.getContext()));
        output = rewriter.create<IE::TransposeOp>(
                                 appendLoc(sharedTransposeOp.getLoc(), "qkv_shared_transpose_{0}", offset), output,
                                 nullptr, newOrderValue)
                         .getOutput();

        if (patternOps.transpose != nullptr) {
            auto branchTransposeOp = patternOps.transpose;
            output = rewriter.create<IE::TransposeOp>(
                                     appendLoc(branchTransposeOp->getLoc(), "qkv_batch_transpose_{0}", offset), output,
                                     nullptr, branchTransposeOp.getOrderValueAttr())
                             .getOutput();
            branchTransposeOp.getOutput().replaceAllUsesWith(output);
        } else {
            gatherOp.getOutput().replaceAllUsesWith(output);
        }
    }

    return;
}

/*
    Perform the following conversion, where FakeQuantize and Transpose(V) are optional.
             input                               input
               |                             /      |      \
             RootOps               RootOpsTile RootOpsTile RootOpsTile
               |                          |         |         |
           Transpose                  Transpose Transpose Transpose
          /        |                      |         |         |
    FakeQuantize Gather             FakeQuantize FakeQuantize |
       / \       |                        |         |         |
  Gather Gather  Transpose     =>         Q         K         V
      |    |       |                        \       |        /
      Q    K       V                            Attention
       \   |      /
        Attention
*/
mlir::LogicalResult AdjustQKVTransposeForUnrollSDPA::matchAndRewrite(IE::TransposeOp transposeOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    // Match root ops when walking backwards from Transpose input: (optional) AffineReshape -> (optional) Add ->
    // (optional) AffineReshape -> FullyConnected.
    auto matchRootOpsPattern = [&](IE::TransposeOp transposeOp, PatternOpsWithRootOp& patternOpsWithRootOp) {
        auto currentOp = transposeOp.getInput().getDefiningOp();
        if (auto affineReshapeOp = mlir::dyn_cast_if_present<IE::AffineReshapeOp>(currentOp)) {
            if (!affineReshapeOp.getOutput().hasOneUse()) {
                return false;
            }

            patternOpsWithRootOp.affineReshape1.first = affineReshapeOp;
            currentOp = affineReshapeOp.getInput().getDefiningOp();
        }

        if (auto addOp = mlir::dyn_cast_if_present<IE::AddOp>(currentOp)) {
            if (!addOp.getOutput().hasOneUse() || !vpux::Const::isConstValue(addOp.getInput2())) {
                return false;
            }

            patternOpsWithRootOp.rootAdd.first = addOp;
            currentOp = addOp.getInput1().getDefiningOp();
        }

        if (auto affineReshapeOp = mlir::dyn_cast_if_present<IE::AffineReshapeOp>(currentOp)) {
            if (!affineReshapeOp.getOutput().hasOneUse()) {
                return false;
            }

            patternOpsWithRootOp.affineReshape2.first = affineReshapeOp;
            currentOp = affineReshapeOp.getInput().getDefiningOp();
        }

        auto fcOp = mlir::dyn_cast_if_present<IE::FullyConnectedOp>(currentOp);
        if (fcOp == nullptr || !vpux::Const::isConstValue(fcOp.getWeights()) || fcOp.getBias() != nullptr) {
            return false;
        }

        patternOpsWithRootOp.rootFC.first = fcOp;
        return true;
    };

    auto matchAttentionPattern = [&](IE::TransposeOp transposeOp, PatternOpsWithRootOp& patternOpsWithRootOp) {
        SmallVector<mlir::Operation*> userOps;
        for (auto userOp : transposeOp.getOutput().getUsers()) {
            if (auto fq = mlir::dyn_cast_if_present<IE::FakeQuantizeOp>(userOp)) {
                if (!IE::isPerTensorFQ({fq}) || !IE::hasStaticLowAndHighValues(fq)) {
                    return false;
                }

                for (auto fqUserOp : fq.getOutput().getUsers()) {
                    auto patternOps = PatternOps{};
                    patternOps.fq = fq;
                    userOps.push_back(fqUserOp);
                    patternOpsWithRootOp.patternOpsVec.push_back(patternOps);
                }

                continue;
            }

            userOps.push_back(userOp);
            patternOpsWithRootOp.patternOpsVec.push_back(PatternOps{});
        }
        // Expect three branches for Q, K, V
        if (userOps.size() != 3) {
            return false;
        }

        for (auto [index, userOp] : userOps | indexed) {
            auto gather = mlir::dyn_cast_if_present<IE::GatherOp>(userOp);
            if (gather == nullptr) {
                return false;
            }
            patternOpsWithRootOp.patternOpsVec[index].gather = gather;

            mlir::Value attentionInput = gather.getOutput();
            if (gather.getOutput().hasOneUse()) {
                if (auto transpose =
                            mlir::dyn_cast_if_present<IE::TransposeOp>(*gather.getOutput().getUsers().begin())) {
                    if (!transpose.getOutput().hasOneUse()) {
                        return false;
                    }

                    patternOpsWithRootOp.patternOpsVec[index].transpose = transpose;
                    attentionInput = transpose.getOutput();
                }
            }
            if (!attentionInput.hasOneUse() ||
                !mlir::isa_and_present<IE::AttentionOp>(*attentionInput.getUsers().begin())) {
                return false;
            }

            auto gatherOutShape = getShape(gather.getOutput());
            // Only support 4D attention and the num_heads must be greater than 1
            if (gatherOutShape.size() != 4 || gatherOutShape[Dims4D::Act::N] != 1 ||
                gatherOutShape[Dims4D::Act::C] == 1) {
                return false;
            }
        }

        // Sort the patternOpsVec based on the order of Q, K, V in AttentionOp inputs
        SmallVector<PatternOps> orderedPatternOpsVec(3, PatternOps{});
        for (auto& patternOps : patternOpsWithRootOp.patternOpsVec) {
            auto attentionInput =
                    patternOps.transpose != nullptr ? patternOps.transpose.getOutput() : patternOps.gather.getOutput();
            auto attentionOp = mlir::dyn_cast_if_present<IE::AttentionOp>(*attentionInput.getUsers().begin());
            VPUX_THROW_UNLESS(attentionOp != nullptr, "Expected AttentionOp as user of Transpose/Gather output");
            if (attentionOp.getInputQ() == attentionInput) {
                orderedPatternOpsVec[0] = patternOps;
            } else if (attentionOp.getInputK() == attentionInput) {
                orderedPatternOpsVec[1] = patternOps;
            } else if (attentionOp.getInputV() == attentionInput) {
                orderedPatternOpsVec[2] = patternOps;
            } else {
                return false;
            }
        }
        if (llvm::any_of(orderedPatternOpsVec, [](const PatternOps& patternOps) {
                return patternOps.gather == nullptr;
            })) {
            return false;
        }

        patternOpsWithRootOp.patternOpsVec = std::move(orderedPatternOpsVec);
        return true;
    };

    PatternOpsWithRootOp patternOpsWithRootOp;
    patternOpsWithRootOp.transposeOp.first = transposeOp;
    if (!matchRootOpsPattern(transposeOp, patternOpsWithRootOp) ||
        !matchAttentionPattern(transposeOp, patternOpsWithRootOp)) {
        return mlir::failure();
    }

    auto canGatherBeEliminated = [&](const PatternOps& patternOps, size_t index) {
        auto gatherOp = patternOps.gather;
        if (gatherOp == nullptr) {
            return false;
        }

        auto indicesConst = gatherOp.getIndices().getDefiningOp<Const::DeclareOp>();
        if (indicesConst == nullptr || getShape(gatherOp.getIndices()).totalSize() != 1) {
            return false;
        }
        const auto splatIndices = indicesConst.getContentAttr().fold().getSplatValue<int64_t>();
        if (splatIndices != checked_cast<int64_t>(index)) {
            return false;
        }

        const auto axisVal = gatherOp.getAxisValue();
        const auto batchDims = gatherOp.getBatchDims();
        const auto indicesRank = gatherOp.getIndicesRank();
        if (axisVal != 0 || batchDims != 0 || indicesRank != 0) {
            return false;
        }

        // Propagate gatherDim and unrolledDim through the path to check if the slice happens on constant weights of a
        // FullyConnected
        const auto gatherInputShape = getShape(gatherOp.getInput());
        const auto gatherOutShape = getShape(gatherOp.getOutput());
        if (gatherOutShape.size() != 4) {
            return false;
        }

        const auto gatherAxis = Dim(axisVal);
        std::optional<Dim> gatherDim = gatherAxis;
        auto unrolledDim = Dims4D::Act::C;
        if (unrolledDim.ind() >= gatherAxis.ind()) {
            unrolledDim = Dim(unrolledDim.ind() + 1);
        }
        if (gatherInputShape.size() <= checked_cast<size_t>(unrolledDim.ind())) {
            return false;
        }

        auto getTransposeInputDim = [&](IE::TransposeOp transposeOp,
                                        std::optional<Dim> outputDim) -> std::optional<Dim> {
            if (!outputDim.has_value()) {
                return std::nullopt;
            }

            const auto affineMap = transposeOp.getOrderValue();
            if (!affineMap.has_value()) {
                return std::nullopt;
            }

            if (checked_cast<size_t>(outputDim.value().ind()) >= affineMap.value().getNumResults()) {
                return std::nullopt;
            }

            const auto dimExpr =
                    mlir::dyn_cast<mlir::AffineDimExpr>(affineMap.value().getResult(outputDim.value().ind()));
            if (dimExpr == nullptr) {
                return std::nullopt;
            }
            return Dim(dimExpr.getPosition());
        };

        auto getAffineReshapeInputDim = [](IE::AffineReshapeOp affineReshapeOp,
                                           std::optional<Dim> outputDim) -> std::optional<Dim> {
            if (!outputDim.has_value()) {
                return std::nullopt;
            }

            const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.getDimMapping());
            for (auto inputDim : irange(dimMapping.size())) {
                const auto& outputDims = dimMapping[inputDim];
                if (llvm::is_contained(outputDims, outputDim.value().ind())) {
                    return Dim(checked_cast<int64_t>(inputDim));
                }
            }
            return std::nullopt;
        };

        // Propagate gatherDim and unrolledDim through the path to check if the slice happens on constant weights of a
        // FullyConnected
        gatherDim = getTransposeInputDim(transposeOp, gatherDim);
        if (!gatherDim.has_value()) {
            return false;
        }
        patternOpsWithRootOp.transposeOp.second = gatherDim.value();
        auto maybeUnrolledDim = getTransposeInputDim(transposeOp, unrolledDim);
        if (patternOpsWithRootOp.affineReshape1.first != nullptr) {
            auto affineReshapeOp = patternOpsWithRootOp.affineReshape1.first;
            gatherDim = getAffineReshapeInputDim(affineReshapeOp, gatherDim);
            if (!gatherDim.has_value()) {
                return false;
            }
            patternOpsWithRootOp.affineReshape1.second = gatherDim.value();
            maybeUnrolledDim = getAffineReshapeInputDim(affineReshapeOp, maybeUnrolledDim);
        }
        if (patternOpsWithRootOp.rootAdd.first != nullptr) {
            if (!gatherDim.has_value()) {
                return false;
            }
            patternOpsWithRootOp.rootAdd.second = gatherDim.value();
        }
        if (patternOpsWithRootOp.affineReshape2.first != nullptr) {
            auto affineReshapeOp = patternOpsWithRootOp.affineReshape2.first;
            gatherDim = getAffineReshapeInputDim(affineReshapeOp, gatherDim);
            if (!gatherDim.has_value()) {
                return false;
            }
            patternOpsWithRootOp.affineReshape2.second = gatherDim.value();
            maybeUnrolledDim = getAffineReshapeInputDim(affineReshapeOp, maybeUnrolledDim);
        }

        if (!gatherDim.has_value() || !maybeUnrolledDim.has_value()) {
            return false;
        }

        auto fcOutShape = getShape(patternOpsWithRootOp.rootFC.first.getOutput());
        patternOpsWithRootOp.rootFC.second = gatherDim.value();
        return fcOutShape.size() == 2 && gatherDim.value() == Dim(1) && maybeUnrolledDim.value() == Dim(1) &&
               fcOutShape[gatherDim.value()] % 3 == 0;
    };
    for (const auto& indexedPatternOps : patternOpsWithRootOp.patternOpsVec | indexed) {
        if (!canGatherBeEliminated(indexedPatternOps.value(), indexedPatternOps.index())) {
            return mlir::failure();
        }
    }

    rebuildSubgraph(rewriter, patternOpsWithRootOp);

    return mlir::success();
}

class AdjustQKVTransposeForUnrollSDPAPass final :
        public IE::impl::AdjustQKVTransposeForUnrollSDPABase<AdjustQKVTransposeForUnrollSDPAPass> {
public:
    explicit AdjustQKVTransposeForUnrollSDPAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustQKVTransposeForUnrollSDPAPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AdjustQKVTransposeForUnrollSDPA>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustQKVTransposeForUnrollSDPAPass(Logger log) {
    return std::make_unique<AdjustQKVTransposeForUnrollSDPAPass>(log);
}
