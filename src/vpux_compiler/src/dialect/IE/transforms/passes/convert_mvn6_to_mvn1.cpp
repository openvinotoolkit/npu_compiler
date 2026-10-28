//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTMVN6TOMVN1
#define GEN_PASS_DEF_CONVERTMVN6TOMVN1
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertMVN6ToMVN1
// Motivation:
// For MVN operation there are two kernels, namely, MVN1, and MVN6.
// MVN6 is more general in terms of norm-axes (can be any, can be non-adjacent,
// also supports optional SCALE, BIAS); but contains less optimizations than MVN1.
// The general (any axes) implementation of MVN6 is slower. Therefore this pass
// attempts to convert MVN6 to MVN1 in cases when the conversion is possible.
// This often requires inserting Transpose operations before and after MVN1 to
// ensure that the axes used for normalization are the ones supported by MVN1.
//
class ConvertMVN6ToMVN1 final : public mlir::OpRewritePattern<IE::MVN6Op> {
public:
    ConvertMVN6ToMVN1(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MVN6Op>(ctx), _log(log) {
        setDebugName("ConvertMVN6ToMVN1");
    }

    mlir::LogicalResult matchAndRewrite(IE::MVN6Op origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertMVN6ToMVN1::matchAndRewrite(IE::MVN6Op origOp, mlir::PatternRewriter& rewriter) const {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto inputShape = inputType.getShape();
    auto inputShapeVal = inputShape.raw();
    const auto inputShapeSize = inputShape.size();
    const auto inRank = inputType.getRank();

    if (origOp.getScale() || origOp.getBias()) {
        _log.nest().trace("MVN6 got scale/bias, cannot convert to MVN1.");
        return mlir::failure();
    }

    if (inputShapeSize < 2 || inputShapeSize > 5) {
        _log.nest().trace("MVN6 -> MVN1 conversion pass supports only 2D, 3D, 4D or 5D cases. Got {0}D input shape",
                          inputShapeSize);
        return mlir::failure();
    }

    const auto epsMode = origOp.getEpsMode();
    const auto eps = origOp.getEpsAttr().getValueAsDouble();
    const bool normalizeVariance = origOp.getNormalizeVariance();
    bool acrossChannels = false;
    SmallVector<int64_t> axesAttr;

    if (epsMode != IE::MvnEpsMode::INSIDE_SQRT) {
        _log.nest().trace("MVN-1 does not support OUTSIDE_SQRT eps mode, unless small enough 'eps' values. If "
                          "OUTSIDE_SQRT is not supported, we should do MVNFusion pass");

        const double epsThreshold = 1e-3;
        if (eps > epsThreshold) {
            _log.nest().trace("For small enough 'eps' values, can treat OUTSIDE_SQRT mode as INSIDE_SQRT. Can not "
                              "convert because of large epsilon value: {0} vs {1}",
                              eps, epsThreshold);
            return mlir::failure();
        }
    }

    if (origOp.getAxes() != nullptr && !origOp.getAxesValue().has_value()) {
        auto axesConst = origOp.getAxes().getDefiningOp<Const::DeclareOp>();
        if (axesConst == nullptr) {
            return mlir::failure();
        }

        const auto axesContent = axesConst.getContent();
        axesAttr = to_small_vector(axesContent.getValues<int64_t>());

        for (auto& axis : axesAttr) {
            if (axis < 0) {
                axis += inRank;
            }
        }
        std::sort(axesAttr.begin(), axesAttr.end());
    } else if (origOp.getAxes() == nullptr && origOp.getAxesValue().has_value()) {
        axesAttr = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
    } else {
        return mlir::failure();
    }

    bool needsTranspose = false;

    // 4D input and axis is 1, we need a Transpose to transpose dim C
    // to spatial dim
    IE::TransposeOp transposeIn;
    mlir::AffineMapAttr inverseTransposeOrder;
    SmallVector<int64_t> origInputShapeVal(inputShapeVal.begin(), inputShapeVal.end());
    if (inputShapeSize == 4 && axesAttr.size() == 1 && (axesAttr[0] == 1 || axesAttr[0] == 2)) {
        needsTranspose = true;
        _log.trace("Transpose dim C to spatial dim");
        mlir::AffineMapAttr transposeOrder;
        if (axesAttr[0] == 1) {
            transposeOrder = mlir::AffineMapAttr::get(
                    mlir::AffineMap::getPermutationMap(SmallVector<uint32_t>{0, 2, 3, 1}, getContext()));
            // Inverse permutation of {0,2,3,1} is {0,3,1,2}
            inverseTransposeOrder = mlir::AffineMapAttr::get(
                    mlir::AffineMap::getPermutationMap(SmallVector<uint32_t>{0, 3, 1, 2}, getContext()));
        } else {
            // Ensure that the more efficient mvn kernel is used when axesAttr[0] == 2.
            // This aligns with the behaviour of NBPerf.
            transposeOrder = mlir::AffineMapAttr::get(
                    mlir::AffineMap::getPermutationMap(SmallVector<uint32_t>{0, 1, 3, 2}, getContext()));
            // Inverse permutation of {0,1,3,2} is {0,1,3,2}
            inverseTransposeOrder = mlir::AffineMapAttr::get(
                    mlir::AffineMap::getPermutationMap(SmallVector<uint32_t>{0, 1, 3, 2}, getContext()));
        }
        const auto transposeLoc = appendLoc(origOp->getLoc(), "transpose_mvn_in");
        rewriter.setInsertionPoint(origOp);
        transposeIn = IE::TransposeOp::create(rewriter, transposeLoc, origOp.getInput(), nullptr, transposeOrder);
        inputShapeVal = mlir::cast<vpux::NDTypeInterface>(transposeIn.getOutput().getType()).getShape().raw();
        axesAttr.clear();
        axesAttr.push_back(inputShapeSize - 1);
    }

    Shape newInShape(inputShapeVal.begin(), inputShapeVal.end());

    if ((inputShapeSize == 2 || inputShapeSize == 3 || inputShapeSize == 4) && axesAttr.size() == 1 &&
        static_cast<uint32_t>(axesAttr[0]) == (inputShapeSize - 1)) {
        acrossChannels = false;
        if (inputShape.size() == 4) {
            newInShape = {inputShapeVal[0], inputShapeVal[1] * inputShapeVal[2], 1, inputShapeVal[3]};
        } else if (inputShape.size() == 3) {
            newInShape = {inputShapeVal[0], inputShapeVal[1], inputShapeVal[2], 1};
        } else if (inputShape.size() == 2) {
            newInShape = {1, inputShapeVal[0], inputShapeVal[1], 1};
        } else {
            return mlir::failure();
        }
    } else if (inputShapeSize == 3) {
        if (axesAttr.size() == 2 && axesAttr[0] == 1 && axesAttr[1] == 2) {
            newInShape = {1, inputShapeVal[0], inputShapeVal[1], inputShapeVal[2]};
        } else if (axesAttr.size() == 1 && axesAttr[0] == 1 && inputShapeVal[2] == 1) {
            // AxBx1 with axis [1] -> Ax1xBx1, MVN1 (across_channels=false) normalizes over H,W = B,1
            newInShape = {inputShapeVal[0], 1, inputShapeVal[1], inputShapeVal[2]};
        }
    } else if (inputShapeSize == 4) {
        if (axesAttr.size() == 3 && axesAttr[0] == 1 && axesAttr[1] == 2 && axesAttr[2] == 3) {
            acrossChannels = true;
        } else if (axesAttr.size() == 2 && axesAttr[0] == 2 && axesAttr[1] == 3) {
            acrossChannels = false;
        } else {
            _log.nest().trace("MVN-1 layer supports only normalization across channel or spatial dimension, in this "
                              "case we should do MVNFusion pass");
            return mlir::failure();
        }
    } else if (inputShapeSize == 5 && axesAttr.size() == 3 && axesAttr[0] == 2 && axesAttr[1] == 3 &&
               axesAttr[2] == 4) {
        if (inputShape.size() == 5) {
            newInShape = {inputShapeVal[0], inputShapeVal[1], inputShapeVal[2], inputShapeVal[3] * inputShapeVal[4]};

        } else {
            _log.nest().trace("Unexpected input shape");
            return mlir::failure();
        }
    }

    const auto normVarianceAttr = mlir::BoolAttr::get(getContext(), normalizeVariance);
    const auto acrossChannelsAttr = mlir::BoolAttr::get(getContext(), acrossChannels);
    const auto epsAttr = getFPAttr(getContext(), eps);

    if (newInShape.size() == 4) {
        const auto origLoc = origOp->getLoc();

        bool isDynamic = newInShape.isDynamic();

        mlir::Value reshapeInput;
        mlir::Value reshapeOutput;
        mlir::Value reshapeInputSource = needsTranspose ? transposeIn.getOutput() : origOp.getInput();

        const auto uniqueSuffix = vpux::stringifyPrimaryLocationSanitized(origOp->getLoc());
        const auto mvnLoc = appendLoc(origLoc, "mvn_" + uniqueSuffix);
        const auto transposeLoc = appendLoc(origLoc, "transpose_mvn_out_" + uniqueSuffix);

        if (isDynamic) {
            auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());

            const auto dynamicReshapeInLoc = appendLoc(origLoc, "dynamic_reshape_mvn_" + uniqueSuffix + "_input");
            const auto dynamicReshapeOutLoc = appendLoc(origLoc, "dynamic_reshape_mvn_" + uniqueSuffix + "_output");

            auto type = origOp.getInput().getType();
            const auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type);
            VPUX_THROW_WHEN(boundedType == nullptr, "Expected BoundedTensorType but got {0}", type);
            const auto bounds = boundedType.getBounds().raw();

            SmallVector<int64_t> newBounds;
            const size_t origRank = bounds.size();
            const size_t newRank = newInShape.size();
            for (size_t i = 0; i < newRank; ++i) {
                std::optional<size_t> origIdx;
                if (i < origRank) {
                    origIdx = origRank - 1 - i;
                }
                size_t newIdx = newRank - 1 - i;
                auto newInShapeRaw = newInShape.raw();
                if (newInShapeRaw[newIdx] != mlir::ShapedType::kDynamic) {
                    newBounds.insert(newBounds.begin(), newInShapeRaw[newIdx]);
                } else if (origIdx.has_value()) {
                    newBounds.insert(newBounds.begin(), bounds[*origIdx]);
                } else {
                    VPUX_THROW("Failed to compute new bounds for DynamicReshape: newIdx={}, origIdx={}, "
                               "newInShapeRaw={}, bounds={}",
                               newIdx, origIdx.has_value() ? std::to_string(*origIdx) : "none",
                               printToString("{}", newInShapeRaw), printToString("{}", bounds));
                }
            }

            auto newShapeValues = IE::replaceDynamicDimsWithValue<int32_t>(to_small_vector(newInShape), -1);

            auto si32Type = mlir::IntegerType::get(rewriter.getContext(), 32, mlir::IntegerType::Signed);
            auto shapeType = mlir::RankedTensorType::get({static_cast<int64_t>(newShapeValues.size())}, si32Type);
            auto shapeAttr = mlir::DenseElementsAttr::get(shapeType, llvm::ArrayRef<int32_t>(newShapeValues));
            auto shapeConst = vpux::Const::DeclareOp::create(rewriter, origLoc, shapeType,
                                                             vpux::Const::ContentAttr::get(shapeAttr));

            auto reshapeInputShapeAttr = rewriter.getI64ArrayAttr(newInShape.raw());
            auto reshapeInputBoundsAttr = rewriter.getI64ArrayAttr(newBounds);
            auto ndType = mlir::cast<vpux::NDTypeInterface>(inType);
            auto newType =
                    Core::BoundedTensorType::get(vpux::getTensorType(newInShape, ndType.getElementType(),
                                                                     vpux::DimsOrder::fromNumDims(newInShape.size()),
                                                                     ndType.getMemSpace(), /*Bounds=*/{}, {}),
                                                 vpux::BoundsRef(newBounds));

            reshapeInput = IE::DynamicReshapeOp::create(rewriter, dynamicReshapeInLoc, newType, reshapeInputSource,
                                                        shapeConst.getOutput(), reshapeInputShapeAttr,
                                                        reshapeInputBoundsAttr, nullptr)
                                   .getOutput();

            auto mvnOp =
                    IE::MVNOp::create(rewriter, mvnLoc, reshapeInput, acrossChannelsAttr, normVarianceAttr, epsAttr);

            auto inputShapeValues = IE::replaceDynamicDimsWithValue<int32_t>(to_small_vector(inputShape), -1);

            auto finalShapeType =
                    mlir::RankedTensorType::get({static_cast<int64_t>(inputShapeValues.size())}, si32Type);
            auto finalShapeAttr =
                    mlir::DenseElementsAttr::get(finalShapeType, llvm::ArrayRef<int32_t>(inputShapeValues));
            auto finalShapeConst = vpux::Const::DeclareOp::create(rewriter, origLoc, finalShapeType,
                                                                  vpux::Const::ContentAttr::get(finalShapeAttr));
            auto finalOutputShapeAttr = rewriter.getI64ArrayAttr(inputShapeVal);
            auto finalOutputBoundsAttr = rewriter.getI64ArrayAttr(bounds);

            reshapeOutput = IE::DynamicReshapeOp::create(rewriter, dynamicReshapeOutLoc, inType, mvnOp.getOutput(),
                                                         finalShapeConst.getOutput(), finalOutputShapeAttr,
                                                         finalOutputBoundsAttr, nullptr)
                                    .getOutput();

        } else {
            auto staticReshapeInput = IE::ReshapeOp::create(rewriter, origLoc, reshapeInputSource,
                                                            getIntArrayAttr(getContext(), newInShape))
                                              .getOutput();

            auto mvnOp = IE::MVNOp::create(rewriter, origLoc, staticReshapeInput, acrossChannelsAttr, normVarianceAttr,
                                           epsAttr);

            reshapeOutput = IE::ReshapeOp::create(rewriter, origLoc, mvnOp.getOutput(),
                                                  getIntArrayAttr(getContext(), inputShapeVal))
                                    .getOutput();
        }

        if (needsTranspose) {
            auto transposeOut =
                    IE::TransposeOp::create(rewriter, transposeLoc, reshapeOutput, nullptr, inverseTransposeOrder);
            rewriter.replaceOp(origOp, transposeOut.getOutput());
        } else {
            rewriter.replaceOp(origOp, reshapeOutput);
        }
        return mlir::success();
    } else {
        _log.nest().trace("MVN6 -> MVN1 conversion pass not applied");
        return mlir::failure();
    }
}

//
// ConvertMVN6ToMVN1Pass
//

class ConvertMVN6ToMVN1Pass final : public IE::impl::ConvertMVN6ToMVN1Base<ConvertMVN6ToMVN1Pass> {
public:
    explicit ConvertMVN6ToMVN1Pass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertMVN6ToMVN1Pass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertMVN6ToMVN1>(&ctx, _log);

    auto func = getOperation();
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createConvertMVN6ToMVN1Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertMVN6ToMVN1Pass(Logger log) {
    return std::make_unique<ConvertMVN6ToMVN1Pass>(log);
}
