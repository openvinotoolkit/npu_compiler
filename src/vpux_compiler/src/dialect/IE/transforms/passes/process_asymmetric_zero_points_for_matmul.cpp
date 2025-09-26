//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>

namespace vpux::IE {
#define GEN_PASS_DECL_PROCESSASYMMETRICZEROPOINTSFORMATMUL
#define GEN_PASS_DEF_PROCESSASYMMETRICZEROPOINTSFORMATMUL
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

template <typename InOp_t, typename OutOp_t>
OutOp_t checkOp(InOp_t inOp, const std::function<mlir::Value(InOp_t)>& valGetter,
                const std::function<bool(OutOp_t)>& opChecker) {
    if (inOp == nullptr) {
        return nullptr;
    }
    mlir::Value val = valGetter(inOp);
    if (val == nullptr) {
        return nullptr;
    }
    auto op = val.getDefiningOp<OutOp_t>();
    if (op == nullptr) {
        return nullptr;
    }
    if (!opChecker(op)) {
        return nullptr;
    }
    return op;
}

mlir::Value getConvWeights(IE::ConvolutionOp convOp) {
    return convOp.getFilter();
}

bool checkReshapeToNxCx1x1(IE::AffineReshapeOp reshapeOp) {
    const auto inShape = getShape(reshapeOp.getInput());
    if (inShape.size() != 4) {
        return false;
    }
    const auto outShape = getShape(reshapeOp.getOutput());
    if (outShape.size() != 4) {
        return false;
    }
    const auto expectedOutShape1 = Shape{inShape[Dims4D::Act::H], inShape[Dims4D::Act::W], 1, 1};
    const auto expectedOutShape2 = Shape{inShape[Dims4D::Act::C], inShape[Dims4D::Act::W], 1, 1};
    return outShape == expectedOutShape1 || outShape == expectedOutShape2;
}

bool checkReshapeTo1xHx1xW(IE::AffineReshapeOp reshapeOp) {
    const auto inShape = getShape(reshapeOp.getInput());
    if (inShape.size() != 4) {
        return false;
    }
    const auto outShape = getShape(reshapeOp.getOutput());
    if (outShape.size() != 4) {
        return false;
    }
    const auto expectedOutShape1 = Shape{1, inShape[Dims4D::Act::H], 1, inShape[Dims4D::Act::W]};
    return outShape == expectedOutShape1;
}

mlir::Value getReshapeInput(IE::AffineReshapeOp reshapeOp) {
    return reshapeOp.getInput();
}

bool checkTranspose(IE::TransposeOp transposeOp) {
    if (!transposeOp.getOutput().hasOneUse()) {
        return false;
    }
    const auto inShape = getShape(transposeOp.getInput());
    if (inShape.size() != 4) {
        return false;
    }
    const auto outShape = getShape(transposeOp.getOutput());
    if (outShape.size() != 4) {
        return false;
    }
    if (!transposeOp.getOrderValue().has_value()) {
        return false;
    }
    const auto expectedOutShape1 = Shape{1, 1, inShape[Dims4D::Act::W], inShape[Dims4D::Act::H]};
    const auto expectedOutShape2 = Shape{1, inShape[Dims4D::Act::W], 1, inShape[Dims4D::Act::C]};
    const auto order = DimsOrder::fromAffineMap(transposeOp.getOrderValue().value());
    return (outShape == expectedOutShape1 && order == DimsOrder::NCWH) ||
           (outShape == expectedOutShape2 && order == DimsOrder::NWHC);
}

mlir::Value getTransposeInput(IE::TransposeOp transposeOp) {
    return transposeOp.getInput();
}

mlir::quant::QuantizedType getQuantizedElementTypeFromFakeQuantize(IE::FakeQuantizeOp fqOp) {
    auto outLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    const auto realType = mlir::cast<vpux::NDTypeInterface>(fqOp.getInput().getType());
    const auto realElemType = mlir::cast<mlir::FloatType>(realType.getElementType());
    return getQuantizedType(outLowConst.getContentAttr(), outHighConst.getContentAttr(), fqOp.getLevels(),
                            fqOp.getLowFpType(), realElemType, false, fqOp.getLoc(), fqOp.getAutoBroadcast(), true);
}

bool checkFQ(IE::FakeQuantizeOp fqOp) {
    if (fqOp.getLevels() != 256 || !fqOp.getOutputLow().hasOneUse() || !fqOp.getOutputHigh().hasOneUse()) {
        return false;
    }
    auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();

    if (inLowConst == nullptr || inHighConst == nullptr) {
        return false;
    }
    const auto outQuantizeElemType = getQuantizedElementTypeFromFakeQuantize(fqOp);
    if (outQuantizeElemType == nullptr) {
        return false;
    }
    // If zp is not 128 it requires fix.
    const auto uniformQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outQuantizeElemType);
    if (uniformQuantType != nullptr) {
        return !uniformQuantType.isSigned() && uniformQuantType.getStorageTypeIntegralWidth() == 8 &&
               uniformQuantType.getZeroPoint() != 128;
    }

    const auto uniformQuantPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(outQuantizeElemType);
    if (uniformQuantPerAxisType == nullptr) {
        return false;
    }
    // If any of the axis zero point is not 128, that still requires zero point fix.
    auto zeroPointsPerAxis = uniformQuantPerAxisType.getZeroPoints();
    return !uniformQuantPerAxisType.isSigned() && uniformQuantPerAxisType.getStorageTypeIntegralWidth() == 8 &&
           std::any_of(zeroPointsPerAxis.begin(), zeroPointsPerAxis.end(), [](int64_t val) {
               return val != 128;
           });
}

NDTypeInterface inferReduceSumOutputType(NDTypeInterface inputType) {
    // Add to outShape the values with indices not found in axes_set.
    size_t axes = 1;
    SmallVector<int64_t> outShape;
    auto inShape = inputType.getShape();
    for (size_t i = 0; i < inShape.size(); i++) {
        if (i != axes) {
            outShape.push_back(inShape[Dim(i)]);
        } else {
            outShape.push_back(1);
        }
    }

    if (outShape.size() == 0) {
        outShape.push_back(1);
    }
    auto outputType = mlir::RankedTensorType::get(outShape, inputType.getElementType());
    return mlir::cast<NDTypeInterface>(outputType);
}

IE::FakeQuantizeOp getMatchingFakeQuantizeOp(IE::ConvolutionOp convOp) {
    // Do pattern checks and quantization checks

    // We want to reach FakeQuantize there may be multiple optional different ops (reshape/transpose)
    auto maybeFQ = checkOp<IE::ConvolutionOp, IE::FakeQuantizeOp>(convOp, getConvWeights, checkFQ);
    if (maybeFQ != nullptr) {
        return maybeFQ;
    }

    auto maybeReshapeToNxCx1x1 =
            checkOp<IE::ConvolutionOp, IE::AffineReshapeOp>(convOp, getConvWeights, checkReshapeToNxCx1x1);
    maybeFQ = checkOp<IE::AffineReshapeOp, IE::FakeQuantizeOp>(maybeReshapeToNxCx1x1, getReshapeInput, checkFQ);

    if (maybeFQ != nullptr) {
        return maybeFQ;
    }

    // TransposeOp may or may not be in the graph depending on transpose_b.
    auto maybeTransposeOp =
            checkOp<IE::AffineReshapeOp, IE::TransposeOp>(maybeReshapeToNxCx1x1, getReshapeInput, checkTranspose);
    maybeFQ = checkOp<IE::TransposeOp, IE::FakeQuantizeOp>(maybeTransposeOp, getTransposeInput, checkFQ);
    if (maybeFQ != nullptr) {
        return maybeFQ;
    }

    // Some graphs have reshape to 1xHx1xW
    auto maybeReshapeTo1xHx1xW =
            checkOp<IE::TransposeOp, IE::AffineReshapeOp>(maybeTransposeOp, getTransposeInput, checkReshapeTo1xHx1xW);
    maybeFQ = checkOp<IE::AffineReshapeOp, IE::FakeQuantizeOp>(maybeReshapeTo1xHx1xW, getReshapeInput, checkFQ);

    return maybeFQ;
}

vpux::Byte estimateNewOpsSize(IE::ConvolutionOp convOp) {
    // We will ignore optimizations/transposes for calculation for now.
    const auto convOutType = mlir::cast<NDTypeInterface>(convOp->getResult(0).getType());
    const auto convInputType = mlir::cast<NDTypeInterface>(convOp.getInput().getType());
    const auto convFilterType = mlir::cast<NDTypeInterface>(convOp.getFilter().getType());
    // First calculate reduceSum input which goes through 2 transformation
    // 1. ConvertBatchedLayerTo1N, Transpose to DimsOrder::HCNW, which we will calculate
    // 2. Optional AdjustConvolutionInputShape, Does not affect C dim, will not affects sizes, wont calculate
    const auto transposedType = convInputType.changeDimsOrder(DimsOrder::HCNW);
    const auto reduceSumInputSize = transposedType.getTotalAllocSize();

    const auto reduceSumOutputType = inferReduceSumOutputType(transposedType);
    const auto multiplyInputType = reduceSumOutputType;
    const auto OC = convFilterType.getShape()[Dims4D::Filter::OC];
    const auto newScalesShape = mlir::RankedTensorType::get(SmallVector<int64_t, 4>{1, OC, 1, 1},
                                                            mlir::Float16Type::get(convOp.getContext()));
    const auto multiplyInput2Type = mlir::cast<NDTypeInterface>(newScalesShape);
    const auto multiplyOutputType = convOutType;
    const auto multiplySize = multiplyInputType.getTotalAllocSize() + multiplyInput2Type.getTotalAllocSize();
    const auto addSize = multiplyOutputType.getTotalAllocSize() * 2;  // Inputs are same size for the Add.
    return reduceSumInputSize + multiplySize + addSize;
}

std::vector<double> rewriteFQOutputParams(IE::FakeQuantizeOp fakeQuantize, mlir::PatternRewriter& rewriter) {
    auto oLoConst = fakeQuantize.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto oHiConst = fakeQuantize.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    const auto quantizedElemType = getQuantizedElementTypeFromFakeQuantize(fakeQuantize);
    const auto [scales, zeroPoints] = extractScalesAndZeroPoints(quantizedElemType);
    auto diff = std::vector<double>(scales.size());
    std::transform(scales.begin(), scales.end(), zeroPoints.begin(), diff.begin(), [](double scale, double zeroPoint) {
        return (zeroPoint - 128.0) * scale;
    });

    auto insertionPointBackup = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfter(oHiConst);
    // Change zero point to 128 by modifying outputLow/High
    auto oldOutLoContent = oLoConst.getContentAttr().fold();
    auto oldOutLoValues = to_small_vector(oldOutLoContent.getValues<vpux::type::float16>());

    std::transform(oldOutLoValues.begin(), oldOutLoValues.end(), diff.begin(), oldOutLoValues.begin(),
                   [](vpux::type::float16 origOutLo, vpux::type::float16 diff) {
                       return origOutLo + diff;
                   });

    auto newOutLoType =
            mlir::RankedTensorType::get(mlir::cast<NDTypeInterface>(oLoConst.getOutput().getType()).getShape(),
                                        mlir::Float16Type::get(rewriter.getContext()));
    auto newOutLoValuesAttr = mlir::DenseElementsAttr::get(newOutLoType, ArrayRef(oldOutLoValues));
    auto newLoInput = rewriter.create<Const::DeclareOp>(oLoConst.getLoc(), oLoConst.getType(),
                                                        Const::ContentAttr::get(newOutLoValuesAttr));

    // Change outHigh to make zp =128
    auto oldOutHiContent = oHiConst.getContentAttr().fold();
    auto oldOutHiValues = to_small_vector(oldOutHiContent.getValues<vpux::type::float16>());
    std::transform(oldOutHiValues.begin(), oldOutHiValues.end(), diff.begin(), oldOutHiValues.begin(),
                   [](vpux::type::float16 origOutHi, vpux::type::float16 diff) {
                       return origOutHi + diff;
                   });
    auto newOutHiType =
            mlir::RankedTensorType::get(mlir::cast<NDTypeInterface>(oHiConst.getOutput().getType()).getShape(),
                                        mlir::Float16Type::get(rewriter.getContext()));
    auto newOutHiValuesAttr = mlir::DenseElementsAttr::get(newOutHiType, ArrayRef(oldOutHiValues));
    auto newHiInput = rewriter.create<Const::DeclareOp>(oHiConst.getLoc(), oHiConst.getType(),
                                                        Const::ContentAttr::get(newOutHiValuesAttr));

    rewriter.replaceOp(oLoConst, newLoInput);
    rewriter.replaceOp(oHiConst, newHiInput);
    rewriter.restoreInsertionPoint(insertionPointBackup);

    return diff;
}

std::tuple<bool, mlir::TypedValue<mlir::RankedTensorType>, IE::AffineReshapeOp>
applyReshapeFromAdjustConvolutionInputShape(IE::TransposeOp transposeInput, mlir::PatternRewriter& rewriter) {
    auto reduceSumInput = transposeInput.getOutput();
    bool isReshaped{false};
    const ShapeRef transposedShape = getShape(transposeInput.getOutput());
    IE::AffineReshapeOp reshapeInput;
    if (transposedShape[Dims4D::Act::H] % 4 == 0) {
        const Shape shapeNxCxHx4 = {
                transposedShape[Dims4D::Act::N],
                transposedShape[Dims4D::Act::C],
                transposedShape[Dims4D::Act::H] / 4,
                transposedShape[Dims4D::Act::W] * 4,
        };
        SmallVector<SmallVector<int64_t>> inDimMapping = {{0}, {1}, {2, 3}, {3}};

        // And that transformation is typically done by AdjustConvolutionInputShape
        reshapeInput = rewriter.create<IE::AffineReshapeOp>(appendLoc(transposeInput.getLoc(), "to [N, C, H/4, 4]"),
                                                            transposeInput.getOutput(),
                                                            getIntArrayOfArray(rewriter.getContext(), inDimMapping),
                                                            getIntArrayAttr(rewriter.getContext(), shapeNxCxHx4));
        reduceSumInput = reshapeInput.getOutput();
        isReshaped = true;
    }
    return std::make_tuple(isReshaped, reduceSumInput, reshapeInput);
}

IE::MultiplyOp createOpsToCalculateFix(IE::ConvolutionOp convOp,
                                       mlir::TypedValue<::mlir::RankedTensorType> reduceSumInput,
                                       std::vector<double>& diff, mlir::PatternRewriter& rewriter) {
    // Input -> [IE.ReduceSum (axes = [1])] -> IE.Multiply ([128 - zp] * scales)
    // Reduce   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    SmallVector<int64_t> reductionAxes = {1};
    auto reduce = rewriter.create<IE::ReduceSumOp>(appendLoc(convOp.getLoc(), "_reduce"), reduceSumInput, nullptr,
                                                   getIntArrayAttr(rewriter.getContext(), ArrayRef(reductionAxes)),
                                                   /*keep_dims=*/true, nullptr, nullptr);
    // Input -> IE.ReduceSum (axes = [1]) -> IE.Multiply ([128 - zp] * scales)
    // Rescale values                                     ^^^^^^^^^^^^^^^^^^^
    // SmallVector<float> newScales = {(zp - 128) * scale};
    // Negate the value to replace IE.Subtract with IE.Add
    SmallVector<vpux::type::float16> newScales(diff.size());
    std::transform(diff.begin(), diff.end(), newScales.begin(), [](auto diffVal) {
        return -diffVal;
    });
    auto OC = mlir::cast<NDTypeInterface>(convOp.getFilter().getType()).getShape()[Dims4D::Filter::OC];
    const auto newScalesShape = mlir::RankedTensorType::get(SmallVector<int64_t, 4>{1, OC, 1, 1},
                                                            mlir::Float16Type::get(rewriter.getContext()));
    const auto newScalesAttr = mlir::DenseElementsAttr::get(newScalesShape, ArrayRef(newScales));
    auto newScalesVal = rewriter.create<Const::DeclareOp>(appendLoc(convOp.getLoc(), "new_scales"), newScalesShape,
                                                          Const::ContentAttr::get(newScalesAttr));
    // Input -> IE.ReduceSum (axes = [1]) -> [IE.Multiply] ([128 - zp] * scales)
    // Multiply reduced sum by scales        ^^^^^^^^^^^^^
    auto rescale = rewriter.create<IE::MultiplyOp>(
            appendLoc(convOp.getLoc(), "rescale"), reduce.getOutput(), newScalesVal.getOutput(),
            IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), IE::AutoBroadcastType::NUMPY),
            /*postOp=*/nullptr,
            /*clamp=*/nullptr, nullptr, nullptr);
    return rescale;
}

IE::AffineReshapeOp rollbackAdjustConvolutionInputShapeReshape(IE::AddOp subtract, mlir::PatternRewriter& rewriter) {
    const ShapeRef subtractShape = getShape(subtract.getOutput());
    const Shape shapeNxCx4Hx1 = {
            subtractShape[Dims4D::Act::N],
            subtractShape[Dims4D::Act::C],
            subtractShape[Dims4D::Act::H] * 4,
            subtractShape[Dims4D::Act::W] / 4,
    };

    SmallVector<SmallVector<int64_t>> outDimMapping = {{0}, {1}, {2}, {2, 3}};

    auto reshapeOutput =
            rewriter.create<IE::AffineReshapeOp>(appendLoc(subtract.getLoc(), "to [N, C, 4H, 1]"), subtract.getOutput(),
                                                 getIntArrayOfArray(rewriter.getContext(), outDimMapping),
                                                 getIntArrayAttr(rewriter.getContext(), shapeNxCx4Hx1));
    return reshapeOutput;
}

bool isConversionBeneficial(IE::ConvolutionOp convOp, double decompositionEnablementRatio) {
    const auto convInputSize = mlir::cast<NDTypeInterface>(convOp.getInput().getType()).getTotalAllocSize();
    const auto convFilterSize = mlir::cast<NDTypeInterface>(convOp.getFilter().getType()).getTotalAllocSize();
    const auto convSize = convFilterSize + convInputSize;
    const auto newOpSize = estimateNewOpsSize(convOp);
    const auto convToNewOpRatio = (double)convSize.count() / newOpSize.count();

    // Most of the time runtime dequantization is more efficient than this method, however for some layers new ops added
    // with this are really small compared to original matmul, so if new ops are smaller
    // 1/_decompositionEnablementRatio(250) of original matmul we use this method
    return convToNewOpRatio >= decompositionEnablementRatio;
}

bool isOneByOneConvolution(IE::ConvolutionOp convOp) {
    auto filterShape = mlir::cast<NDTypeInterface>(convOp.getFilter().getType()).getShape();
    bool fourDFilter = filterShape.size() == 4;
    bool matmulOr1x1Conv =
            fourDFilter && (filterShape[Dims4D::Filter::KY] == 1 && filterShape[Dims4D::Filter::KX] == 1);
    // Only convs with 1x1 kernel or matmuls ( which are mapped to 1x1 kernel convs) are supported.
    return matmulOr1x1Conv;
}

class FixMatmulZeroPointRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    FixMatmulZeroPointRewriter(mlir::MLIRContext* ctx, double decompositionEnablementRatio, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx),
              _decompositionEnablementRatio(decompositionEnablementRatio),
              _log(log) {
        setDebugName("FixMatmulZeroPointRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    double _decompositionEnablementRatio;
    Logger _log;
};

mlir::LogicalResult FixMatmulZeroPointRewriter::matchAndRewrite(IE::ConvolutionOp convOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), convOp->getName(), convOp->getLoc());

    auto nestedLog = _log.nest();

    auto oneByOneConv = isOneByOneConvolution(convOp);
    if (!oneByOneConv) {
        return matchFailed(nestedLog, rewriter, convOp, "Only convolutions with 1x1 kernels are supported");
    }

    auto asymmetricFQ = getMatchingFakeQuantizeOp(convOp);
    if (asymmetricFQ == nullptr) {
        return matchFailed(nestedLog, rewriter, convOp, "Could not find asymmetric FQ");
    }

    // Most of the time runtime dequantization is more efficient than this method, however for some layers new ops added
    // with this are really small compared to original matmul, so if new ops are smaller
    // 1/_decompositionEnablementRatio(250) of original matmul we use this method
    if (!isConversionBeneficial(convOp, _decompositionEnablementRatio)) {
        return matchFailed(nestedLog, rewriter, convOp, "Conversion is not beneficial.");
    }

    // We change outputHigh/Low of FQ so original matmul runs as if it was symmetric, then we will apply a fix later
    // using negative diff(zeroPoint - 128.0) * scale; )
    auto diff = rewriteFQOutputParams(asymmetricFQ, rewriter);

    auto matmulInput = convOp.getInput();

    // We have to apply transformations which are normally done by other passes (ConvertBatchedLayerTo1N and
    // AdjustConvolutionInputShape)
    // Difference is  transpose/reshape that would be originally added after Conv must be
    // added after AddOp in this pass.
    // Also pattern matching here is complex and complexity is contained here while
    // other passes keep their simple approach.

    // This part is originally  covered by ConvertBatchedLayerTo1N
    const auto orderHCNW = mlir::AffineMapAttr::get(DimsOrder::HCNW.toAffineMap(rewriter.getContext()));
    auto transposeInput = rewriter.create<IE::TransposeOp>(appendLoc(matmulInput.getLoc(), "in_to_HCNW"), matmulInput,
                                                           nullptr, orderHCNW);

    auto reduceSumInput = transposeInput.getOutput();
    auto [isReshaped, reduceSumInputNew, reshapeInput] =
            applyReshapeFromAdjustConvolutionInputShape(transposeInput, rewriter);
    if (isReshaped) {
        reduceSumInput = reduceSumInputNew;
    }

    auto rescale = createOpsToCalculateFix(convOp, reduceSumInput, diff, rewriter);

    auto convInput = isReshaped ? reshapeInput.getOutput() : transposeInput.getOutput();

    auto newConvOp = rewriter.create<IE::ConvolutionOp>(
            convOp->getLoc(), convInput, convOp.getFilter(), convOp.getBias(), convOp.getStrides(),
            convOp.getPadsBegin(), convOp.getPadsEnd(), convOp.getDilations(), convOp.getPostOpAttr(),
            convOp.getClampAttr(), convOp.getStaticScaleAttr(), convOp.getOutputPaddingAttr(),
            convOp.getInputPaddingAttr());

    // IE.Add with a negative operand instead of IE.Subtract
    // Subtract the fix, to get original result .
    auto subtract = rewriter.create<IE::AddOp>(
            appendLoc(convOp.getLoc(), "subtract_reduction"), newConvOp->getResult(0), rescale.getOutput(),
            IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), IE::AutoBroadcastType::NUMPY),
            /*postOp=*/nullptr,
            /*clamp=*/nullptr, nullptr, nullptr);

    auto endTransposeInput = subtract.getOutput();
    auto endTransposeInputLoc = subtract.getLoc();
    // Roll back reshape from AdjustConvolutionInputShape
    if (reshapeInput) {
        auto reshapeOutput = rollbackAdjustConvolutionInputShapeReshape(subtract, rewriter);
        endTransposeInput = reshapeOutput.getOutput();
        endTransposeInputLoc = reshapeOutput.getLoc();
    }
    // Roll back the transposition from ConvertBatchedLayerTo1N
    auto transposeOutput = rewriter.create<IE::TransposeOp>(appendLoc(endTransposeInputLoc, "out_to_HCNW"),
                                                            endTransposeInput, nullptr, orderHCNW);
    _log.trace("Matmul decomposition is applied to fix zero point of weights related to conv : {0}", convOp->getLoc());
    rewriter.replaceOp(convOp, transposeOutput.getOutput());
    return mlir::success();
}

class ProcessAsymmetricZeroPointsForMatmulPass final :
        public IE::impl::ProcessAsymmetricZeroPointsForMatmulBase<ProcessAsymmetricZeroPointsForMatmulPass> {
public:
    explicit ProcessAsymmetricZeroPointsForMatmulPass(double decompositionEnablementRatio, Logger log)
            : _decompositionEnablementRatio{decompositionEnablementRatio} {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    double _decompositionEnablementRatio;
};

mlir::LogicalResult ProcessAsymmetricZeroPointsForMatmulPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (matmulMixedPrecisionDecompositionRatio.hasValue()) {
        _decompositionEnablementRatio = matmulMixedPrecisionDecompositionRatio.getValue();
    }

    return mlir::success();
}

void ProcessAsymmetricZeroPointsForMatmulPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FixMatmulZeroPointRewriter>(&ctx, _decompositionEnablementRatio, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createProcessAsymmetricZeroPointsForMatmulPass(
        double decompositionEnablementRatio, Logger log) {
    return std::make_unique<ProcessAsymmetricZeroPointsForMatmulPass>(decompositionEnablementRatio, log);
}
