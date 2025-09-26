//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/fft_ops_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/op.hpp>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTFFTTOCONV
#define GEN_PASS_DEF_CONVERTFFTTOCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

constexpr float FftPi = 3.141592653589793238462643f;

inline int64_t getRdftOutAxisLength(int64_t inAxisLength) {
    return (inAxisLength / 2 + 1);
}

using AnyRankedTensor = mlir::TypedValue<mlir::RankedTensorType>;

auto reshapeRestoreAndComplexAdd(mlir::PatternRewriter& rewriter, mlir::Location loc, AnyRankedTensor input,
                                 mlir::Value expectedInput, bool addComplex, bool isRdftLastAxisCut, Logger log) {
    log.trace("reshapeRestoreAndComplexAdd : {0}", input);
    const auto shapeTypeExpect = mlir::cast<vpux::NDTypeInterface>(expectedInput.getType()).getShape();
    const auto shapeTypeInput = mlir::cast<vpux::NDTypeInterface>(input.getType()).getShape();
    auto newShape = to_small_vector(shapeTypeExpect);
    auto shapeInput = to_small_vector(shapeTypeInput);
    if (addComplex) {
        newShape.push_back(2);
    }
    if (isRdftLastAxisCut) {
        auto rank = newShape.size();
        auto axisLengthOut = newShape[rank - 2];
        axisLengthOut = getRdftOutAxisLength(axisLengthOut);
        newShape[rank - 2] = axisLengthOut;
    }
    if (newShape != shapeInput) {
        const auto newShapeAttr = getIntArrayAttr(rewriter.getContext(), newShape);
        auto reshapeOp = rewriter.create<IE::ReshapeOp>(loc, input, nullptr, false, newShapeAttr);
        return reshapeOp.getOutput();
    }
    return input;
}

auto reshapeRestoreIrdftLastAxis(mlir::PatternRewriter& rewriter, mlir::Location loc, AnyRankedTensor input,
                                 mlir::Value expectedInput, Logger log) {
    log.trace("reshapeRestoreIrdftLastAxis : {0}", input);
    const auto outResShapeType = mlir::cast<vpux::NDTypeInterface>(expectedInput.getType()).getShape();
    auto newShape = to_small_vector(outResShapeType);
    newShape.pop_back();  // delete complex representation
    auto axisShape = (newShape.back() - 1) * 2;
    newShape.pop_back();
    newShape.push_back(axisShape);
    const auto newShapeAttr = getIntArrayAttr(rewriter.getContext(), newShape);
    auto reshapeOp = rewriter.create<IE::ReshapeOp>(loc, input, nullptr, false, newShapeAttr);
    return reshapeOp.getOutput();
}

auto reshapeToMatMulNeedRank(mlir::PatternRewriter& rewriter, mlir::Location loc, AnyRankedTensor input,
                             bool cutComplex, Logger log) {
    log.trace("reshapeToMatMulNeedRank: {0}", input);
    bool shapeWasChanged = false;
    const int64_t targetRank = 3;
    const auto shape = mlir::cast<vpux::NDTypeInterface>(input.getType()).getShape();
    auto newShape = to_small_vector(shape);
    if (cutComplex) {
        newShape.pop_back();
        auto lastSize = newShape.end()[-1];
        newShape.pop_back();
        lastSize = lastSize * 2;
        newShape.push_back(lastSize);
        shapeWasChanged = true;
    }
    const auto rank = newShape.size();
    if (rank > targetRank) {
        auto newShapeResized = SmallVector<int64_t>(newShape.end() - targetRank, newShape.end());
        auto newShapeRemains = SmallVector<int64_t>(newShape.begin(), newShape.end() - targetRank);
        auto newCumulateSize = newShapeResized[0];
        for (auto sz : newShapeRemains) {
            newCumulateSize = newCumulateSize * sz;
        }
        newShapeResized[0] = newCumulateSize;
        newShape = std::move(newShapeResized);
        shapeWasChanged = true;
    }
    if (rank < targetRank) {
        auto newShapeResized = newShape;
        for (auto i = newShape.size(); i < targetRank; i++) {
            newShapeResized.insert(newShapeResized.begin(), 1);
        }
        newShape = std::move(newShapeResized);
        shapeWasChanged = true;
    }
    if (shapeWasChanged) {
        const auto newShapeAttr = getIntArrayAttr(rewriter.getContext(), newShape);
        auto reshapeOp = rewriter.create<IE::ReshapeOp>(loc, input, nullptr, false, newShapeAttr);
        return reshapeOp.getOutput();
    }
    return input;
}

mlir::Value complexFFTGetTwiddleFactors(mlir::Location loc, int64_t axisLength, mlir::Type dtype,
                                        mlir::PatternRewriter& rewriter, bool isInverseFFT, bool isRdftLastAxisCut) {
    const auto complexNoScale = 2;
    int64_t axisLengthOut = axisLength;
    if (isRdftLastAxisCut) {
        axisLengthOut = getRdftOutAxisLength(axisLength);
    }
    int64_t size = (axisLengthOut * complexNoScale * axisLength * complexNoScale);
    auto lengthAxis = static_cast<float>(axisLength);
    float twiddleConst = -2.0f * FftPi;
    float filterConst = 1.0f;
    if (isInverseFFT) {
        twiddleConst = 2.0f * FftPi;
        filterConst = 1.0f / lengthAxis;
    }
    std::vector<float> twiddleVals(size);
    int64_t idx = 0;
    for (int64_t k = 0; k < axisLengthOut; k++) {
        for (int64_t c = 0; c < axisLength; c++) {
            float crtProdPos = static_cast<float>(c * k);
            float angle = twiddleConst * crtProdPos / lengthAxis;
            twiddleVals[axisLength * 2 + idx] = std::sin(angle) * filterConst;
            twiddleVals[idx++] = std::cos(angle) * filterConst;
            twiddleVals[axisLength * 2 + idx] = std::cos(angle) * filterConst;
            twiddleVals[idx++] = -std::sin(angle) * filterConst;
        }
        idx += (axisLength * 2);
    }
    const SmallVector<int64_t> twiddleShape({axisLengthOut * complexNoScale, axisLength * complexNoScale});
    const auto twiddleType = mlir::RankedTensorType::get(twiddleShape, mlir::Float32Type::get(dtype.getContext()));
    return Const::createConst(rewriter, loc, twiddleType, ArrayRef(twiddleVals), [&](Const::ContentSetup& setup) {
        return setup.castElemType(dtype);
    });
}

mlir::Value fftGetTwiddleFactorsForRdftRealInput(mlir::Location loc, int64_t axisLength, mlir::Type dtype,
                                                 mlir::PatternRewriter& rewriter, bool /*isInverseFFT*/,
                                                 bool isRdftLastAxisCut) {
    const auto complexNoScale = 2;
    int64_t axisLengthOut = axisLength;
    if (isRdftLastAxisCut) {
        axisLengthOut = getRdftOutAxisLength(axisLength);
    }
    int64_t size = (axisLengthOut * complexNoScale * axisLength);
    auto floatAxisLength = static_cast<float>(axisLength);
    const float twiddleConst = -2.0f * FftPi;
    std::vector<float> twiddleVals(size);
    int64_t idx = 0;
    for (int64_t k = 0; k < axisLengthOut; k++) {
        for (int64_t c = 0; c < axisLength; c++) {
            float crtProdPos = static_cast<float>(c * k);
            float angle = twiddleConst * crtProdPos / floatAxisLength;
            twiddleVals[axisLength + idx] = std::sin(angle);
            twiddleVals[idx++] = std::cos(angle);
        }
        idx += axisLength;
    }
    const SmallVector<int64_t> twiddleShape({axisLengthOut * complexNoScale, axisLength});
    const auto twiddleType = mlir::RankedTensorType::get(twiddleShape, mlir::Float32Type::get(dtype.getContext()));
    return Const::createConst(rewriter, loc, twiddleType, ArrayRef(twiddleVals), [&](Const::ContentSetup& setup) {
        return setup.castElemType(dtype);
    });
}

mlir::Value fftGetTwiddleFactorsForIrdftRealOutput(mlir::Location loc, int64_t axisLength, mlir::Type dtype,
                                                   mlir::PatternRewriter& rewriter) {
    auto inAxisLength = axisLength;
    auto outAxisLength = (inAxisLength - 1) * 2;
    const auto complexNoScale = 2;
    int64_t size = (inAxisLength * complexNoScale * outAxisLength);
    auto floatAxisLength = static_cast<float>(outAxisLength);
    const float twiddleConst = 2.0f * FftPi;
    const float filterConst = 1.0f / floatAxisLength;
    std::vector<float> twiddleVals(size, 0.0f);
    int64_t idx = 0;
    for (int64_t k = 0; k < outAxisLength; k++) {
        for (int64_t c = 0; c < inAxisLength; c++) {
            float crtProdPos = static_cast<float>(c * k);
            float angle = twiddleConst * crtProdPos / floatAxisLength;
            float valR = std::cos(angle) * filterConst;
            float valI = -std::sin(angle) * filterConst;
            if ((c > 0) && (c < (outAxisLength - inAxisLength + 1))) {
                auto hermitianSymmetricIdx = outAxisLength - c;
                crtProdPos = static_cast<float>(hermitianSymmetricIdx * k);
                angle = twiddleConst * crtProdPos / floatAxisLength;

                float valHermitianR = std::cos(angle) * filterConst;
                float valHermitianI = -std::sin(angle) * filterConst;

                valR = valR + valHermitianR;
                valI = valI - valHermitianI;
            }
            twiddleVals[idx++] = valR;
            twiddleVals[idx++] = valI;
        }
    }
    const SmallVector<int64_t> twiddleShape({outAxisLength, inAxisLength * complexNoScale});
    const auto twiddleType = mlir::RankedTensorType::get(twiddleShape, mlir::Float32Type::get(dtype.getContext()));
    return Const::createConst(rewriter, loc, twiddleType, ArrayRef(twiddleVals), [&](Const::ContentSetup& setup) {
        return setup.castElemType(dtype);
    });
}

auto fftOneAxisDecompose(mlir::PatternRewriter& rewriter, mlir::Location loc, AnyRankedTensor inIter,
                         int64_t axisLength, const DimArr& curOrder, int64_t lastComplexAxes, int64_t axis,
                         bool isInverseFFT, bool inputIsComplex, bool isRdftLastAxisCut, Logger log,
                         mlir::Value (*getTwiddleFactors)(mlir::Location, int64_t, mlir::Type, mlir::PatternRewriter&,
                                                          bool, bool)) {
    log.trace("fftOneAxisDecompose: {0}", inIter);
    const auto inType = mlir::cast<vpux::NDTypeInterface>(inIter.getType());
    // Reorder input in order to move axis on last dimension
    auto transposesOut = inIter;
    auto axisPermutation = curOrder;
    auto perm = axisPermutation[axis];
    axisPermutation[axis] = axisPermutation[lastComplexAxes];
    axisPermutation[lastComplexAxes] = perm;
    auto dstOrder = DimsOrder::fromPermutation(axisPermutation);
    auto orderOutputAttr = mlir::AffineMapAttr::get(dstOrder.toAffineMap(rewriter.getContext()));
    if (axis < lastComplexAxes) {
        auto transposeOp =
                rewriter.create<IE::TransposeOp>(appendLoc(loc, "TransposeIn"), inIter, nullptr, orderOutputAttr);
        transposesOut = transposeOp.getOutput();
    }
    auto reshapeOut =
            reshapeToMatMulNeedRank(rewriter, appendLoc(loc, "ReshapeIn"), transposesOut, inputIsComplex, log);
    // produce constant twiddle factors with ouput type precision. Keep consistent precision.
    auto twiddleConstant = getTwiddleFactors(appendLoc(loc, "TwiddleFactors"), axisLength, inType.getElementType(),
                                             rewriter, isInverseFFT, isRdftLastAxisCut);
    auto multiplyOp = rewriter.create<IE::MatMulOp>(appendLoc(loc, "MatMulOp"), reshapeOut, twiddleConstant, false,
                                                    true, nullptr);
    // restore original shape
    auto reshapeRestoredOut =
            reshapeRestoreAndComplexAdd(rewriter, appendLoc(loc, "ReshapeOut"), multiplyOp.getOutput(), transposesOut,
                                        !inputIsComplex, isRdftLastAxisCut, log);
    // restore original order
    auto transposesOutRestored = reshapeRestoredOut;
    if (axis < lastComplexAxes) {
        if (!inputIsComplex) {
            const auto rank = mlir::cast<vpux::NDTypeInterface>(transposesOutRestored.getType()).getRank();
            axisPermutation.push_back(Dim(rank - 1));
            dstOrder = DimsOrder::fromPermutation(axisPermutation);
            orderOutputAttr = mlir::AffineMapAttr::get(dstOrder.toAffineMap(rewriter.getContext()));
        }
        auto transposesOpRestored = rewriter.create<IE::TransposeOp>(appendLoc(loc, "TransposeOut"), reshapeRestoredOut,
                                                                     nullptr, orderOutputAttr);
        transposesOutRestored = transposesOpRestored.getOutput();
    }
    return transposesOutRestored;
}

auto complexFFTDecompose(mlir::PatternRewriter& rewriter, mlir::Location loc, AnyRankedTensor input,
                         ArrayRef<int64_t> axes, bool isInverseFFT, bool isRdft, Logger log) {
    log.trace("complexFFTDecompose: {0}", input);
    Logger _log = log.nest();
    auto inIter = input;
    const auto inType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    auto rank = inType.getRank();
    auto lastComplexAxes = rank - 2;
    const auto curOrder = DimsOrder::fromValue(inIter).toPermutation();
    auto shape = to_small_vector(inType.getShape());
    for (auto axis : axes | indexed) {
        const auto fftLoc = appendLoc(loc, "ComplexFft_axis_{0}", axis.index());
        inIter = fftOneAxisDecompose(rewriter, fftLoc, inIter, shape[axis.value()], curOrder, lastComplexAxes,
                                     axis.value(), isInverseFFT, true, (axis.index() == (axes.size() - 1)) && isRdft,
                                     _log, complexFFTGetTwiddleFactors);
    }
    return inIter;
}

auto rdftFirstAxisDecompose(mlir::PatternRewriter& rewriter, mlir::Location loc, AnyRankedTensor input, int64_t axis,
                            bool isRdftLastAxisCut, Logger log) {
    log.trace("rdftFirstAxisDecompose: {0}", input);
    Logger _log = log.nest();
    const auto rdftLoc = appendLoc(loc, "FirstAxis");
    const auto inType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    auto rank = inType.getRank();
    auto lastComplexAxes = rank - 1;
    const auto curOrder = DimsOrder::fromValue(input).toPermutation();
    auto shape = to_small_vector(inType.getShape());
    return fftOneAxisDecompose(rewriter, rdftLoc, input, shape[axis], curOrder, lastComplexAxes, axis, false, false,
                               isRdftLastAxisCut, _log, fftGetTwiddleFactorsForRdftRealInput);
}

auto irdftLastAxisDecompose(mlir::PatternRewriter& rewriter, mlir::Location loc, AnyRankedTensor input, int64_t axis,
                            Logger log) {
    log.trace("irdftLastAxisDecompose: {0}", input);
    Logger _log = log.nest();
    const auto irdftLoc = appendLoc(loc, "LastAxis");
    const auto inType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    auto rank = inType.getRank();
    auto lastComplexAxes = rank - 2;
    auto curOrder = DimsOrder::fromValue(input).toPermutation();
    auto shape = to_small_vector(inType.getShape());
    // Reorder input in order to move axis on last dimension
    auto transposesOut = input;
    auto axisPermutation = std::move(curOrder);
    auto perm = axisPermutation[axis];
    axisPermutation[axis] = axisPermutation[lastComplexAxes];
    axisPermutation[lastComplexAxes] = perm;
    auto dstOrder = DimsOrder::fromPermutation(axisPermutation);
    auto orderOutputAttr = mlir::AffineMapAttr::get(dstOrder.toAffineMap(rewriter.getContext()));
    if (axis < lastComplexAxes) {
        auto transposeOp =
                rewriter.create<IE::TransposeOp>(appendLoc(irdftLoc, "TransposeIn"), input, nullptr, orderOutputAttr);
        transposesOut = transposeOp.getOutput();
    }
    // MatMull request rank=3. Reshape in order to accumulate rank >3 in rank=3 and cut complex representation
    auto reshapeOut = reshapeToMatMulNeedRank(rewriter, appendLoc(irdftLoc, "ReshapeIn"), transposesOut, true, _log);
    // produce constant twiddle factors with ouput type precision. Keep consistent precision.
    auto twiddleConstant = fftGetTwiddleFactorsForIrdftRealOutput(appendLoc(irdftLoc, "TwiddleFactors"), shape[axis],
                                                                  inType.getElementType(), rewriter);
    // mat mull for complex number
    auto multiplyOp = rewriter.create<IE::MatMulOp>(appendLoc(irdftLoc, "MatMulOp"), reshapeOut, twiddleConstant, false,
                                                    true, nullptr);
    // reshape to output size representation
    auto reshapeRestoredOut = reshapeRestoreIrdftLastAxis(rewriter, appendLoc(irdftLoc, "ReshapeOut"),
                                                          multiplyOp.getOutput(), transposesOut, _log);
    // transpose back
    auto transposesOutRestored = reshapeRestoredOut;
    if (axis < lastComplexAxes) {
        axisPermutation.pop_back();  // delete complex representation
        dstOrder = DimsOrder::fromPermutation(axisPermutation);
        orderOutputAttr = mlir::AffineMapAttr::get(dstOrder.toAffineMap(rewriter.getContext()));
        auto transposesOpRestored = rewriter.create<IE::TransposeOp>(appendLoc(irdftLoc, "TransposeOut"),
                                                                     reshapeRestoredOut, nullptr, orderOutputAttr);
        transposesOutRestored = transposesOpRestored.getOutput();
    }
    return transposesOutRestored;
}

//
// ComplexFFTOpConverter
//

template <class T>
class ComplexFFTOpConverter final : public mlir::OpRewritePattern<T> {
public:
    ComplexFFTOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<T>(ctx), _log(log) {
        this->setDebugName("ConvertFFTToConvPass");
    }

public:
    mlir::LogicalResult matchAndRewrite(T op, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got Operation: '{1}'", this->getDebugName(), op);
        const auto dftLoc = appendLoc(op.getLoc(), "ComplexDftOrIdft");
        bool complexInputType = true;
        auto params = fftExtractParams(op.getLoc(), op, complexInputType);
        if (mlir::failed(params)) {
            return mlir::failure();
        }
        auto axes = params.value().axes;
        auto inIter =
                complexFFTDecompose(rewriter, dftLoc, op.getInput(), axes, mlir::isa<IE::IDFTOp>(op), false, _log);
        rewriter.replaceOp(op, inIter);

        return mlir::success();
    }

private:
    Logger _log;
};

//
// RDFTOpConverter
//

class RDFTOpConverter final : public mlir::OpRewritePattern<IE::RDFTOp> {
public:
    RDFTOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::RDFTOp>(ctx), _log(log) {
        setDebugName("RDFTOpConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::RDFTOp op, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got Operation: '{1}'", getDebugName(), op);
        const auto rdftLoc = appendLoc(op.getLoc(), "RdftSplit");
        bool complexInputType = false;
        auto params = fftExtractParams(op.getLoc(), op, complexInputType);
        if (mlir::failed(params)) {
            return mlir::failure();
        }
        auto axes = params.value().axes;
        // first axis will have different approach as input is not complex number
        auto inIter = rdftFirstAxisDecompose(rewriter, rdftLoc, op.getInput(), axes[0], axes.size() == 1, _log);
        // run rest of axes in complex representation
        if (axes.size() > 1) {
            axes.erase(axes.begin());
            inIter = complexFFTDecompose(rewriter, rdftLoc, inIter, axes, false, true, _log);
        }
        rewriter.replaceOp(op, inIter);
        return mlir::success();
    }

private:
    Logger _log;
};

//
// IRDFTOpConverter
//

class IRDFTOpConverter final : public mlir::OpRewritePattern<IE::IRDFTOp> {
public:
    IRDFTOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::IRDFTOp>(ctx), _log(log) {
        setDebugName("IRDFTOpConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::IRDFTOp op, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got Operation: '{1}'", getDebugName(), op);
        const auto irdftLoc = appendLoc(op.getLoc(), "IrdftSplit");
        bool complexInputType = true;
        auto params = fftExtractParams(op.getLoc(), op, complexInputType);
        if (mlir::failed(params)) {
            return mlir::failure();
        }
        auto axes = params.value().axes;
        auto inIter = op.getInput();
        auto lastAxis = axes.back();
        // produce complex part for all axis except last
        if (axes.size() > 1) {
            axes.pop_back();  // delete last axes that will produce real output
            inIter = complexFFTDecompose(rewriter, irdftLoc, inIter, axes, true, false, _log);
        }
        // last axes processing, with convert to Real output.
        inIter = irdftLastAxisDecompose(rewriter, irdftLoc, inIter, lastAxis, _log);
        rewriter.replaceOp(op, inIter);
        return mlir::success();
    }

private:
    Logger _log;
};

//
// ConvertFFTToConvPass
//

class ConvertFFTToConvPass final : public IE::impl::ConvertFFTToConvBase<ConvertFFTToConvPass> {
public:
    explicit ConvertFFTToConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

template <class ConcreteOp>
bool isLegalOp(ConcreteOp op) {
    bool complexInputType = true;
    // RDFT input is real, so rank of input tensor map with parameter rank considered.
    if (mlir::isa<IE::RDFTOp>(op)) {
        complexInputType = false;
    }
    auto params = fftExtractParams(op.getLoc(), op, complexInputType);
    auto axes = params.value().axes;
    auto signalSize = params.value().signalSize;
    // Calculate output shape if signal_size params is default
    const auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto shape = to_small_vector(inType.getShape());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
    auto outShape = to_small_vector(outType.getShape());
    if (mlir::isa<IE::IRDFTOp>(op)) {
        shape.pop_back();
        const auto lastAxis = axes.back();
        shape[lastAxis] = (shape[lastAxis] - 1) * 2;
    }
    if (mlir::isa<IE::RDFTOp>(op)) {
        const auto lastAxis = axes.back();
        // integer / 2 can leave false situation when signalSize attribute is on
        if ((signalSize.back() != -1) && (signalSize.back() != shape[lastAxis])) {
            return true;
        }
        shape[lastAxis] = getRdftOutAxisLength(shape[lastAxis]);
        shape.push_back(2);
    }
    return (shape != outShape);
}

void ConvertFFTToConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::DFTOp>(&isLegalOp<IE::DFTOp>);
    target.addDynamicallyLegalOp<IE::IDFTOp>(&isLegalOp<IE::IDFTOp>);
    target.addDynamicallyLegalOp<IE::RDFTOp>(&isLegalOp<IE::RDFTOp>);
    target.addDynamicallyLegalOp<IE::IRDFTOp>(&isLegalOp<IE::IRDFTOp>);
    target.addLegalOp<IE::TransposeOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::MatMulOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::SliceOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ComplexFFTOpConverter<IE::DFTOp>>(&ctx, _log);
    patterns.add<ComplexFFTOpConverter<IE::IDFTOp>>(&ctx, _log);
    patterns.add<RDFTOpConverter>(&ctx, _log);
    patterns.add<IRDFTOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertFFTToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertFFTToConvPass(Logger log) {
    return std::make_unique<ConvertFFTToConvPass>(log);
}
