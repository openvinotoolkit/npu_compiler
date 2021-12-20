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

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/IE/loop.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

constexpr int64_t MAX_KERNEL = 11;
constexpr int64_t PADDING_RIGHT = 1;
constexpr int64_t PADDING_BOT = 3;

struct Factors {
    int64_t larger = 0;
    int64_t smaller = 0;

    Factors() {
    }
    Factors(int64_t larger, int64_t smaller): larger(larger), smaller(smaller) {
    }
};

bool checkFactors(const Factors& factors, int64_t kernelSize = 0) {
    const auto hasZeroFactors = factors.larger == 0 || factors.smaller == 0;
    const auto factorLessThanKernelSize = factors.larger * factors.smaller < kernelSize;
    const auto hasUnsupportedFactors = factors.larger > MAX_KERNEL || factors.smaller > MAX_KERNEL;
    const auto hasBadFactors = factors.larger * factors.smaller > (kernelSize + factors.smaller / 2);
    return !(hasZeroFactors || factorLessThanKernelSize || hasUnsupportedFactors || hasBadFactors);
    // those last 2 checks have the main scope of finding the best suited factors:
    // if one of the last 2 checks fails it means that the gap between product of
    // those 2 factors and original kernel size is too big, which generates larger overlapping area
}

SmallVector<Factors> getFactorsList(int64_t n) {
    SmallVector<Factors> factors;
    for (int64_t i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) {
            factors.emplace_back(n / i, i);  // larger, smaller
        }
    }
    return factors;
}

Factors getFactorsAround(int64_t kernelSize, int64_t pad) {
    const auto& candidateFactors = getFactorsList(kernelSize + pad);
    if (!candidateFactors.empty()) {
        return candidateFactors.back();
    }
    return {};
}

Factors getFactors(int64_t kernelSize) {
    const auto& allFactors = getFactorsList(kernelSize);
    if (!allFactors.empty() && checkFactors(allFactors.back(), kernelSize)) {
        return allFactors.back();
    }

    int64_t padValue = 1;
    while (padValue < kernelSize) {
        const auto& factors = getFactorsAround(kernelSize, padValue);
        if (checkFactors(factors, kernelSize)) {
            return factors;
        }
        padValue++;
    }
    VPUX_THROW("All factors failed check");
}

void getFactorsForSecondDimension(std::array<int64_t, 4>& padding, std::array<int64_t, 2>& firstOpKernel,
                                  std::array<int64_t, 2>& sequencedOpKernel, int32_t smallDim, Logger log,
                                  ArrayRef<int64_t> kernelSize) {
    const auto factorsSecondDim = getFactors(kernelSize[smallDim]);  // toggling between the two kernel sizes
    log.trace("Second Dimension kernel[{0}]= {1}, larger factor: {2} , smaller factor: {3}", smallDim,
              kernelSize[smallDim], factorsSecondDim.larger, factorsSecondDim.smaller);

    VPUX_THROW_UNLESS((factorsSecondDim.larger <= MAX_KERNEL) && (factorsSecondDim.smaller <= MAX_KERNEL),
                      "Second dimension factors ({1}, {2})  are larger than MAX_KERNEL {0}", MAX_KERNEL,
                      factorsSecondDim.larger, factorsSecondDim.smaller);
    firstOpKernel[smallDim] = factorsSecondDim.larger;
    sequencedOpKernel[smallDim] = factorsSecondDim.smaller;
    auto multipliedFactors = firstOpKernel[smallDim] * sequencedOpKernel[smallDim];

    padding[PADDING_BOT] = (multipliedFactors > kernelSize[smallDim]) ? 1 : 0;
}

void calculateKernelsAndPadding(ArrayRef<int64_t> kernelSize, std::array<int64_t, 4>& padding,
                                std::array<int64_t, 2>& firstOpKernel, std::array<int64_t, 2>& sequencedOpKernel,
                                Logger log) {
    const auto KY = kernelSize[Dims4D::Kernel::Y.ind()];
    const auto KX = kernelSize[Dims4D::Kernel::X.ind()];

    // figure out the bigger kernel dimension width or height when having an asymmetric kernel
    auto largerKernelSize = KX;
    auto largeDim = Dims4D::Kernel::X.ind();
    auto smallDim = Dims4D::Kernel::Y.ind();
    auto asymmetricCase = (KX != KY);
    auto asymmetricBothKernelsLarge = (asymmetricCase && (KX > MAX_KERNEL) && (KY > MAX_KERNEL));

    // deal with asymetric kernels when one dim is larger than MAX_KERNEL
    if (asymmetricCase && (KX < KY)) {
        largerKernelSize = KY;
        largeDim = Dims4D::Kernel::Y.ind();
        smallDim = Dims4D::Kernel::X.ind();
    }
    const auto factors = getFactors(largerKernelSize);

    log.trace("Large Dimension kernelSize[{0}] = {1}, larger factor: {2} , smaller factor: {3}", largeDim,
              largerKernelSize, factors.larger, factors.smaller);
    VPUX_THROW_UNLESS((factors.larger <= MAX_KERNEL) && (factors.smaller <= MAX_KERNEL),
                      "Large dimension factors ({1}, {2})  are larger the MAX_KERNEL {0}", MAX_KERNEL, factors.larger,
                      factors.smaller);

    // cascading supported ops
    // first op kernel [factors.larger, factorsSecondDim.larger] - firstOpKernel
    // sequenced op kernel [factors.smaller, factorsSecondDim.smaller] - sequencedOpKernel
    // Padding quantity relationship is (input size + pad) / k = output size, padding config is TRUE, FALSE
    firstOpKernel[largeDim] = factors.larger;  // first was the large dimension
    sequencedOpKernel[largeDim] = factors.smaller;
    auto multipliedFactors = firstOpKernel[largeDim] * sequencedOpKernel[largeDim];

    if (asymmetricCase) {
        if (asymmetricBothKernelsLarge) {
            getFactorsForSecondDimension(padding, firstOpKernel, sequencedOpKernel, smallDim, log, kernelSize);
        } else {
            firstOpKernel[smallDim] = kernelSize[smallDim];
            sequencedOpKernel[smallDim] =
                    1;  // the smallDim was not factorized, the multiplication kSize*1 covers the second op

            padding[PADDING_BOT] = 0;
        }
        // factors multiplied > kernel, we need padding
        padding[PADDING_RIGHT] = (multipliedFactors > kernelSize[largeDim]) ? 1 : 0;

        if (largeDim != Dims4D::Kernel::X.ind()) {
            // change the padding on the other dimensions as largeDim was not on the width dimension - PADD_RIGHT
            std::swap(padding[PADDING_RIGHT], padding[PADDING_BOT]);
        }
    } else {
        firstOpKernel[smallDim] = factors.larger;  // largeDim has the same kernel size as smallDim
        sequencedOpKernel[smallDim] = factors.smaller;
        padding[PADDING_RIGHT] = padding[PADDING_BOT] = (multipliedFactors > kernelSize[largeDim]) ? 1 : 0;
    }
}

//
// AveragePoolRewriter
//

class AveragePoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AveragePoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
        setDebugName("AveragePoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AveragePoolRewriter::matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got AveragePool layer at '{1}'", getDebugName(), origOp->getLoc());

    std::array<int64_t, 4> calculatedPadding = {0, 0, 0, 0};
    std::array<int64_t, 2> firstOpKernel, sequencedOpKernel = {1, 1};

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    calculateKernelsAndPadding(kernelSize, calculatedPadding, firstOpKernel, sequencedOpKernel, _log.nest(2));

    auto* ctx = origOp->getContext();

    const auto firstOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[2], calculatedPadding[0]}));
    const auto firstOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[3], calculatedPadding[1]}));

    const auto firstOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(firstOpKernel));
    const auto sequencedOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(sequencedOpKernel));

    auto firstOp = rewriter.create<IE::AvgPoolOp>(origOp->getLoc(), origOp.input(), firstOpKernelAttr,
                                                  firstOpKernelAttr, firstOpPadBegin, firstOpPadEnd,
                                                  origOp.rounding_typeAttr(), origOp.exclude_padsAttr());

    calculatedPadding = {0, 0, 0, 0};
    const auto sequencedOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[2], calculatedPadding[0]}));
    const auto sequencedOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[3], calculatedPadding[1]}));
    rewriter.replaceOpWithNewOp<IE::AvgPoolOp>(origOp, origOp.getType(), firstOp.output(), sequencedOpKernelAttr,
                                               sequencedOpKernelAttr, sequencedOpPadBegin, sequencedOpPadEnd,
                                               origOp.rounding_typeAttr(), origOp.exclude_padsAttr());

    return mlir::success();
}

//
// MaxPoolRewriter
//

class MaxPoolRewriter final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
        setDebugName("MaxPoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool layer at '{1}'", getDebugName(), origOp->getLoc());

    std::array<int64_t, 4> calculatedPadding = {0, 0, 0, 0};
    std::array<int64_t, 2> firstOpKernel, sequencedOpKernel = {1, 1};

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    calculateKernelsAndPadding(kernelSize, calculatedPadding, firstOpKernel, sequencedOpKernel, _log.nest(2));

    mlir::MLIRContext* ctx = origOp->getContext();

    const auto origStridesAttr = origOp.strides();
    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    std::array<int64_t, 4> origPadding = {padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]};
    std::array<int64_t, 4> inputPadding = calculatedPadding;

    auto stridesAttr = getIntArrayAttr(ctx, makeArrayRef(firstOpKernel));
    const auto origStrides = parseIntArrayAttr<int64_t>(origStridesAttr);

    if (origStrides[Dims4D::Strides::X.ind()] != kernelSize[Dims4D::Kernel::X.ind()] ||
        origStrides[Dims4D::Strides::Y.ind()] != kernelSize[Dims4D::Kernel::Y.ind()]) {
        inputPadding = origPadding;
        stridesAttr = origStridesAttr;
    }

    mlir::ArrayAttr firstOpPadBegin, firstOpPadEnd;
    bool unsuportedPad = false;
    bool isSupportedYPadding = (inputPadding[0] < firstOpKernel[Dims4D::Kernel::Y.ind()] / 2) &&
                               (inputPadding[1] < firstOpKernel[Dims4D::Kernel::Y.ind()] / 2);
    bool isSupportedXPadding = (inputPadding[2] < firstOpKernel[Dims4D::Kernel::X.ind()] / 2) &&
                               (inputPadding[3] < firstOpKernel[Dims4D::Kernel::X.ind()] / 2);
    bool allPaddingsEqual = std::all_of(inputPadding.cbegin(), inputPadding.cend(), [&inputPadding](int64_t inPad) {
        return inPad == inputPadding[0];
    });

    if (!isSupportedXPadding && !isSupportedYPadding && allPaddingsEqual) {
        unsuportedPad = true;
        firstOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({inputPadding[2] / 2, inputPadding[0] / 2}));
        firstOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({inputPadding[3] / 2, inputPadding[1] / 2}));
    } else {
        firstOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({inputPadding[2], inputPadding[0]}));
        firstOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({inputPadding[3], inputPadding[1]}));
    }
    const auto firstOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(firstOpKernel));
    auto sequencedOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(sequencedOpKernel));

    auto firstOp = rewriter.create<IE::MaxPoolOp>(origOp.getLoc(), origOp.input(), firstOpKernelAttr, stridesAttr,
                                                  firstOpPadBegin, firstOpPadEnd, origOp.rounding_type(),
                                                  origOp.post_opAttr());

    stridesAttr = sequencedOpKernelAttr;

    if (origStrides[Dims4D::Strides::X.ind()] != kernelSize[Dims4D::Kernel::X.ind()] ||
        origStrides[Dims4D::Strides::Y.ind()] != kernelSize[Dims4D::Kernel::Y.ind()]) {
        // in this case stride shall be taken into account and pyramid cascading does not work
        // use expression orig_kernel = sum (k1, k2, ..., ki)
        sequencedOpKernel[Dims4D::Kernel::X.ind()] =
                kernelSize[Dims4D::Kernel::X.ind()] - firstOpKernel[Dims4D::Kernel::X.ind()] + 1;
        sequencedOpKernel[Dims4D::Kernel::Y.ind()] =
                kernelSize[Dims4D::Kernel::Y.ind()] - firstOpKernel[Dims4D::Kernel::Y.ind()] + 1;
        stridesAttr = origStridesAttr;
        calculatedPadding = {0, 0, 0, 0};
    }
    if (unsuportedPad) {
        calculatedPadding[0] = inputPadding[0] - inputPadding[0] / 2;
        calculatedPadding[1] = inputPadding[1] - inputPadding[1] / 2;
        calculatedPadding[2] = inputPadding[2] - inputPadding[2] / 2;
        calculatedPadding[3] = inputPadding[3] - inputPadding[3] / 2;
    }

    const auto sequencedOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[2], calculatedPadding[0]}));
    const auto sequencedOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[3], calculatedPadding[1]}));
    sequencedOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(sequencedOpKernel));
    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(origOp, firstOp.output(), sequencedOpKernelAttr, stridesAttr,
                                               sequencedOpPadBegin, sequencedOpPadEnd, origOp.rounding_type(),
                                               origOp.post_opAttr());
    return mlir::success();
}

//
// HandleLargeKernelsPass
//

class HandleLargeKernelsPass final : public IE::HandleLargeKernelsBase<HandleLargeKernelsPass> {
public:
    explicit HandleLargeKernelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void HandleLargeKernelsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto hasSupportedKernels = [](const SmallVector<int64_t>& kernelSize) {
        const auto KY = kernelSize[Dims4D::Kernel::Y.ind()];
        const auto KX = kernelSize[Dims4D::Kernel::X.ind()];

        return KY <= MAX_KERNEL && KX <= MAX_KERNEL;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::AvgPoolOp>([&](IE::AvgPoolOp op) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_size());
        if (hasSupportedKernels(kernelSize)) {
            return true;
        }
        const auto inDataType = op.input().getType().cast<mlir::ShapedType>();
        const auto inDataShape = inDataType.getShape();
        const auto strides = parseIntArrayAttr<int64_t>(op.strides());
        const auto maxKernelSizeSupported = 121;  // we can only get 2 factors and max kernel should be 11 * 11 = 121
        auto unsupportedKernelCheck = [&](int32_t kernelInd, int32_t actInd, int32_t strideInd) {
            return ((kernelSize[kernelInd] < inDataShape[actInd] && kernelSize[kernelInd] != strides[strideInd]) ||
                    kernelSize[kernelInd] > maxKernelSizeSupported);
        };

        if (unsupportedKernelCheck(Dims4D::Kernel::X.ind(), Dims4D::Act::W.ind(), Dims4D::Strides::X.ind())) {
            _log.trace("AvgPool operation unsupported by HandleLargeKernel pass");
            return true;
        }
        if (unsupportedKernelCheck(Dims4D::Kernel::Y.ind(), Dims4D::Act::H.ind(), Dims4D::Strides::Y.ind())) {
            _log.trace("AvgPool operation unsupported by HandleLargeKernel pass");
            return true;
        }
        return false;
    });
    target.addDynamicallyLegalOp<IE::MaxPoolOp>([&](IE::MaxPoolOp op) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_size());
        return hasSupportedKernels(kernelSize);
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<AveragePoolRewriter>(&ctx, _log);
    patterns.insert<MaxPoolRewriter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createHandleLargeKernelsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createHandleLargeKernelsPass(Logger log) {
    return std::make_unique<HandleLargeKernelsPass>(log);
}
