//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <openvino/op/op.hpp>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU::arch37xx {
#define GEN_PASS_DECL_SPLITREALDFTOPS
#define GEN_PASS_DEF_SPLITREALDFTOPS
#include "vpux/compiler/NPU37XX/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU::arch37xx

using namespace vpux;

namespace {

// A twiddle factor, in fast Fourier transform (FFT) algorithms, is any of the trigonometric constant coefficients that
// are multiplied by the data in the course of the algorithm. More specifically, "twiddle factors" originally referred
// to the root-of-unity complex multiplicative constants in the butterfly operations of the Cooley–Tukey FFT algorithm,
// used to recursively combine smaller discrete Fourier transforms.
template <typename T, typename D>
mlir::Value fftGetTwiddleFactors(mlir::Location loc, T op, ArrayRef<int64_t> axes, ArrayRef<int64_t> signalSize,
                                 mlir::Type dtype, D /*elementType*/, mlir::PatternRewriter& rewriter) {
    const auto outType = op.getOutput().getType().template cast<vpux::NDTypeInterface>();
    auto shape = to_small_vector(outType.getShape());
    // full RDFT last axis shape size need
    if (mlir::isa<VPU::RDFTOp>(op)) {
        const auto inType = op.getInput().getType().template cast<vpux::NDTypeInterface>();
        shape = to_small_vector(inType.getShape());
        for (size_t i = 0; i < axes.size(); ++i) {
            if (signalSize[i] != -1) {
                shape[axes[i]] = signalSize[i];
            }
        }
        shape.push_back(2);
    }
    int64_t size = 0;
    const auto complexNoScale = 2;
    for (auto axis : axes) {
        size += (shape[axis] * shape[axis] * complexNoScale);
    }
    const float pi = 3.141592653589793238462643f;
    float twiddleConst = -2.0f * pi;
    if (mlir::isa<VPU::IDFTOp>(op) || mlir::isa<VPU::IRDFTOp>(op)) {
        twiddleConst = 2.0f * pi;
    }
    std::vector<D> twiddleVals(size);
    int64_t idx = 0;
    for (auto axis : axes) {
        float lengthAxis = static_cast<float>(shape[axis]);
        for (int64_t k = 0; k < shape[axis]; k++) {
            for (int64_t c = 0; c < shape[axis]; c++) {
                float crtProdPos = static_cast<float>(c * k);
                float angle = twiddleConst * crtProdPos / lengthAxis;
                twiddleVals[idx++] = static_cast<D>(std::cos(angle));
                twiddleVals[idx++] = static_cast<D>(std::sin(angle));
            }
        }
    }
    const llvm::SmallVector<int64_t> twiddleShape({size});
    const auto twiddleType = mlir::RankedTensorType::get(twiddleShape, dtype);
    return Const::createConst(rewriter, loc, twiddleType, ArrayRef(twiddleVals));
}

template <typename T>
mlir::Value fftGetTwiddleFactorsByType(T op, ArrayRef<int64_t> axes, ArrayRef<int64_t> signalSize,
                                       mlir::PatternRewriter& rewriter) {
    auto* ctx = op->getContext();
    // produce constant twiddle factors with ouput type precision. Keep consistent precision.
    const auto outType = op.getOutput().getType().template cast<vpux::NDTypeInterface>();
    auto outElementType = outType.getElementType();
    if (outElementType.isF16()) {
        vpux::type::float16 elementType = 0;
        return fftGetTwiddleFactors(op.getLoc(), op, axes, signalSize, mlir::Float16Type::get(ctx), elementType,
                                    rewriter);
    } else {
        float elementType = 0;
        return fftGetTwiddleFactors(op.getLoc(), op, axes, signalSize, mlir::Float32Type::get(ctx), elementType,
                                    rewriter);
    }
}

//
// SplitRDFT
//

// VPU.RDFT = {VPU.RDFTUncutOp->VPU.SliceOp}
class SplitRDFTToComponents final : public mlir::OpRewritePattern<VPU::RDFTOp> {
public:
    SplitRDFTToComponents(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::RDFTOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::RDFTOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Convert to computational form. In conformity with RDFT operation definition, it is necessary to convert input to
// complex number, apply DFT transformation and after last axes processing cut off symmetric part. There are, and it
// will be implemented more sw kernel that will optimize algorithm with dedicate first axis processing that will not
// make any unnecessary data movement. Same for last axis (not produced unnecessary part).
mlir::LogicalResult SplitRDFTToComponents::matchAndRewrite(VPU::RDFTOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found RDFT Operation '{0}'", origOp->getLoc());
    auto* ctx = origOp->getContext();
    const auto outputShape = getShape(origOp.getOutput()).raw();

    auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesAttr());
    auto signalSize = parseIntArrayAttr<int64_t>(origOp.getSignalSizeAttr());
    auto twiddleConstantOp = fftGetTwiddleFactorsByType(origOp, axes, signalSize, rewriter);

    auto rdft = rewriter.create<VPU::RDFTUncutOp>(origOp->getLoc(), origOp.getInput(), twiddleConstantOp,
                                                  origOp.getAxesAttr(), origOp.getSignalSizeAttr());
    SmallVector<int64_t> offsets(outputShape.size(), 0);
    SmallVector<int64_t> sizes(outputShape.begin(), outputShape.end());
    rewriter.replaceOpWithNewOp<VPU::SliceOp>(origOp, rdft.getOutput(), getIntArrayAttr(ctx, offsets),
                                              getIntArrayAttr(ctx, sizes));
    return mlir::success();
}

//
// Split DFT and IDFT
//

template <class T>
class SplitComplexFFTToComponents final : public mlir::OpRewritePattern<T> {
public:
    SplitComplexFFTToComponents(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<T>(ctx), _log(log) {
    }

public:
    // Convert to computational form DFT and IDFT op. Constant part and dynamic part
    mlir::LogicalResult matchAndRewrite(T origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("Found DFT or IDFT Operation '{0}'", origOp->getLoc());

        auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesAttr());
        auto signalSize = parseIntArrayAttr<int64_t>(origOp.getSignalSizeAttr());
        auto twiddleConstantOp = fftGetTwiddleFactorsByType(origOp, axes, signalSize, rewriter);

        rewriter.replaceOpWithNewOp<T>(origOp, origOp.getInput(), twiddleConstantOp, origOp.getAxesAttr(),
                                       origOp.getSignalSizeAttr());
        return mlir::success();
    }

private:
    Logger _log;
};

//
// RewriteIRDFT
//

// VPU.IRDFT = {VPU.IDFTOp->VPU.IRDFTLastAxisOp}
class SplitIRDFTToComponents final : public mlir::OpRewritePattern<VPU::IRDFTOp> {
public:
    SplitIRDFTToComponents(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::IRDFTOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::IRDFTOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Convert to computational form. In conformity with IRDFT operation definition it contain IDFT operation and after, cut
// off complex part of result. Special filter is created for not processing unnecessary imaginary part on last axis
// processing (this filter is in fact IRDFT filter when params contain just 1 axis).
mlir::LogicalResult SplitIRDFTToComponents::matchAndRewrite(VPU::IRDFTOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IRDFT Operation '{0}'", origOp->getLoc());
    auto* ctx = origOp->getContext();
    // remove last axis and signal from axis and signal and keep for IRDFTLastAxisOp
    auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesAttr());
    auto signalSize = parseIntArrayAttr<int64_t>(origOp.getSignalSizeAttr());
    auto lastAxis = SmallVector<int64_t>{axes.back()};
    auto lastSignalSize = SmallVector<int64_t>{signalSize.back()};
    axes.pop_back();
    signalSize.pop_back();
    auto irdftInput = origOp.getInput();
    if (!axes.empty()) {
        auto twiddleConstantOp = fftGetTwiddleFactorsByType(origOp, axes, signalSize, rewriter);
        irdftInput = rewriter.create<VPU::IDFTOp>(origOp->getLoc(), origOp.getInput(), twiddleConstantOp,
                                                  getIntArrayAttr(ctx, axes), getIntArrayAttr(ctx, signalSize));
    }
    auto twiddleConstantLastAxisOp = fftGetTwiddleFactorsByType(origOp, lastAxis, lastSignalSize, rewriter);
    rewriter.replaceOpWithNewOp<VPU::IRDFTLastAxisOp>(origOp, irdftInput, twiddleConstantLastAxisOp,
                                                      getIntArrayAttr(ctx, lastAxis),
                                                      getIntArrayAttr(ctx, lastSignalSize));
    return mlir::success();
}

//
// SplitRealDFTOpsPass
//

class SplitRealDFTOpsPass final : public VPU::arch37xx::impl::SplitRealDFTOpsBase<SplitRealDFTOpsPass> {
public:
    explicit SplitRealDFTOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

template <class ConcreteOp>
bool isLegalOp(ConcreteOp op) {
    return (op.getTwiddleFactors() != nullptr);
}

void SplitRealDFTOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);

    target.addDynamicallyLegalOp<VPU::DFTOp>(&isLegalOp<VPU::DFTOp>);
    target.addDynamicallyLegalOp<VPU::IDFTOp>(&isLegalOp<VPU::IDFTOp>);
    target.addIllegalOp<VPU::RDFTOp>();
    target.addIllegalOp<VPU::IRDFTOp>();
    target.addLegalOp<VPU::RDFTUncutOp>();
    target.addLegalOp<VPU::IRDFTLastAxisOp>();
    target.addLegalOp<VPU::SliceOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SplitRDFTToComponents>(&ctx, _log);
    patterns.add<SplitIRDFTToComponents>(&ctx, _log);
    patterns.add<SplitComplexFFTToComponents<VPU::DFTOp>>(&ctx, _log);
    patterns.add<SplitComplexFFTToComponents<VPU::IDFTOp>>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        _log.debug("Failed to split RealDFTOps into parts.");
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitRealDFTOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::arch37xx::createSplitRealDFTOpsPass(Logger log) {
    return std::make_unique<SplitRealDFTOpsPass>(log);
}
