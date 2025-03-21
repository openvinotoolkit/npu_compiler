//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/m2i_utils.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/enums.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPU::arch40xx {
#define GEN_PASS_DECL_CONVERTM2IOPS
#define GEN_PASS_DEF_CONVERTM2IOPS
#include "vpux/compiler/NPU40XX/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU::arch40xx

using namespace vpux;
using namespace VPU;

namespace {

//
// ConvertM2iResizeToTask
//

class ConvertM2iResizeToTask final : public mlir::OpRewritePattern<VPU::M2IResizeOp> {
public:
    ConvertM2iResizeToTask(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::M2IResizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::M2IResizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertM2iResizeToTask::matchAndRewrite(VPU::M2IResizeOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto elType = origOp.getInput().getType().cast<vpux::NDTypeInterface>().getElementType();
    M2iColorFmt fmt;

    if (elType.isUnsignedInteger(8)) {
        // If last axes value is the last shape dim => planar
        const auto axes = parseIntArrayAttr<int64_t>(origOp.getAxes());
        const auto axesSize = axes.size();
        const auto outType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
        if (axes[axesSize - 1] == outType.getRank() - 1) {
            fmt = M2iColorFmt::PL_RGB24;
        } else {
            fmt = M2iColorFmt::IL_RGB888;
        }
    } else if (elType.isF16()) {
        fmt = M2iColorFmt::PL_FP16_RGB;
    } else {
        VPUX_THROW("m2i unsupported format {0}", elType);
    }
    auto chromaOrder = 0;  // YCbCr format: 0 - YCbCr, 1 - YCrCb;   RGB format: 0 - RGB, 1 - BGR
    auto lumaOrder = 0;    // YCbCr format: 0 - Y before CbCr, 1 - Y after CbCr
    auto m2iOp = rewriter.create<VPU::M2ITaskOp>(origOp->getLoc(), origOp.getType(), origOp.getInput(), false, false,
                                                 fmt, fmt, chromaOrder, chromaOrder, lumaOrder, lumaOrder,
                                                 origOp.getSizes(), origOp.getAxes(), nullptr, origOp.getInterp());
    rewriter.replaceOp(origOp, m2iOp.getOutput());

    return mlir::success();
}

//
// ConvertM2iCscToTask
//

class ConvertM2iCscToTask final : public mlir::OpRewritePattern<VPU::M2IColorConvertOp> {
public:
    ConvertM2iCscToTask(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::M2IColorConvertOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::M2IColorConvertOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertM2iCscToTask::matchAndRewrite(VPU::M2IColorConvertOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto inColorOrder = getM2iColorOrderReg(origOp.getInFmtAttr().getValue());
    auto outColorOrder = getM2iColorOrderReg(origOp.getOutFmtAttr().getValue());
    auto inLumaOrder = getM2iLumaOrderReg(origOp.getInFmtAttr().getValue());
    auto outLumaOrder = getM2iLumaOrderReg(origOp.getOutFmtAttr().getValue());

    auto iFmt = IEtoM2iColorFmt(origOp.getInFmtAttr().getValue());
    auto oFmt = IEtoM2iColorFmt(origOp.getOutFmtAttr().getValue());

    auto m2iOp = rewriter.create<VPU::M2ITaskOp>(origOp->getLoc(), origOp.getType(), origOp.getInput(), true, false,
                                                 iFmt, oFmt, inColorOrder, outColorOrder, inLumaOrder, outLumaOrder,
                                                 nullptr, nullptr, nullptr);
    rewriter.replaceOp(origOp, m2iOp.getOutput());

    return mlir::success();
}

//
// ConvertM2iNormToTask
//

class ConvertM2iNormToTask final : public mlir::OpRewritePattern<VPU::M2INormOp> {
public:
    ConvertM2iNormToTask(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::M2INormOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::M2INormOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertM2iNormToTask::matchAndRewrite(VPU::M2INormOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto coeffsAttr = VPU::getM2INormCoeffsAttr(getContext(), origOp);

    auto fmt = M2iColorFmt::PL_FP16_RGB;  // any planar FP16 format
    auto chromaOrder = 0;                 // YCbCr format: 0 - YCbCr, 1 - YCrCb;   RGB format: 0 - RGB, 1 - BGR
    auto lumaOrder = 0;                   // YCbCr format: 0 - Y before CbCr, 1 - Y after CbCr
    auto m2iOp = rewriter.create<VPU::M2ITaskOp>(origOp->getLoc(), origOp.getType(), origOp.getInput(), false, true,
                                                 fmt, fmt, chromaOrder, chromaOrder, lumaOrder, lumaOrder, nullptr,
                                                 nullptr, coeffsAttr);
    rewriter.replaceOp(origOp, m2iOp.getOutput());

    return mlir::success();
}

//
// ConvertM2IOpsPass
//

class ConvertM2IOpsPass final : public VPU::arch40xx::impl::ConvertM2IOpsBase<ConvertM2IOpsPass> {
public:
    explicit ConvertM2IOpsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertM2IOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertM2iCscToTask>(&ctx, _log);
    patterns.add<ConvertM2iResizeToTask>(&ctx, _log);
    patterns.add<ConvertM2iNormToTask>(&ctx, _log);

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<VPU::M2ITaskOp>();
    target.addIllegalOp<VPU::M2IColorConvertOp>();
    target.addIllegalOp<VPU::M2IResizeOp>();
    target.addIllegalOp<VPU::M2INormOp>();

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertM2IOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::arch40xx::createConvertM2IOpsPass(Logger log) {
    return std::make_unique<ConvertM2IOpsPass>(log);
}
