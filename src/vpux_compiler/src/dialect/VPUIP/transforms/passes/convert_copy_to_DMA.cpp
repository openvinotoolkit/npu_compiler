//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTTRANSFEROPSTODMAS
#define GEN_PASS_DEF_CONVERTTRANSFEROPSTODMAS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// TimestampRewrite
//

class TimestampRewrite final : public mlir::OpRewritePattern<VPUIP::TimestampOp> {
public:
    TimestampRewrite(mlir::MLIRContext* ctx, vpux::VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPUIP::TimestampOp>(ctx), _log(log), _arch(arch) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::TimestampOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    vpux::VPU::ArchKind _arch;
};

mlir::LogicalResult TimestampRewrite::matchAndRewrite(VPUIP::TimestampOp origOp,
                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Timestamp Operation '{0}'", origOp->getLoc());

    auto origType = origOp.getType().cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(origType.getNumElements() == 1, "Got wrong elements number for TimestampOp");

    const auto timerType = origType.changeMemSpace(VPU::MemoryKind::Register);

    uint32_t hwAddress = 0;

    switch (_arch) {
    case VPU::ArchKind::NPU37XX:
        hwAddress = VPUIP::HW_TIMER_ABSOLUTE_ADDR_37XX;
        VPUX_THROW_UNLESS(origType.getElementType() == getUInt64Type(getContext()),
                          "Got wrong element type for TimestampOp");
        break;
    default:
        VPUX_THROW("Unsuported architecture for TimestampRewrite");
    }

    auto bufferOp = rewriter.create<VPURT::DeclareBufferOp>(origOp->getLoc(), timerType, VPURT::BufferSection::Register,
                                                            hwAddress);

    rewriter.replaceOpWithNewOp<VPUIP::NNDMAOp>(origOp, bufferOp.getBuffer(), origOp.getOutputBuff());

    _log.trace("Replaced with 'VPURT::DeclareBufferOp'");

    return mlir::success();
}

//
// CopyOpRewrite
//

class CopyOpRewrite final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CopyOpRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CopyOpRewrite::matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found CopyOp Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPUIP::NNDMAOp>(origOp, origOp.getInput(), origOp.getOutputBuff());

    return mlir::success();
}

//
// ConvertTransferOpsToDMAsPass
//

class ConvertTransferOpsToDMAsPass final :
        public VPUIP::impl::ConvertTransferOpsToDMAsBase<ConvertTransferOpsToDMAsPass> {
public:
    explicit ConvertTransferOpsToDMAsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertTransferOpsToDMAsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPUIP::CopyOp>();
    target.addIllegalOp<VPUIP::TimestampOp>();
    target.addLegalOp<VPUIP::NNDMAOp>();
    target.addLegalOp<VPURT::DeclareBufferOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<TimestampRewrite>(&ctx, arch, _log);
    patterns.insert<CopyOpRewrite>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertTransferOpsToDMAsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertTransferOpsToDMAsPass(Logger log) {
    return std::make_unique<ConvertTransferOpsToDMAsPass>(log);
}
