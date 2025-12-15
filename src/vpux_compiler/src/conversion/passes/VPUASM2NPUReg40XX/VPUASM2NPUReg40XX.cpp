//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/act_kernel_invocation_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/act_kernel_range_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/act_shave_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/barrier_configure_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/dma_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/managed_barrier_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/managed_mapped_inference_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/mapped_inference_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/mi_version_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/nnrt_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/work_item_rewriter.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"

#include "vpux/compiler/conversion.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
#define GEN_PASS_DECL_CONVERTVPUASM2NPUREG40XX
#define GEN_PASS_DEF_CONVERTVPUASM2NPUREG40XX
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;
using namespace vpuasm2npureg40xx;

namespace {

//
// ConvertVPUASM2NPUReg40XXPass
//

class ConvertVPUASM2NPUReg40XXPass final : public impl::ConvertVPUASM2NPUReg40XXBase<ConvertVPUASM2NPUReg40XXPass> {
public:
    ConvertVPUASM2NPUReg40XXPass(Logger log, uint32_t modelIdentifier): _modelIdentifier(modelIdentifier) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    uint32_t _modelIdentifier;
};

void ConvertVPUASM2NPUReg40XXPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto& ctx = getContext();
    mlir::func::FuncOp netFunc;
    net::NetworkInfoOp netInfo;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, netFunc);

    mlir::ConversionTarget target(ctx);

    target.addLegalDialect<NPUReg40XX::NPUReg40XXDialect>();
    target.addLegalDialect<VPUASM::VPUASMDialect>();

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    ELF::SymbolReferenceMap symRefMap(elfMain, true);

    mlir::RewritePatternSet patternNNDMA(&ctx);
    patternNNDMA.add<NNDMARewriter>(&ctx, _log, symRefMap);
    target.addIllegalOp<VPUASM::NNDMAOp>();
    if (mlir::failed(mlir::applyPartialConversion(netFunc, target, std::move(patternNNDMA)))) {
        signalPassFailure();
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<BarrierRewriter>(&ctx, _log);
    patterns.add<ActShaveRtRewriter>(&ctx, _log);
    patterns.add<ActKernelInvocationRewriter>(&ctx, _log, symRefMap);
    patterns.add<ActKernelRangeRewriter>(&ctx, _log, symRefMap);
    patterns.add<NNRTConfigRewriter>(&ctx, _log, symRefMap);
    patterns.add<ManagedBarrierRewriter>(&ctx, _log);
    patterns.add<MappedInferenceRewriter>(&ctx, _log, symRefMap);
    patterns.add<ManagedMappedInferenceRewriter>(&ctx, _log, _modelIdentifier);
    patterns.add<MappedInferenceVersionRewriter>(&ctx, _log);
    patterns.add<WorkItemRewriter>(&ctx, _log);

    target.addIllegalOp<VPUASM::WorkItemOp>();
    target.addIllegalOp<VPUASM::MappedInferenceOp>();
    target.addIllegalOp<VPUASM::ConfigureBarrierOp>();
    target.addIllegalOp<VPUASM::ActShaveRtOp>();
    target.addIllegalOp<VPUASM::ActKernelInvocationOp>();
    target.addIllegalOp<VPUASM::ActKernelRangeOp>();
    target.addIllegalOp<VPUASM::NNrtConfigOp>();
    target.addIllegalOp<VPUASM::ManagedBarrierOp>();
    target.addIllegalOp<VPUASM::ManagedMappedInferenceOp>();
    target.addIllegalOp<VPUASM::MappedInferenceVersionOp>();

    target.addDynamicallyLegalOp<VPUASM::PlatformInfoOp>([&](VPUASM::PlatformInfoOp op) {
        return config::getArch(op.getOperation()) != config::ArchKind::UNKNOWN;
    });

    if (mlir::failed(mlir::applyPartialConversion(netFunc, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertVPUASM2NPUReg40XXPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUASM2NPUReg40XXPass(Logger log, uint32_t modelIdentifier) {
    return std::make_unique<ConvertVPUASM2NPUReg40XXPass>(log, modelIdentifier);
}
