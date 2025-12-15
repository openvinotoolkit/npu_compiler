//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/dialect.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/act_kernel_invocation_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/act_kernel_range_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/act_shave_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/barrier_configure_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/dma_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/managed_barrier_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/managed_mapped_inference_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/mapped_inference_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/mi_version_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/nnrt_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/work_item_rewriter.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"

#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/conversion.hpp"

namespace vpux {
#define GEN_PASS_DECL_CONVERTVPUASM2NPUREG50XX
#define GEN_PASS_DEF_CONVERTVPUASM2NPUREG50XX
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpuasm2npureg50xx;

namespace {
//
// ConvertVPUASM2NPUReg50XXPass
//

class ConvertVPUASM2NPUReg50XXPass final : public impl::ConvertVPUASM2NPUReg50XXBase<ConvertVPUASM2NPUReg50XXPass> {
public:
    explicit ConvertVPUASM2NPUReg50XXPass(Logger log, uint32_t modelIdentifier): _modelIdentifier(modelIdentifier) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    uint32_t _modelIdentifier;
    void safeRunOnModule() final;
};

void ConvertVPUASM2NPUReg50XXPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto& ctx = getContext();
    mlir::func::FuncOp netFunc;
    net::NetworkInfoOp netInfo;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, netFunc);

    mlir::ConversionTarget target(ctx);

    target.addLegalDialect<NPUReg50XX::NPUReg50XXDialect>();
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
    patterns.add<ActShaveRtRewriter>(&ctx, _log);
    patterns.add<ActKernelInvocationRewriter>(&ctx, _log, symRefMap);
    patterns.add<ActKernelRangeRewriter>(&ctx, _log, symRefMap);
    patterns.add<MappedInferenceRewriter>(&ctx, _log, symRefMap);
    patterns.add<NNRTConfigRewriter>(&ctx, _log, symRefMap);
    patterns.add<WorkItemRewriter>(&ctx, _log);
    patterns.add<ManagedMappedInferenceRewriter>(&ctx, _log, _modelIdentifier);
    patterns.add<ManagedBarrierRewriter>(&ctx, _log);
    patterns.add<BarrierRewriter>(&ctx, _log);
    patterns.add<MappedInferenceVersionRewriter>(&ctx, _log);

    target.addIllegalOp<VPUASM::MappedInferenceVersionOp>();
    target.addIllegalOp<VPUASM::ConfigureBarrierOp>();
    target.addIllegalOp<VPUASM::ManagedBarrierOp>();
    target.addIllegalOp<VPUASM::ManagedMappedInferenceOp>();
    target.addIllegalOp<VPUASM::WorkItemOp>();
    target.addIllegalOp<VPUASM::NNrtConfigOp>();
    target.addIllegalOp<VPUASM::MappedInferenceOp>();
    target.addIllegalOp<VPUASM::ActShaveRtOp>();
    target.addIllegalOp<VPUASM::ActKernelInvocationOp>();
    target.addIllegalOp<VPUASM::ActKernelRangeOp>();

    target.addDynamicallyLegalOp<VPUASM::PlatformInfoOp>([&](VPUASM::PlatformInfoOp op) {
        return config::getArch(op) != config::ArchKind::UNKNOWN;
    });

    if (mlir::failed(mlir::applyPartialConversion(netFunc, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertVPUASM2NPUReg50XXPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUASM2NPUReg50XXPass(Logger log, uint32_t modelIdentifier) {
    return std::make_unique<ConvertVPUASM2NPUReg50XXPass>(log, modelIdentifier);
}
