//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/act_kernel_invocation_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/act_kernel_range_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/act_shave_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/barrier_configure_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/dma_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/dpu_pvp_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/managed_barrier_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/managed_mapped_inference_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/mapped_inference_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/mi_version_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/nnrt_rewriter.hpp"

#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/dialect.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp"
#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/work_item_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"

#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"

#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/conversion.hpp"

namespace vpux {
#define GEN_PASS_DECL_CONVERTVPUASM2NPUREG60XX
#define GEN_PASS_DEF_CONVERTVPUASM2NPUREG60XX
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;
using namespace vpuasm2npureg60xx;

namespace {

//
// ConvertVPUASM2NPUReg60XXPass
//

class ConvertVPUASM2NPUReg60XXPass final : public impl::ConvertVPUASM2NPUReg60XXBase<ConvertVPUASM2NPUReg60XXPass> {
public:
    explicit ConvertVPUASM2NPUReg60XXPass(Logger log, uint32_t modelIdentifier): _modelIdentifier(modelIdentifier) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    uint32_t _modelIdentifier;
};

void ConvertVPUASM2NPUReg60XXPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto& ctx = getContext();
    auto netFunc = net::getMainFunc(moduleOp);

    mlir::ConversionTarget target(ctx);

    target.addLegalDialect<NPUReg60XX::NPUReg60XXDialect>();
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
    patterns.add<BarrierRewriter>(&ctx, _log);
    patterns.add<MappedInferenceRewriter>(&ctx, _log, symRefMap);
    patterns.add<NNRTConfigRewriter>(&ctx, _log, symRefMap);
    patterns.add<MappedInferenceVersionRewriter>(&ctx, _log);
    patterns.add<ManagedBarrierRewriter>(&ctx, _log);
    patterns.add<ManagedMappedInferenceRewriter>(&ctx, _log, _modelIdentifier);
    patterns.add<WorkItemRewriter>(&ctx, _log);
    patterns.add<DpuPVPRewriter>(&ctx, _log, elfMain);

    target.addDynamicallyLegalOp<VPUASM::PlatformInfoOp>([&](VPUASM::PlatformInfoOp op) {
        return config::getArch(op) != config::ArchKind::UNKNOWN;
    });

    target.addIllegalOp<VPUASM::WorkItemOp>();
    target.addIllegalOp<VPUASM::ManagedMappedInferenceOp>();
    target.addIllegalOp<VPUASM::ManagedBarrierOp>();
    target.addIllegalOp<VPUASM::MappedInferenceVersionOp>();
    target.addIllegalOp<VPUASM::NNrtConfigOp>();
    target.addIllegalOp<VPUASM::MappedInferenceOp>();
    target.addIllegalOp<VPUASM::ConfigureBarrierOp>();
    target.addIllegalOp<VPUASM::ActShaveRtOp>();
    target.addIllegalOp<VPUASM::ActKernelInvocationOp>();
    target.addIllegalOp<VPUASM::ActKernelRangeOp>();
    target.addIllegalOp<VPUASM::DpuPVPOp>();
    if (mlir::failed(mlir::applyPartialConversion(netFunc, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertVPUASM2NPUReg60XXPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUASM2NPUReg60XXPass(Logger log, uint32_t modelIdentifier) {
    return std::make_unique<ConvertVPUASM2NPUReg60XXPass>(log, modelIdentifier);
}
