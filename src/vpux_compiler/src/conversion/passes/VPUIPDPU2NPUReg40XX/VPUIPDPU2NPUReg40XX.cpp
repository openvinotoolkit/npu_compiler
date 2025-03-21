//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg40XX/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg40XX/dpu_variant_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/utils/dry_run_utils.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>

namespace vpux {
#define GEN_PASS_DECL_CONVERTVPUIPDPU2NPUREG40XX
#define GEN_PASS_DEF_CONVERTVPUIPDPU2NPUREG40XX
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;
using namespace vpux::VPURegMapped;
using namespace vpux::vpuipdpu2npureg40xx;
using namespace npu40xx;

namespace {

//
// ConvertVPUIPDPU2NPUReg40XXPass
//

class ConvertVPUIPDPU2NPUReg40XXPass final :
        public impl::ConvertVPUIPDPU2NPUReg40XXBase<ConvertVPUIPDPU2NPUReg40XXPass> {
public:
    explicit ConvertVPUIPDPU2NPUReg40XXPass(Logger log, VPU::DPUDryRunMode dpuDryRunMode)
            : _dpuDryRunMode(dpuDryRunMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

    explicit ConvertVPUIPDPU2NPUReg40XXPass(Logger log): _dpuDryRunMode(VPU::DPUDryRunMode::NONE) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;
    VPU::DPUDryRunMode _dpuDryRunMode;
};

mlir::LogicalResult ConvertVPUIPDPU2NPUReg40XXPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (dpuDryRun.hasValue()) {
        _dpuDryRunMode = VPU::getDPUDryRunMode(dpuDryRun.getValue());
    }

    return mlir::success();
}

void ConvertVPUIPDPU2NPUReg40XXPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    auto arch = VPU::getArch(moduleOp);
    if (arch != VPU::ArchKind::NPU40XX) {
        _log.error("Unsupported architecture for ConvertVPUIPDPU2NPUReg40XXPass: {0}! Required architecture: NPU40XX.",
                   arch);
        signalPassFailure();
        return;
    }

    auto& ctx = getContext();
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp cnnOp;

    IE::CNNNetworkOp::getFromModule(moduleOp, cnnOp, netFunc);

    mlir::ConversionTarget target(ctx);

    target.addLegalDialect<ELF::ELFDialect>();
    target.addLegalDialect<NPUReg40XX::NPUReg40XXDialect>();
    target.addLegalDialect<VPUASM::VPUASMDialect>();
    target.addIllegalDialect<VPUIPDPU::VPUIPDPUDialect>();

    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<DPUVariantRewriter>(&ctx, _log, _dpuDryRunMode);
    patterns.add<DPUInvariantRewriter>(&ctx, _log, _dpuDryRunMode);

    if (mlir::failed(mlir::applyPartialConversion(netFunc, target, std::move(patterns)))) {
        signalPassFailure();
    }

    return;
}

}  // namespace

//
// createConvertVPUIPDPU2NPUReg40XXPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUIPDPU2NPUReg40XXPass(Logger log, VPU::DPUDryRunMode dpuDryRunMode) {
    return std::make_unique<ConvertVPUIPDPU2NPUReg40XXPass>(log, dpuDryRunMode);
}
