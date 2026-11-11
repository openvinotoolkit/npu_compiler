//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/types.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg60XX/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg60XX/dpu_variant_rewriter.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
#define GEN_PASS_DECL_CONVERTVPUIPDPU2NPUREG60XX
#define GEN_PASS_DEF_CONVERTVPUIPDPU2NPUREG60XX
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;
using namespace vpux::VPURegMapped;
using namespace vpux::vpuipdpu2npureg60xx;

namespace {

//
// ConvertVPUIPDPU2NPUReg60XXPass
//

class ConvertVPUIPDPU2NPUReg60XXPass final :
        public impl::ConvertVPUIPDPU2NPUReg60XXBase<ConvertVPUIPDPU2NPUReg60XXPass> {
public:
    explicit ConvertVPUIPDPU2NPUReg60XXPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ConvertVPUIPDPU2NPUReg60XXPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    auto arch = config::getArch(moduleOp);
    if (arch != config::ArchKind::NPU60XX) {
        _log.error("Unsupported architecture for ConvertVPUIPDPU2NPUReg60XXPass: {0}! Required architecture: NPU60XX",
                   arch);
        signalPassFailure();
        return;
    }

    auto& ctx = getContext();
    auto netFunc = net::getMainFunc(moduleOp);

    mlir::ConversionTarget target(ctx);

    target.addLegalDialect<ELF::ELFDialect>();
    target.addLegalDialect<NPUReg60XX::NPUReg60XXDialect>();
    target.addLegalDialect<VPUASM::VPUASMDialect>();
    target.addIllegalDialect<VPUIPDPU::VPUIPDPUDialect>();

    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<DPUVariantRewriter>(&ctx, _log);
    patterns.add<DPUInvariantRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(netFunc, target, std::move(patterns)))) {
        signalPassFailure();
    }

    return;
}

}  // namespace

//
// createConvertVPUIPDPU2NPUReg60XXPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUIPDPU2NPUReg60XXPass(Logger log) {
    return std::make_unique<ConvertVPUIPDPU2NPUReg60XXPass>(log);
}
