//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/types.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg50XX/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg50XX/dpu_variant_rewriter.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include <npu_40xx_nnrt.hpp>

namespace vpux {
#define GEN_PASS_DECL_CONVERTVPUIPDPU2NPUREG50XX
#define GEN_PASS_DEF_CONVERTVPUIPDPU2NPUREG50XX
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;
using namespace vpux::VPURegMapped;
using namespace vpux::vpuipdpu2npureg50xx;
using namespace npu40xx;

namespace {

//
// ConvertVPUIPDPU2NPUReg50XXPass
//

class ConvertVPUIPDPU2NPUReg50XXPass final :
        public impl::ConvertVPUIPDPU2NPUReg50XXBase<ConvertVPUIPDPU2NPUReg50XXPass> {
public:
    ConvertVPUIPDPU2NPUReg50XXPass(Logger log,
                                   VPURegMapped::NPU5PPEBackwardsCompatibilityMode npu5PPEBackwardsCompatibilityMode)
            : _npu5PPEBackwardsCompatibilityMode(npu5PPEBackwardsCompatibilityMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) override;

private:
    void safeRunOnModule() final;
    VPURegMapped::NPU5PPEBackwardsCompatibilityMode _npu5PPEBackwardsCompatibilityMode =
            VPURegMapped::NPU5PPEBackwardsCompatibilityMode::DISABLED;
};

mlir::LogicalResult ConvertVPUIPDPU2NPUReg50XXPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (npu5PPEBackwardsCompatibilityModeOpt.hasValue()) {
        _npu5PPEBackwardsCompatibilityMode = npu5PPEBackwardsCompatibilityModeOpt.getValue();
    }

    return mlir::success();
}

void ConvertVPUIPDPU2NPUReg50XXPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    auto& ctx = getContext();
    mlir::func::FuncOp netFunc;
    net::NetworkInfoOp netInfo;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, netFunc);

    mlir::ConversionTarget target(ctx);

    target.addLegalDialect<ELF::ELFDialect>();
    target.addLegalDialect<NPUReg50XX::NPUReg50XXDialect>();
    target.addLegalDialect<VPUASM::VPUASMDialect>();
    target.addIllegalDialect<VPUIPDPU::VPUIPDPUDialect>();

    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<DPUVariantRewriter>(&ctx, _log, _npu5PPEBackwardsCompatibilityMode);
    patterns.add<DPUInvariantRewriter>(&ctx, _log, _npu5PPEBackwardsCompatibilityMode);

    if (mlir::failed(mlir::applyPartialConversion(netFunc, target, std::move(patterns)))) {
        signalPassFailure();
    }

    return;
}

}  // namespace

//
// createConvertVPUIPDPU2NPUReg50XXPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUIPDPU2NPUReg50XXPass(
        Logger log, vpux::VPURegMapped::NPU5PPEBackwardsCompatibilityMode npu5PPEBackwardsCompatibilityMode) {
    return std::make_unique<ConvertVPUIPDPU2NPUReg50XXPass>(log, npu5PPEBackwardsCompatibilityMode);
}
