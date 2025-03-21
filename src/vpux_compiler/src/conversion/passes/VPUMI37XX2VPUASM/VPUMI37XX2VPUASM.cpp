//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/VPUMI37XX2VPUASM/symbolization_pattern.hpp"
#include "vpux/compiler/conversion/passes/VPUMI37XX2VPUASM/symbolization_type_converter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/barrier_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/declare_buffer_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/declare_const_buffer_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/declare_task_buffer_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/dma_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/dpu_variant_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/kernel_data_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/kernel_entry_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/kernel_invocation_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/kernel_params_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/kernel_range_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/kernel_text_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/mapped_inference_rewriter.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/kernel_params_utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/symbolization.hpp"

#include <mlir/IR/IRMapping.h>

namespace vpux {
#define GEN_PASS_DECL_CONVERTVPUMI37XX2VPUASM
#define GEN_PASS_DEF_CONVERTVPUMI37XX2VPUASM
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;
using namespace vpumi37xx2vpuasm;

namespace {

//
// ConvertVPUMI37XX2VPUASMPass
//

class ConvertVPUMI37XX2VPUASMPass final : public impl::ConvertVPUMI37XX2VPUASMBase<ConvertVPUMI37XX2VPUASMPass> {
public:
    explicit ConvertVPUMI37XX2VPUASMPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    static bool isDeclareBufferDistributed(VPURT::DeclareBufferOp op);
};

bool ConvertVPUMI37XX2VPUASMPass::isDeclareBufferDistributed(VPURT::DeclareBufferOp op) {
    if (!op.getType().isa<VPUIP::DistributedBufferType>())
        return false;

    auto res = op.getResult();

    auto symbolUser = std::find_if(res.user_begin(), res.user_end(), [](mlir::Operation* use) {
        return mlir::isa<VPUASM::SymbolizeValueOp>(use);
    });

    return symbolUser != res.user_end();
}

void ConvertVPUMI37XX2VPUASMPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto& ctx = getContext();
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp cnnOp;

    IE::CNNNetworkOp::getFromModule(moduleOp, cnnOp, netFunc);

    llvm::DenseMap<mlir::Value, mlir::SymbolRefAttr> symbolNameMappings;
    std::unordered_map<ELF::SectionSignature, ELF::ElfSectionInterface> sectionMap;

    SymbolizationTypeConverter typeConverter;

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<VPUMI37XX::VPUMI37XXDialect>();
    target.addIllegalDialect<Const::ConstDialect>();
    target.addIllegalDialect<VPURT::VPURTDialect>();
    target.addIllegalOp<VPUIP::PPETaskOp>();

    target.addLegalDialect<VPUASM::VPUASMDialect>();
    target.addLegalDialect<ELF::ELFDialect>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();

    // DPU temporaries until we can integrate VPUIPDPU
    target.addDynamicallyLegalOp<VPURT::DeclareBufferOp>(isDeclareBufferDistributed);

    SymbolizationPatternSet patterns(&ctx);

    patterns.add<DeclareBufferRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<DeclareConstBufferRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<DeclareTaskBufferRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<NNDMARewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<KernelTextRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<KernelDataRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<KernelEntryRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<KernelParamsRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<KernelRangeRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<KernelInvocationRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<DPUVariantRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<DPUInvariantRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<BarrierRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<MappedInferenceRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);

    if (mlir::failed(
                mlir::applyFullConversion(netFunc, target, SymbolizationPatternSet::freeze(std::move(patterns))))) {
        signalPassFailure();
    }

    return;
}

}  // namespace

//
// createConvertVPUMI37XX2VPUASMPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUMI37XX2VPUASMPass(Logger log) {
    return std::make_unique<ConvertVPUMI37XX2VPUASMPass>(log);
}
