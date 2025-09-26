//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_type_converter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/act_shave_runtime_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/barrier_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/bootstrap_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/declare_buffer_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/declare_const_buffer_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/declare_task_buffer_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/dma_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/dpu_variant_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/enqueue_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_data_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_entry_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_invocation_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_params_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_range_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_text_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/m2i_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/mapped_inference_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/mapped_inference_version_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/platform_info_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/profiling_metadata_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/shave_stack_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/view_task_range_rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/symbolization.hpp"

namespace vpux {
#define GEN_PASS_DECL_CONVERTVPUMI40XX2VPUASM
#define GEN_PASS_DEF_CONVERTVPUMI40XX2VPUASM
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;
using namespace vpumi40xx2vpuasm;

namespace {

//
// ConvertVPUMI40XX2VPUASMPass
//

class ConvertVPUMI40XX2VPUASMPass final : public impl::ConvertVPUMI40XX2VPUASMBase<ConvertVPUMI40XX2VPUASMPass> {
public:
    ConvertVPUMI40XX2VPUASMPass(Logger log, bool disableDmaSwFifo): _disableDmaSwFifo(disableDmaSwFifo) {
        Base::initLogger(log, Base::getArgumentName());
    }

    ConvertVPUMI40XX2VPUASMPass(Logger log, bool enablePWLM, bool disableDmaSwFifo)
            : _disableDmaSwFifo(disableDmaSwFifo) {
        Base::initLogger(log, Base::getArgumentName());
        enablePWLMOpt = enablePWLM;
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) override;

private:
    void safeRunOnModule() final;
    bool _disableDmaSwFifo;
};

mlir::LogicalResult ConvertVPUMI40XX2VPUASMPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    return mlir::success();
}

void ConvertVPUMI40XX2VPUASMPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto& ctx = getContext();
    mlir::func::FuncOp netFunc;
    net::NetworkInfoOp netInfo;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, netFunc);

    llvm::DenseMap<mlir::Value, mlir::SymbolRefAttr> symbolNameMappings;
    std::unordered_map<ELF::SectionSignature, ELF::ElfSectionInterface> sectionMap;

    SymbolizationTypeConverter typeConverter;

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<VPUMI40XX::VPUMI40XXDialect>();
    target.addIllegalDialect<Const::ConstDialect>();
    target.addIllegalDialect<VPURT::VPURTDialect>();

    target.addLegalDialect<VPUASM::VPUASMDialect>();
    target.addLegalDialect<ELF::ELFDialect>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();
    target.addLegalOp<VPURegMapped::TaskBufferLayoutOp>();

    {
        // don't use rewriter infrastructure here as it's not "regular" lowering
        // instead of lowering some operation VPUMI40XX -> VPUASM with symbol name
        // lower VPUMI40XX::OpRanges -> mlir::func::ReturnOp with no arguments
        auto opRanges = mlir::cast<VPUMI40XX::OpRanges>(netFunc.getBlocks().front().getTerminator());
        mlir::OpBuilder(opRanges).create<mlir::func::ReturnOp>(opRanges.getLoc());
        opRanges.erase();
    }

    mlir::OpBuilder builderFunc(&(netFunc.getBody().front().back()));

    // don't use rewriter infrastructure here as it's not "regular" lowering
    // ELF::ABIVersionOp does not need op symbolization
    // ELF::ABIVersionOp is moved into proper ELF section at this point
    auto abiVersionOps = to_small_vector(netFunc.getOps<ELF::ABIVersionOp>());
    VPUX_THROW_UNLESS(abiVersionOps.size() == 1, "Expected exactly one ELF ABIVersionOp. Got {0}",
                      abiVersionOps.size());
    auto abiVersionOp = abiVersionOps[0];
    ELF::moveOpToSection(abiVersionOp.getOperation(), sectionMap, builderFunc);

    auto compilerHashOps = to_small_vector(netFunc.getOps<ELF::CompilerHashOp>());
    VPUX_THROW_UNLESS(compilerHashOps.size() == 1, "Expected exactly one ELF CompilerHashOp. Got {0}",
                      compilerHashOps.size());
    auto compilerHashOp = compilerHashOps[0];
    ELF::moveOpToSection(compilerHashOp.getOperation(), sectionMap, builderFunc);

    // E-141668:
    // insertion of PerformanceMetricsOp in IR should be done in a dedicated pass
    auto perfMetricsOp = builderFunc.create<ELF::PerformanceMetricsOp>(builderFunc.getUnknownLoc());
    ELF::moveOpToSection(perfMetricsOp.getOperation(), sectionMap, builderFunc);

    ELF::insertELFMain(netFunc);

    SymbolizationPatternSet patterns(&ctx);
    patterns.add<DeclareBufferRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<DeclareConstBufferRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<DeclareTaskBufferRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<NNDMARewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<M2IRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<KernelTextRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<KernelDataRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<KernelEntryRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<KernelParamsRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<ActShaveRtRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<KernelRangeRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<KernelInvocationRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<DPUVariantRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<DPUInvariantRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<ViewTaskRangeRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<EnqueueRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<BarrierRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log, enablePWLMOpt);
    patterns.add<MappedInferenceRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log,
                                          _disableDmaSwFifo);
    patterns.add<ProfilingMetadataRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<BootstrapRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<PlatformInfoRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<MappedInferenceVersionRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<ShaveStackRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);

    if (mlir::failed(
                mlir::applyFullConversion(netFunc, target, SymbolizationPatternSet::freeze(std::move(patterns))))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertVPUMI40XX2VPUASMPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUMI40XX2VPUASMPass(Logger log, bool disableDmaSwFifo) {
    return std::make_unique<ConvertVPUMI40XX2VPUASMPass>(log, disableDmaSwFifo);
}

std::unique_ptr<mlir::Pass> vpux::createConvertVPUMI40XX2VPUASMPass(bool enablePWLM, Logger log,
                                                                    bool disableDmaSwFifo) {
    return std::make_unique<ConvertVPUMI40XX2VPUASMPass>(log, enablePWLM, disableDmaSwFifo);
}
