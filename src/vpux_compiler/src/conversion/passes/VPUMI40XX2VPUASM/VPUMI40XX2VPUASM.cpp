//
// Copyright (C) 2023-2026 Intel Corporation
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
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/mapped_inference_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/mapped_inference_version_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/platform_info_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/profiling_metadata_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/shave_stack_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/view_task_range_rewriter.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/symbolization.hpp"

namespace vpux {
#define GEN_PASS_DECL_CONVERTVPUMI40XX2VPUASM
#define GEN_PASS_DEF_CONVERTVPUMI40XX2VPUASM
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;
using namespace vpumi40xx2vpuasm;

namespace {

// Single-pass symbol name pre-population.
// Walks all operations once and populates the symbol mapper for all op types
void populateSymbolMappings(mlir::func::FuncOp netFunc, mlir::MLIRContext* ctx,
                            llvm::DenseMap<mlir::Value, mlir::SymbolRefAttr>& mapper) {
    // Resolve ELF::MainOp once
    auto elfMains = to_small_vector(netFunc.getOps<ELF::MainOp>());
    mlir::Block& opsBlock = elfMains.empty() ? netFunc.getBody().front() : *elfMains[0].getBody();

    // Per-type counters for ops that use counter-based naming
    size_t declareBufferCounter = 0;
    size_t declareConstBufferCounter = 0;

    for (auto& op : opsBlock) {
        // Ops implementing SymbolicNameOpInterface build their own name (includes
        // index-based generic ops and singleton overrides such as MappedInferenceOp).
        if (auto symNameOp = mlir::dyn_cast<VPUMI40XX::SymbolicNameOpInterface>(&op)) {
            auto sym = symNameOp.buildSymbolicName(ctx);
            if (sym) {
                mapper.try_emplace(op.getResult(0), sym);
            }
        } else {
            llvm::TypeSwitch<mlir::Operation*>(&op)
                    // DeclareTaskBuffer uses taskType string as suffix
                    .Case<VPUMI40XX::DeclareTaskBufferOp>([&](VPUMI40XX::DeclareTaskBufferOp taskBufOp) {
                        auto sym = buildSymbolicName(taskBufOp, ctx,
                                                     VPURegMapped::stringifyTaskType(taskBufOp.getTaskType()).str());
                        if (sym) {
                            mapper.try_emplace(taskBufOp->getResult(0), sym);
                        }
                    })
                    // DeclareBuffer uses counter, skips MAC_Accumulators
                    .Case<VPURT::DeclareBufferOp>([&](VPURT::DeclareBufferOp bufOp) {
                        size_t counter = declareBufferCounter++;
                        if (bufOp.getSection() == VPURT::BufferSection::MAC_Accumulators) {
                            // Null symbol — matches original DeclareBufferRewriter behavior
                            return;
                        }
                        auto sym = buildSymbolicName(bufOp, ctx, std::nullopt, counter);
                        if (sym) {
                            mapper.try_emplace(bufOp->getResult(0), sym);
                        }
                    })
                    // Const::DeclareOp uses counter
                    .Case<Const::DeclareOp>([&](Const::DeclareOp constOp) {
                        size_t counter = declareConstBufferCounter++;
                        auto sym = buildSymbolicName(constOp, ctx, std::nullopt, counter);
                        if (sym) {
                            mapper.try_emplace(constOp->getResult(0), sym);
                        }
                    })
                    // EnqueueOp uses the default index-based name (foreign dialect, no interface)
                    .Case<VPURegMapped::EnqueueOp>([&](mlir::Operation* matched) {
                        auto sym = buildSymbolicName(matched, ctx);
                        if (sym) {
                            mapper.try_emplace(matched->getResult(0), sym);
                        }
                    })
                    // ViewTaskRange depends on mapper being already populated for referenced ops
                    .Case<VPURegMapped::ViewTaskRangeOp>([&](VPURegMapped::ViewTaskRangeOp viewOp) {
                        auto it = mapper.find(viewOp.getFirst());
                        if (it != mapper.end()) {
                            auto sym = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(it->getSecond());
                            if (sym) {
                                mapper.try_emplace(viewOp->getResult(0), sym);
                            }
                        }
                    });
        }
    }
}

//
// ConvertVPUMI40XX2VPUASMPass
//

class ConvertVPUMI40XX2VPUASMPass final : public impl::ConvertVPUMI40XX2VPUASMBase<ConvertVPUMI40XX2VPUASMPass> {
public:
    ConvertVPUMI40XX2VPUASMPass(Logger log, bool disableDmaSwFifo): _disableDmaSwFifo(disableDmaSwFifo) {
        Base::initLogger(log, Base::getArgumentName());
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
    auto netFunc = net::getMainFunc(moduleOp);
    bool skipBarrierLowering = false;

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

    // E-141668:
    // insertion of PerformanceMetricsOp in IR should be done in a dedicated pass
    auto perfMetricsOp = builderFunc.create<ELF::PerformanceMetricsOp>(builderFunc.getUnknownLoc());
    ELF::moveOpToSection(perfMetricsOp.getOperation(), sectionMap, builderFunc);

    ELF::insertELFMain(netFunc);

    // Populate all symbol name mappings in a single pass over the IR
    populateSymbolMappings(netFunc, &ctx, symbolNameMappings);

    SymbolizationPatternSet patterns(&ctx);
    patterns.add<DeclareBufferRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<DeclareConstBufferRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<DeclareTaskBufferRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
    patterns.add<NNDMARewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log);
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
    patterns.add<BarrierRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log,
                                  skipBarrierLowering);
    patterns.add<MappedInferenceRewriter>(netFunc, typeConverter, symbolNameMappings, sectionMap, &ctx, _log,
                                          _disableDmaSwFifo, skipBarrierLowering);
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
