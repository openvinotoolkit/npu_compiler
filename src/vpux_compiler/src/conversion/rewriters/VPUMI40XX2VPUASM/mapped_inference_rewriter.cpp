//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/mapped_inference_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"
#include "vpux/compiler/utils/analysis.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> MappedInferenceRewriter::symbolize(
        VPUMI40XX::MappedInferenceOp op, SymbolMapper& mapper, mlir::ConversionPatternRewriter& rewriter) const {
    // Original name before mapper update
    auto result = op.getResult();
    mlir::StringAttr symName = findSym(result).getRootReference();

    auto miFormat = config::getNPUConstraints(op->getContext()).mappedInferenceFormat;
    if (miFormat == config::NPUConstraints::MappedInferenceFormat::MappedInference) {
        auto newOp = symbolizeMappedInference(op, symName, rewriter);
        mapper[result] = moveOpToSection(newOp.getOperation(), *_sectionMap, rewriter);
        assert(mapper[result] != nullptr);

        if (newOp.getManagedMappedInference().has_value()) {
            newOp.setManagedMappedInferenceAttr(
                    ELF::cloneSectionSymbol(mapper[result], newOp.getManagedMappedInference().value()));
        }
        if (op.getWorkItemCount()) {
            symbolizeManagedMappedInference(op, symName, rewriter);
        }
    } else {
        symbolizeManagedMappedInference(op, symName, rewriter);
    }

    rewriter.eraseOp(op);

    return SymbolizationResult();
}

mlir::StringAttr MappedInferenceRewriter::getManagedMappedInferenceSymbolName(mlir::MLIRContext* ctx,
                                                                              mlir::StringAttr symName) {
    return mlir::StringAttr::get(ctx, (llvm::Twine(symName.getValue()) + "_managed").str());
}

VPUASM::MappedInferenceOp MappedInferenceRewriter::symbolizeMappedInference(
        VPUMI40XX::MappedInferenceOp op, mlir::StringAttr symName, mlir::ConversionPatternRewriter& rewriter) const {
    mlir::MLIRContext* ctx = rewriter.getContext();

    SmallVector<mlir::Attribute> dmasAttrVec;
    dmasAttrVec.reserve(op.getDmaTasks().size());
    for (auto tileDmas : op.getDmaTasks()) {
        SmallVector<mlir::Attribute> tileSyms;
        tileSyms.reserve(tileDmas.size());
        for (auto dma : tileDmas) {
            tileSyms.push_back(findSym(dma));
        }
        dmasAttrVec.push_back(mlir::ArrayAttr::get(ctx, tileSyms));
    }

    llvm::SmallVector<mlir::Attribute> invariantTasks;
    invariantTasks.reserve(op.getInvariantTasks().size());
    for (auto invariantTask : op.getInvariantTasks()) {
        invariantTasks.push_back(findSym(invariantTask));
    }

    llvm::SmallVector<mlir::Attribute> variantTasks;
    variantTasks.reserve(op.getVariantTasks().size());
    for (auto variantTask : op.getVariantTasks()) {
        variantTasks.push_back(findSym(variantTask));
    }

    // ActKernelRanges and ActKernelInvocations attributes are 2-D arrays containing respective counts for each tile and
    // each shave in that tile. For any given tile, the new attributes of the rewritten MappedInferenceOp
    // (ActKernelRanges and ActKernelInvocations) will contain only the information on counts stored at index 0,
    // corresponding to SHAVE0 for that tile. This is because for the non-WLM flow, dedicated SW FIFOs are not
    // supported, hence only 0'th index may contain non-zero values, and relevant FW headers (act_kernel_ranges and
    // act_kernel_invocations) are linear structures that only contain counts per tile. Therefore, it is not needed to
    // further propagate the per-shave information. For the WLM flow, the content of these attributes is irrelevant at
    // this point of compilation as WLM flow uses ManagedMappedInference structure.

    mlir::SmallVector<int64_t> rangeCount(op.getActKernelRangesCount().size(), 0);
    for (auto [tileIdx, countPerTile] : llvm::enumerate(op.getActKernelRangesCount())) {
        const auto perTileAttr = mlir::cast<mlir::ArrayAttr>(countPerTile);
        rangeCount[tileIdx] = mlir::cast<mlir::IntegerAttr>(perTileAttr[0]).getInt();
    }

    mlir::SmallVector<int64_t> invoCount(op.getActKernelInvocationsCount().size(), 0);
    for (auto [tileIdx, countPerTile] : llvm::enumerate(op.getActKernelInvocationsCount())) {
        const auto perTileAttr = mlir::cast<mlir::ArrayAttr>(countPerTile);
        invoCount[tileIdx] = mlir::cast<mlir::IntegerAttr>(perTileAttr[0]).getInt();
    }

    llvm::SmallVector<mlir::Attribute> actKernelRanges;
    for (auto actKernelRangesPerTile : llvm::enumerate(op.getActKernelRanges())) {
        if (rangeCount[actKernelRangesPerTile.index()]) {
            auto actKernelRangeName = findSym(actKernelRangesPerTile.value().front());
            actKernelRanges.push_back(actKernelRangeName);
        }
    }

    llvm::SmallVector<mlir::Attribute> actKernelInvocations;
    for (auto actKernelInvoPerTile : llvm::enumerate(op.getActKernelInvocations())) {
        if (invoCount[actKernelInvoPerTile.index()]) {
            auto actKernelInvocationName = findSym(actKernelInvoPerTile.value().front());
            actKernelInvocations.push_back(actKernelInvocationName);
        }
    }

    llvm::SmallVector<mlir::Attribute> actShaveStacks;
    actShaveStacks.reserve(op.getActShaveStacks().size());
    for (auto actShaveStack : op.getActShaveStacks()) {
        actShaveStacks.push_back(findSym(actShaveStack));
    }

    mlir::ArrayAttr dmasAttr = dmasAttrVec.empty() ? nullptr : mlir::ArrayAttr::get(ctx, dmasAttrVec);
    mlir::ArrayAttr invariantTasksAttr = invariantTasks.size() ? mlir::ArrayAttr::get(ctx, invariantTasks) : nullptr;
    mlir::ArrayAttr variantTasksAttr = variantTasks.size() ? mlir::ArrayAttr::get(ctx, variantTasks) : nullptr;
    mlir::ArrayAttr actKernelRangesAttr = actKernelRanges.size() ? mlir::ArrayAttr::get(ctx, actKernelRanges) : nullptr;
    mlir::ArrayAttr actKernelRangesCountAttr = rewriter.getI64ArrayAttr(ArrayRef(rangeCount));
    mlir::ArrayAttr actKernelInvocationsAttr =
            actKernelInvocations.size() ? mlir::ArrayAttr::get(ctx, actKernelInvocations) : nullptr;
    mlir::ArrayAttr actKernelInvoCountAttr = rewriter.getI64ArrayAttr(ArrayRef(invoCount));
    mlir::SymbolRefAttr mediaTasksAttr = op.getMediaTasks() ? findSym(op.getMediaTasks()) : nullptr;
    mlir::SymbolRefAttr barrierTasksAttr =
            (!op.getBarrierTasks() || _skipBarrierLowering) ? nullptr : findSym(op.getBarrierTasks());
    mlir::SymbolRefAttr actShaveRtAttr = op.getActShaveRt() ? findSym(op.getActShaveRt()) : nullptr;
    mlir::ArrayAttr actShaveStacksAttr = actShaveStacks.size() ? mlir::ArrayAttr::get(ctx, actShaveStacks) : nullptr;
    mlir::SymbolRefAttr dmaHwpBase = op.getDmaHwpBase() ? findSym(op.getDmaHwpBase()) : nullptr;
    mlir::SymbolRefAttr workpointCfg = op.getHwpWorkpointCfg() ? findSym(op.getHwpWorkpointCfg()) : nullptr;
    mlir::SymbolRefAttr mappedInferenceVersion =
            op.getMappedInferenceVersion() ? findSym(op.getMappedInferenceVersion()) : nullptr;

    mlir::FlatSymbolRefAttr managedMPISymRef;
    if (op.getWorkItemCount()) {
        auto managedMPISymName = getManagedMappedInferenceSymbolName(ctx, symName);
        managedMPISymRef = mlir::FlatSymbolRefAttr::get(managedMPISymName);
    }

    return rewriter.create<VPUASM::MappedInferenceOp>(
            op.getLoc(), symName, dmasAttr, invariantTasksAttr, variantTasksAttr, actKernelRangesAttr,
            actKernelInvocationsAttr, mediaTasksAttr, barrierTasksAttr, actShaveRtAttr, actShaveStacksAttr,
            managedMPISymRef, op.getDmaCountAttr(), op.getInvariantCountAttr(), op.getVariantCountAttr(),
            actKernelRangesCountAttr, actKernelInvoCountAttr, op.getMediaCountAttr(), op.getBarrierCountAttr(),
            dmaHwpBase, workpointCfg, mappedInferenceVersion);
}

void MappedInferenceRewriter::symbolizeManagedMappedInference(VPUMI40XX::MappedInferenceOp op, mlir::StringAttr symName,
                                                              mlir::ConversionPatternRewriter& rewriter) const {
    mlir::MLIRContext* ctx = rewriter.getContext();
    auto moduleOp = getModuleOp(op);
    auto tileOp = config::getTileExecutor(moduleOp);
    auto tileCount = static_cast<size_t>(tileOp.getCount());

    auto nnrtConfigSymName = mlir::StringAttr::get(ctx, (llvm::Twine(symName.getValue()) + "_nnrtConfigManaged").str());
    auto nnRtConfigSymRef = mlir::FlatSymbolRefAttr::get(nnrtConfigSymName);
    auto isActKernelInvocations = llvm::any_of(op.getActKernelInvocationsCount(), [](mlir::Attribute tileAttr) {
        return llvm::any_of(mlir::cast<mlir::ArrayAttr>(tileAttr), [](mlir::Attribute countAttr) {
            return mlir::cast<mlir::IntegerAttr>(countAttr).getInt() > 0;
        });
    });

    llvm::SmallVector<mlir::Attribute> actShaveStacks;
    actShaveStacks.reserve(op.getActShaveStacks().size());
    for (auto actShaveStack : op.getActShaveStacks()) {
        actShaveStacks.push_back(findSym(actShaveStack));
    }

    mlir::SymbolRefAttr fullStackFramesSectionName = nullptr;
    // If more than two shaves per tile, additional operation needed to handle extra stack frames
    const size_t defaultStacksNum = 2;
    auto shvPerTile = static_cast<size_t>(tileOp.getSubExecutor(config::ExecutorKind::SHAVE_ACT).getCount());
    if (shvPerTile > defaultStacksNum) {
        auto stackFramesSymName = mlir::StringAttr::get(ctx, (llvm::Twine(symName.getValue()) + "_stackFrames").str());
        // If shave stack frames are in CMX, get hardcoded addresses, else if shave stack frames are in DDR
        // allocate empty array of proper size, actual addresses are patched by per-entry relocations.
        auto addresses = actShaveStacks.size() ? SmallVector<uint32_t>(actShaveStacks.size(), 0)
                                               : VPUASM::getCMXStackFrames(moduleOp);
        auto stackFrames =
                rewriter.create<VPUASM::StackFrameAddrsOp>(op.getLoc(), stackFramesSymName, std::move(addresses));
        fullStackFramesSectionName =
                ELF::cloneSectionSymbol(moveOpToSection(stackFrames.getOperation(), *_sectionMap, rewriter),
                                        mlir::FlatSymbolRefAttr::get(stackFramesSymName));
    }

    mlir::SymbolRefAttr barrierTasksAttr =
            (!op.getBarrierTasks() || _skipBarrierLowering) ? nullptr : findSym(op.getBarrierTasks());
    mlir::SymbolRefAttr actShaveRtAttr = op.getActShaveRt() ? findSym(op.getActShaveRt()) : nullptr;
    mlir::ArrayAttr actShaveStacksAttr = actShaveStacks.size() ? mlir::ArrayAttr::get(ctx, actShaveStacks) : nullptr;
    mlir::SymbolRefAttr dmaHwpBase = op.getDmaHwpBase() ? findSym(op.getDmaHwpBase()) : nullptr;
    mlir::SymbolRefAttr workpointCfg = op.getHwpWorkpointCfg() ? findSym(op.getHwpWorkpointCfg()) : nullptr;
    mlir::SymbolRefAttr mappedInferenceVersion =
            op.getMappedInferenceVersion() ? findSym(op.getMappedInferenceVersion()) : nullptr;

    auto nnRtConfig = rewriter.create<VPUASM::NNrtConfigOp>(op.getLoc(), nnrtConfigSymName, isActKernelInvocations,
                                                            actShaveRtAttr, actShaveStacksAttr,
                                                            fullStackFramesSectionName, dmaHwpBase, workpointCfg);
    auto fullNNRtConfigSectionName = moveOpToSection(nnRtConfig.getOperation(), *_sectionMap, rewriter);
    assert(fullNNRtConfigSectionName != nullptr);

    SmallVector<mlir::Attribute> managedDmasAttrVec;
    managedDmasAttrVec.reserve(op.getDmaTasks().size());
    for (auto tileDmas : op.getDmaTasks()) {
        SmallVector<mlir::Attribute> tileSyms;
        for (auto dma : tileDmas) {
            auto dmaTask = mlir::cast<VPUMI40XX::NNDMAOp>(dma.getDefiningOp());
            if (!dmaTask.getTaskLink().has_value()) {
                continue;
            }
            tileSyms.push_back(findSym(dma));
        }
        managedDmasAttrVec.push_back(mlir::ArrayAttr::get(ctx, tileSyms));
    }
    auto managedDmasAttr = mlir::ArrayAttr::get(ctx, managedDmasAttrVec);

    mlir::SymbolRefAttr nullAttr;
    auto workItemCount = op.getWorkItemCount().value_or(0);
    auto workItems = workItemCount ? findSym(op.getWorkItemTasks()) : nullAttr;

    auto bootstrapBarriersCount = checked_cast<int>(op.getBootstrapBarriersCount().value_or(0));
    auto finalBarrierId = checked_cast<int>(op.getFinalBarrierId().value_or(0));

    auto bootstrapWorkItemTasksCount = op.getBootstrapWorkItemsCount().value_or(0);
    mlir::SymbolRefAttr bootstrapBarriers = op.getBootstrapBarriers() ? findSym(op.getBootstrapBarriers()) : nullptr;

    auto fillBits = [](uint8_t numberOfElements) {
        return static_cast<uint8_t>((1 << numberOfElements) - 1);
    };

    uint8_t media_used = 0;
    if (op.getMediaCount()) {
        media_used = fillBits(1);
    }
    uint8_t dpu_used = fillBits(tileCount);
    uint8_t activeDmaDDR = 0;
    uint8_t activeDMACMX = 0;
    for (auto dmaTile : op.getDmaCount()) {
        auto dmaTileArr = mlir::cast<mlir::ArrayAttr>(dmaTile);
        if (mlir::cast<mlir::IntegerAttr>(dmaTileArr[static_cast<size_t>(VPUMI40XX::DmaNnSrcType::DDR)]).getInt() > 0) {
            activeDmaDDR++;
        }
        if (mlir::cast<mlir::IntegerAttr>(dmaTileArr[static_cast<size_t>(VPUMI40XX::DmaNnSrcType::CMX_NN)]).getInt() >
            0) {
            activeDMACMX++;
        }
    }

    uint8_t dma_from_ddr_used = fillBits(activeDmaDDR);
    uint8_t dma_from_cmx_used = fillBits(activeDMACMX);

    uint8_t activeShaves = 0;
    for (auto countPerTile : op.getActKernelRangesCount()) {
        for (auto countPerShave : mlir::cast<mlir::ArrayAttr>(countPerTile)) {
            if (mlir::cast<mlir::IntegerAttr>(countPerShave).getInt() > 0) {
                activeShaves++;
            }
        }
    }

    // All attributes which needed for new WLM execution flow with initial barrier programming
    // If we do not have barrier configuration tasks -> we have old flow
    mlir::SymbolRefAttr barriersReprogrammings =
            op.getNumOfBarrierReprogrammings() ? findSym(op.getNumOfBarrierReprogrammings()) : nullptr;
    mlir::SymbolRefAttr barrierConfigurationDescs =
            op.getBarrierConfigurationTasks() ? findSym(op.getBarrierConfigurationTasks()) : nullptr;
    auto barrierConfigurationCount = op.getBarrierConfigurationTasksCount().value_or(0);
    size_t barrierReprogrammingCount = 0;
    size_t barrierConfigurationStride = 0;
    if (barrierConfigurationDescs != nullptr) {
        auto numberOfAvailablePhysicalBarriers = VPUIP::getNumAvailableBarriers(op);
        barrierConfigurationStride = barrierConfigurationCount / numberOfAvailablePhysicalBarriers;
        barrierReprogrammingCount = numberOfAvailablePhysicalBarriers;
    }

    uint8_t actshv_used = fillBits(activeShaves);

    auto workloadManagementBarrierProgrammingMode = op.getWorkloadManagementBarrierProgrammingMode().value_or(
            VPURegMapped::WorkloadManagementBarrierProgrammingMode::LEGACY);

    mlir::StringAttr managedMPISymName = getManagedMappedInferenceSymbolName(ctx, symName);
    auto managedMPI = rewriter.create<VPUASM::ManagedMappedInferenceOp>(
            op.getLoc(), managedMPISymName, managedDmasAttr, workItems, barrierTasksAttr, bootstrapBarriers,
            nnRtConfigSymRef, barrierConfigurationDescs, barriersReprogrammings, op.getDmaCountAttr(), workItemCount,
            op.getBarrierCount(), finalBarrierId, bootstrapBarriersCount, bootstrapWorkItemTasksCount,
            barrierConfigurationCount, barrierReprogrammingCount, barrierConfigurationStride, actshv_used, dpu_used,
            media_used, dma_from_ddr_used, dma_from_cmx_used, mappedInferenceVersion,
            workloadManagementBarrierProgrammingMode, _disableDmaSwFifo);
    moveOpToSection(managedMPI.getOperation(), *_sectionMap, rewriter);

    managedMPI.setNnrtConfigAttr(ELF::cloneSectionSymbol(fullNNRtConfigSectionName, managedMPI.getNnrtConfigAttr()));
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
