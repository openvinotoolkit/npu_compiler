//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/mapped_inference_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/analysis.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

llvm::SmallVector<mlir::FlatSymbolRefAttr> MappedInferenceRewriter::getSymbolicNames(VPUMI40XX::MappedInferenceOp,
                                                                                     size_t) {
    return {mlir::FlatSymbolRefAttr::get(getContext(), "MappedInference")};
}

mlir::FailureOr<SymbolizationResult> MappedInferenceRewriter::symbolize(
        VPUMI40XX::MappedInferenceOp op, SymbolMapper& mapper, mlir::ConversionPatternRewriter& rewriter) const {
    mlir::MLIRContext* ctx = rewriter.getContext();
    auto result = op.getResult();

    llvm::SmallVector<llvm::SmallVector<mlir::Attribute>> dmas(op.getDmaTasks().size());
    for (auto tileDmas : llvm::enumerate(op.getDmaTasks())) {
        dmas[tileDmas.index()].resize(tileDmas.value().size());
        for (auto dma : llvm::enumerate(tileDmas.value())) {
            auto dmaName = findSym(dma.value());

            dmas[tileDmas.index()][dma.index()] = dmaName;
        }
    }

    llvm::SmallVector<mlir::Attribute> invariantTasks(op.getInvariantTasks().size());
    for (auto invariantTask : llvm::enumerate(op.getInvariantTasks())) {
        auto invariantTaskName = findSym(invariantTask.value());

        invariantTasks[invariantTask.index()] = invariantTaskName;
    }

    llvm::SmallVector<mlir::Attribute> variantTasks(op.getVariantTasks().size());
    for (auto variantTask : llvm::enumerate(op.getVariantTasks())) {
        auto variantTaskName = findSym(variantTask.value());

        variantTasks[variantTask.index()] = variantTaskName;
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
    for (auto actKernRangeCountPerTile : llvm::enumerate(op.getActKernelRangesCount())) {
        auto countsPerTile = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(actKernRangeCountPerTile.value()));
        rangeCount[actKernRangeCountPerTile.index()] = countsPerTile[0];
    }

    mlir::SmallVector<int64_t> invoCount(op.getActKernelInvocationsCount().size(), 0);
    for (auto actKernInvoCountPerTile : llvm::enumerate(op.getActKernelInvocationsCount())) {
        auto countsPerTile = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(actKernInvoCountPerTile.value()));
        invoCount[actKernInvoCountPerTile.index()] = countsPerTile[0];
    }

    llvm::SmallVector<mlir::Attribute> actKernelRanges;
    for (auto actKernelRangesPerTile : llvm::enumerate(op.getActKernelRanges())) {
        if (rangeCount[actKernelRangesPerTile.index()]) {
            auto actKernelRangeName = findSym(actKernelRangesPerTile.value().front());
            actKernelRanges.emplace_back(actKernelRangeName);
        }
    }

    llvm::SmallVector<mlir::Attribute> actKernelInvocations;
    for (auto actKernelInvoPerTile : llvm::enumerate(op.getActKernelInvocations())) {
        if (invoCount[actKernelInvoPerTile.index()]) {
            auto actKernelInvocationName = findSym(actKernelInvoPerTile.value().front());
            actKernelInvocations.emplace_back(actKernelInvocationName);
        }
    }

    llvm::SmallVector<mlir::Attribute> actShaveStacks(op.getActShaveStacks().size());
    for (auto actShaveStack : llvm::enumerate(op.getActShaveStacks())) {
        auto actShaveStacksName = findSym(actShaveStack.value());

        actShaveStacks[actShaveStack.index()] = actShaveStacksName;
    }

    SmallVector<mlir::Attribute> dmasAttrVec;
    for (const auto& dma : dmas) {
        dmasAttrVec.push_back(mlir::ArrayAttr::get(ctx, dma));
    }

    mlir::StringAttr symName = findSym(result).getRootReference();
    mlir::ArrayAttr dmasAttr = dmasAttrVec.size() ? mlir::ArrayAttr::get(ctx, dmasAttrVec) : nullptr;
    mlir::ArrayAttr invariantTasksAttr = invariantTasks.size() ? mlir::ArrayAttr::get(ctx, invariantTasks) : nullptr;
    mlir::ArrayAttr variantTasksAttr = variantTasks.size() ? mlir::ArrayAttr::get(ctx, variantTasks) : nullptr;
    mlir::ArrayAttr actKernelRangesAttr = actKernelRanges.size() ? mlir::ArrayAttr::get(ctx, actKernelRanges) : nullptr;
    mlir::ArrayAttr actKernelRangesCountAttr = rewriter.getI64ArrayAttr(ArrayRef(rangeCount));
    mlir::ArrayAttr actKernelInvocationsAttr =
            actKernelInvocations.size() ? mlir::ArrayAttr::get(ctx, actKernelInvocations) : nullptr;
    mlir::ArrayAttr actKernelInvoCountAttr = rewriter.getI64ArrayAttr(ArrayRef(invoCount));
    mlir::SymbolRefAttr mediaTasksAttr = op.getMediaTasks() ? findSym(op.getMediaTasks()) : nullptr;
    mlir::SymbolRefAttr barrierTasksAttr = op.getBarrierTasks() ? findSym(op.getBarrierTasks()) : nullptr;
    mlir::SymbolRefAttr actShaveRtAttr = op.getActShaveRt() ? findSym(op.getActShaveRt()) : nullptr;
    mlir::ArrayAttr actShaveStacksAttr = actShaveStacks.size() ? mlir::ArrayAttr::get(ctx, actShaveStacks) : nullptr;
    mlir::SymbolRefAttr dmaHwpBase = op.getDmaHwpBase() ? findSym(op.getDmaHwpBase()) : nullptr;
    mlir::SymbolRefAttr workpointCfg = op.getHwpWorkpointCfg() ? findSym(op.getHwpWorkpointCfg()) : nullptr;
    mlir::SymbolRefAttr mappedInferenceVersion =
            op.getMappedInferenceVersion() ? findSym(op.getMappedInferenceVersion()) : nullptr;

    mlir::FlatSymbolRefAttr managedMPISymRef;
    auto managedMPISymName = mlir::StringAttr::get(rewriter.getContext(), symName.str() + std::string("_managed"));
    if (op.getWorkItemCount()) {
        managedMPISymRef = mlir::FlatSymbolRefAttr::get(managedMPISymName);
    }

    auto newOp = rewriter.create<VPUASM::MappedInferenceOp>(
            op.getLoc(), symName, dmasAttr, invariantTasksAttr, variantTasksAttr, actKernelRangesAttr,
            actKernelInvocationsAttr, mediaTasksAttr, barrierTasksAttr, actShaveRtAttr, actShaveStacksAttr,
            managedMPISymRef, op.getDmaCountAttr(), op.getInvariantCountAttr(), op.getVariantCountAttr(),
            actKernelRangesCountAttr, actKernelInvoCountAttr, op.getMediaCountAttr(), op.getBarrierCountAttr(),
            dmaHwpBase, workpointCfg, mappedInferenceVersion);
    mapper[result] = moveOpToSection(newOp.getOperation(), *_sectionMap, rewriter);
    if (newOp.getManagedMappedInference().has_value()) {
        newOp.setManagedMappedInferenceAttr(
                ELF::cloneSectionSymbol(mapper[result], newOp.getManagedMappedInference().value()));
    }

    // TODO: E#100357

    if (op.getWorkItemCount()) {
        auto nnrtConfigSymName =
                mlir::StringAttr::get(rewriter.getContext(), symName.str() + std::string("_nnrtConfigManaged"));
        auto nnRtConfigSymRef = mlir::FlatSymbolRefAttr::get(nnrtConfigSymName);
        bool isActKernelInvocation = op.getActKernelInvocationsCount() ? true : false;
        auto nnRtConfig =
                rewriter.create<VPUASM::NNrtConfigOp>(op.getLoc(), nnrtConfigSymName, isActKernelInvocation,
                                                      actShaveRtAttr, actShaveStacksAttr, dmaHwpBase, workpointCfg);
        auto fullNNRtConfigSectionName = moveOpToSection(nnRtConfig.getOperation(), *_sectionMap, rewriter);

        llvm::SmallVector<llvm::SmallVector<mlir::Attribute>> managedDmas;
        for (auto tileDmas : llvm::enumerate(op.getDmaTasks())) {
            auto& newList = managedDmas.emplace_back();
            for (auto dma : llvm::enumerate(tileDmas.value())) {
                auto dmaTask = mlir::cast<VPUMI40XX::NNDMAOp>(dma.value().getDefiningOp());
                if (!dmaTask.getTaskLink().has_value()) {
                    continue;
                }

                auto dmaName = findSym(dma.value());
                newList.push_back(dmaName);
            }
        }

        SmallVector<mlir::Attribute> managedDmasAttrVec;
        for (const auto& dma : managedDmas) {
            managedDmasAttrVec.push_back(mlir::ArrayAttr::get(ctx, dma));
        }
        mlir::ArrayAttr managedDmasAttr =
                managedDmasAttrVec.size() ? mlir::ArrayAttr::get(ctx, managedDmasAttrVec) : nullptr;

        auto workItems = findSym(op.getWorkItemTasks());

        auto workItemCount = 0;
        if (op.getWorkItemCount().has_value()) {
            workItemCount = op.getWorkItemCount().value();
        }
        auto bootstrapBarriersCount = 0;
        if (op.getBootstrapBarriersCount().has_value()) {
            bootstrapBarriersCount = op.getBootstrapBarriersCount().value();
        }
        auto finalBarrierId = 0;
        if (op.getFinalBarrierId().has_value()) {
            finalBarrierId = op.getFinalBarrierId().value();
        }

        auto bootstrapWorkItemTasksCount = op.getBootsrapWorkItemsCount().value_or(0);
        mlir::SymbolRefAttr bootstrapBarriers =
                op.getBootstrapBarriers() ? findSym(op.getBootstrapBarriers()) : nullptr;

        auto fillBits = [](uint8_t numberOfElements) {
            return static_cast<uint8_t>((1 << numberOfElements) - 1);
        };

        uint8_t media_used = 0;
        if (op.getMediaCount()) {
            media_used = fillBits(1);
        }
        auto module = getModuleOp(op);
        auto tileCount = static_cast<size_t>(config::getTileExecutor(module).getCount());
        uint8_t dpu_used = fillBits(tileCount);
        auto dmaCount = parseIntArrayOfArrayAttr<int64_t>(op.getDmaCount());

        uint8_t activeDmaDDR = 0;
        uint8_t activeDMACMX = 0;
        for (auto dmaTileIndex : irange(dmaCount.size())) {
            if (dmaCount[dmaTileIndex][static_cast<size_t>(VPUMI40XX::DmaNnSrcType::DDR)] > 0) {
                activeDmaDDR++;
            }
            if (dmaCount[dmaTileIndex][static_cast<size_t>(VPUMI40XX::DmaNnSrcType::CMX_NN)] > 0) {
                activeDMACMX++;
            }
        }

        uint8_t dma_from_ddr_used = fillBits(activeDmaDDR);
        uint8_t dma_from_cmx_used = fillBits(activeDMACMX);

        auto actKernelRangesCountVec = parseIntArrayOfArrayAttr<int64_t>(op.getActKernelRangesCount());
        uint8_t activeShaves = 0;
        for (const auto& rangeCountPerTile : actKernelRangesCountVec) {
            for (const auto& rangeCountPerList : rangeCountPerTile) {
                if (rangeCountPerList > 0) {
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

        auto managedMPI = rewriter.create<VPUASM::ManagedMappedInferenceOp>(
                op.getLoc(), managedMPISymName, managedDmasAttr, workItems, barrierTasksAttr, bootstrapBarriers,
                nnRtConfigSymRef, barrierConfigurationDescs, barriersReprogrammings, op.getDmaCountAttr(),
                workItemCount, op.getBarrierCount(), finalBarrierId, bootstrapBarriersCount,
                bootstrapWorkItemTasksCount, barrierConfigurationCount, barrierReprogrammingCount,
                barrierConfigurationStride, actshv_used, dpu_used, media_used, dma_from_ddr_used, dma_from_cmx_used,
                mappedInferenceVersion, workloadManagementBarrierProgrammingMode, _disableDmaSwFifo);
        moveOpToSection(managedMPI.getOperation(), *_sectionMap, rewriter);

        managedMPI.setNnrtConfigAttr(
                ELF::cloneSectionSymbol(fullNNRtConfigSectionName, managedMPI.getNnrtConfigAttr()));
    }

    rewriter.eraseOp(op);

    return SymbolizationResult();
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
