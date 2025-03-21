//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

//
// MappedInferenceOp
//

void vpux::NPUReg40XX::MappedInferenceOp::serializeCached(elf::writer::BinaryDataSection<uint8_t>& binDataSection,
                                                          ELF::SymbolReferenceMap& symRefMap) {
    // E#156570
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    bool isActShaveProfilingEnabled =
            vpux::getProfilingSection(moduleOp, profiling::ExecutorType::ACTSHAVE).has_value();

    nn_public::VpuMappedInference mi = {};

    auto mpiVersionRef = symRefMap.lookupSymbol(getMappedInferenceVersion());
    auto mpiVersionOp = mlir::cast<NPUReg40XX::MappedInferenceVersionOp>(mpiVersionRef);

    mi.vpu_nnrt_api_ver = VPU_CONCAT_NNRT_API_VER(mpiVersionOp.getMajor(), mpiVersionOp.getMinor());

    mi.barrier_configs.count = getBarrierCount();
    mi.media_tasks.count = getMediaCount();

    auto dmaDDRCountVec = parseIntArrayAttr<int64_t>(getDmaDDRCount());
    size_t totalDDRDmaCount = 0;
    VPUX_THROW_WHEN(dmaDDRCountVec.size() > nn_public::VPU_MAX_DMA_ENGINES, "Too many DMA DDR lists");
    for (size_t listIdx = 0; listIdx < dmaDDRCountVec.size(); ++listIdx) {
        mi.dma_tasks_ddr_[listIdx].count = dmaDDRCountVec[listIdx];
        totalDDRDmaCount += mi.dma_tasks_ddr_[listIdx].count;
    }

    auto dmaCMXCountVec = parseIntArrayAttr<int64_t>(getDmaCMXCount());
    size_t totalCMXDmaCount = 0;
    VPUX_THROW_WHEN(dmaCMXCountVec.size() > nn_public::VPU_MAX_DMA_ENGINES, "Too many DMA CMX lists");
    for (size_t listIdx = 0; listIdx < dmaCMXCountVec.size(); ++listIdx) {
        mi.dma_tasks_cmx_[listIdx].count = dmaCMXCountVec[listIdx];
        totalCMXDmaCount += mi.dma_tasks_cmx_[listIdx].count;
    }

    auto invariantCountVec = parseIntArrayAttr<int64_t>(getInvariantCount());
    VPUX_THROW_WHEN(invariantCountVec.size() > nn_public::VPU_MAX_TILES, "Too many Invariant lists");
    for (size_t listIdx = 0; listIdx < invariantCountVec.size(); ++listIdx) {
        mi.invariants[listIdx].count = invariantCountVec[listIdx];
    }

    auto variantCountVec = parseIntArrayAttr<int64_t>(getVariantCount());
    VPUX_THROW_WHEN(variantCountVec.size() > nn_public::VPU_MAX_TILES, "Too many Variant lists");
    for (size_t listIdx = 0; listIdx < variantCountVec.size(); ++listIdx) {
        mi.variants[listIdx].count = variantCountVec[listIdx];
    }

    auto actKernelRangesCountVec = parseIntArrayAttr<int64_t>(getActKernelRangesCount());
    VPUX_THROW_WHEN(actKernelRangesCountVec.size() > nn_public::VPU_MAX_TILES, "Too many ActKernelRange lists");
    for (size_t listIdx = 0; listIdx < actKernelRangesCountVec.size(); ++listIdx) {
        mi.act_kernel_ranges[listIdx].count = actKernelRangesCountVec[listIdx];
    }

    auto actKernelInvocationsCountVec = parseIntArrayAttr<int64_t>(getActKernelInvocationsCount());
    VPUX_THROW_WHEN(actKernelInvocationsCountVec.size() > nn_public::VPU_MAX_TILES, "Too many ActKernelInvo lists");
    for (size_t listIdx = 0; listIdx < actKernelInvocationsCountVec.size(); ++listIdx) {
        mi.act_kernel_invocations[listIdx].count = actKernelInvocationsCountVec[listIdx];
    }

    std::optional<uint64_t> stackSize;
    if (getActShaveStacks().has_value()) {
        auto stackRef = symRefMap.lookupSymbol(getActShaveStacks()->begin()->dyn_cast<mlir::SymbolRefAttr>());
        auto stackOp = mlir::cast<VPUASM::ShaveStackFrameOp>(stackRef);
        stackSize = stackOp.getStackSize();
    }
    // NPU40XX does not have stack frames provided by compiler
    // they are resolved by shave driver when initialized.

    auto isActKernelInvocations = getActKernelInvocationsCount().size() > 0;
    NPUReg40XX::fillNNrtConfig<NPUReg40XX::ActShaveRtOp>(mi.shv_rt_configs, getOperation(), getActShaveRt(), stackSize,
                                                         isActShaveProfilingEnabled, isActKernelInvocations,
                                                         std::nullopt);

    if (getManagedMappedInference().has_value()) {
        mi.managed_inference.count = 1;
    }

    // Look only at the DMA tasks belonging to the first (and only) DMA engine
    std::tie(mi.task_storage_counts_.dma_ddr_count, mi.task_storage_counts_.dma_cmx_count) =
            VPUMI40XX::compute_dma_split(totalDDRDmaCount, totalCMXDmaCount);
    auto archKind = VPU::ArchKind::NPU40XX;
    mi.task_storage_counts_.dpu_invariant_count =
            VPURegMapped::getDefaultTaskListCount(VPURegMapped::TaskType::DPUInvariant, archKind);
    mi.task_storage_counts_.dpu_variant_count =
            VPURegMapped::getDefaultTaskListCount(VPURegMapped::TaskType::DPUVariant, archKind);
    mi.task_storage_counts_.act_range_count =
            VPURegMapped::getDefaultTaskListCount(VPURegMapped::TaskType::ActKernelRange, archKind);
    mi.task_storage_counts_.act_invo_count =
            VPURegMapped::getDefaultTaskListCount(VPURegMapped::TaskType::ActKernelInvocation, archKind);
    mi.task_storage_counts_.media_count = VPURegMapped::getDefaultTaskListCount(VPURegMapped::TaskType::M2I, archKind);

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&mi);
    binDataSection.appendData(ptrCharTmp, getBinarySize(archKind));
}

size_t vpux::NPUReg40XX::MappedInferenceOp::getBinarySize(VPU::ArchKind) {
    return sizeof(nn_public::VpuMappedInference);
}

size_t vpux::NPUReg40XX::MappedInferenceOp::getAlignmentRequirements(VPU::ArchKind) {
    return alignof(nn_public::VpuMappedInference);
}

namespace {
size_t getSymRefOffsetForReloc(NPUReg40XX::MappedInferenceOp op, mlir::SymbolRefAttr ref) {
    auto dmaTaskReferenceOffset = offsetof(nn_public::VpuTaskReference<nn_public::VpuDMATask>, address);
    for (size_t dmaEngine = 0; dmaEngine < op.getDmaTasksAttr().size(); dmaEngine++) {
        auto dmaGroup = op.getDmaTasksAttr()[dmaEngine].cast<mlir::ArrayAttr>();
        auto dmaCounts = op.getDmaCount();
        auto dmaCountList = parseIntArrayAttr<int64_t>(dmaCounts[dmaEngine].cast<mlir::ArrayAttr>());
        auto dmaTaskIdxOffset = (dmaEngine * sizeof(nn_public::VpuTaskReference<nn_public::VpuDMATask>));

        // by default we expect that getDmaTasksAttr
        // will return a list that looks like
        // index 0 -> DDR dmas
        // index 1 -> CMX dmas
        auto cmxIdx = 1;
        if (dmaCountList[0] != 0) {
            auto dmaDDR = dmaGroup[0].cast<mlir::SymbolRefAttr>();
            if (ref == dmaDDR) {
                return offsetof(nn_public::VpuMappedInference, dma_tasks_ddr_[0]) + dmaTaskIdxOffset +
                       dmaTaskReferenceOffset;
            }
        } else {
            cmxIdx = 0;
        }

        // could be cmx dma
        // if there are no DDR dmas for the current task it means that cmx index would be 0
        if (dmaCountList[1] != 0) {
            auto dmaCMX = dmaGroup[cmxIdx].cast<mlir::SymbolRefAttr>();
            if (ref == dmaCMX) {
                return offsetof(nn_public::VpuMappedInference, dma_tasks_cmx_[0]) + dmaTaskIdxOffset +
                       dmaTaskReferenceOffset;
            }
        }
    }
    if (ref == op.getBarrierTasks()) {
        return offsetof(nn_public::VpuMappedInference, barrier_configs) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuBarrierCountConfig>, address);
    }

    if (ref == op.getMediaTasks()) {
        return offsetof(nn_public::VpuMappedInference, media_tasks) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuMediaTask>, address);
    }

    if (ref == op.getActShaveRt()) {
        return offsetof(nn_public::VpuMappedInference, shv_rt_configs) +
               offsetof(nn_public::VpuNNShaveRuntimeConfigs, act_rt_window_base);
    }

    if (ref == op.getDmaHwpBase()) {
        return offsetof(nn_public::VpuMappedInference, logaddr_dma_hwp_);
    }

    if (ref == op.getHwpWorkpointCfg()) {
        return offsetof(nn_public::VpuMappedInference, hwp_workpoint_cfg_addr);
    }

    if (op.getActShaveStacksAttr()) {
        auto shaveStacksRef = llvm::find(op.getActShaveStacksAttr(), ref);
        if (shaveStacksRef != op.getActShaveStacksAttr().end()) {
            auto index = shaveStacksRef - op.getActShaveStacksAttr().begin();
            auto shaveRtConfigOffset = offsetof(nn_public::VpuMappedInference, shv_rt_configs);
            auto stackFramesOffset = offsetof(nn_public::VpuNNShaveRuntimeConfigs, stack_frames);
            auto arrayIdxOffset = index * sizeof(nn_public::VpuNNShaveRuntimeConfigs::stack_frames[0]);
            return shaveRtConfigOffset + stackFramesOffset + arrayIdxOffset;
        }
    }

    auto getTileIdx = [](auto arrayIdx, mlir::ArrayAttr countAttr) -> size_t {
        auto count = parseIntArrayAttr<int64_t>(countAttr);

        auto usedTilesNum = arrayIdx + 1;
        for (size_t countIdx = 0; countIdx < count.size() && usedTilesNum; ++countIdx) {
            if (count[countIdx]) {
                --usedTilesNum;
            }
            if (!usedTilesNum) {
                return countIdx;
            }
        }

        VPUX_THROW_WHEN(usedTilesNum, "Cannot identify the tile corresponding for task of index {0}", arrayIdx);

        return 0;
    };

    if (op.getActKernelInvocationsAttr()) {
        auto shaveInvoRef = llvm::find(op.getActKernelInvocationsAttr(), ref);
        if (shaveInvoRef != op.getActKernelInvocationsAttr().end()) {
            auto index = getTileIdx(shaveInvoRef - op.getActKernelInvocationsAttr().begin(),
                                    op.getActKernelInvocationsCount());
            auto invosOffset = offsetof(nn_public::VpuMappedInference, act_kernel_invocations);
            auto arrayIdxOffset = (index * sizeof(nn_public::VpuTaskReference<nn_public::VpuActKernelInvocation>));
            auto vpuPtrOffset = offsetof(nn_public::VpuTaskReference<nn_public::VpuActKernelInvocation>, address);
            return invosOffset + arrayIdxOffset + vpuPtrOffset;
        }
    }

    if (op.getActKernelRangesAttr()) {
        auto shaveRangeRef = llvm::find(op.getActKernelRangesAttr(), ref);
        if (shaveRangeRef != op.getActKernelRangesAttr().end()) {
            auto index = getTileIdx(shaveRangeRef - op.getActKernelRangesAttr().begin(), op.getActKernelRangesCount());
            auto rangesOffset = offsetof(nn_public::VpuMappedInference, act_kernel_ranges);
            auto arrayIdxOffset = (index * sizeof(nn_public::VpuTaskReference<nn_public::VpuActKernelRange>));
            auto vpuPtrOffset = offsetof(nn_public::VpuTaskReference<nn_public::VpuActKernelRange>, address);
            return rangesOffset + arrayIdxOffset + vpuPtrOffset;
        }
    }

    if (op.getVariantTasks()) {
        auto variantsRef = llvm::find(op.getVariantTasksAttr(), ref);
        if (variantsRef != op.getVariantTasksAttr().end()) {
            auto index = getTileIdx(variantsRef - op.getVariantTasksAttr().begin(), op.getVariantCount());
            auto variantsOffset = offsetof(nn_public::VpuMappedInference, variants);
            auto arrayIdxOffset = (index * sizeof(nn_public::VpuTaskReference<nn_public::VpuDPUVariant>));
            auto vpuPtrOffset = offsetof(nn_public::VpuTaskReference<nn_public::VpuDPUVariant>, address);
            return variantsOffset + arrayIdxOffset + vpuPtrOffset;
        }
    }

    if (op.getInvariantTasks()) {
        auto invariantsRef = llvm::find(op.getInvariantTasksAttr(), ref);
        if (invariantsRef != op.getInvariantTasksAttr().end()) {
            auto index = getTileIdx(invariantsRef - op.getInvariantTasksAttr().begin(), op.getInvariantCount());
            auto invariantsOffset = offsetof(nn_public::VpuMappedInference, invariants);
            auto arrayIdxOffset = (index * sizeof(nn_public::VpuTaskReference<nn_public::VpuDPUInvariant>));
            auto vpuPtrOffset = offsetof(nn_public::VpuTaskReference<nn_public::VpuDPUInvariant>, address);
            return invariantsOffset + arrayIdxOffset + vpuPtrOffset;
        }
    }

    if (op.getManagedMappedInferenceAttr()) {
        if (ref == op.getManagedMappedInferenceAttr()) {
            return offsetof(nn_public::VpuMappedInference, managed_inference) +
                   offsetof(nn_public::VpuTaskReference<uint8_t>, address);
        }
    }

    VPUX_THROW("Provided SymbolRefAttr is not linked to the MappedInference Op or getSymRefOffsetForReloc does not "
               "support it");
}
}  // namespace

std::vector<ELF::RelocationInfo> vpux::NPUReg40XX::MappedInferenceOp::getRelocationInfo(
        ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    auto thisMI = *(this);
    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    if (auto dmaTasks = getDmaTasks().value_or(nullptr)) {
        for (auto dmaList : dmaTasks) {
            auto dmaListArrayAttr = dmaList;
            dmaListArrayAttr.walkImmediateSubElements(
                    [&](mlir::Attribute attr) {
                        if (auto symRef = attr.dyn_cast<mlir::SymbolRefAttr>()) {
                            relocs.emplace_back(symRef, targetSection, getSymRefOffsetForReloc(thisMI, symRef),
                                                ELF::RelocationType::R_VPU_64,
                                                ELF::getOffsetOfSymRef(symRefMap, symRef),
                                                "Dma list in mapped inference reloc");
                        }
                    },
                    [](mlir::Type) {});
        }
    }

    if (auto invariantTasks = getInvariantTasks().value_or(nullptr)) {
        auto invTasksSubElemIf = invariantTasks;
        invTasksSubElemIf.walkImmediateSubElements(
                [&](mlir::Attribute attr) {
                    if (auto symRef = attr.dyn_cast<mlir::SymbolRefAttr>()) {
                        relocs.emplace_back(symRef, targetSection, getSymRefOffsetForReloc(thisMI, symRef),
                                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, symRef),
                                            "Invariant task in mapped inference reloc");
                    }
                },
                [](mlir::Type) {});
    }

    if (auto variantTasks = getVariantTasks().value_or(nullptr)) {
        auto varTasksSubElemIf = variantTasks;
        varTasksSubElemIf.walkImmediateSubElements(
                [&](mlir::Attribute attr) {
                    if (auto symRef = attr.dyn_cast<mlir::SymbolRefAttr>()) {
                        relocs.emplace_back(symRef, targetSection, getSymRefOffsetForReloc(thisMI, symRef),
                                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, symRef),
                                            "Variant task in mapped inference reloc");
                    }
                },
                [](mlir::Type) {});
    }

    if (auto actKernelRanges = getActKernelRanges().value_or(nullptr)) {
        auto akrTasksSubElemIf = actKernelRanges;
        akrTasksSubElemIf.walkImmediateSubElements(
                [&](mlir::Attribute attr) {
                    if (auto symRef = attr.dyn_cast<mlir::SymbolRefAttr>()) {
                        relocs.emplace_back(symRef, targetSection, getSymRefOffsetForReloc(thisMI, symRef),
                                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, symRef),
                                            "Act kernel range in mapped inference reloc");
                    }
                },
                [](mlir::Type) {});
    }

    if (auto actKernelInvos = getActKernelInvocations().value_or(nullptr)) {
        auto akiTasksSubElemIf = actKernelInvos;
        akiTasksSubElemIf.walkImmediateSubElements(
                [&](mlir::Attribute attr) {
                    if (auto symRef = attr.dyn_cast<mlir::SymbolRefAttr>()) {
                        relocs.emplace_back(symRef, targetSection, getSymRefOffsetForReloc(thisMI, symRef),
                                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, symRef),
                                            "Act kernel invocation in mapped inference reloc");
                    }
                },
                [](mlir::Type) {});
    }

    if (auto mediaTasks = getMediaTasks().value_or(nullptr)) {
        relocs.emplace_back(mediaTasks, targetSection, getSymRefOffsetForReloc(thisMI, mediaTasks),
                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, mediaTasks),
                            "mediaTasks in mapped inference reloc");
    }

    if (auto barrierTasks = getBarrierTasks().value_or(nullptr)) {
        relocs.emplace_back(barrierTasks, targetSection, getSymRefOffsetForReloc(thisMI, barrierTasks),
                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, barrierTasks),
                            "barrierTasks in mapped inference reloc");
    }

    if (auto actShaveRt = getActShaveRt().value_or(nullptr)) {
        relocs.emplace_back(actShaveRt, targetSection, getSymRefOffsetForReloc(thisMI, actShaveRt),
                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, actShaveRt),
                            "actShaveRt in mapped inference reloc");
    }

    if (auto actShaveStacks = getActShaveStacks().value_or(nullptr)) {
        auto shvStacksTasksSubElemIf = actShaveStacks;
        shvStacksTasksSubElemIf.walkImmediateSubElements(
                [&](mlir::Attribute attr) {
                    if (auto symRef = attr.dyn_cast<mlir::SymbolRefAttr>()) {
                        auto stacks = symRefMap.lookupSymbol(symRef);
                        auto stackOp = mlir::cast<VPUASM::ShaveStackFrameOp>(stacks);
                        auto stackSize = stackOp.getStackSize();
                        // SHAVE stack grows backwards!
                        // set the addend to the top of the allocated section so it does not override
                        // outside of its buffer
                        auto addend = ELF::getOffsetOfSymRef(symRefMap, symRef) + stackSize;

                        relocs.emplace_back(symRef, targetSection, getSymRefOffsetForReloc(thisMI, symRef),
                                            ELF::RelocationType::R_VPU_32, addend,
                                            "Act shave stack in mapped inference reloc");
                    }
                },
                [](mlir::Type) {});
    }

    if (auto dmaHwpBase = getDmaHwpBase().value_or(nullptr)) {
        relocs.emplace_back(dmaHwpBase, targetSection, getSymRefOffsetForReloc(thisMI, dmaHwpBase),
                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, dmaHwpBase),
                            "dmaHwpBase in mapped inference reloc");
    }

    if (auto hwpWorkpointCfg = getHwpWorkpointCfg().value_or(nullptr)) {
        relocs.emplace_back(hwpWorkpointCfg, targetSection, getSymRefOffsetForReloc(thisMI, hwpWorkpointCfg),
                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, hwpWorkpointCfg),
                            "hwpWorkpointCfg in mapped inference reloc");
    }

    if (auto managedMPI = getManagedMappedInference().value_or(nullptr)) {
        relocs.emplace_back(managedMPI, targetSection, getSymRefOffsetForReloc(thisMI, managedMPI),
                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, managedMPI),
                            "managedMPI in mapped inference reloc");
    }

    return relocs;
}

elf::Version NPUReg40XX::MappedInferenceOp::getVersion() {
    // PV version of Mapped inference.
    // Don't expect this to be changed as it is fixed
    auto returnVersion = elf::Version(NNRT_API_UD2024_44_MAJOR_VERSION, NNRT_API_UD2024_44_MINOR_VERSION,
                                      NNRT_API_UD2024_44_PATCH_VERSION);
    return returnVersion;
}
