//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/utils.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <npu_40xx_nnrt.hpp>
#include <optional>

using namespace vpux;
using namespace npu40xx;

//
// MappedInferenceOp
//

void vpux::NPUReg50XX::MappedInferenceOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto mappedInference = getProperties().getDescriptor();

    VPUX_THROW_UNLESS(sizeof(nn_public::VpuMappedInference) == mappedInference.size(),
                      "HW VpuMappedInference size {0} != regMapped representation size {1}.",
                      sizeof(nn_public::VpuMappedInference), mappedInference.size());

    auto serializedDescriptor = mappedInference.getStorage();
    binDataSection.appendData(serializedDescriptor.data(), getBinarySize(config::ArchKind::NPU50XX));
}

size_t vpux::NPUReg50XX::MappedInferenceOp::getBinarySize(config::ArchKind) {
    return sizeof(nn_public::VpuMappedInference);
}

size_t vpux::NPUReg50XX::MappedInferenceOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(nn_public::VpuMappedInference);
}

namespace {
size_t getSymRefOffsetForReloc(NPUReg50XX::MappedInferenceOp op, mlir::SymbolRefAttr ref) {
    auto dmaTaskReferenceOffset = offsetof(nn_public::VpuTaskReference<nn_public::VpuDMATask>, address);
    for (size_t dmaEngine = 0; dmaEngine < op.getDmaTasksAttr().size(); dmaEngine++) {
        auto dmaGroup = mlir::cast<mlir::ArrayAttr>(op.getDmaTasksAttr()[dmaEngine]);
        auto dmaCounts = op.getDmaCount();
        auto dmaCountList = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(dmaCounts[dmaEngine]));
        auto dmaTaskIdxOffset = (dmaEngine * sizeof(nn_public::VpuTaskReference<nn_public::VpuDMATask>));

        // by default we expect that getDmaTasksAttr
        // will return a list that looks like
        // index 0 -> DDR dmas
        // index 1 -> CMX dmas
        auto cmxIdx = 1;
        if (dmaCountList[0] != 0) {
            auto dmaDDR = mlir::cast<mlir::SymbolRefAttr>(dmaGroup[0]);
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
            auto dmaCMX = mlir::cast<mlir::SymbolRefAttr>(dmaGroup[cmxIdx]);
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

std::vector<ELF::RelocationInfo> vpux::NPUReg50XX::MappedInferenceOp::getRelocationInfo(
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
                        if (auto symRef = mlir::dyn_cast<mlir::SymbolRefAttr>(attr)) {
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
                    if (auto symRef = mlir::dyn_cast<mlir::SymbolRefAttr>(attr)) {
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
                    if (auto symRef = mlir::dyn_cast<mlir::SymbolRefAttr>(attr)) {
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
                    if (auto symRef = mlir::dyn_cast<mlir::SymbolRefAttr>(attr)) {
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
                    if (auto symRef = mlir::dyn_cast<mlir::SymbolRefAttr>(attr)) {
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
                    if (auto symRef = mlir::dyn_cast<mlir::SymbolRefAttr>(attr)) {
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

void NPUReg50XX::MappedInferenceOp::setVersion(const elf::Version& version) {
    auto descriptor = getProperties().getDescriptor();
    const auto serializedVersion = VPU_CONCAT_NNRT_API_VER(version.getMajor(), version.getMinor());
    descriptor.write<NPUReg50XX::Fields::miVpuNNRTApiVer>(serializedVersion);
    getProperties().setDescriptor(std::move(descriptor));
}

void vpux::NPUReg50XX::MappedInferenceOp::build(
        mlir::OpBuilder&, mlir::OperationState& state, mlir::StringAttr sym_name, mlir::ArrayAttr dmaCount,
        mlir::ArrayAttr dmaDDRCount, mlir::ArrayAttr dmaCMXCount, mlir::ArrayAttr invariantCount,
        mlir::ArrayAttr variantCount, mlir::ArrayAttr actKernelRangesCount, mlir::ArrayAttr actKernelInvocationsCount,
        mlir::IntegerAttr mediaCount, mlir::IntegerAttr barrierCount, mlir::SymbolRefAttr mappedInferenceVersion,
        mlir::SymbolRefAttr actShaveRt, mlir::ArrayAttr actShaveStacks, mlir::SymbolRefAttr dmaHwpBase,
        mlir::SymbolRefAttr hwpWorkpointCfg, mlir::ArrayAttr dmaTasks, mlir::ArrayAttr invariantTasks,
        mlir::ArrayAttr variantTasks, mlir::ArrayAttr actKernelRanges, mlir::ArrayAttr actKernelInvocations,
        mlir::SymbolRefAttr mediaTasks, mlir::SymbolRefAttr barrierTasks, mlir::SymbolRefAttr managedMappedInference,
        vpux::NPUReg50XX::Descriptors::VpuMappedInference&& descriptor) {
    auto& props = state.getOrAddProperties<Properties>();

    props.sym_name = sym_name;
    props.mappedInferenceVersion = mappedInferenceVersion;
    props.dmaCount = dmaCount;
    props.dmaDDRCount = dmaDDRCount;
    props.dmaCMXCount = dmaCMXCount;
    props.invariantCount = invariantCount;
    props.variantCount = variantCount;
    props.actKernelRangesCount = actKernelRangesCount;
    props.actKernelInvocationsCount = actKernelInvocationsCount;
    props.mediaCount = mediaCount;
    props.barrierCount = barrierCount;
    props.actShaveRt = actShaveRt;
    props.actShaveStacks = actShaveStacks;
    props.dmaHwpBase = dmaHwpBase;
    props.hwpWorkpointCfg = hwpWorkpointCfg;
    props.managedMappedInference = managedMappedInference;
    props.dmaTasks = dmaTasks;
    props.invariantTasks = invariantTasks;
    props.variantTasks = variantTasks;
    props.actKernelRanges = actKernelRanges;
    props.actKernelInvocations = actKernelInvocations;
    props.mediaTasks = mediaTasks;
    props.barrierTasks = barrierTasks;
    props.descriptor = std::move(descriptor);
}
