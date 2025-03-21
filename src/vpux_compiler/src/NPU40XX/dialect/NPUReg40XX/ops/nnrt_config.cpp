//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"

#include <npu_40xx_nnrt.hpp>
#include <optional>

using namespace vpux;
using namespace npu40xx;

//
// NNrtConfigOp
//

void vpux::NPUReg40XX::NNrtConfigOp::serializeCached(elf::writer::BinaryDataSection<uint8_t>& binDataSection,
                                                     ELF::SymbolReferenceMap& symRefMap) {
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    bool isActShaveProfilingEnabled =
            vpux::getProfilingSection(moduleOp, profiling::ExecutorType::ACTSHAVE).has_value();

    std::optional<uint64_t> stackSize;
    if (getActShaveStacks().has_value()) {
        auto stackRef = symRefMap.lookupSymbol(getActShaveStacks()->begin()->dyn_cast<mlir::SymbolRefAttr>());
        auto stackOp = mlir::cast<VPUASM::ShaveStackFrameOp>(stackRef);
        stackSize = stackOp.getStackSize();
    }
    // NPU40XX does not have stack frames provided by compiler
    // they are resolved by shave driver when initialized.

    npu40xx::nn_public::VpuNNRTConfig nnrtConfig = {};
    NPUReg40XX::fillNNrtConfig<NPUReg40XX::ActShaveRtOp>(nnrtConfig.shv_rt_configs, getOperation(), getActShaveRt(),
                                                         stackSize, isActShaveProfilingEnabled,
                                                         getIsActKernelInvocations(), std::nullopt);

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&nnrtConfig);
    binDataSection.appendData(ptrCharTmp, getBinarySize(VPU::ArchKind::NPU40XX));
}

size_t vpux::NPUReg40XX::NNrtConfigOp::getBinarySize(VPU::ArchKind) {
    return sizeof(npu40xx::nn_public::VpuNNRTConfig);
}

size_t vpux::NPUReg40XX::NNrtConfigOp::getAlignmentRequirements(VPU::ArchKind) {
    return alignof(nn_public::VpuNNRTConfig);
}

namespace {
size_t getSymRefOffsetForReloc(NPUReg40XX::NNrtConfigOp op, mlir::SymbolRefAttr ref) {
    if (ref == op.getActShaveRt()) {
        return offsetof(nn_public::VpuNNRTConfig, shv_rt_configs) +
               offsetof(nn_public::VpuNNShaveRuntimeConfigs, act_rt_window_base);
    }

    if (op.getActShaveStacksAttr()) {
        auto shaveStacksRef = llvm::find(op.getActShaveStacksAttr(), ref);
        if (shaveStacksRef != op.getActShaveStacksAttr().end()) {
            auto index = shaveStacksRef - op.getActShaveStacksAttr().begin();
            auto shaveRtConfigOffset = offsetof(nn_public::VpuNNRTConfig, shv_rt_configs);
            auto stackFramesOffset = offsetof(nn_public::VpuNNShaveRuntimeConfigs, stack_frames);
            auto arrayIdxOffset = index * sizeof(nn_public::VpuNNShaveRuntimeConfigs::stack_frames[0]);
            return shaveRtConfigOffset + stackFramesOffset + arrayIdxOffset;
        }
    }

    if (ref == op.getDmaHwpBase()) {
        return offsetof(nn_public::VpuNNRTConfig, logaddr_dma_hwp);
    }

    if (ref == op.getHwpWorkpointCfg()) {
        return offsetof(nn_public::VpuNNRTConfig, hwp_workpoint_cfg_addr);
    }

    VPUX_THROW("Provided SymbolRefAttr is not linked to the NNRTConfig Op or getSymRefOffsetForReloc does not "
               "support it");
}
}  // namespace

std::vector<ELF::RelocationInfo> vpux::NPUReg40XX::NNrtConfigOp::getRelocationInfo(ELF::SymbolReferenceMap& symRefMap) {
    auto thisNNRTConfig = *(this);
    std::vector<ELF::RelocationInfo> relocs;
    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    if (auto actShaveRt = getActShaveRt().value_or(nullptr)) {
        relocs.emplace_back(actShaveRt, targetSection, getSymRefOffsetForReloc(thisNNRTConfig, actShaveRt),
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

                        relocs.emplace_back(symRef, targetSection, getSymRefOffsetForReloc(thisNNRTConfig, symRef),
                                            ELF::RelocationType::R_VPU_32, addend,
                                            "Act shave stack in mapped inference reloc");
                    }
                },
                [](mlir::Type) {});
    }

    if (auto dmaHwpBase = getDmaHwpBase().value_or(nullptr)) {
        relocs.emplace_back(dmaHwpBase, targetSection, getSymRefOffsetForReloc(thisNNRTConfig, dmaHwpBase),
                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, dmaHwpBase));
    }

    if (auto hwpWorkpointCfg = getHwpWorkpointCfg().value_or(nullptr)) {
        relocs.emplace_back(hwpWorkpointCfg, targetSection, getSymRefOffsetForReloc(thisNNRTConfig, hwpWorkpointCfg),
                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, hwpWorkpointCfg));
    }

    return relocs;
}

elf::Version NPUReg40XX::NNrtConfigOp::getVersion() {
    // PV version of VpuNNShaveRuntimeConfigs.
    // Don't expect this to be changed as it is fixed
    auto returnVersion = elf::Version(NNRT_API_UD2024_44_MAJOR_VERSION, NNRT_API_UD2024_44_MINOR_VERSION,
                                      NNRT_API_UD2024_44_PATCH_VERSION);
    return returnVersion;
}
