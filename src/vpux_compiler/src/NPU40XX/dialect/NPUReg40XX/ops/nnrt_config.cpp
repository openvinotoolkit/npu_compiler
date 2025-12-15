//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <npu_40xx_nnrt.hpp>
#include <optional>

using namespace vpux;
using namespace npu40xx;

//
// NNrtConfigOp
//

void vpux::NPUReg40XX::NNrtConfigOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto nnrtCfg = getProperties().getDescriptor();

    VPUX_THROW_UNLESS(sizeof(nn_public::VpuNNRTConfig) == nnrtCfg.size(),
                      "HW VpuNNRTConfig size {0} != regMapped representation size {1}.",
                      sizeof(nn_public::VpuNNRTConfig), nnrtCfg.size());

    auto serializedDescriptor = nnrtCfg.getStorage();
    binDataSection.appendData(serializedDescriptor.data(), getBinarySize(config::ArchKind::NPU40XX));
}

size_t vpux::NPUReg40XX::NNrtConfigOp::getBinarySize(config::ArchKind) {
    return sizeof(npu40xx::nn_public::VpuNNRTConfig);
}

size_t vpux::NPUReg40XX::NNrtConfigOp::getAlignmentRequirements(config::ArchKind) {
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
                    if (auto symRef = mlir::dyn_cast<mlir::SymbolRefAttr>(attr)) {
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

void vpux::NPUReg40XX::NNrtConfigOp::build(mlir::OpBuilder&, mlir::OperationState& state, mlir::StringAttr sym_name,
                                           mlir::UnitAttr isActKernelInvocations, mlir::SymbolRefAttr actShaveRt,
                                           mlir::ArrayAttr actShaveStacks, mlir::SymbolRefAttr dmaHwpBase,
                                           mlir::SymbolRefAttr hwpWorkpointCfg,
                                           vpux::NPUReg40XX::Descriptors::VpuNNRTConfig&& descriptor) {
    auto& props = state.getOrAddProperties<Properties>();

    props.sym_name = sym_name;
    props.isActKernelInvocations = isActKernelInvocations;
    props.actShaveRt = actShaveRt;
    props.actShaveStacks = actShaveStacks;
    props.dmaHwpBase = dmaHwpBase;
    props.hwpWorkpointCfg = hwpWorkpointCfg;
    props.descriptor = std::move(descriptor);
}
