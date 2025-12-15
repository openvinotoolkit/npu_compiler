//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/blob_writer.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/BuiltinTypes.h>

#include <npu_37xx_nnrt.hpp>

#include <algorithm>
#include <cstdint>

#include "external/runtime_dpu_parser_imports.cpp.inc"

using namespace vpux;
using namespace npu37xx;

#define SLICE_LENGTH \
    (2 * 1024 * 1024)  // TODO: E#54008	Don't hardcode this . Check if we can do it via relocation from CMX symbol size?

void vpux::VPUMI37XX::DPUInvariantOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuDPUInvariant taskWrapper{};
    nn_public::VpuDPUInvariantRegisters& registers = taskWrapper.registers_;

    memset(reinterpret_cast<void*>(&registers), 0, sizeof(registers));

    SetupInput(*this, registers);
    SetupWeights(*this, registers);
    SetupKernel(*this, registers);
    SetupOutput(*this, registers);

    auto taskType = getNceTaskType();
    switch (taskType) {
    case VPUIP::NCETaskType::CONV:
        SetupInvariant_Convolution(*this, registers);
        break;
    case VPUIP::NCETaskType::MAXPOOL:
        SetupInvariant_MaxPool(*this, registers);
        break;
    case VPUIP::NCETaskType::AVEPOOL:
        registers.elops_wload.elops_wload_bf.pool_wt_rd_dis = 1;
        SetupInvariant_DwConvolution(*this, registers);
        break;
    case VPUIP::NCETaskType::DWCONV:
        SetupInvariant_DwConvolution(*this, registers);
        break;
    case VPUIP::NCETaskType::ELTWISE:
        SetupInvariant_Eltwise(*this, registers);
        break;
    default:
        VPUX_THROW("Unsupported task type {0}", taskType);
        break;
    }

    Setup_PPE(*this, registers);

    // imported from update_invariant
    {
        // TODO: E#54009 hardcoded and directly copied from POC runtime...
        registers.base_offset_a = 0x200;
        registers.base_offset_b = 0x602;

        registers.act_offset[2] = 0;
        registers.act_offset[3] = 0;

        registers.se_sp_addr[1].se_addr = ((1 * SLICE_LENGTH) >> 4);
        registers.se_sp_addr[2].se_addr = ((1 * SLICE_LENGTH) >> 4);
        registers.se_sp_addr[3].se_addr = ((1 * SLICE_LENGTH) >> 4);

        if (getNceTaskType() == VPUIP::NCETaskType::ELTWISE) {
            auto weightsOffs = mlir::cast<VPURT::DeclareBufferOp>(getWeights().getDefiningOp()).getByteOffset();
            auto actOffs = mlir::cast<VPURT::DeclareBufferOp>(getInput().getDefiningOp()).getByteOffset();

            registers.tensor_start =
                    checked_cast<uint32_t>(std::max(static_cast<int64_t>(0), (actOffs - weightsOffs) >> 4));
            registers.weight_start =
                    checked_cast<uint32_t>(std::max(static_cast<int64_t>(0), (weightsOffs - actOffs) >> 4));
        }

        llvm::SmallVector<mlir::Value> outputs(getOutputBuffs());
        llvm::sort(outputs.begin(), outputs.end(), [](mlir::Value lhs, mlir::Value rhs) {
            auto lhsIdx = mlir::cast<vpux::NDTypeInterface>(lhs.getType()).getMemSpace().getIndex().value_or(0);
            auto rhsIdx = mlir::cast<vpux::NDTypeInterface>(rhs.getType()).getMemSpace().getIndex().value_or(0);
            return lhsIdx < rhsIdx;
        });

        auto firstIndex = mlir::cast<vpux::NDTypeInterface>(outputs[0].getType()).getMemSpace().getIndex().value_or(0);
        for (size_t idx = 1; idx < outputs.size(); ++idx) {
            auto outIdx =
                    mlir::cast<vpux::NDTypeInterface>(outputs[idx].getType()).getMemSpace().getIndex().value_or(0);
            if (outIdx == firstIndex) {
                continue;
            }

            registers.odu_cast[idx - 1].odu_cast_bf.cast_enable = 1;
            registers.odu_cast[idx - 1].odu_cast_bf.cast_offset = ((outIdx - firstIndex) * SLICE_LENGTH) >> 4;
        }
    }

    auto bufOp = getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(bufOp, "Parent of DPU op is not a declareBufferOp");

    taskWrapper.cluster_ = bufOp.getNonEmptySectionIndex()[0];  // TODO:E#54007 use from memref and don't assume index 0

    taskWrapper.is_cont_conv_ = getIsContinued();

    if (auto profBuffer = getProfilingData()) {
        auto profBufOp = profBuffer.getDefiningOp<VPURT::DeclareBufferOp>();
        taskWrapper.hwp_cmx_base_offset_ = profBufOp.getByteOffset();
    } else {
        taskWrapper.hwp_cmx_base_offset_ = -1;
    }

    // Barriers setup
    taskWrapper.barriers_.wait_mask_ = VPUMI37XX::computeMask(getWaitBarriers());
    taskWrapper.barriers_.post_mask_ = VPUMI37XX::computeMask(getUpdateBarriers());

    taskWrapper.barriers_.group_ = 0;
    taskWrapper.barriers_.mask_ = 0;

    std::tie(taskWrapper.barriers_.group_, taskWrapper.barriers_.mask_) =
            ELF::reduceWaitMaskTo8bit(taskWrapper.barriers_.wait_mask_);

    taskWrapper.barriers_sched_.start_after_ = getStartAfter();
    taskWrapper.barriers_sched_.clean_after_ = getCleanAfter();

    uint32_t variant_count = 0;
    for (auto& user : getResult().getUses()) {
        auto owner = user.getOwner();
        if (mlir::dyn_cast_or_null<VPUMI37XX::DPUVariantOp>(owner)) {
            variant_count++;
        }
    }

    taskWrapper.variant_count_ = variant_count;

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&taskWrapper);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUMI37XX::DPUInvariantOp::getBinarySize() {
    return sizeof(nn_public::VpuDPUInvariant);
}

size_t vpux::VPUMI37XX::DPUInvariantOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuDPUInvariant);
}

size_t vpux::VPUMI37XX::DPUInvariantOp::getOffsetOfWithinOperation(mlir::Value /*value*/) {
    VPUX_THROW("getOffset is not supported for DPUInvariant");
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::DPUInvariantOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_EXECINSTR | ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::DPUInvariantOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_DPU);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::DPUInvariantOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

void vpux::VPUMI37XX::DPUVariantOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto invariant = getInvariant().getDefiningOp<vpux::VPUMI37XX::DPUInvariantOp>();
    VPUX_THROW_WHEN(invariant == nullptr, "variant not pointing to an invariant");

    nn_public::VpuDPUVariant taskWrapper{};
    memset(reinterpret_cast<void*>(&taskWrapper), 0, sizeof(taskWrapper));

    parseVariant(*this, invariant, taskWrapper);

    taskWrapper.invariant_index_ = invariant.getType().getValue();
    taskWrapper.invariant_ = invariant.getType().getValue();
    taskWrapper.cluster_ = 0;

    auto opType = invariant.getNceTaskType();
    if (opType == VPUIP::NCETaskType::ELTWISE) {
        auto weightsOffs = mlir::cast<VPURT::DeclareBufferOp>(invariant.getWeights().getDefiningOp()).getByteOffset();
        auto actOffs = mlir::cast<VPURT::DeclareBufferOp>(invariant.getInput().getDefiningOp()).getByteOffset();

        auto weight_start = std::max(static_cast<int64_t>(0), (weightsOffs - actOffs) >> 4);
        taskWrapper.weight_table_offset_ += weight_start;
    }

    // in case of disabled profiling runtime expects default (empty) value
    // to be -1 instead of 0
    taskWrapper.wload_id_ = getWorkloadId().value_or(-1);

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&taskWrapper);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUMI37XX::DPUVariantOp::getBinarySize() {
    return sizeof(nn_public::VpuDPUVariant);
}

size_t vpux::VPUMI37XX::DPUVariantOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuDPUVariant);
}

size_t vpux::VPUMI37XX::DPUVariantOp::getOffsetOfWithinOperation(mlir::Value value) {
    if (value == getInvariant()) {
        return offsetof(nn_public::VpuDPUVariant, invariant_);
    }

    VPUX_THROW("Provided Value is not linked to the DPUVariant Op or getOffset does not support it");
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::DPUVariantOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_EXECINSTR | ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::DPUVariantOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_DPU);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::DPUVariantOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}
