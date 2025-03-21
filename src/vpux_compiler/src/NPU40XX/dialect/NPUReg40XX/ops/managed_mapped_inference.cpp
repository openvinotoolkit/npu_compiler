//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/utils/ELF/utils.hpp"
#include "vpux/utils/core/error.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

//
// ManagedMappedInferenceOp
//

void vpux::NPUReg40XX::ManagedMappedInferenceOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto managedMpiDescriptor = getDescriptor().getRegMapped();

    VPUX_THROW_UNLESS(sizeof(nn_public::VpuManagedMappedInference) == managedMpiDescriptor.size(),
                      "HW VpuManagedMappedInference size {0} != regMapped representation size {1}.",
                      sizeof(nn_public::VpuManagedMappedInference), managedMpiDescriptor.size());

    auto serializedDescriptor = managedMpiDescriptor.getStorage();
    binDataSection.appendData(serializedDescriptor.data(), getBinarySize(VPU::ArchKind::NPU40XX));
}

size_t vpux::NPUReg40XX::ManagedMappedInferenceOp::getBinarySize(VPU::ArchKind) {
    return sizeof(npu40xx::nn_public::VpuManagedMappedInference);
}

size_t vpux::NPUReg40XX::ManagedMappedInferenceOp::getAlignmentRequirements(VPU::ArchKind) {
    return alignof(nn_public::VpuManagedMappedInference);
}

namespace {
size_t getSymRefOffsetForReloc(NPUReg40XX::ManagedMappedInferenceOp op, mlir::SymbolRefAttr ref) {
    if (ref == op.getBarrierTasksAttr()) {
        return offsetof(nn_public::VpuManagedMappedInference, task_configs) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuTaskBarrierMap>, address);
    }

    if (ref == op.getWorkItemsAttr()) {
        return offsetof(nn_public::VpuManagedMappedInference, work_items) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuWorkItem>, address);
    }

    if (ref == op.getBootstrapTasksAttr()) {
        return offsetof(nn_public::VpuManagedMappedInference, initial_barriers) +
               offsetof(nn_public::VpuTaskReference<uint32_t>, address);
    }

    if (ref == op.getNnrtConfigAttr()) {
        return offsetof(nn_public::VpuManagedMappedInference, nnrt_config) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuNNRTConfig>, address);
    }

    if (ref == op.getBarrierConfigurationTasksAttr()) {
        return offsetof(nn_public::VpuManagedMappedInference, barriers_configuration) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuBarrierConfiguration>, address);
    }

    if (ref == op.getNumOfBarrierReprogrammingsAttr()) {
        return offsetof(nn_public::VpuManagedMappedInference, num_of_barrier_reprogrammings) +
               offsetof(nn_public::VpuTaskReference<uint32_t>, address);
    }

    VPUX_THROW(
            "Provided SymbolRefAttr is not linked to the ManagedMappedInference Op or getSymRefOffsetForReloc does not "
            "support it");
}
}  // namespace

std::vector<ELF::RelocationInfo> NPUReg40XX::ManagedMappedInferenceOp::getRelocationInfo(
        ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    auto thisMMI = *(this);
    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    if (auto workItemTasks = getWorkItems().value_or(nullptr)) {
        relocs.emplace_back(workItemTasks, targetSection, getSymRefOffsetForReloc(thisMMI, workItemTasks),
                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, workItemTasks),
                            "workItemTasks for managed mapped inference reloc");
    }

    if (auto barriersTasks = getBarrierTasks().value_or(nullptr)) {
        relocs.emplace_back(barriersTasks, targetSection, getSymRefOffsetForReloc(thisMMI, barriersTasks),
                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, barriersTasks),
                            "barriersTasks for managed mapped inference reloc");
    }

    if (auto bootstrapTasks = getBootstrapTasks().value_or(nullptr)) {
        relocs.emplace_back(bootstrapTasks, targetSection, getSymRefOffsetForReloc(thisMMI, bootstrapTasks),
                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, bootstrapTasks),
                            "bootstrapTasks for managed mapped inference reloc");
    }

    if (auto nnrtConfig = getNnrtConfig()) {
        relocs.emplace_back(nnrtConfig, targetSection, getSymRefOffsetForReloc(thisMMI, nnrtConfig),
                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, nnrtConfig));
    }

    if (auto barrierProgramming = getBarrierConfigurationTasks().value_or(nullptr)) {
        relocs.emplace_back(barrierProgramming, targetSection, getSymRefOffsetForReloc(thisMMI, barrierProgramming),
                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, barrierProgramming),
                            "barrierConfiguration for managed mapped inference reloc");
    }

    if (auto barrierProgrammingStrides = getNumOfBarrierReprogrammings().value_or(nullptr)) {
        relocs.emplace_back(barrierProgrammingStrides, targetSection,
                            getSymRefOffsetForReloc(thisMMI, barrierProgrammingStrides), ELF::RelocationType::R_VPU_64,
                            ELF::getOffsetOfSymRef(symRefMap, barrierProgrammingStrides),
                            "numOfBarrierReprogrammings for managed mapped inference reloc");
    }

    return relocs;
}

void NPUReg40XX::ManagedMappedInferenceOp::setVersion(const elf::Version& version) {
    auto descriptor = getDescriptor().getRegMapped();
    const auto serializedVersion = VPU_CONCAT_NNRT_API_VER(version.getMajor(), version.getMinor());
    descriptor.write<NPUReg40XX::Fields::MMI_vpu_nnrt_api_ver>(serializedVersion);

    auto managedMappedInferenceDescriptorAttr =
            NPUReg40XX::VpuManagedMappedInferenceAttr::get(getContext(), std::move(descriptor));
    setDescriptorAttr(managedMappedInferenceDescriptorAttr);
}
