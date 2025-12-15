//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include <npu_40xx_nnrt.hpp>
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"

using namespace vpux;
using namespace npu40xx;

//
// ActKernelRangeOp
//

void vpux::NPUReg50XX::ActKernelRangeOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto actKernRangeDescriptor = getProperties().getDescriptor();

    VPUX_THROW_UNLESS(sizeof(nn_public::VpuActKernelRange) == actKernRangeDescriptor.size(),
                      "HW VpuActKernelRange size {0} != regMapped representation size {1}.",
                      sizeof(nn_public::VpuActKernelRange), actKernRangeDescriptor.size());

    auto serializedActKernRangeDesc = actKernRangeDescriptor.getStorage();
    binDataSection.appendData(serializedActKernRangeDesc.data(), getBinarySize(config::ArchKind::NPU50XX));
}

size_t vpux::NPUReg50XX::ActKernelRangeOp::getBinarySize(config::ArchKind) {
    return sizeof(nn_public::VpuActKernelRange);
}

size_t vpux::NPUReg50XX::ActKernelRangeOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(nn_public::VpuActKernelRange);
}

std::vector<ELF::RelocationInfo> vpux::NPUReg50XX::ActKernelRangeOp::getRelocationInfo(
        ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    if (auto kernelText = getKernelText().value_or(nullptr)) {
        relocs.emplace_back(
                kernelText, targetSection,
                offsetof(nn_public::VpuActKernelRange, text_window_base) + offsetof(nn_public::VpuPtr<void>, ptr),
                ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, kernelText),
                "Kernel text (ptr in text_window_base) for act kernel range reloc");
    }
    return relocs;
}

void NPUReg50XX::ActKernelRangeOp::build(mlir::OpBuilder&, mlir::OperationState& state, mlir::StringAttr symName,
                                         vpux::NPUReg50XX::Descriptors::VpuActKernelRange&& descriptor,
                                         mlir::SymbolRefAttr taskLocation, mlir::SymbolRefAttr kernelText,
                                         mlir::SymbolRefAttr kernelEntry) {
    auto& props = state.getOrAddProperties<Properties>();

    props.sym_name = symName;
    props.descriptor = std::move(descriptor);
    props.task_location = taskLocation;
    props.kernel_text = kernelText;
    props.kernel_entry = kernelEntry;
}
