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
// ActKernelInvocationOp
//

void vpux::NPUReg50XX::ActKernelInvocationOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto actKernInvoDescriptor = getProperties().getDescriptor();

    VPUX_THROW_UNLESS(sizeof(nn_public::VpuActKernelInvocation) == actKernInvoDescriptor.size(),
                      "HW VpuActKernelInvocation size {0} != regMapped representation size {1}.",
                      sizeof(nn_public::VpuActKernelInvocation), actKernInvoDescriptor.size());

    auto serializedActKernInvoDesc = actKernInvoDescriptor.getStorage();
    binDataSection.appendData(serializedActKernInvoDesc.data(), getBinarySize(config::ArchKind::NPU50XX));
}

size_t vpux::NPUReg50XX::ActKernelInvocationOp::getBinarySize(config::ArchKind) {
    return sizeof(nn_public::VpuActKernelInvocation);
}

size_t vpux::NPUReg50XX::ActKernelInvocationOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(nn_public::VpuActKernelInvocation);
}

namespace {
size_t getSymRefOffsetForReloc(NPUReg50XX::ActKernelInvocationOp op, mlir::SymbolRefAttr ref) {
    constexpr auto ptrOffset = offsetof(nn_public::VpuPtr<void>, ptr);

    if (ref == op.getKernelRange()) {
        return offsetof(nn_public::VpuActKernelInvocation, range) + ptrOffset;
    } else if (ref == op.getKernelParams()) {
        return offsetof(nn_public::VpuActKernelInvocation, kernel_args) + ptrOffset;
    } else if (ref == op.getKernelData()) {
        return offsetof(nn_public::VpuActKernelInvocation, data_window_base) + ptrOffset;
    } else if (ref == op.getProfilingData()) {
        return offsetof(nn_public::VpuActKernelInvocation, perf_packet_out) + ptrOffset;
    }

    VPUX_THROW("Provided SymbolRefAttr is not linked to the ActKernelInvocation Op or getSymRefOffsetForReloc does not "
               "support "
               "it");
}
}  // namespace

std::vector<ELF::RelocationInfo> vpux::NPUReg50XX::ActKernelInvocationOp::getRelocationInfo(
        ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    auto thisInvo = *(this);
    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    relocs.emplace_back(getKernelRange(), targetSection, getSymRefOffsetForReloc(thisInvo, getKernelRange()),
                        ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET,
                        ELF::getOffsetOfSymRef(symRefMap, getKernelRange()),
                        "Kernel range act kernel invocation reloc");

    if (auto kernelData = getKernelData().value_or(nullptr)) {
        relocs.emplace_back(kernelData, targetSection, getSymRefOffsetForReloc(thisInvo, kernelData),
                            ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, kernelData),
                            "Kernel data act kernel invocation reloc");
    }

    relocs.emplace_back(getKernelParams(), targetSection, getSymRefOffsetForReloc(thisInvo, getKernelParams()),
                        ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, getKernelParams()),
                        "Kernel params act kernel invocation reloc");

    if (auto profilingData = getProfilingData().value_or(nullptr)) {
        relocs.emplace_back(profilingData, targetSection, getSymRefOffsetForReloc(thisInvo, profilingData),
                            ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET,
                            ELF::getOffsetOfSymRef(symRefMap, profilingData),
                            "Profiling data act kernel invocation reloc");
    }

    if (auto nextLink = getNextLinkAttr()) {
        auto addend = ELF::getOffsetOfSymRef(symRefMap, nextLink);
        relocs.push_back(ELF::RelocationInfo(nextLink, targetSection,
                                             offsetof(nn_public::VpuActKernelInvocation, next_aki_wl_addr),
                                             ELF::RelocationType::R_VPU_32_BIT_OR_B21_B26_UNSET, addend));
    }

    return relocs;
}

void vpux::NPUReg50XX::ActKernelInvocationOp::build(mlir::OpBuilder&, mlir::OperationState& state,
                                                    mlir::StringAttr symName,
                                                    vpux::NPUReg50XX::Descriptors::VpuActKernelInvocation&& descriptor,
                                                    mlir::SymbolRefAttr taskLocation, mlir::SymbolRefAttr nextLink,
                                                    mlir::SymbolRefAttr kernelRange, mlir::SymbolRefAttr kernelData,
                                                    mlir::SymbolRefAttr kernelParams,
                                                    mlir::SymbolRefAttr profilingData) {
    auto& props = state.getOrAddProperties<Properties>();

    props.sym_name = symName;
    props.descriptor = std::move(descriptor);
    props.task_location = taskLocation;
    props.next_link = nextLink;
    props.kernel_range = kernelRange;
    props.kernel_data = kernelData;
    props.kernel_params = kernelParams;
    props.profiling_data = profilingData;
}
