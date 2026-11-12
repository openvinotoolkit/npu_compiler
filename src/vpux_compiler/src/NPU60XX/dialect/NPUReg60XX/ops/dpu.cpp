//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>

#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/utils/core/error.hpp"

#include <cstdint>
#include <vector>

#include <npu_60xx_nnrt.hpp>

using namespace vpux;
using namespace npu60xx;

namespace vpux {
namespace NPUReg60XX {

void DPUInvariantOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto invariantDesc = getProperties().getDescriptor();

    VPUX_THROW_UNLESS(sizeof(nn_public::VpuDPUInvariant) == invariantDesc.size(),
                      "HW VpuDPUInvariant size {0} != regMapped representation size {1}.",
                      sizeof(nn_public::VpuDPUInvariant), invariantDesc.size());

    auto serializedInvariantDesc = invariantDesc.getStorage();
    binDataSection.appendData(serializedInvariantDesc.data(), serializedInvariantDesc.size());
}

size_t DPUInvariantOp::getBinarySize(config::ArchKind) {
    return sizeof(nn_public::VpuDPUInvariant);
}

std::vector<ELF::RelocationInfo> DPUInvariantOp::getRelocationInfo(ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    auto regsOffset = offsetof(nn_public::VpuDPUInvariant, registers_);
    auto opType = getNceTaskType();

    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    //
    // Input Relocs
    //
    if (auto inputSymRef = getInput().value_or(nullptr)) {
        auto addend = ELF::getOffsetOfSymRef(symRefMap, inputSymRef);
        // SOH disabled for VPU40XX

        if (getInputStorageElementTable().value_or(nullptr)) {
            relocs.emplace_back(
                    inputSymRef, targetSection, regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, act_offset),
                    ELF::RelocationType::R_VPU_LO_21, addend, "Input (act_offset) in DPUInvariant op reloc");
        } else {
            relocs.emplace_back(inputSymRef, targetSection,
                                regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, tensor_start),
                                ELF::RelocationType::R_VPU_LO_21_RSHIFT_4, addend,
                                "Input (tensor_start) in DPUInvariant op reloc");
        }
    }

    //
    // Input Sparsity Map Relocs
    //
    if (auto inputSparsityMap = getInputSparsityMap().value_or(nullptr)) {
        auto addend = ELF::getOffsetOfSymRef(symRefMap, inputSparsityMap);
        relocs.emplace_back(inputSparsityMap, targetSection,
                            regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, sparsity_addr),
                            ELF::RelocationType::R_VPU_LO_21, addend,
                            "Input sparsity map (sparsity_addr) in DPUInvariant op reloc");

        if (opType == VPUIP::NCETaskType::ELTWISE) {
            auto elop_addend = addend;
            if (auto tensorBSparsityMapSymRef = getWeightsSparsityMap().value_or(nullptr)) {
                elop_addend = ELF::getOffsetOfSymRef(symRefMap, tensorBSparsityMapSymRef);
            }
            relocs.emplace_back(inputSparsityMap, targetSection,
                                regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, elops_sparsity_addr),
                                ELF::RelocationType::R_VPU_LO_21, elop_addend,
                                "Input sparsity map (for ELTWISE, elops_sparsity_addr) in DPUInvariant op reloc");
        }
    }

    //
    // Input SE Table Relocs
    //
    if (auto inputSETable = getInputStorageElementTable().value_or(nullptr)) {
        auto addend = ELF::getOffsetOfSymRef(symRefMap, inputSETable);
        relocs.emplace_back(
                inputSETable, targetSection, regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, se_addr),
                ELF::RelocationType::R_VPU_LO_21, addend, "Input SE table (se_addr) in DPUInvariant op reloc");
    }

    //
    // Weights Relocs
    //
    // wt_offset needs to be set even if there is no weights operand in MAXPOOL & shouldn't be set for
    // ELTWISE
    if ((getWeights().value_or(nullptr) && opType != VPUIP::NCETaskType::ELTWISE) ||
        (opType == VPUIP::NCETaskType::MAXPOOL && getInput().value_or(nullptr))) {
        // Relocation needs to setup wt_offset to the base CMX addr
        auto addend = 0;
        auto newInput = getInput().value();
        auto weights = getWeights().value_or(nullptr);

        if ((getIsZeroOffsetWeightsTable() || !getWeightTable().has_value()) && weights &&
            !getWeightsSparsityMap().has_value()) {
            newInput = weights;
            addend = ELF::getOffsetOfSymRef(symRefMap, weights);
        }

        relocs.emplace_back(
                newInput, targetSection, regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, wt_offset),
                ELF::RelocationType::R_VPU_LO_21, addend,
                "Weights (wt_offset) in DPUInvariant op reloc");  // Using input as source just as a placeholder
    }

    //
    // Tensor2 Relocs
    //
    // tensor2_start needs to be set for dual elops (ELTWISE with 2 input tensors) in case of per output channel
    // scaling, i.e. when weightTable is provided
    if (opType == VPUIP::NCETaskType::ELTWISE && (getWeightTable().has_value() || getWeightTableScale().has_value())) {
        if (auto weights = getWeights().value_or(nullptr)) {
            auto addend = ELF::getOffsetOfSymRef(symRefMap, weights);
            relocs.emplace_back(weights, targetSection,
                                regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, tensor2_start),
                                ELF::RelocationType::R_VPU_LO_21_RSHIFT_4, addend,
                                "Weights (for ELTWISE, tensor2_start) in DPUInvariant op reloc");
        }
    }

    //
    // sprLookupTable Relocs
    //
    if (auto sprLookupTable = getSprLookupTable().value_or(nullptr)) {
        auto addend = ELF::getOffsetOfSymRef(symRefMap, sprLookupTable);
        relocs.emplace_back(sprLookupTable, targetSection,
                            regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, ppe_lut_ptr),
                            ELF::RelocationType::R_VPU_16_LSB_21_RSHIFT_5, addend,
                            "Spr lookup table (ppe_lut_ptr) in DPUInvariant op reloc");
    }

    //
    // palletLookupTable Relocs
    //
    if (auto palletLookupTable = getPalletLookupTable().value_or(nullptr)) {
        auto addend = ELF::getOffsetOfSymRef(symRefMap, palletLookupTable);
        relocs.emplace_back(palletLookupTable, targetSection,
                            regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, odu_cfg_2),
                            ELF::RelocationType::R_VPU_16_LSB_21_RSHIFT_5_LSHIFT_16, addend,
                            "Pallet lookup table (plt_ptr) in DPUInvariant op reloc");
    }

    //
    // Output Relocs
    //
    // no output in case of continued conv
    if (!getIsContinued()) {
        if (auto output = getOutput().value_or(nullptr)) {
            auto addend = ELF::getOffsetOfSymRef(symRefMap, output);
            relocs.emplace_back(
                    output, targetSection, regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, odu_ac_base),
                    ELF::RelocationType::R_VPU_LO_21, addend, "Output (odu_ac_base) in DPUInvariant op reloc");
        }
        if (auto outputSparsityMap = getOutputSparsityMap().value_or(nullptr)) {
            auto addend = ELF::getOffsetOfSymRef(symRefMap, outputSparsityMap);
            relocs.emplace_back(outputSparsityMap, targetSection,
                                regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, sp_base),
                                ELF::RelocationType::R_VPU_LO_21_MULTICAST_BASE, addend,
                                "Output sparsity map (sp_base) in DPUInvariant op reloc");
        }
        if (auto outputMinPerXY = getMinPerXy().value_or(nullptr)) {
            auto addend = ELF::getOffsetOfSymRef(symRefMap, outputMinPerXY);
            relocs.emplace_back(outputMinPerXY, targetSection,
                                regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, odu_mn_base),
                                ELF::RelocationType::R_VPU_LO_21, addend,
                                "Output min per xy values (odu_mn_base) in DPUInvariant op reloc");
        }
        if (auto outputMaxPerXY = getMaxPerXy().value_or(nullptr)) {
            auto addend = ELF::getOffsetOfSymRef(symRefMap, outputMaxPerXY);
            relocs.emplace_back(outputMaxPerXY, targetSection,
                                regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, odu_mx_base),
                                ELF::RelocationType::R_VPU_LO_21, addend,
                                "Output max per xy values (odu_mx_base) in DPUInvariant op reloc");
        }
        if (auto outputMinMaxPerTensor = getMinMaxPerTensor().value_or(nullptr)) {
            auto firstMinMaxSymRef = mlir::cast<mlir::SymbolRefAttr>(getMinMaxPerTensorAttr()[0]);
            auto addend = ELF::getOffsetOfSymRef(symRefMap, firstMinMaxSymRef);
            relocs.emplace_back(firstMinMaxSymRef, targetSection,
                                regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, odu_mt_base),
                                ELF::RelocationType::R_VPU_LO_21, addend,
                                "Output min&max per tensor values (odu_mt_base) in DPUInvariant op reloc");
        }
    }

    //
    // HWP Relocs
    //
    if (getProfilingData().has_value()) {
        // Relocation needs to setup hwp_cmx_mem_addr to the base CMX addr
        auto addend = 0;
        relocs.emplace_back(
                getInput().value_or(nullptr), targetSection,
                regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, hwp_cmx_mem_addr),
                ELF::RelocationType::R_VPU_CMX_LOCAL_RSHIFT_5, addend,
                "HWP (hwp_cmx_mem_addr) in DPUInvariant op reloc");  // Using input as source just as a placeholder
    }

    // set invariant pointer. workaround preemtion: #E-97614
    auto addend = ELF::getOffsetOfSymRef(symRefMap, getTaskLocation().value());
    relocs.emplace_back(getTaskLocation().value(), targetSection,
                        offsetof(nn_public::VpuDPUInvariant, registers_.tensor_mode),
                        ELF::RelocationType::R_VPU_16_LSB_21_RSHIFT_5_LSHIFT_16, addend,
                        "Invariant pointer in DPUInvariant op reloc");

    //
    // dynamicSequenceLength Relocs
    //
    if (auto dynamicSequenceLength = getDynamicSequenceLength().value_or(nullptr)) {
        auto addend = ELF::getOffsetOfSymRef(symRefMap, dynamicSequenceLength);
        relocs.emplace_back(dynamicSequenceLength, targetSection,
                            regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, dyn_scale),
                            ELF::RelocationType::R_VPU_32_OR_LO_19_LSB_21_RSHIFT_2, addend,
                            "Dynamic sequence length (dyn_scale_fetch_addr) in DPUInvariant op reloc");
    }

    return relocs;
}

size_t DPUInvariantOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(nn_public::VpuDPUInvariant);
}

void DPUVariantOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto variantDesc = getProperties().getDescriptor();

    VPUX_THROW_UNLESS(sizeof(nn_public::VpuDPUVariant) == variantDesc.size(),
                      "HW VpuDPUVariant size {0} != regMapped representation size {1}.",
                      sizeof(nn_public::VpuDPUVariant), variantDesc.size());

    auto serializedVariantDesc = variantDesc.getStorage();
    binDataSection.appendData(serializedVariantDesc.data(), serializedVariantDesc.size());
}

size_t DPUVariantOp::getBinarySize(config::ArchKind) {
    return sizeof(nn_public::VpuDPUVariant);
}

std::vector<ELF::RelocationInfo> DPUVariantOp::getRelocationInfo(ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    auto regsOffset = offsetof(nn_public::VpuDPUVariant, registers_);
    auto opType = getNceTaskType();

    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    // Important!
    // weight_start needs to be preset at serialization to the value of variant.weight_table_offset_
    if (auto weightTable = getWeightTable().value_or(nullptr)) {
        // weight_start reloc R_32_SUM with weight_table sym
        auto addend = ELF::getOffsetOfSymRef(symRefMap, weightTable);
        relocs.emplace_back(
                weightTable, targetSection, regsOffset + offsetof(nn_public::VpuDPUVariantRegisters, weight_start),
                ELF::RelocationType::R_VPU_LO_21_SUM, addend, "Weights (weight_start) in DPUVariant op reloc");
    } else {
        if (opType == VPUIP::NCETaskType::ELTWISE && !getWeightTableScale().has_value()) {
            if (auto weights = getWeights().value_or(nullptr)) {
                auto addend = ELF::getOffsetOfSymRef(symRefMap, weights);
                relocs.emplace_back(weights, targetSection,
                                    regsOffset + offsetof(nn_public::VpuDPUVariantRegisters, weight_start),
                                    ELF::RelocationType::R_VPU_LO_21_RSHIFT_4, addend,
                                    "Weights (for ELTWISE, weight_start) in DPUVariant op reloc");
            }
        }
        if (auto wtDataPtr = getWeightTableDataPtr().value_or(nullptr)) {
            auto addend = ELF::getOffsetOfSymRef(symRefMap, wtDataPtr);
            relocs.emplace_back(wtDataPtr, targetSection,
                                regsOffset + offsetof(nn_public::VpuDPUVariantRegisters, workload_start1),
                                ELF::RelocationType::R_VPU_16_LSB_21_RSHIFT_5_LSHIFT_16_SUM, addend,
                                "WeightTableDataPtr (weight_d_ptr_start) in DPUVariant op reloc");
        } else if (auto wtZp = getWeightZeroPoints().value_or(nullptr)) {
            auto addend = ELF::getOffsetOfSymRef(symRefMap, wtZp);
            relocs.emplace_back(wtZp, targetSection,
                                regsOffset + offsetof(nn_public::VpuDPUVariantRegisters, workload_start1),
                                ELF::RelocationType::R_VPU_16_LSB_21_RSHIFT_5_LSHIFT_16_SUM, addend,
                                "WeightZeroPoints (weight_d_ptr_start) in DPUVariant op reloc");
        }
        if (auto wtSpPtr = getWeightTableSpPtr().value_or(nullptr)) {
            auto addend = ELF::getOffsetOfSymRef(symRefMap, wtSpPtr);
            relocs.emplace_back(wtSpPtr, targetSection,
                                regsOffset + offsetof(nn_public::VpuDPUVariantRegisters, weight_num),
                                ELF::RelocationType::R_VPU_16_LSB_21_RSHIFT_5_LSHIFT_16_SUM, addend,
                                "WeightTableSpPtr (weight_sp_ptr_start) in DPUVariant op reloc");
        }
        auto wtScale = getWeightTableScale().value_or(nullptr);
        auto wtBias = getWeightTableBias().value_or(nullptr);
        if (wtScale && wtBias) {
            auto addend = ELF::getOffsetOfSymRef(symRefMap, wtScale);
            relocs.emplace_back(wtScale, targetSection,
                                regsOffset + offsetof(nn_public::VpuDPUVariantRegisters, weight_start),
                                ELF::RelocationType::R_VPU_16_LSB_21_RSHIFT_5_LSHIFT_16_SUM, addend,
                                "WeightTableScale (weight_start) in DPUVariant op reloc");
            addend = ELF::getOffsetOfSymRef(symRefMap, wtBias);
            relocs.emplace_back(wtBias, targetSection,
                                regsOffset + offsetof(nn_public::VpuDPUVariantRegisters, weight_start),
                                ELF::RelocationType::R_VPU_16_LSB_21_RSHIFT_5_SUM, addend,
                                "WeightTableBias (weight_start) in DPUVariant op reloc");
        }
    }

    // Set the additional invariant pointer in the VpuDPUVariant struct
    if (auto linkedInvariant = getInvariantTaskLocation()) {
        auto addend = ELF::getOffsetOfSymRef(symRefMap, linkedInvariant);
        relocs.emplace_back(linkedInvariant, targetSection,
                            offsetof(nn_public::VpuDPUVariant, invariant_) +
                                    offsetof(nn_public::VpuPtr<nn_public::VpuDPUInvariant>, ptr),
                            ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET, addend,
                            "Invariant pointer in DPUVariant op reloc");

        relocs.emplace_back(linkedInvariant, targetSection,
                            regsOffset + offsetof(nn_public::VpuDPUVariantRegisters, invar_ptr),
                            ELF::RelocationType::R_VPU_16_LSB_21_RSHIFT_5, addend,
                            "Invariant pointer register in DPUVariant op reloc");
    }

    // set variant pointer. workaround preemtion: #E-97614
    auto addend = ELF::getOffsetOfSymRef(symRefMap, getTaskLocation().value());
    relocs.emplace_back(getTaskLocation().value(), targetSection,
                        offsetof(nn_public::VpuDPUVariant, registers_.dpu_cfg),
                        ELF::RelocationType::R_VPU_16_LSB_21_RSHIFT_5_LSHIFT_CUSTOM, addend,
                        "Variant pointer in DPUVariant op reloc");

    if (auto nextLink = getNextLinkAttr()) {
        auto addend = ELF::getOffsetOfSymRef(symRefMap, nextLink);
        relocs.emplace_back(nextLink, targetSection, regsOffset + offsetof(nn_public::VpuDPUVariantRegisters, var_cfg),
                            ELF::RelocationType::R_VPU_16_LSB_21_RSHIFT_5_LSHIFT_16, addend,
                            "Next link (var_cfg) in DPUVariant op reloc");
    }

    return relocs;
}

size_t DPUVariantOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(nn_public::VpuDPUVariant);
}

void vpux::NPUReg60XX::DPUInvariantOp::build(
        mlir::OpBuilder&, mlir::OperationState& state, mlir::StringAttr symName, mlir::TypeAttr taskIndex,
        vpux::NPUReg60XX::Descriptors::DpuInvariantRegister&& descriptor, mlir::SymbolRefAttr taskLocation,
        mlir::SymbolRefAttr input, mlir::SymbolRefAttr inputSparsityMap, mlir::SymbolRefAttr inputStorageElementTable,
        mlir::SymbolRefAttr weights, mlir::SymbolRefAttr weightsSparsityMap, mlir::SymbolRefAttr weightTable,
        mlir::SymbolRefAttr weightTableDataPtr, mlir::SymbolRefAttr weightTableScale,
        mlir::SymbolRefAttr sprLookupTable, mlir::SymbolRefAttr palletLookupTable, mlir::SymbolRefAttr output,
        mlir::SymbolRefAttr outputSparsityMap, mlir::SymbolRefAttr profilingData, bool isZeroOffsetWeightsTable,
        mlir::SymbolRefAttr maxPerXY, mlir::SymbolRefAttr minPerXY, mlir::ArrayAttr minMaxPerTensor,
        mlir::SymbolRefAttr dynamicSequenceLength, VPUIP::NCETaskTypeAttr nceTaskType, bool isContinued) {
    auto& props = state.getOrAddProperties<Properties>();

    props.sym_name = symName;
    props.task_index = taskIndex;
    props.descriptor = std::move(descriptor);
    props.task_location = taskLocation;
    props.input = input;
    props.input_sparsity_map = inputSparsityMap;
    props.input_storage_element_table = inputStorageElementTable;
    props.weights = weights;
    props.weights_sparsity_map = weightsSparsityMap;
    props.weight_table = weightTable;
    props.weight_table_data_ptr = weightTableDataPtr;
    props.weight_table_scale = weightTableScale;
    props.spr_lookup_table = sprLookupTable;
    props.pallet_lookup_table = palletLookupTable;
    props.output = output;
    props.output_sparsity_map = outputSparsityMap;
    props.profiling_data = profilingData;
    props.is_zero_offset_weights_table = isZeroOffsetWeightsTable;
    props.max_per_xy = maxPerXY;
    props.min_per_xy = minPerXY;
    props.min_max_per_tensor = minMaxPerTensor;
    props.dynamic_sequence_length = dynamicSequenceLength;
    props.nce_task_type = nceTaskType;
    props.is_continued = isContinued;
}

void vpux::NPUReg60XX::DPUVariantOp::build(mlir::OpBuilder&, mlir::OperationState& state, mlir::StringAttr symName,
                                           mlir::SymbolRefAttr nextLink, mlir::TypeAttr taskIndex,
                                           vpux::NPUReg60XX::Descriptors::DpuVariantRegister&& descriptor,
                                           mlir::SymbolRefAttr taskLocation, mlir::SymbolRefAttr invariantTaskLocation,
                                           mlir::SymbolRefAttr weights, mlir::SymbolRefAttr weightTable,
                                           mlir::SymbolRefAttr weightTableDataPtr, mlir::SymbolRefAttr weightTableSpPtr,
                                           mlir::SymbolRefAttr weightTableScale, mlir::SymbolRefAttr weightTableBias,
                                           mlir::SymbolRefAttr weightZeroPoints, VPUIP::NCETaskTypeAttr nceTaskType,
                                           mlir::IntegerAttr workloadId, mlir::ArrayAttr variantPrimitiveId) {
    auto& props = state.getOrAddProperties<Properties>();

    props.sym_name = symName;
    props.next_link = nextLink;
    props.task_index = taskIndex;
    props.descriptor = std::move(descriptor);
    props.task_location = taskLocation;
    props.invariant_task_location = invariantTaskLocation;
    props.weights = weights;
    props.weight_table = weightTable;
    props.weight_table_data_ptr = weightTableDataPtr;
    props.weight_table_sp_ptr = weightTableSpPtr;
    props.weight_table_scale = weightTableScale;
    props.weight_table_bias = weightTableBias;
    props.weight_zero_points = weightZeroPoints;
    props.nce_task_type = nceTaskType;
    props.workload_id = workloadId;
    props.variant_primitive_id = variantPrimitiveId;
}

}  // namespace NPUReg60XX
}  // namespace vpux
