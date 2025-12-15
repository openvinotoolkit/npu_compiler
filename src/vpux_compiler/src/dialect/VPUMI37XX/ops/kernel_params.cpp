//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <kernels/inc/common_types.h>

using namespace vpux;

//
// KernelParamsOp
//

void vpux::VPUMI37XX::KernelParamsOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto params = getKernelParams();

    auto dense_elem_data = params.getValues<uint8_t>();

    auto data_vector = std::vector<uint8_t>(dense_elem_data.begin(), dense_elem_data.end());

    // serialize actual kernel params
    binDataSection.appendData(data_vector.data(), data_vector.size());

    // serialize IO dims/strides - only when shave kernel is not MLIR-compiled
    if (!getIsCompiled()) {
        std::vector<uint8_t> inputDimsVector, outputDimsVector;
        std::vector<uint8_t> inputStridesVector, outputStridesVector;
        const auto inputMemrefVals = getInputs();
        const auto outputMemrefVals = getOutputs();

        auto insertDimsIntoVector = [](std::vector<uint8_t>& dimsVector, mlir::Value val) {
            const auto shape = getShape(val);
            const auto inOrderDims = DimsOrder::fromValue(val);
            const auto memShape = inOrderDims.toMemoryOrder(shape);

            for (auto& memDim : memShape | reversed) {
                auto dim = checked_cast<int32_t>(memDim);
                ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&dim), sizeof(dim));
                dimsVector.insert(dimsVector.end(), valueAsArray.begin(), valueAsArray.end());
            }
        };

        auto insertStridesIntoVector = [](std::vector<uint8_t>& stridesVector, mlir::Value val) {
            const auto strides = getMemStrides(val);
            for (auto&& stride : strides | reversed) {
                ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&stride), sizeof(stride));
                stridesVector.insert(stridesVector.end(), valueAsArray.begin(), valueAsArray.end());
            }
        };

        // input Dims & Strides
        for (const auto inputMemrefVal : inputMemrefVals) {
            insertDimsIntoVector(inputDimsVector, inputMemrefVal);
            insertStridesIntoVector(inputStridesVector, inputMemrefVal);
        }

        // output Dims & Strides
        for (const auto outputMemrefVal : outputMemrefVals) {
            insertDimsIntoVector(outputDimsVector, outputMemrefVal);
            insertStridesIntoVector(outputStridesVector, outputMemrefVal);
        }

        binDataSection.appendData(inputDimsVector.data(), inputDimsVector.size());
        binDataSection.appendData(inputStridesVector.data(), inputStridesVector.size());
        binDataSection.appendData(outputDimsVector.data(), outputDimsVector.size());
        binDataSection.appendData(outputStridesVector.data(), outputStridesVector.size());
    }
}

size_t vpux::VPUMI37XX::KernelParamsOp::getBinarySize() {
    if (getIsCompiled()) {
        return getParamsStructSize();
    }

    const auto inputMemrefVals = getInputs();
    const auto outputMemrefVals = getOutputs();

    size_t inputDimsSize = 0;
    size_t inputStridesSize = 0;

    for (const auto inputMemrefVal : inputMemrefVals) {
        inputDimsSize += sizeof(int32_t) * getShape(inputMemrefVal).size();
        inputStridesSize += sizeof(int64_t) * getMemStrides(inputMemrefVal).size();
    }

    size_t outputDimsSize = 0;
    size_t outputStridesSize = 0;

    for (const auto outputMemrefVal : outputMemrefVals) {
        outputDimsSize += sizeof(int32_t) * getShape(outputMemrefVal).size();
        outputStridesSize += sizeof(int64_t) * getMemStrides(outputMemrefVal).size();
    }

    return getParamsStructSize() + inputDimsSize + outputDimsSize + inputStridesSize + outputStridesSize;
}

size_t vpux::VPUMI37XX::KernelParamsOp::getParamsStructSize() {
    auto params = getKernelParams();
    auto dense_elem_data = params.getValues<uint8_t>();
    auto data_vector = std::vector<uint8_t>(dense_elem_data.begin(), dense_elem_data.end());

    return data_vector.size();
}

size_t vpux::VPUMI37XX::KernelParamsOp::getOffsetOfWithinOperation(mlir::Value) {
    VPUX_THROW("getOffset is not supported for KernelParams Op");
}

// The parameter structs for the sw layers must be 64Byte aligned as an ActShave requirement
size_t vpux::VPUMI37XX::KernelParamsOp::getAlignmentRequirements() {
    return ELFNPU37XX::VPUX_DEFAULT_ALIGNMENT;
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::KernelParamsOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::KernelParamsOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::KernelParamsOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}
