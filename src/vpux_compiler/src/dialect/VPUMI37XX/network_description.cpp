//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <openvino/core/partial_shape.hpp>

#include <intel_npu/prefix.hpp>

#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>
#include <vpux_headers/serial_metadata.hpp>

#include "vpux/compiler/dialect/VPUMI37XX/network_description.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"

#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <algorithm>

using namespace vpux;

using intel_npu::IODescriptor;
using intel_npu::NetworkMetadata;

namespace {

ov::element::Type_t extractPrecisionFromDType(elf::DType dtype) {
    static const EnumMap<elf::DType, ov::element::Type_t> dataTypeMapping = {
            {elf::DType::DType_FP64, ov::element::Type_t::f64},   {elf::DType::DType_FP32, ov::element::Type_t::f32},
            {elf::DType::DType_FP16, ov::element::Type_t::f16},   {elf::DType::DType_U64, ov::element::Type_t::u64},
            {elf::DType::DType_U32, ov::element::Type_t::u32},    {elf::DType::DType_U16, ov::element::Type_t::u16},
            {elf::DType::DType_U8, ov::element::Type_t::u8},      {elf::DType::DType_U4, ov::element::Type_t::u4},
            {elf::DType::DType_I64, ov::element::Type_t::i64},    {elf::DType::DType_I32, ov::element::Type_t::i32},
            {elf::DType::DType_I16, ov::element::Type_t::i16},    {elf::DType::DType_I8, ov::element::Type_t::i8},
            {elf::DType::DType_I4, ov::element::Type_t::i4},      {elf::DType::DType_BIN, ov::element::Type_t::u1},
            {elf::DType::DType_BFP16, ov::element::Type_t::bf16}, {elf::DType::DType_I4X, ov::element::Type_t::nf4}};

    VPUX_THROW_WHEN(dataTypeMapping.count(dtype) == 0,
                    "Missing precision upon attempting to convert the value from ELF to OV format");

    return dataTypeMapping.at(dtype);
}

const EnumMap<elf::OVNodeType, ov::element::Type_t> mapElementTypeOV = {
        {elf::OVNodeType::OVNodeType_UNDEFINED, ov::element::Type_t::undefined},
        {elf::OVNodeType::OVNodeType_DYNAMIC, ov::element::Type_t::dynamic},
        {elf::OVNodeType::OVNodeType_BOOLEAN, ov::element::Type_t::boolean},
        {elf::OVNodeType::OVNodeType_BF16, ov::element::Type_t::bf16},
        {elf::OVNodeType::OVNodeType_F16, ov::element::Type_t::f16},
        {elf::OVNodeType::OVNodeType_F32, ov::element::Type_t::f32},
        {elf::OVNodeType::OVNodeType_F64, ov::element::Type_t::f64},
        {elf::OVNodeType::OVNodeType_I4, ov::element::Type_t::i4},
        {elf::OVNodeType::OVNodeType_I8, ov::element::Type_t::i8},
        {elf::OVNodeType::OVNodeType_I16, ov::element::Type_t::i16},
        {elf::OVNodeType::OVNodeType_I32, ov::element::Type_t::i32},
        {elf::OVNodeType::OVNodeType_I64, ov::element::Type_t::i64},
        {elf::OVNodeType::OVNodeType_U1, ov::element::Type_t::u1},
        {elf::OVNodeType::OVNodeType_U4, ov::element::Type_t::u4},
        {elf::OVNodeType::OVNodeType_U8, ov::element::Type_t::u8},
        {elf::OVNodeType::OVNodeType_U16, ov::element::Type_t::u16},
        {elf::OVNodeType::OVNodeType_U32, ov::element::Type_t::u32},
        {elf::OVNodeType::OVNodeType_U64, ov::element::Type_t::u64},
        {elf::OVNodeType::OVNodeType_NF4, ov::element::Type_t::nf4},
};

/**
 * @brief Extracts the I/O metadata from ELF specific structures and converts them into OpenVINO specific ones.
 *
 * @param descriptorsFromCompiler The input or output descriptors as seen by the compiler. The object uses ELF specific
 * structures.
 * @param descriptorsFromIRModel The input or output descriptors as seen inside the original IR model (unaltered by the
 * compiler). The object uses ELF specific structures.
 * @returns A vector of descriptors containing OpenVINO specific structures. The vector is meant to represent either the
 * inputs or the outputs of a network, depending on the type of the given arguments.
 */
std::vector<IODescriptor> convertIODescriptors(const std::vector<elf::TensorRef>& descriptorsFromCompiler,
                                               const std::optional<std::vector<elf::OVNode>>& descriptorsFromIRModel) {
    std::vector<IODescriptor> convertedIODescriptors;

    const size_t descriptorsFromCompilerCount = descriptorsFromCompiler.size();
    size_t descriptorsFromIRModelCount = 0;

    if (descriptorsFromIRModel.has_value()) {
        // In addition to the inputs/outputs found in the IR model, the compiler may also place additional I/O such as
        // states and shape tensors. Thus, we cannot have less I/O entries originating from the compiler compared to the
        // IR model.
        descriptorsFromIRModelCount = descriptorsFromIRModel->size();
        VPUX_THROW_WHEN(descriptorsFromIRModelCount > descriptorsFromCompilerCount,
                        "The number of inputs/outputs extracted from the compiled model is less than the number found "
                        "in the IR model");
    }

    convertedIODescriptors.reserve(descriptorsFromCompilerCount);
    for (auto descriptorIndex : irange(descriptorsFromCompilerCount)) {
        IODescriptor convertedIODescriptor;

        // From the "elf::TensorRef" structure we may build the following "intel_npu::IODescriptor" fields:
        //  * nameFromCompiler
        //  * precision
        //  * shapeFromCompiler
        //  * isStateInput/isStateOutput/isShapeTensor
        const elf::TensorRef& descriptorFromCompiler = descriptorsFromCompiler.at(descriptorIndex);
        convertedIODescriptor.nameFromCompiler = descriptorFromCompiler.name;

        // Flags will be used instead of indices for informing the type of the current entry
        if (intel_npu::isStateInputName(convertedIODescriptor.nameFromCompiler)) {
            convertedIODescriptor.nameFromCompiler =
                    convertedIODescriptor.nameFromCompiler.substr(intel_npu::READVALUE_PREFIX.length());
            convertedIODescriptor.isStateInput = true;
        } else if (intel_npu::isStateOutputName(convertedIODescriptor.nameFromCompiler)) {
            convertedIODescriptor.nameFromCompiler =
                    convertedIODescriptor.nameFromCompiler.substr(intel_npu::ASSIGN_PREFIX.length());
            convertedIODescriptor.isStateOutput = true;
        } else if (intel_npu::isShapeTensorName(convertedIODescriptor.nameFromCompiler)) {
            convertedIODescriptor.nameFromCompiler =
                    convertedIODescriptor.nameFromCompiler.substr(intel_npu::SHAPE_TENSOR_PREFIX.length());
            convertedIODescriptor.isShapeTensor = true;
        }

        const std::vector<size_t> dataDims(descriptorFromCompiler.dimensions,
                                           descriptorFromCompiler.dimensions + descriptorFromCompiler.dimensions_size);

        convertedIODescriptor.shapeFromCompiler = ov::PartialShape(dataDims);
        convertedIODescriptor.precision = extractPrecisionFromDType(descriptorFromCompiler.data_type);

        if (descriptorsFromIRModel.has_value() && descriptorIndex < descriptorsFromIRModelCount) {
            // The states and shape tensors are appended as inputs/outputs by the compiler, the IR model cannot contain
            // them in the parameters/results entries.
            VPUX_THROW_WHEN(convertedIODescriptor.isStateInput || convertedIODescriptor.isStateOutput ||
                                    convertedIODescriptor.isShapeTensor,
                            "The inputs/outputs found in the IR model cannot be states or shape tensors");

            // From the "elf::OVNode" structure we may build the following "intel_npu::IODescriptor" fields:
            //  * nodeFriendlyName
            //  * outputTensorNames
            const elf::OVNode& descriptorFromIRModel = descriptorsFromIRModel->at(descriptorIndex);

            convertedIODescriptor.nodeFriendlyName = std::string(descriptorFromIRModel.friendly_name);
            convertedIODescriptor.outputTensorNames = [&descriptorFromIRModel]() {
                std::unordered_set<std::string> retTensorNames;
                for (auto i : irange(descriptorFromIRModel.tensor_names_count)) {
                    retTensorNames.insert(std::string(descriptorFromIRModel.tensor_names[i]));
                }
                return retTensorNames;
            }();
            convertedIODescriptor.shapeFromIRModel = std::optional([&descriptorFromIRModel]() {
                const auto dynamicDim = std::numeric_limits<uint64_t>::max();
                ov::PartialShape retShape;
                for (auto i : irange(descriptorFromIRModel.shape_size)) {
                    retShape.reserve(descriptorFromIRModel.shape_size);
                    if (descriptorFromIRModel.shape[i] != dynamicDim) {
                        retShape.push_back(checked_cast<int64_t>(descriptorFromIRModel.shape[i]));
                    } else {
                        retShape.push_back(-1);
                    }
                }
                return retShape;
            }());

            VPUX_THROW_WHEN(convertedIODescriptor.precision != mapElementTypeOV.at(descriptorFromIRModel.type),
                            "Precision mismatch between the compiler specific metadata and IR model one.\nName from "
                            "compiler: {0}, precision: {1}\nNode friendly name: {2}, precision: {3}",
                            convertedIODescriptor.nameFromCompiler,
                            ov::element::Type(convertedIODescriptor.precision).get_type_name(),
                            convertedIODescriptor.nodeFriendlyName,
                            ov::element::Type(mapElementTypeOV.at(descriptorFromIRModel.type)).get_type_name());
        }

        convertedIODescriptors.push_back(std::move(convertedIODescriptor));
    }

    return convertedIODescriptors;
}

}  // namespace

NetworkMetadata vpux::VPUMI37XX::getNetworkMetadata(mlir::ArrayRef<uint8_t> blob) {
    NetworkMetadata network;

    OV_ITT_TASK_CHAIN(NETWORK_DESCRIPTION, itt::domains::VPUXPlugin, "NetworkDescription::NetworkDescription",
                      "elfReader");
    VPUX_THROW_UNLESS(!blob.empty(), "Got NULL pointer");

    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(blob.data(), blob.size());
    elf::Reader<elf::ELF_Bitness::Elf64> reader(&accessor);

    std::shared_ptr<elf::NetworkMetadata> metadata;

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "getSection&getHeader");
    for (auto secIndex : irange(reader.getSectionsNum())) {
        const auto& section = reader.getSection(secIndex);

        const auto secHeader = section.getHeader();
        if (secHeader->sh_type == static_cast<elf::Elf_Word>(vpux::ELFNPU37XX::SectionTypeAttr::VPU_SHT_NETDESC)) {
            metadata = elf::MetadataSerialization::deserialize(section.getData<uint8_t>(), secHeader->sh_size);
            break;
        }
    }

    VPUX_THROW_UNLESS(metadata != nullptr, "METADATA NOT FOUND IN ELF");
    network.name = metadata->mIdentification.blob_name;

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "convertIODescriptors");

    network.inputs = convertIODescriptors(metadata->mNetInputs, std::optional(metadata->mOVParameters));
    network.outputs = convertIODescriptors(metadata->mNetOutputs, std::optional(metadata->mOVResults));
    network.profilingOutputs = convertIODescriptors(metadata->mProfilingOutputs, std::nullopt);

    VPUX_THROW_UNLESS(!network.outputs.empty(), "Metadata structure does not contain info on outputs");

    network.numStreams = metadata->mResourceRequirements.nn_slice_count_;

    network.bindRelatedDescriptors();

    return network;
}
