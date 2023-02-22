//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/ELF/metadata.hpp"

using namespace vpux;

void copy_str(char* dst, const std::string& src) {
    auto str_len = src.size() < elf::MAX_STRING_LEN ? src.size() : elf::MAX_STRING_LEN - 1;

    memcpy(dst, src.data(), str_len);
    dst[str_len] = '\0';
}

elf::DType ELF::createDType(mlir::Type type) {
    if (type.isF64()) {
        return elf::DType::DType_FP64;
    } else if (type.isF32()) {
        return elf::DType::DType_FP32;
    } else if (type.isF16()) {
        return elf::DType::DType_FP16;
    } else if (type.isBF16()) {
        return elf::DType::DType_BFP16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int64_t))) {
        return elf::DType::DType_I64;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int32_t))) {
        return elf::DType::DType_I32;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int16_t))) {
        return elf::DType::DType_I16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return elf::DType::DType_I8;
    } else if (type.isSignedInteger(4)) {
        return elf::DType::DType_I4;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint64_t))) {
        return elf::DType::DType_U64;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint32_t))) {
        return elf::DType::DType_U32;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint16_t))) {
        return elf::DType::DType_U16;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint8_t))) {
        return elf::DType::DType_U8;
    } else if (type.isInteger(4)) {
        return elf::DType::DType_U4;
    } else if (type.isInteger(2)) {
        return elf::DType::DType_I2;
    } else if (type.isInteger(1)) {
        return elf::DType::DType_BIN;
    } else if (type.isa<mlir::quant::QuantizedType>()) {
        return createDType(type.cast<mlir::quant::QuantizedType>().getStorageType());
    } else {
        VPUX_THROW("Unsupported element type {0}", type);
    }
}

elf::TensorRef ELF::createTensorRef(vpux::NDTypeInterface type, StringRef name) {
    elf::TensorRef out;

    copy_str(out.name, name.str());

    // dtype
    out.data_type = ELF::createDType(type.getElementType());

    // dims
    const auto shape = type.getShape();
    VPUX_THROW_WHEN(shape.empty(), "Shape variable is empty");
    out.dimensions_size = shape.size();

    for (auto sh_pair : shape | indexed) {
        const auto ind = checked_cast<uint32_t>(sh_pair.index());
        auto sh = sh_pair.value();
        out.dimensions[ind] = checked_cast<uint32_t>(sh);
    }

    // strides
    auto strides = type.getStrides();
    out.strides_size = strides.size();

    Strides temp;
    temp.push_back(type.getElemTypeSize());
    temp.append(strides.begin(), strides.end());

    for (auto iterator : temp | indexed) {
        auto val = iterator.value();
        auto index = iterator.index();

        if (val.count() % CHAR_BIT == 0) {
            checked_cast<float>(Byte(val).count());
        }

        out.strides[index] = checked_cast<float>(val.count()) / CHAR_BIT;
    }

    // dimsOrder
    out.order = type.getDimsOrder().code();

    return out;
}

elf::TensorRef ELF::createTensorRef(mlir::Value val, StringRef name) {
    return createTensorRef(val.getType().cast<vpux::NDTypeInterface>(), name);
}

elf::NetworkMetadata ELF::constructMetadata(mlir::ModuleOp module, IE::CNNNetworkOp netOp, mlir::FuncOp netFunc,
                                            const std::vector<vpux::PreProcessInfo>& preprocessInfo,
                                            const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                            const std::vector<std::shared_ptr<const ov::Node>>& results) {
    auto inputsInfo = netOp.getInputsInfo();
    auto outputsInfo = netOp.getOutputsInfo();
    auto profilingOutputsInfo = netOp.getProfilingOutputsInfo();

    elf::NetworkMetadata metadata;

    copy_str(metadata.blob_name, module.getName().getValueOr("network").str());

    metadata.net_input_count = inputsInfo.size();
    metadata.in_tenosr_count = inputsInfo.size();

    metadata.net_output_count = outputsInfo.size();
    metadata.out_tensor_count = outputsInfo.size();

    metadata.profiling_output_count = profilingOutputsInfo.size();

    // input
    for (const auto& p : inputsInfo | indexed) {
        const auto index = checked_cast<uint32_t>(p.index());
        auto userInfo = p.value();
        const auto val = netFunc.getArgument(index);

        const auto userType = userInfo.userType().cast<vpux::NDTypeInterface>();

        metadata.net_input[index] = createTensorRef(val, userInfo.name());
        metadata.in_tensor_desc[index] = createTensorRef(userType, userInfo.name());
    }

    // output
    for (const auto& p : outputsInfo | indexed) {
        const auto index = p.index();
        const auto funcArgIndex = inputsInfo.size() + index;

        auto userInfo = p.value();
        const auto val = netFunc.getArgument(checked_cast<uint32_t>(funcArgIndex));

        const auto userType = userInfo.userType().cast<vpux::NDTypeInterface>();

        metadata.net_output[index] = createTensorRef(val, userInfo.name());
        metadata.out_tensor_desc[index] = createTensorRef(userType, userInfo.name());
    }

    // profiling
    for (const auto& p : profilingOutputsInfo | indexed) {
        const auto index = p.index();
        const auto funcArgInd = inputsInfo.size() + outputsInfo.size() + index;

        const auto val = netFunc.getArgument(checked_cast<uint32_t>(funcArgInd));

        metadata.profiling_output[index] = createTensorRef(val, p.value().name());
    }

    // ov parameters
    metadata.ov_parameters_count = parameters.size();
    for (const auto& node : parameters | indexed) {
        VPUX_THROW_WHEN(node.value() == nullptr, "Null OV node");
        auto node_val = node.value();
        auto index = node.index();

        elf::OVNode tmp_node;
        tmp_node.type = ELF::mapElementType.at(node_val->get_element_type());

        // name strings
        copy_str(tmp_node.friendly_name, node_val->get_friendly_name());

        tmp_node.input_name[0] = '\0';

        const auto tmpTensorNames = node_val->get_output_tensor(0).get_names();
        tmp_node.tensor_names_count = tmpTensorNames.size();
        for (auto tensor_name : tmpTensorNames | indexed) {
            copy_str(tmp_node.tensor_names[tensor_name.index()], tensor_name.value());
        }

        // shape
        auto shape = node_val->get_output_partial_shape(0).get_shape();
        tmp_node.shape_size = shape.size();

        for (const auto& sh_iterator : shape | indexed) {
            tmp_node.shape[sh_iterator.index()] = sh_iterator.value();
        }

        metadata.ov_parameters[index] = tmp_node;
    }

    // ov results
    metadata.ov_results_count = results.size();
    for (const auto& node : results | indexed) {
        VPUX_THROW_WHEN(node.value() == nullptr, "Null OV node");
        auto node_val = node.value();
        auto index = node.index();

        elf::OVNode tmp_node;
        tmp_node.type = ELF::mapElementType.at(node_val->get_element_type());

        // name strings
        copy_str(tmp_node.friendly_name, node_val->get_friendly_name());

        const auto tmpInputName = ngraph::op::util::create_ie_output_name(node_val->input_value(0));
        copy_str(tmp_node.input_name, tmpInputName);

        const auto tmpTensorNames = node_val->get_output_tensor(0).get_names();
        tmp_node.tensor_names_count = tmpTensorNames.size();
        for (auto tensor_name : tmpTensorNames | indexed) {
            copy_str(tmp_node.tensor_names[tensor_name.index()], tensor_name.value());
        }

        auto shape = node_val->get_output_partial_shape(0).get_shape();
        tmp_node.shape_size = shape.size();

        for (const auto& sh_iterator : shape | indexed) {
            tmp_node.shape[sh_iterator.index()] = sh_iterator.value();
        }

        metadata.ov_results[index] = tmp_node;
    }

    // preprocess info

    metadata.pre_process_info_count = preprocessInfo.size();
    for (const auto& pr : preprocessInfo | indexed) {
        auto pr_val = pr.value();

        elf::PreprocessingInfo tmp_preprocessInfo;

        copy_str(tmp_preprocessInfo.input_name, pr_val._inputName);

        tmp_preprocessInfo.input_format = ELF::mapPreProcessColorFormat.at(pr_val._inputFormat);
        tmp_preprocessInfo.output_format = ELF::mapPreProcessColorFormat.at(pr_val._outputFormat);
        tmp_preprocessInfo.algorithm = ELF::mapPreProcessResizeAlgorithm.at(pr_val._algorithm);

        metadata.pre_process_info[pr.index()] = tmp_preprocessInfo;
    }

    return metadata;
}
