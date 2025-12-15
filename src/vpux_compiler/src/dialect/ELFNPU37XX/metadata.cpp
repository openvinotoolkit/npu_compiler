//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux_headers/metadata_primitives.hpp"

#include <intel_npu/prefix.hpp>

#include <llvm/Support/Format.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>

using namespace vpux;

static void copy_str(char* dst, const std::string& src, bool throwOnErr = false) {
    VPUX_THROW_WHEN(throwOnErr && (src.size() >= elf::MAX_STRING_LEN), "Target char array is too small");
    auto str_len = src.size() < elf::MAX_STRING_LEN ? src.size() : elf::MAX_STRING_LEN - 1;

    memcpy(dst, src.data(), str_len);
    dst[str_len] = '\0';
}

elf::DType ELFNPU37XX::createDType(mlir::Type type) {
    if (type.isF64()) {
        return elf::DType::DType_FP64;
    } else if (type.isF32()) {
        return elf::DType::DType_FP32;
    } else if (type.isF16()) {
        return elf::DType::DType_FP16;
    } else if (type.isBF16()) {
        return elf::DType::DType_BFP16;
    } else if (mlir::isa<mlir::Float8E4M3FNType>(type)) {
        return elf::DType::DType_F8E4M3FN;
    } else if (mlir::isa<mlir::Float8E5M2Type>(type)) {
        return elf::DType::DType_F8E5M2;
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
    } else if (type.isSignedInteger(2)) {
        return elf::DType::DType_I2;
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
        return elf::DType::DType_I2X;
    } else if (type.isInteger(1)) {
        return elf::DType::DType_BIN;
    } else if (mlir::isa<mlir::quant::QuantizedType>(type)) {
        return createDType(mlir::cast<mlir::quant::QuantizedType>(type).getStorageType());
    } else if (mlir::isa<type::QuantileFloatType>(type)) {
        return elf::DType::DType_I4X;
    } else {
        VPUX_THROW("Unsupported element type {0}", type);
    }
}

elf::TensorRef ELFNPU37XX::createTensorRef(NDTypeInterface type, StringRef name) {
    elf::TensorRef out{};

    copy_str(out.name, name.str());

    // dtype
    out.data_type = ELFNPU37XX::createDType(type.getElementType());

    // dims
    const auto shape = type.getShape();
    out.dimensions_size = checked_cast<uint32_t>(shape.size());

    VPUX_THROW_UNLESS(shape.size() < elf::MAX_TENSOR_REF_DIMS, "Shape rank of the type {0} is too high {1} >= {2}",
                      type, shape.size(), elf::MAX_TENSOR_REF_DIMS);

    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type)) {
        auto bounds = boundedType.getBounds();
        for (auto [ind, dim] : bounds | indexed) {
            out.dimensions[ind] = checked_cast<uint32_t>(bounds[Dim(ind)]);
        }
    } else {
        for (auto [ind, dim] : shape | indexed) {
            out.dimensions[ind] = checked_cast<uint32_t>(dim);
        }
    }

    // getStrides returns upper bounds for dynamic shapes
    auto strides = type.getStrides();
    out.strides_size = checked_cast<uint32_t>(strides.size());

    Strides temp;
    temp.push_back(type.getElemTypeSize());
    temp.append(strides.begin(), strides.end());
    VPUX_THROW_UNLESS(strides.size() <= elf::MAX_TENSOR_REF_STRIDES,
                      "Too many strides for tensor '{0}'. Actual: {1}, expected not more than: {2}", name,
                      strides.size(), elf::MAX_TENSOR_REF_STRIDES);

    for (auto iterator : temp | indexed) {
        auto val = iterator.value();
        auto index = iterator.index();

        out.strides[index] = checked_cast<uint64_t>(val.count());
    }

    // dimsOrder
    out.order = type.getDimsOrder().code();

    return out;
}

elf::TensorRef ELFNPU37XX::createTensorRef(mlir::Value val, StringRef name) {
    return createTensorRef(mlir::cast<NDTypeInterface>(val.getType()), name);
}

elf::OVNodeType ELFNPU37XX::createOVNodeType(mlir::Type type) {
    // The order of the if else statements is important, first the float types are checked, then signed integers of
    // specified length, and then all the integers, both unsigned and signless, with special cases for BOOL and 1-bit
    // integers
    if (type.isF64()) {
        return elf::OVNodeType::OVNodeType_F64;
    } else if (type.isF32()) {
        return elf::OVNodeType::OVNodeType_F32;
    } else if (type.isF16()) {
        return elf::OVNodeType::OVNodeType_F16;
    } else if (type.isBF16()) {
        return elf::OVNodeType::OVNodeType_BF16;
    } else if (mlir::isa<mlir::Float8E4M3FNType>(type)) {
        return elf::OVNodeType::OVNodeType_F8E4M3FN;
    } else if (mlir::isa<mlir::Float8E5M2Type>(type)) {
        return elf::OVNodeType::OVNodeType_F8E5M2;
    } else if (type.isSignedInteger(64)) {
        return elf::OVNodeType::OVNodeType_I64;
    } else if (type.isSignedInteger(32)) {
        return elf::OVNodeType::OVNodeType_I32;
    } else if (type.isSignedInteger(16)) {
        return elf::OVNodeType::OVNodeType_I16;
    } else if (type.isSignedInteger(8)) {
        return elf::OVNodeType::OVNodeType_I8;
    } else if (type.isSignedInteger(4)) {
        return elf::OVNodeType::OVNodeType_I4;
    } else if (type.isSignedInteger(2)) {
        return elf::OVNodeType::OVNodeType_I2;
    } else if (type.isSignlessInteger(8)) {
        // In frontend signless 8-bit integer is used for BOOL, to distinguish it from U8
        // This if else statement should come before the check for U8 to distinguish between these two types
        return elf::OVNodeType::OVNodeType_BOOLEAN;
    } else if (type.isInteger(64)) {
        return elf::OVNodeType::OVNodeType_U64;
    } else if (type.isInteger(32)) {
        return elf::OVNodeType::OVNodeType_U32;
    } else if (type.isInteger(16)) {
        return elf::OVNodeType::OVNodeType_U16;
    } else if (type.isInteger(8)) {
        return elf::OVNodeType::OVNodeType_U8;
    } else if (type.isInteger(4)) {
        return elf::OVNodeType::OVNodeType_U4;
    } else if (type.isInteger(2)) {
        return elf::OVNodeType::OVNodeType_U2;
    } else if (type.isInteger(1)) {
        // Both signed and unsigned 1-bit integers are converted to U1
        return elf::OVNodeType::OVNodeType_U1;
    } else if (mlir::isa<type::QuantileFloatType>(type)) {
        // 4 bit quantile float dtype
        return elf::OVNodeType::OVNodeType_NF4;
    } else {
        VPUX_THROW("Unsupported type : '{0}'", type);
    }
}

namespace {

void setOVNodeType(elf::OVNode& node, net::DataInfoOp dataInfo) {
    auto userType = mlir::cast<NDTypeInterface>(dataInfo.getUserType());
    node.type = ELFNPU37XX::createOVNodeType(userType.getElementType());
}

void setOVNodeNames(elf::OVNode& node, net::DataInfoOp dataInfo, const Logger& log) {
    auto primaryName = dataInfo.getName().str();

    // If the friendlyName is not set in DataInfoOp, friendlyName is equal to primary name.
    auto friendlyName = dataInfo.getFriendlyName().has_value() ? dataInfo.getFriendlyName().value().str() : primaryName;
    copy_str(node.friendly_name, friendlyName);

    // If the inputName is not set in DataInfoOp, inputName is equal to primary name.
    auto inputName = dataInfo.getInputName().has_value() ? dataInfo.getInputName().value().str() : primaryName;
    copy_str(node.input_name, inputName);

    node.tensor_names_count = 0;
    if (dataInfo.getTensorNames().has_value()) {
        const auto tmpTensorNames = dataInfo.getTensorNames().value();
        node.tensor_names_count = checked_cast<uint32_t>(tmpTensorNames.size());
        if (tmpTensorNames.size() > elf::MAX_METADATA_IO) {
            log.warning("OV Node \"{0}\" has {1} tensor names. Trimming to the maximum limit of {2}", primaryName,
                        node.tensor_names_count, elf::MAX_METADATA_IO);
            node.tensor_names_count = elf::MAX_METADATA_IO;
        }
        for (auto i : irange(node.tensor_names_count)) {
            copy_str(node.tensor_names[i], mlir::cast<mlir::StringAttr>(tmpTensorNames[i]).str());
        }
    }
}

void setOVNodeShape(elf::OVNode& node, net::DataInfoOp dataInfo) {
    // If the originalShape is not set in DataInfo, originalShape is the same as shape of userType
    auto shape = dataInfo.getOriginalShape().has_value()
                         ? mlir::cast<NDTypeInterface>(dataInfo.getOriginalShape().value()).getShape()
                         : mlir::cast<NDTypeInterface>(dataInfo.getUserType()).getShape();
    node.shape_size = checked_cast<uint32_t>(shape.size());
    for (const auto& sh_iterator : shape | indexed) {
        auto dim = sh_iterator.value();
        auto ind = sh_iterator.index();

        if (dim >= 0) {
            node.shape[ind] = checked_cast<uint64_t>(dim);
        } else if (mlir::ShapedType::isDynamic(dim)) {
            node.shape[ind] = std::numeric_limits<uint64_t>::max();
        } else {
            VPUX_THROW(
                    "Unexpected dim value {0}. It must be a positive number or mlir::ShapedType::kDynamic to represent "
                    "a dynamic dim",
                    dim);
        }
    }
}

void createOVNodes(std::vector<elf::OVNode>& nodes, ArrayRef<net::DataInfoOp> dataInfoVector, const Logger& log) {
    for (auto dataInfo : dataInfoVector) {
        // Serialize metadata only for model primary parameters and results, skip state and shape nodes
        const auto name = dataInfo.getName().str();
        if (intel_npu::isStateInputName(name) || intel_npu::isStateOutputName(name) ||
            intel_npu::isShapeTensorName(name) || intel_npu::isInitInputWeightsName(name) ||
            intel_npu::isInitOutputWeightsName(name) || intel_npu::isMainInputWeightsName(name)) {
            continue;
        }

        elf::OVNode tmpNode{};

        setOVNodeType(tmpNode, dataInfo);
        setOVNodeNames(tmpNode, dataInfo, log);
        setOVNodeShape(tmpNode, dataInfo);

        nodes.push_back(tmpNode);
    }
};

std::string stringifyOVNodeType(elf::OVNodeType val) {
    switch (val) {
    case elf::OVNodeType::OVNodeType_F64:
        return "F64";
    case elf::OVNodeType::OVNodeType_F32:
        return "F32";
    case elf::OVNodeType::OVNodeType_F16:
        return "F16";
    case elf::OVNodeType::OVNodeType_BF16:
        return "BF16";
    case elf::OVNodeType::OVNodeType_F8E4M3FN:
        return "F8E4M3FN";
    case elf::OVNodeType::OVNodeType_F8E5M2:
        return "F8E5M2";
    case elf::OVNodeType::OVNodeType_I64:
        return "I64";
    case elf::OVNodeType::OVNodeType_I32:
        return "I32";
    case elf::OVNodeType::OVNodeType_I16:
        return "I16";
    case elf::OVNodeType::OVNodeType_I8:
        return "I8";
    case elf::OVNodeType::OVNodeType_I4:
        return "I4";
    case elf::OVNodeType::OVNodeType_I2:
        return "I2";
    case elf::OVNodeType::OVNodeType_U64:
        return "U64";
    case elf::OVNodeType::OVNodeType_U32:
        return "U32";
    case elf::OVNodeType::OVNodeType_U16:
        return "U16";
    case elf::OVNodeType::OVNodeType_U8:
        return "U8";
    case elf::OVNodeType::OVNodeType_U4:
        return "U4";
    case elf::OVNodeType::OVNodeType_U2:
        return "U2";
    case elf::OVNodeType::OVNodeType_U1:
        return "U1";
    case elf::OVNodeType::OVNodeType_BOOLEAN:
        return "BOOLEAN";
    case elf::OVNodeType::OVNodeType_NF4:
        return "NF4";
    default:
        return "";
    }
}

std::string stringifyPlatform(mlir::ModuleOp module, const Logger& log) {
    auto platform = config::getPlatform(module);
    if (platform.has_value()) {
        return config::stringifyPlatform(platform.value()).str();
    }
    log.warning("Target platform is not defined. Serializing TEST blob.");
    // Unknown platform indicates test configuration
    return "TEST";
}

std::string namesToString(elf::TensorName* names, uint32_t size) {
    std::stringstream names_str_stream;
    bool first = true;
    for (uint32_t i = 0; i < size; i++) {
        if (!first) {
            names_str_stream << ", ";
        }
        names_str_stream << i << ":\"" << names[i] << "\"";
        first = false;
    }
    return names_str_stream.str();
}

std::string shapeToString(uint64_t* shape, uint32_t size) {
    std::stringstream shape_str_stream;
    shape_str_stream << "[";
    bool first = true;
    for (uint32_t i = 0; i < size; i++) {
        if (!first) {
            shape_str_stream << ",";
        }
        shape_str_stream << shape[i];
        first = false;
    }
    shape_str_stream << "]";
    return shape_str_stream.str();
}

void printOVNodes(const std::vector<elf::OVNode>& nodes, const Logger& log) {
    for (const auto& p : nodes | indexed) {
        auto node = p.value();
        log.debug("{0}:friendly_name: \"{1}\"", llvm::format_decimal(p.index(), 3), node.friendly_name);
        log.nest(2).debug("input_name: \"{0}\"", node.input_name);
        log.nest(2).debug("tensor_names: {0}", namesToString(node.tensor_names, node.tensor_names_count));
        log.nest(2).debug("shape: {0}", shapeToString(node.shape, node.shape_size));
        log.nest(2).debug("type: {0}", stringifyOVNodeType(node.type));
    }
}

// Metadata parameter passed as pointer due to large size of `elf::NetworkMetadata` structure
void printMetadata(elf::NetworkMetadata* metadata, const Logger& log) {
    log.debug("mOVParameters:");
    printOVNodes(metadata->mOVParameters, log);

    log.debug("mOVResults:");
    printOVNodes(metadata->mOVResults, log);
}

}  // namespace

std::unique_ptr<elf::NetworkMetadata> ELFNPU37XX::constructMetadata(mlir::ModuleOp module, Logger log) {
    log.setName("constructMetadata");

    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp netFunc;
    net::NetworkInfoOp::getFromModule(module, netInfo, netFunc);

    auto inputsInfo = netInfo.getInputsDataInfo();
    auto outputsInfo = netInfo.getOutputsDataInfo();
    auto profilingOutputsInfo = netInfo.getProfilingOutputsDataInfo();

    // We are returning a unique_ptr to the heap allocated metadata due to its large size.
    // Returning the metadata struct by value can cause a stack overflow on certain systems.
    auto metadataPtr = std::make_unique<elf::NetworkMetadata>();
    auto& metadata = *metadataPtr.get();

    // Copy arch_name and throw if it doesn't fit into the buffer.
    // arch_name must not be truncated to ensure proper operation of the ELF loader.
    copy_str(metadata.mIdentification.arch_name, stringifyPlatform(module, log), true);
    // Copy blob_name and throw if it doesn't fit into the buffer.
    // blob_name must not be truncated to ensure proper operation of the driver.
    copy_str(metadata.mIdentification.blob_name, module.getName().value_or("network").str(), true);

    metadata.mNetInputs.resize(inputsInfo.size());
    metadata.mInTensorDescriptors.resize(inputsInfo.size());

    metadata.mNetOutputs.resize(outputsInfo.size());
    metadata.mOutTensorDescriptors.resize(outputsInfo.size());

    metadata.mProfilingOutputs.resize(profilingOutputsInfo.size());

    const auto architecture = config::getArch(module);

    const bool isLLVMMainForHostCompile =
            (config::getCompilationMode(module) == config::CompilationMode::HostCompile) &&
            (module->getParentOfType<mlir::ModuleOp>() == nullptr);

    auto setTensor = [&](elf::TensorRef& netInput, elf::TensorRef& tensorDesc, NDTypeInterface type,
                         net::DataInfoOp userInfo) {
        const auto userType = mlir::cast<NDTypeInterface>(userInfo.getUserType());

        // For dynamic shape, userType is required as it has both size and bounds.
        netInput = createTensorRef(isLLVMMainForHostCompile ? userType : type, userInfo.getName());
        tensorDesc = createTensorRef(userType, userInfo.getName());
    };

    if (architecture >= config::ArchKind::NPU40XX) {
        if (!isLLVMMainForHostCompile) {
            auto ioBindings = VPUASM::IOBindingsOp::getFromModule(module);
            auto inputDeclarations =
                    to_small_vector(ioBindings.getInputDeclarations().front().getOps<VPUASM::DeclareBufferOp>());

            for (const auto& p : inputsInfo | indexed) {
                const auto index = checked_cast<uint32_t>(p.index());
                auto inputDeclaration = inputDeclarations[index];
                auto declaredInputType = mlir::cast<NDTypeInterface>(inputDeclaration.getBufferType().getMemref());

                setTensor(metadata.mNetInputs[index], metadata.mInTensorDescriptors[index], declaredInputType,
                          p.value());
            }

            auto outDeclarations =
                    to_small_vector(ioBindings.getOutputDeclarations().front().getOps<VPUASM::DeclareBufferOp>());
            for (const auto& p : outputsInfo | indexed) {
                const auto index = p.index();
                auto outDeclaration = outDeclarations[index];
                auto declaredOutType = mlir::cast<NDTypeInterface>(outDeclaration.getBufferType().getMemref());

                setTensor(metadata.mNetOutputs[index], metadata.mOutTensorDescriptors[index], declaredOutType,
                          p.value());
            }

            // profiling
            auto profilingDeclarations = to_small_vector(
                    ioBindings.getProfilingBuffDeclarations().front().getOps<VPUASM::DeclareBufferOp>());
            for (const auto& p : profilingOutputsInfo | indexed) {
                const auto index = p.index();
                auto profilingDeclaration = profilingDeclarations[index];

                auto declaredProfileBuffType =
                        mlir::cast<NDTypeInterface>(profilingDeclaration.getBufferType().getMemref());

                metadata.mProfilingOutputs[index] = createTensorRef(declaredProfileBuffType, p.value().getName());
            }
        } else {
            // Host Compile does not create IOBindingsOp in its pipeline for a entry function
            // Will check if IOBindingsOp can be created.

            // input

            for (const auto& p : inputsInfo | indexed) {
                const auto index = checked_cast<uint32_t>(p.index());
                // Refer to userType as it provides all information (shape, bounds for dynamic shape)
                // No need to pass a function arg
                setTensor(metadata.mNetInputs[index], metadata.mInTensorDescriptors[index], {}, p.value());
            }

            // output
            for (const auto& p : outputsInfo | indexed) {
                const auto index = p.index();
                // Refer to userType as it provides all information (shape, bounds for dynamic shape)
                // No need to pass a function arg
                setTensor(metadata.mNetOutputs[index], metadata.mOutTensorDescriptors[index], {}, p.value());
            }

            // Currently, profiling feature is not supported
            VPUX_THROW_UNLESS(profilingOutputsInfo.size() == 0, "Profiling is not supported for HostCompile Mode");
        }
    } else {
        // input
        for (const auto& p : inputsInfo | indexed) {
            const auto index = checked_cast<uint32_t>(p.index());
            setTensor(metadata.mNetInputs[index], metadata.mInTensorDescriptors[index],
                      netFunc.getArgument(index).getType(), p.value());
        }

        // output
        for (const auto& p : outputsInfo | indexed) {
            const auto index = p.index();
            const auto funcArgIndex = checked_cast<uint32_t>(inputsInfo.size() + index);
            setTensor(metadata.mNetOutputs[index], metadata.mOutTensorDescriptors[index],
                      netFunc.getArgument(funcArgIndex).getType(), p.value());
        }

        // profiling
        for (const auto& p : profilingOutputsInfo | indexed) {
            const auto index = p.index();
            const auto funcArgInd = inputsInfo.size() + outputsInfo.size() + index;

            const auto val = netFunc.getArgument(checked_cast<uint32_t>(funcArgInd));

            metadata.mProfilingOutputs[index] = createTensorRef(val, p.value().getName());
        }
    }

    // ov parameters
    createOVNodes(metadata.mOVParameters, inputsInfo, log);

    // ov results
    createOVNodes(metadata.mOVResults, outputsInfo, log);

    printMetadata(&metadata, log);

    return metadataPtr;
}

void vpux::ELFNPU37XX::setResourceRequirement(mlir::ModuleOp moduleOp, elf::NetworkMetadata& metadata) {
    auto nBarrs = VPUIP::getNumAvailableBarriers(moduleOp);
    metadata.mResourceRequirements.nn_barriers_ = checked_cast<uint8_t>(nBarrs);
    metadata.mResourceRequirements.nn_slice_count_ = checked_cast<uint8_t>(VPUIP::getNumTilesUsed(moduleOp));
    metadata.mResourceRequirements.nn_slice_length_ =
            checked_cast<uint32_t>(config::getAvailableMemory(moduleOp, VPU::MemoryKind::CMX_NN).getByteSize());
}
