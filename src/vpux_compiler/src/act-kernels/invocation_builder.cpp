//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/act_kernels/invocation_builder.h"

#include <kernels/inc/common_types.h>

#include <vpux/compiler/dialect/VPUIP/ops.hpp>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>

using namespace vpux;

void InvocationBuilder::parseBasicAttrTypes(mlir::Attribute attr) {
    if (auto val = attr.dyn_cast_or_null<mlir::IntegerAttr>()) {
        appendValue(_scalarStorage, val.getValue().getSExtValue());
    } else if (auto val = attr.dyn_cast_or_null<mlir::FloatAttr>()) {
        appendValue(_scalarStorage, static_cast<float>(val.getValue().convertToDouble()));
    } else {
        VPUX_THROW("Act Shave Invocation: cannot store arg of type {0}", attr.getType());
    }
}

void InvocationBuilder::addArg(mlir::Attribute attr) {
    if (auto arr = attr.dyn_cast_or_null<mlir::ArrayAttr>()) {
        auto vals = arr.getValue();
        for (auto val : vals) {
            parseBasicAttrTypes(val);
        }
    } else {
        parseBasicAttrTypes(attr);
    }
}

SmallVector<uint8_t> InvocationBuilder::store() const {
    SmallVector<uint8_t> serialStorage(_scalarStorage.begin(), _scalarStorage.end());
    serialStorage.insert(serialStorage.end(), _arrayStorage.begin(), _arrayStorage.end());

    const auto patchBase = _win_e_offset + _scalarStorage.size() + mvds::nce2p7::ACT_KERNEL_DATA_WINDOW;
    for (auto& patchCallback : _deferredPointers) {
        patchCallback(serialStorage, patchBase);
    }

    return serialStorage;
}

sw_params::DataType mvDTypeToDataType(const MVCNN::DType& mvDType) {
    switch (mvDType) {
    case MVCNN::DType::DType_FP64:
        return sw_params::DataType::NN_FP64;
    case MVCNN::DType::DType_FP32:
        return sw_params::DataType::NN_FP32;
    case MVCNN::DType::DType_FP16:
        return sw_params::DataType::NN_FP16;
    case MVCNN::DType::DType_BFP16:
        return sw_params::DataType::NN_BF16;
    case MVCNN::DType::DType_FP8:
        return sw_params::DataType::NN_FP8;
    case MVCNN::DType::DType_U64:
        return sw_params::DataType::NN_U64;
    case MVCNN::DType::DType_U32:
        return sw_params::DataType::NN_U32;
    case MVCNN::DType::DType_U16:
        return sw_params::DataType::NN_U16;
    case MVCNN::DType::DType_U8:
        return sw_params::DataType::NN_U8;
    case MVCNN::DType::DType_I64:
        return sw_params::DataType::NN_I64;
    case MVCNN::DType::DType_I32:
        return sw_params::DataType::NN_I32;
    case MVCNN::DType::DType_I16:
        return sw_params::DataType::NN_I16;
    case MVCNN::DType::DType_I8:
        return sw_params::DataType::NN_I8;
    case MVCNN::DType::DType_I4:
        return sw_params::DataType::NN_I4;
    case MVCNN::DType::DType_I2:
        return sw_params::DataType::NN_I2;
    case MVCNN::DType::DType_BIN:
        return sw_params::DataType::NN_BIN;
    case MVCNN::DType::DType_NOT_SET:
        return sw_params::DataType::NN_UNDEFINED;
    default:
        VPUX_THROW("Data type handling is not implemented {0}", mvDType);
    }
}

void InvocationBuilder::addTensorArg(mlir::Value value, const MVCNN::TensorReference* tensorRef) {
    VPUX_THROW_UNLESS(tensorRef != nullptr, "Got NULL tensor reference");

    sw_params::MemRefData memrefData{};

    const auto shape = getShape(value);
    memrefData.numDims = checked_cast<uint32_t>(shape.size());

    // order
    const auto inOrder = DimsOrder::fromValue(value);
    const auto memShape = inOrder.toMemoryOrder(shape);
    memrefData.dimsOrder = inOrder.invertedCode();

    // dims
    const auto dimsPatcher = [](sw_params::MemRefData& memrefData, uint32_t updateTo) {
        memrefData.dimsAddr = updateTo;
    };
    _deferredPointers.push_back(createPatchPoint<sw_params::MemRefData>(dimsPatcher));

    for (auto& dim : memShape | reversed) {
        appendValue(_arrayStorage, checked_cast<int32_t>(dim));
    }

    // strides
    auto stridesPatcher = [](sw_params::MemRefData& memrefData, uint32_t updateTo) {
        memrefData.stridesAddr = updateTo;
    };
    _deferredPointers.push_back(createPatchPoint<sw_params::MemRefData>(stridesPatcher));

    const auto memStrides = getMemStrides(value);
    for (auto&& stride : memStrides | reversed) {
        appendValue(_arrayStorage, stride);
    }

    const auto indirectDataRef = tensorRef->data();
    VPUX_THROW_UNLESS(indirectDataRef != nullptr, "Got NULL indirect data reference");

    const auto addr = indirectDataRef->data_index() + tensorRef->leading_offset();

    auto type = value.getType().dyn_cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non vpux::NDTypeInterface '{1}'", value, value.getType());

    auto memKind = type.getMemoryKind();
    // for VPU2.7 only windows within CMX memory supported, so lets guess in which tile this tensor lay
    VPUX_THROW_UNLESS(memKind == VPU::MemoryKind::CMX_NN, "Tensor arg '{0}' should be of CMX_NN memkind, but '{1}'",
                      value, memKind);

    auto memspace = type.getMemSpace();
    VPUX_THROW_UNLESS(memspace != nullptr, "Value '{0}' has non memspace attribute'{1}'", value, value.getType());

    auto mayBeIndex = memspace.getIndex();
    VPUX_THROW_UNLESS(mayBeIndex.hasValue(), "Value '{0}' has no memspace index", value);

    memrefData.dataAddr = checked_cast<uint32_t>(mvds::nce2p7::ACT_KERNEL_CMX_WINDOW +
                                                 mayBeIndex.getValue() * mvds::nce2p7::CMX_SLICE_SIZE + addr);
    memrefData.dataType = mvDTypeToDataType(tensorRef->data_dtype());
    memrefData.location = sw_params::NN_CMX;

    appendValue(_scalarStorage, memrefData);
}
