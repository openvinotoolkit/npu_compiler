//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <converters.hpp>
#include <ie_precision.hpp>
#include <map>

const uint32_t DIM_N = 0, DIM_C = 1, DIM_H = 2, DIM_W = 3, DIM_D = 4;

static const std::map<InferenceEngine::Layout, std::vector<float>> orderMapping = {
        {InferenceEngine::Layout::NCHW, {DIM_N, DIM_C, DIM_H, DIM_W}},
        {InferenceEngine::Layout::NHWC, {DIM_N, DIM_H, DIM_W, DIM_C}},
        {InferenceEngine::Layout::NCDHW, {DIM_N, DIM_C, DIM_D, DIM_H, DIM_W}},
        {InferenceEngine::Layout::NDHWC, {DIM_N, DIM_D, DIM_H, DIM_W, DIM_C}},
        {InferenceEngine::Layout::C, {DIM_C}},
        {InferenceEngine::Layout::CHW, {DIM_C, DIM_H, DIM_W}},
        {InferenceEngine::Layout::NC, {DIM_N, DIM_C}},
};

static const std::map<InferenceEngine::Precision, MVCNN::DType> dataTypeMapping = {
        {InferenceEngine::Precision::FP32, MVCNN::DType::DType_FP32},
        {InferenceEngine::Precision::FP16, MVCNN::DType::DType_FP16},
        {InferenceEngine::Precision::BF16, MVCNN::DType::DType_BFP16},
        {InferenceEngine::Precision::U64, MVCNN::DType::DType_U64},
        {InferenceEngine::Precision::U32, MVCNN::DType::DType_U32},
        {InferenceEngine::Precision::U16, MVCNN::DType::DType_U16},
        {InferenceEngine::Precision::U8, MVCNN::DType::DType_U8},
        {InferenceEngine::Precision::I64, MVCNN::DType::DType_I64},
        {InferenceEngine::Precision::I32, MVCNN::DType::DType_I32},
        {InferenceEngine::Precision::I16, MVCNN::DType::DType_I16},
        {InferenceEngine::Precision::I8, MVCNN::DType::DType_I8},
        {InferenceEngine::Precision::BIN, MVCNN::DType::DType_BIN},
};

namespace {
InferenceEngine::Layout orderVectorToLayout(const std::vector<float>& tensorOrder) {
    std::function<bool(const std::pair<InferenceEngine::Layout, std::vector<float>>&)> mapSearchPredicate =
            [tensorOrder](const std::pair<InferenceEngine::Layout, std::vector<float>>& orderPair) -> bool {
        size_t orderSize = tensorOrder.size();
        size_t pairSize = orderPair.second.size();
        return (orderSize == pairSize) && std::equal(tensorOrder.begin(), tensorOrder.end(), orderPair.second.begin());
    };
    std::map<InferenceEngine::Layout, std::vector<float>>::const_iterator mapIter =
            std::find_if(orderMapping.begin(), orderMapping.end(), mapSearchPredicate);
    if (mapIter == orderMapping.end()) {
        IE_THROW() << "orderToLayout: failed to convert input order";
    }
    return mapIter->first;
}
}  // namespace

InferenceEngine::Layout getLayout(const MVCNN::TensorReference* const tensorRef) {
    IE_ASSERT(tensorRef != nullptr);
    const uint64_t order = tensorRef->order();
    InferenceEngine::Layout ieLayout = InferenceEngine::Layout::ANY;
    if (order == 0x1)
        ieLayout = InferenceEngine::Layout::C;
    else if (order == 0x12)
        ieLayout = InferenceEngine::Layout::NC;
    else if (order == 0x123)
        ieLayout = InferenceEngine::Layout::CHW;
    else if (order == 0x231)
        ieLayout = InferenceEngine::Layout::HWC;
    else if (order == 0x1234)
        ieLayout = InferenceEngine::Layout::NCHW;
    else if (order == 0x1342)
        ieLayout = InferenceEngine::Layout::NHWC;
    else if (order == 0x12345)
        ieLayout = InferenceEngine::Layout::NCDHW;
    else if (order == 0x13452)
        ieLayout = InferenceEngine::Layout::NDHWC;
    else {
        std::vector<float> tensorOrder;
        IE_ASSERT(tensorRef->strides() != nullptr);
        std::copy(tensorRef->strides()->cbegin(), tensorRef->strides()->cend(), std::back_inserter(tensorOrder));
        ieLayout = orderVectorToLayout(tensorOrder);
    }
    return ieLayout;
}

InferenceEngine::Precision MvcnnDTypeToPrecision(const MVCNN::DType& dtype) {
    std::function<bool(const std::pair<InferenceEngine::Precision, MVCNN::DType>&)> mapSearchPredicate =
            [dtype](const std::pair<InferenceEngine::Precision, MVCNN::DType>& dataTypePair) -> bool {
        return dtype == dataTypePair.second;
    };
    std::map<InferenceEngine::Precision, MVCNN::DType>::const_iterator mapIter =
            std::find_if(dataTypeMapping.begin(), dataTypeMapping.end(), mapSearchPredicate);
    if (mapIter == dataTypeMapping.end()) {
        IE_THROW() << "DTypeToPrecision: failed to convert dtype: " << dtype;
    }
    return mapIter->first;
}

std::vector<float> layoutToOrderVector(const InferenceEngine::Layout& tensorLayout) {
    return orderMapping.at(tensorLayout);
}

MVCNN::DType precisionToMvcnnDType(const InferenceEngine::Precision& tensorPrecision) {
    return dataTypeMapping.at(tensorPrecision);
}

mv::DType precisionToDType(const InferenceEngine::Precision& InferenceEnginePrecision) {
    mv::DType mvType;
    switch (InferenceEnginePrecision) {
    case InferenceEngine::Precision::UNSPECIFIED:
        mvType = mv::DType("Default");
        break;
    case InferenceEngine::Precision::I8:
        mvType = mv::DType("Int8");
        break;
    case InferenceEngine::Precision::U8:
        mvType = mv::DType("UInt8");
        break;
    case InferenceEngine::Precision::I32:
        mvType = mv::DType("Int32");
        break;
    case InferenceEngine::Precision::I64:
        mvType = mv::DType("Int64");
        break;
    case InferenceEngine::Precision::FP16:
        mvType = mv::DType("Float16");
        break;
    case InferenceEngine::Precision::BF16:
        mvType = mv::DType("BFloat16");
        break;
    case InferenceEngine::Precision::FP32:
        mvType = mv::DType("Float32");
        break;
    default:
        IE_THROW() << "Data type handling is not implemented" << InferenceEnginePrecision.name();
    }
    return mvType;
}

mv::Order layoutToOrder(const InferenceEngine::Layout& ieLayout) {
    std::ostringstream layoutToOrder;
    layoutToOrder << ieLayout;
    const auto ieLayoutStr = layoutToOrder.str();
    auto mvLayoutStr = layoutToOrder.str();
    const auto replaceDepth = [](const char& dim) -> char {
        if (dim != 'D') {
            return dim;
        } else {
            return 'T';
        }
    };
    std::transform(ieLayoutStr.cbegin(), ieLayoutStr.cend(), mvLayoutStr.begin(), replaceDepth);
    return mv::Order(mvLayoutStr);
}

mv::Shape sizeVectorToShape(InferenceEngine::SizeVector dims) {
    if (dims.empty()) {
        return mv::Shape({1});
    }
    std::reverse(begin(dims), end(dims));
    return mv::Shape(dims);
}

MVCNN::TargetDeviceRevision getDeviceRevision(const InferenceEngine::VPUXConfigParams::VPUXPlatform platform) {
    switch (platform) {
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400_A0:
        return MVCNN::TargetDeviceRevision::TargetDeviceRevision_A0;
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400:
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700:
        return MVCNN::TargetDeviceRevision::TargetDeviceRevision_B0;
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3800:
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3900:
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720:
    default:
        return MVCNN::TargetDeviceRevision::TargetDeviceRevision_NONE;
    }

    return MVCNN::TargetDeviceRevision::TargetDeviceRevision_NONE;
}
