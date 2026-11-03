//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/mixed_precision_convolution.hpp"
#include <random>
#include <vector>

#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

std::string MixedPrecisionConvSubGraphTest::getTestCaseName(
        const testing::TestParamInfo<mixedPrecisionConvSubGraphTestParamsSet>& obj) {
    mixedPrecisionConvSpecificParams mixedPrecisionConvParams;
    ov::element::Type modelType;
    ov::Shape inputShapes;
    std::string targetDevice;

    std::tie(mixedPrecisionConvParams, modelType, inputShapes, targetDevice) = obj.param;

    ov::op::PadType padType = ov::op::PadType::AUTO;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    LowFpType lowFpType;
    size_t quantLevels;
    QuantizationGranularity quantGranularity;

    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, lowFpType, quantLevels, quantGranularity) =
            mixedPrecisionConvParams;

    const std::string sep = "_";

    std::ostringstream result;

    result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
    result << "TestIdx=" << obj.index << sep;

    result << "IS=" << vec2str(inputShapes) << sep;
    result << "K" << vec2str(kernel) << sep;
    result << "S" << vec2str(stride) << sep;
    result << "PB" << vec2str(padBegin) << sep;
    result << "PE" << vec2str(padEnd) << sep;
    result << "D=" << vec2str(dilation) << sep;
    result << "O=" << convOutChannels << sep;
    result << "AP=" << padType << sep;
    result << "LowFpType=" << lowFpType2String(lowFpType) << sep;
    result << "Levels=" << quantLevels << sep;
    result << "QG=" << quantGranularity << sep;
    result << "netPRC=" << modelType << sep;
    result << "device=" << targetDevice;

    return result.str();
}

void MixedPrecisionConvSubGraphTest::SetUp() {
    abs_threshold = 1.0f;

    mixedPrecisionConvSpecificParams mixedPrecisionConvParams;
    std::vector<size_t> inputShape;
    auto modelType = ov::element::dynamic;

    std::tie(mixedPrecisionConvParams, modelType, inputShape, std::ignore) = this->GetParam();

    ov::op::PadType padType = ov::op::PadType::AUTO;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    LowFpType lowFpType;
    size_t quantLevels;
    QuantizationGranularity quantGranularity;

    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, lowFpType, quantLevels, quantGranularity) =
            mixedPrecisionConvParams;

    init_input_shapes(static_shapes_to_test_representation({inputShape}));

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes.front())};

    std::vector<size_t> weightsShapes = {convOutChannels, inputShape[1], kernel[0], kernel[1]};

    std::mt19937 intMersenneEngine(0);
    std::uniform_int_distribution<> int8Dist(-127, 127);
    std::uniform_int_distribution<> int4Dist(-7, 7);
    std::uniform_int_distribution<> nf4Dist(0, 15);
    auto intGen = [&int8Dist, &int4Dist, &nf4Dist, &intMersenneEngine, lowFpType, quantLevels]() {
        if (lowFpType != LowFpType::Undefined) {
            switch (lowFpType) {
            case LowFpType::NF4:
                return static_cast<int8_t>(std::round(nf4Dist(intMersenneEngine)));
            default:
                VPUX_THROW("unknown low fp type!");
            }
        } else {
            if (quantLevels == 16) {
                return static_cast<int8_t>(std::round(int4Dist(intMersenneEngine)));
            } else {
                return static_cast<int8_t>(std::round(int8Dist(intMersenneEngine)));
            }
        }
    };

    std::vector<int8_t> weightsData(ov::shape_size(weightsShapes));
    std::generate(weightsData.begin(), weightsData.end(), intGen);

    std::shared_ptr<ov::op::v0::Constant> weightsConst;
    if (lowFpType != LowFpType::Undefined) {
        switch (lowFpType) {
        case LowFpType::NF4:
            weightsConst = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::nf4, weightsShapes, weightsData);
            break;
        default:
            VPUX_THROW("unknown low fp type!");
        }
    } else {
        weightsConst = quantLevels != 16
                               ? std::make_shared<ov::op::v0::Constant>(ov::element::i8, weightsShapes, weightsData)
                               : std::make_shared<ov::op::v0::Constant>(ov::element::i4, weightsShapes, weightsData);
    }
    const auto weightsConvert = std::make_shared<ov::op::v0::Convert>(weightsConst, ov::element::f16);

    std::vector<double> multiplyData(weightsShapes[0], 0.078740157480314959);
    const auto multiplyConst = std::make_shared<ov::op::v0::Constant>(
            ov::element::f16, ov::Shape({weightsShapes[0], 1, 1, 1}), multiplyData);
    const auto weightsMultiply = std::make_shared<ov::op::v1::Multiply>(weightsConvert, multiplyConst);

    auto conv = std::make_shared<ov::op::v1::Convolution>(params[0], weightsMultiply, stride, padBegin, padEnd,
                                                          dilation, padType);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv)};
    function = std::make_shared<ov::Model>(results, params, "MixedPrecisionConvolution");
}
}  // namespace test
}  // namespace ov
