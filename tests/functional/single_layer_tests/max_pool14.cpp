// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/opsets/opset14_decl.hpp>
#include <openvino/opsets/opset3_decl.hpp>
#include <shared_test_classes/base/ov_subgraph.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/max_pool.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
namespace ov::test {

using MaxPoolV14TestParams = std::tuple<ov::Shape, ov::element::Type, ov::Strides, std::vector<size_t>,
                                        std::vector<size_t>, std::vector<size_t>, std::vector<size_t>,
                                        ov::op::RoundingType, ov::op::PadType, int32_t, bool, std::string>;

class MaxPoolV14LayerTestCommon : public testing::WithParamInterface<MaxPoolV14TestParams>, public VpuOv2LayerTest {
    void SetUp() override {
        const auto& [inputShape, type, strides, dilation, padBegin, padEnd, kernel, roundingType, padType, axis,
                     twoOutputs, _] = this->GetParam();

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{std::make_shared<ov::opset3::Parameter>(type, inputDynamicShapes.at(0))};

        const auto pooling =
                std::make_shared<ov::opset14::MaxPool>(params.at(0), strides, dilation, padBegin, padEnd, kernel,
                                                       roundingType, padType, ov::element::i32, axis);

        ov::ResultVector results;
        if (twoOutputs) {
            results = {std::make_shared<ov::opset3::Result>(pooling->output(0)),
                       std::make_shared<ov::opset3::Result>(pooling->output(1))};
        } else {
            results = {std::make_shared<ov::opset3::Result>(pooling->output(0))};
        }
        function = std::make_shared<ov::Model>(results, params, "MaxPoolV14Test");
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<MaxPoolV14TestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;

        obj.param;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;

        const auto& [inputShape, type, padType, dilation, kernel, padBegin, padEnd, roundingType, strides, axis,
                     twoOutputs, device] = obj.param;
        result << "InputShape=" << ov::test::utils::vec2str(inputShape) << "_";
        result << "InputType=" << type << "_";
        result << "Strides=" << strides << "_";
        result << "Dilation=" << ov::test::utils::vec2str(dilation) << "_";
        result << "PadBegin=" << ov::test::utils::vec2str(padBegin) << "_";
        result << "PadEnd=" << ov::test::utils::vec2str(padEnd) << "_";
        result << "Kernel=" << ov::test::utils::vec2str(kernel) << "_";
        result << "RoundingType=" << roundingType << "_";
        result << "PadType=" << padType << "_";
        result << "Axis=" << axis << "_";
        result << "Device=" << device;
        return result.str();
    };
};

TEST_P(MaxPoolV14LayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(MaxPoolV14LayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
}  // namespace ov::test

using ov::Shape;
using ov::Strides;
using ov::element::Type;
using ov::op::PadType;
using ov::op::RoundingType;
using ov::test::MaxPoolV14LayerTestCommon;
using AttrType = std::vector<size_t>;

const std::vector<ov::Strides> strides = {ov::Strides{2, 2}};
const std::vector<AttrType> dilation = {{1, 1}};
const std::vector<AttrType> padBegin = {{0, 0}};
const std::vector<AttrType> padEnd = {{0, 0}};
const std::vector<AttrType> kernel = {{3, 3}};
const std::vector<Type> inType = {ov::element::f32};

auto combineMaxPoolV14Params(Shape inputShape, std::vector<Type> inType, std::vector<ov::Strides> strides,
                             std::vector<AttrType> dilation, std::vector<AttrType> padBegin,
                             std::vector<AttrType> padEnd, std::vector<AttrType> kernel, RoundingType roundingType,
                             PadType padType, int32_t axis, bool twoOutputs) {
    return ::testing::Combine(
            /* inputShape= */ ::testing::ValuesIn({inputShape}),
            /* inType= */ ::testing::ValuesIn(inType),
            /* strides= */ ::testing::ValuesIn(strides),
            /* dilation= */ ::testing::ValuesIn(dilation),
            /* padBegin= */ ::testing::ValuesIn(padBegin),
            /* padEnd= */ ::testing::ValuesIn(padEnd),
            /* kernel= */ ::testing::ValuesIn(kernel),
            /* roundingType= */ ::testing::Values(roundingType),
            /* padType= */ ::testing::Values(padType),
            /* axis= */ ::testing::Values(axis),
            /* twoOutputs= */ ::testing::Values(twoOutputs),  // when set to false, the MaxPool14 op created is further
                                                              // converted to MaxPool8, later to MaxPool1 and executed
                                                              // on DPU. Otherwise, it is executed on SHAVE as MaxPool8.
            /* device= */ ::testing::Values(DEVICE_NPU));
}

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV14Test, MaxPoolV14LayerTestCommon,
                         combineMaxPoolV14Params(ov::Shape{1, 3, 30, 30}, inType, strides, dilation, padBegin, padEnd,
                                                 kernel, RoundingType::FLOOR, PadType::AUTO, 0, true),
                         MaxPoolV14LayerTestCommon::getTestCaseName);
// GoogleNet model case
INSTANTIATE_TEST_SUITE_P(smoke_GoogleNetCase_MaxPoolV14Test, MaxPoolV14LayerTestCommon,
                         combineMaxPoolV14Params(ov::Shape{1, 64, 112, 112}, inType, strides, dilation, padBegin,
                                                 padEnd, kernel, RoundingType::CEIL_TORCH, PadType::EXPLICIT, 2, true),
                         MaxPoolV14LayerTestCommon::getTestCaseName);
// MaxPoolV14 to MaxPoolV8 to MaxPoolV1 executed on DPU
INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV14ToMaxPoolV8ConversionTest, MaxPoolV14LayerTestCommon,
                         combineMaxPoolV14Params(ov::Shape{1, 128, 20, 20}, inType, {ov::Strides{1, 1}}, dilation,
                                                 padBegin, padEnd, kernel, RoundingType::FLOOR, PadType::EXPLICIT, 2,
                                                 false),
                         MaxPoolV14LayerTestCommon::getTestCaseName);
