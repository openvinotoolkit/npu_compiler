//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"
#include "vpux/utils/core/error.hpp"

#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"

namespace ov::test::subgraph {
enum class QuantType { PER_CHANNEL, PER_TENSOR, UNDEFINED };
using MatMulMixedPrecisionTestParams = std::tuple<ov::Shape, ov::Shape, QuantType>;

class MatMulMixedPrecisionTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<MatMulMixedPrecisionTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulMixedPrecisionTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        VPUX_THROW_UNLESS(inputShapes.size() == 1, "Only 1 input shape is supported");
        const auto& funcInputs = function->inputs();
        VPUX_THROW_UNLESS(funcInputs.size() == 1, "Only 1 input is supported");
        const auto& inputStaticShape = inputShapes[0];
        const auto totalSize =
                std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), 1, std::multiplies<size_t>());
        auto inputTensor = ov::Tensor{ov::element::f32, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
        for (size_t i = 0; i < totalSize; i++) {
            inputData[i] = std::sin(i);
        }
        inputs = {{funcInputs[0].get_node_shared_ptr(), inputTensor}};
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 1);
        ASSERT_EQ(expectedTensors.size(), 1);

        const auto& expected = expectedTensors[0];
        const auto& actual = actualTensors[0];
        ASSERT_EQ(expected.get_size(), actual.get_size());

        const float absThreshold = 0.05f;
        ov::test::utils::compare(actual, expected, absThreshold);
    }

    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam());
        const auto weightsShape = std::get<1>(GetParam());
        auto scalesShape = ov::Shape(weightsShape.size(), 1);
        scalesShape[0] = weightsShape[0];
        init_input_shapes({ov::test::InputShape{{}, std::vector<ov::Shape>{inputShape}}});
        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        const auto weightsSize =
                std::accumulate(weightsShape.cbegin(), weightsShape.cend(), 1, std::multiplies<size_t>());
        const auto scalesSize = std::accumulate(scalesShape.cbegin(), scalesShape.cend(), 1, std::multiplies<size_t>());

        std::vector<int8_t> weightsValues(weightsSize);
        std::vector<float> scaleValues(scalesSize, 1.0f / std::numeric_limits<int8_t>::max());
        for (size_t i = 0; i < weightsValues.size(); ++i) {
            weightsValues[i] = std::numeric_limits<int8_t>::min() + (i % std::numeric_limits<uint8_t>::max());
        }
        const auto weightsConst = ov::op::v0::Constant::create(ov::element::i8, weightsShape, weightsValues);
        const auto scalesConst = ov::op::v0::Constant::create(ov::element::f32, scalesShape, scaleValues);
        const auto convertWeights = std::make_shared<ov::opset1::Convert>(weightsConst->output(0), ov::element::f32);
        const auto scaleWeights =
                std::make_shared<ov::opset1::Multiply>(convertWeights->output(0), scalesConst->output(0));
        const auto matMul =
                std::make_shared<ov::opset1::MatMul>(params[0]->output(0), scaleWeights->output(0), false, true);
        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(matMul->output(0))};
        function = std::make_shared<ov::Model>(results, params, "MatMulMixedPrecision");
    }
};

//
// Platform test definition
//

TEST_P(MatMulMixedPrecisionTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(MatMulMixedPrecisionTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(MatMulMixedPrecisionTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(smoke_MatMulMixedPrecisionOneICSplit, MatMulMixedPrecisionTestCommon,
                         ::testing::Values(MatMulMixedPrecisionTestParams{
                                 {1, 1, 11008},        // input shape
                                 {4096, 11008},        // weights shape
                                 QuantType::UNDEFINED  // field irrelevant for this test case
                         }),
                         MatMulMixedPrecisionTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulMixedPrecisionTwoICSplit, MatMulMixedPrecisionTestCommon,
                         ::testing::Values(MatMulMixedPrecisionTestParams{
                                 {1, 1, 20000},        // input shape
                                 {4096, 20000},        // weights shape
                                 QuantType::UNDEFINED  // field irrelevant for this test case
                         }),
                         MatMulMixedPrecisionTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulMixedPrecisionOneICSplitOneOCSplit, MatMulMixedPrecisionTestCommon,
                         ::testing::Values(MatMulMixedPrecisionTestParams{
                                 {1, 1, 11008},        // input shape
                                 {11008, 11008},       // weights shape
                                 QuantType::UNDEFINED  // field irrelevant for this test case
                         }),
                         MatMulMixedPrecisionTestCommon::getTestCaseName);

// Quantized Output tests

class MatMulMixedPrecisionTestOutputQuantized : public MatMulMixedPrecisionTestCommon {
public:
    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam());
        const auto weightsShape = std::get<1>(GetParam());
        const auto outQuantType = std::get<2>(GetParam());
        assert((outQuantType == QuantType::PER_CHANNEL || outQuantType == QuantType::PER_TENSOR) &&
               "output quantization type must either be per tensor or per channel for this test");

        auto scalesShape = ov::Shape(weightsShape.size(), 1);
        scalesShape[0] = weightsShape[0];
        init_input_shapes({ov::test::InputShape{{}, std::vector<ov::Shape>{inputShape}}});
        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        const auto weightsSize =
                std::accumulate(weightsShape.cbegin(), weightsShape.cend(), 1, std::multiplies<size_t>());
        const auto scalesSize = std::accumulate(scalesShape.cbegin(), scalesShape.cend(), 1, std::multiplies<size_t>());

        std::vector<int8_t> weightsValues(weightsSize);
        std::vector<float> scaleValues(scalesSize, 2.0f / std::numeric_limits<int8_t>::max());
        for (size_t i = 0; i < weightsValues.size(); ++i) {
            weightsValues[i] = std::numeric_limits<int8_t>::min() + (i % std::numeric_limits<uint8_t>::max());
        }

        const auto weightsConst = ov::op::v0::Constant::create(ov::element::i8, weightsShape, weightsValues);
        const auto scalesConst = ov::op::v0::Constant::create(ov::element::f32, scalesShape, scaleValues);
        const auto convertWeights = std::make_shared<ov::opset1::Convert>(weightsConst->output(0), ov::element::f32);
        const auto scaleWeights =
                std::make_shared<ov::opset1::Multiply>(convertWeights->output(0), scalesConst->output(0));
        const auto matMul =
                std::make_shared<ov::opset1::MatMul>(params[0]->output(0), scaleWeights->output(0), false, true);

        const auto matMulOutShape = matMul->output(0).get_shape();
        auto cstShape = std::vector<size_t>(matMulOutShape.begin(), matMulOutShape.end());
        std::vector<float> lowVals;
        std::vector<float> highVals;

        if (outQuantType == QuantType::PER_TENSOR) {
            cstShape.back() = 1;
            lowVals.push_back(-2.0f);
            highVals.push_back(2.0f);
        } else if (outQuantType == QuantType::PER_CHANNEL) {
            lowVals.reserve(cstShape.back());
            highVals.reserve(cstShape.back());

            for (size_t idx = 0; idx < cstShape.back(); ++idx) {
                const float lowVal = -2.0f - static_cast<float>(idx % 2);
                lowVals.push_back(lowVal);
                const float highVal = 2.0f + static_cast<float>(idx % 2);
                highVals.push_back(highVal);
            }
        }

        auto fq = ov::test::utils::make_fake_quantize(matMul->output(0), ov::element::f32, 256, cstShape, lowVals,
                                                      highVals, lowVals, highVals);
        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(fq)};
        function = std::make_shared<ov::Model>(results, params, "MatMulMixedPrecisionOutputScale");
    }
};

//
// Platform test definition
//

TEST_P(MatMulMixedPrecisionTestOutputQuantized, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(MatMulMixedPrecisionTestOutputQuantized, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(MatMulMixedPrecisionTestOutputQuantized, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(smoke_MatMulMixedPrecisionOutputQuantized, MatMulMixedPrecisionTestOutputQuantized,
                         ::testing::Values(
                                 MatMulMixedPrecisionTestParams{
                                         {1, 1, 20000},         // input shape
                                         {16, 20000},           // weights shape
                                         QuantType::PER_TENSOR  // output is quantized per tensor
                                 },
                                 MatMulMixedPrecisionTestParams{
                                         {1, 1, 11008},         // input shape
                                         {16, 11008},           // weights shape
                                         QuantType::PER_TENSOR  // output is quantized per tensor
                                 },

                                 MatMulMixedPrecisionTestParams{
                                         {1, 1, 20000},          // input shape
                                         {16, 20000},            // weights shape
                                         QuantType::PER_CHANNEL  // output is quantized per channel
                                 },
                                 MatMulMixedPrecisionTestParams{
                                         {1, 1, 11008},          // input shape
                                         {16, 11008},            // weights shape
                                         QuantType::PER_CHANNEL  // output is quantized per channel
                                 }),
                         MatMulMixedPrecisionTestOutputQuantized::getTestCaseName);

}  // namespace ov::test::subgraph
