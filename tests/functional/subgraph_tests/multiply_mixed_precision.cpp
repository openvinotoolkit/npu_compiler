//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Subgraph test for QuantizeWithMultiplyRewriter in ConvertToMixedPrecisionPass.
// Verifies that IE.Multiply(fp16, fp16) -> IE.Quantize(u8/i8) is fused into a
// single IE.Multiply(fp16, fp16) with quantized output.

#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include <cmath>
#include <random>

namespace ov::test::subgraph {

// Shape for both inputs to Multiply and FQ output range {inLow, inHigh, outLow, outHigh}.
using MultiplyMixedPrecisionTestParams = std::tuple<ov::Shape, std::vector<float>>;

class MultiplyMixedPrecisionTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<MultiplyMixedPrecisionTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MultiplyMixedPrecisionTestParams>& obj) {
        ov::Shape inputShape;
        std::vector<float> fqRanges;
        std::tie(inputShape, fqRanges) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InputShape={";
        for (size_t i = 0; i < inputShape.size(); ++i) {
            result << inputShape[i];
            if (i + 1 < inputShape.size()) {
                result << "x";
            }
        }
        result << "}" << sep;
        result << "FQ={" << fqRanges.at(0) << "," << fqRanges.at(1) << "," << fqRanges.at(2) << "," << fqRanges.at(3)
               << "}" << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        const auto& fqRanges = std::get<1>(GetParam());

        const float outLow = fqRanges.at(2);
        const float outHigh = fqRanges.at(3);
        // Generate inputs so that in1 * in2 stays within [outLow, outHigh].
        // Use bound = sqrt(min(|outLow|, |outHigh|)) for the signed case so that
        // the worst-case product bound^2 does not exceed either absolute limit.
        const float absOutLow = std::abs(outLow);
        const float absOutHigh = std::abs(outHigh);
        const float bound =
                std::sqrt((outLow >= 0.0f) ? absOutHigh : (absOutLow < absOutHigh ? absOutLow : absOutHigh));
        const float lo = (outLow >= 0.0f) ? 0.0f : -bound;
        const float hi = bound;

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> distrib(lo, hi);

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            auto tensor = ov::Tensor(ov::element::f32, targetInputStaticShapes[i]);
            auto* data = tensor.data<float>();
            for (size_t j = 0; j < tensor.get_size(); ++j) {
                data[j] = distrib(rng);
            }
            inputs.insert({funcInputs[i].get_node_shared_ptr(), tensor});
        }
    }

    void SetUp() override {
        ov::Shape inputShape;
        std::vector<float> fqRanges;
        std::tie(inputShape, fqRanges) = GetParam();

        init_input_shapes({ov::test::InputShape{{}, std::vector<ov::Shape>{inputShape}},
                           ov::test::InputShape{{}, std::vector<ov::Shape>{inputShape}}});

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[1])};

        const auto multiply = std::make_shared<ov::opset1::Multiply>(params[0]->output(0), params[1]->output(0));

        const auto fq = ov::test::utils::make_fake_quantize(multiply->output(0), ov::element::f32,
                                                            /*levels=*/256, {}, {fqRanges.at(0)}, {fqRanges.at(1)},
                                                            {fqRanges.at(2)}, {fqRanges.at(3)});

        function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::opset1::Result>(fq)}, params,
                                               "MultiplyMixedPrecision");
        // Two quantization steps: accounts for rounding in both the reference and the NPU path.
        abs_threshold = (fqRanges.at(3) - fqRanges.at(2)) * 2.0f / (256 - 1);
    }
};

//
// Platform test definitions
//

TEST_P(MultiplyMixedPrecisionTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(MultiplyMixedPrecisionTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(MultiplyMixedPrecisionTest, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<ov::Shape> multiplyInputShapes = {{1, 4, 16, 16}};

const std::vector<std::vector<float>> multiplyFqRanges = {
        {-128.0f, 127.0f, -128.0f, 127.0f},
        {0.0f, 255.0f, 0.0f, 255.0f},
};

INSTANTIATE_TEST_SUITE_P(smoke_MultiplyMixedPrecision, MultiplyMixedPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(multiplyInputShapes),
                                            ::testing::ValuesIn(multiplyFqRanges)),
                         MultiplyMixedPrecisionTest::getTestCaseName);

//
// QuantizedMultiplyWithSI8MatMulConsumerTest
//
// Reproduces the Multiply → FakeQuantize(i8) pattern with a downstream MatMul consumer
// carrying i8 weights.
// QuantizeWithMultiplyRewriter absorbs the i8 Quantize into IE.Multiply.
//
//   Param0 [f32] ──┐
//                  ├─► Multiply ──► FakeQuantize([-128,127]) ──► MatMul ──► Result
//   Param1 [f32] ──┘                                             │
//                              Param2 [i8, K×N] ── Convert(f32)──┘
//

using QuantizedMultiplyWithSI8MatMulConsumerTestParams =
        std::tuple<std::vector<size_t>,  // inputShape (both Multiply inputs)
                   size_t,               // MatMul output dim N
                   std::vector<float>>;  // FQ ranges: {inLow, inHigh, outLow, outHigh}

class QuantizedMultiplyWithSI8MatMulConsumerTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<QuantizedMultiplyWithSI8MatMulConsumerTestParams> {
public:
    static std::string getTestCaseName(
            const testing::TestParamInfo<QuantizedMultiplyWithSI8MatMulConsumerTestParams>& obj) {
        std::vector<size_t> inputShape;
        size_t matmulN;
        std::vector<float> fqRanges;
        std::tie(inputShape, matmulN, fqRanges) = obj.param;

        std::ostringstream shapeStr;
        shapeStr << "{";
        for (size_t i = 0; i < inputShape.size(); ++i) {
            shapeStr << inputShape[i];
            if (i + 1 < inputShape.size()) {
                shapeStr << "x";
            }
        }
        shapeStr << "}";

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InputShape=" << shapeStr.str() << sep;
        result << "MatMulN=" << matmulN << sep;
        result << "FQ={" << fqRanges.at(0) << "," << fqRanges.at(1) << "," << fqRanges.at(2) << "," << fqRanges.at(3)
               << "}" << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        const auto& fqRanges = std::get<2>(GetParam());

        const float outLow = fqRanges.at(2);
        const float outHigh = fqRanges.at(3);
        const float absOutLow = std::abs(outLow);
        const float absOutHigh = std::abs(outHigh);
        const float bound =
                std::sqrt((outLow >= 0.0f) ? absOutHigh : (absOutLow < absOutHigh ? absOutLow : absOutHigh));
        const float lo = (outLow >= 0.0f) ? 0.0f : -bound;
        const float hi = bound;

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> floatDist(lo, hi);
        std::uniform_int_distribution<int> i8Dist(-128, 127);

        for (size_t i = 0; i < 2; ++i) {
            auto tensor = ov::Tensor(ov::element::f32, targetInputStaticShapes[i]);
            auto* data = tensor.data<float>();
            for (size_t j = 0; j < tensor.get_size(); ++j) {
                data[j] = floatDist(rng);
            }
            inputs.insert({funcInputs[i].get_node_shared_ptr(), tensor});
        }
        auto weightTensor = ov::Tensor(ov::element::i8, targetInputStaticShapes[2]);
        auto* wdata = weightTensor.data<int8_t>();
        for (size_t j = 0; j < weightTensor.get_size(); ++j) {
            wdata[j] = static_cast<int8_t>(i8Dist(rng));
        }
        inputs.insert({funcInputs[2].get_node_shared_ptr(), weightTensor});
    }

    void SetUp() override {
        std::vector<size_t> inputShape;
        size_t matmulN;
        std::vector<float> fqRanges;
        std::tie(inputShape, matmulN, fqRanges) = GetParam();

        const size_t K = inputShape.back();
        const std::vector<size_t> weightShape{K, matmulN};

        init_input_shapes(static_shapes_to_test_representation({inputShape, inputShape, weightShape}));

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(inputShape)),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(inputShape)),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape(weightShape))};

        const auto multiplyOp = std::make_shared<ov::op::v1::Multiply>(params[0]->output(0), params[1]->output(0));

        const auto fq = ov::test::utils::make_fake_quantize(multiplyOp->output(0), ov::element::f32,
                                                            /*levels=*/256, {}, {fqRanges.at(0)}, {fqRanges.at(1)},
                                                            {fqRanges.at(2)}, {fqRanges.at(3)});

        const auto weightsF32 = std::make_shared<ov::op::v0::Convert>(params[2]->output(0), ov::element::f32);
        const auto matmulOp = std::make_shared<ov::op::v0::MatMul>(fq->output(0), weightsF32->output(0), false, false);

        function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(matmulOp)}, params,
                                               "QuantizedMultiplyWithSI8MatMulConsumer");

        // Worst-case MatMul accumulation error: K × 0.5 × max_weight_val
        abs_threshold = static_cast<float>(K) * 0.5f * 127.0f;
    }
};

TEST_P(QuantizedMultiplyWithSI8MatMulConsumerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(precommit_QuantizedMultiplyWithSI8MatMulConsumer, QuantizedMultiplyWithSI8MatMulConsumerTest,
                         ::testing::Values(
                                 // K=4 keeps the worst-case MatMul dot product within f16 range .
                                 QuantizedMultiplyWithSI8MatMulConsumerTestParams{
                                         {1, 4, 16, 4}, 4, {-128.0f, 127.0f, -128.0f, 127.0f}}),
                         QuantizedMultiplyWithSI8MatMulConsumerTest::getTestCaseName);

}  // namespace ov::test::subgraph
