//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset6_decl.hpp"

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/opsets/opset1.hpp"

#include <cstdint>
#include <random>
#include <sstream>
#include <vector>

using namespace ov::test::utils;
using namespace ov::test;

namespace WeightsDequantizeDefinition {

enum class ZPType { INT8_T, FLOAT };

union FloatInt8Union {
    FloatInt8Union(int8_t val): int8_val{val} {
    }
    FloatInt8Union(float val): float_val{val} {
    }
    int8_t int8_val;
    float float_val;
};

struct FQ_as_Mul_Sub_dequantize {
    ZPType zp_type;
    FloatInt8Union zp;
    float scale;
    float o_low, o_high;
    size_t levels;
};

using WeightsDequantizeTestParams = std::tuple<FQ_as_Mul_Sub_dequantize, ov::element::Type, std::string>;

class WeightsDequantize : public testing::WithParamInterface<WeightsDequantizeTestParams>, public VpuOv2LayerTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        const auto& test_case = std::get<0>(params);
        const auto& float_element_type = std::get<1>(params);
        targetDevice = std::get<2>(params);
        const ov::Shape weightsShape{4, 3, 1, 1};
        std::vector<int8_t> weights{-108, -120, -124, 8, 4, -106, -88, 113, -74, 54, 127, 0};
        auto i_weights = std::make_shared<ov::op::v0::Constant>(ov::element::i8, weightsShape, weights);
        auto f_weights = std::make_shared<ov::opset6::Convert>(i_weights, float_element_type);
        std::shared_ptr<ov::opset6::Subtract> subtract_zp;
        float zp;
        if (test_case.zp_type == ZPType::FLOAT) {
            auto f_zp = std::make_shared<ov::op::v0::Constant>(float_element_type, ov::Shape{1},
                                                               std::vector<float>{test_case.zp.float_val});
            subtract_zp = std::make_shared<ov::opset6::Subtract>(f_weights, f_zp);
            zp = test_case.zp.float_val;
        } else {
            auto i_zp = std::make_shared<ov::op::v0::Constant>(ov::element::i8, ov::Shape{1},
                                                               std::vector<int8_t>{test_case.zp.int8_val});
            auto f_zp = std::make_shared<ov::opset6::Convert>(i_zp, float_element_type);
            subtract_zp = std::make_shared<ov::opset6::Subtract>(f_weights, f_zp);
            zp = test_case.zp.int8_val;
        }
        auto scale = std::make_shared<ov::op::v0::Constant>(float_element_type, ov::Shape{1},
                                                            std::vector<float>{test_case.scale});

        std::shared_ptr<ov::opset6::Multiply> f_mul;
        if (zp == 0) {
            f_mul = std::make_shared<ov::opset6::Multiply>(f_weights, scale);
        } else {
            f_mul = std::make_shared<ov::opset6::Multiply>(subtract_zp, scale);
        }

        const ov::Shape inputShape{1, 3, 62, 62};
        init_input_shapes(static_shapes_to_test_representation({inputShape}));
        const ov::ParameterVector conv_params{
                std::make_shared<ov::op::v0::Parameter>(float_element_type, ov::Shape(inputShape))};

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ov::op::v1::Convolution>(conv_params[0], f_mul->output(0), strides,
                                                                    pads_begin, pads_end, dilations);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv)};
        function = std::make_shared<ov::Model>(results, conv_params, "WtDQToFQ");
        rel_threshold = 0.5f;
    }

    static std::string getTestCaseName(testing::TestParamInfo<WeightsDequantizeTestParams> obj) {
        auto params = obj.param;
        FQ_as_Mul_Sub_dequantize test_case = std::get<0>(params);
        ov::element::Type precision = std::get<1>(params);
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "ZPType=" << (test_case.zp_type == ZPType::FLOAT ? "float" : "int8_t") << sep;
        result << "ZP=" << (test_case.zp_type == ZPType::FLOAT ? test_case.zp.float_val : test_case.zp.int8_val) << sep;
        result << "scale=" << test_case.scale << sep;
        result << "oLow=" << test_case.o_low << sep;
        result << "oHigh=" << test_case.o_low << sep;
        result << "levels=" << test_case.levels << sep;
        result << "precision=" << precision.get_type_name() << sep;
        return result.str();
    }
};

//
// Platform test definition
//

TEST_P(WeightsDequantize, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(WeightsDequantize, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

// clang-format off
const auto basicCasesM = ::testing::Combine(
        ::testing::ValuesIn(
            std::vector<FQ_as_Mul_Sub_dequantize>{FQ_as_Mul_Sub_dequantize{ZPType::FLOAT, 1.0f, 2, (-128 - 1) * 2, (127 - 1) * 2, 256},
            FQ_as_Mul_Sub_dequantize{ZPType::FLOAT, 1.0f, 2, (-127 - 1) * 2, (127 - 1) * 2, 255},
            FQ_as_Mul_Sub_dequantize{ZPType::FLOAT, 0.0f, 2, (-128 - 0) * 2, (127 - 0) * 2, 256},
            FQ_as_Mul_Sub_dequantize{ZPType::FLOAT, 0.0f, 2, (-127 - 0) * 2, (127 - 0) * 2, 255},
            FQ_as_Mul_Sub_dequantize{ZPType::INT8_T, (int8_t)1, 2, (-128 - 1) * 2, (127 - 1) * 2, 256},
            FQ_as_Mul_Sub_dequantize{ZPType::INT8_T, (int8_t)1, 2, (-127 - 1) * 2, (127 - 1) * 2, 255},
            FQ_as_Mul_Sub_dequantize{ZPType::INT8_T, (int8_t)0, 2, (-128 - 0) * 2, (127 - 0) * 2, 256},
            FQ_as_Mul_Sub_dequantize{ZPType::INT8_T, (int8_t)0, 2, (-127 - 0) * 2, (127 - 0) * 2, 255}}),
        ::testing::ValuesIn(std::vector<ov::element::Type>{ov::element::f16, ov::element::f32}),
        ::testing::Values(test_utils::TARGET_DEVICE));
// clang-format on

INSTANTIATE_TEST_SUITE_P(precommit_WeightsDequantize, WeightsDequantize, basicCasesM,
                         WeightsDequantize::getTestCaseName);

class WeightsAsInputDequantizeSI8Matmul : public VpuOv2LayerTest {
public:
    void SetUp() override {
        targetDevice = test_utils::TARGET_DEVICE;
        rel_threshold = 0.5f;

        const ov::Shape actShape{1, 128, 2048};
        const ov::Shape weightShape{2048, 32};
        constexpr float zp = -17.0f;
        constexpr float scale = 3.3336121123284101e-4f;
        constexpr auto signedInt8WeightType = ov::element::i8;

        init_input_shapes(static_shapes_to_test_representation({actShape, weightShape}));
        const auto act = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.at(0));
        const auto weightsI8 = std::make_shared<ov::op::v0::Parameter>(signedInt8WeightType, inputDynamicShapes.at(1));

        const auto actF32 = std::make_shared<ov::op::v0::Convert>(act, ov::element::f32);
        const auto weightsF32 = std::make_shared<ov::op::v0::Convert>(weightsI8, ov::element::f32);
        const auto zpConst =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1}, std::vector<float>{zp});
        const auto subZp = std::make_shared<ov::op::v1::Subtract>(weightsF32, zpConst);
        const auto scaleConst =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1}, std::vector<float>{scale});
        const auto dequantizedWeights = std::make_shared<ov::op::v1::Multiply>(subZp, scaleConst);

        const auto matmul = std::make_shared<ov::op::v0::MatMul>(actF32, dequantizedWeights, false, false);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(matmul)};
        function =
                std::make_shared<ov::Model>(results, ov::ParameterVector{act, weightsI8}, "InputDequantizeSI8Matmul");
    }
};

TEST_F(WeightsAsInputDequantizeSI8Matmul, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(WeightsAsInputDequantizeSI8Matmul, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace WeightsDequantizeDefinition

// Sub-byte (u4/i4) weights dequantization via Convert -> Subtract(scalar ZP) -> Multiply(per-channel scale)
// followed by Reshape and MatMul. Tests correct handling of sub-byte zero-points by the compiler.
namespace SubByteWeightsDequantizeDefinition {

struct SubByteWtDqParams {
    size_t outputChannels;         // C — rows in the weight matrix (MatMul output features)
    size_t inputFeatures;          // K — shared inner dimension
    ov::element::Type weightType;  // u4 or i4
    bool perChannelZp;             // false: scalar ZP {1}, true: per-channel ZP {C,1,1}
};

class SubByteWeightsDequantize :
        public testing::WithParamInterface<SubByteWtDqParams>,
        public ov::test::VpuOv2LayerTest {
public:
    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 1);
        ASSERT_EQ(expectedTensors.size(), 1);
        ASSERT_EQ(expectedTensors[0].get_size(), actualTensors[0].get_size());
        const float absThreshold = 0.5f;
        ov::test::utils::compare(actualTensors[0], expectedTensors[0], absThreshold);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor inputTensor = ov::test::utils::create_and_fill_tensor_real_distribution(
                funcInputs[0].get_element_type(), targetInputStaticShapes[0], -1.0f, 1.0f, 42);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), inputTensor});
    }

    void SetUp() override {
        /* Sub-byte weights dequantization subgraph:
         *
         *   weights[C,1,K]   zp[1] or zp[C,1,1]   scale[C,1,1](f16, per-channel)
         *        |                |                          |
         *   Convert(f16)     Convert(f16)                   |
         *           \            /                          |
         *            Subtract --------------------------Multiply
         *                                                   |
         *                                             Reshape[C,K]
         *                                                   |
         *   Input[1,1,K](f16)                               |
         *              \                                    /
         *               MatMul(transposeB=true)
         *                         |
         *                    Output[1,1,C]
         *
         * Weights are stored in 3-D group-quant layout [C, G, K] with G=1
         * (one group per channel), which is equivalent to per-channel
         * quantization after group-quant unrolling.
         * ZP is either scalar {1} or per-channel {C,1,1} depending on params.
         */
        const auto& p = GetParam();

        const ov::Shape inputShape{1, 1, p.inputFeatures};
        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));
        const auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(0));

        const ov::Shape weightsShape{p.outputChannels, 1, p.inputFeatures};
        const ov::Shape zpShape = p.perChannelZp ? ov::Shape{p.outputChannels, 1, 1} : ov::Shape{1};
        const ov::Shape scaleShape{p.outputChannels, 1, 1};

        const bool isUnsigned = (p.weightType == ov::element::u4);
        const int quantMin = isUnsigned ? 0 : -8;
        const int quantMax = isUnsigned ? 15 : 7;
        constexpr float minScale = 0.001f;
        constexpr float maxScale = 0.05f;

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> zpDist(quantMin, quantMax);
        std::uniform_int_distribution<int> wtDist(quantMin, quantMax);
        std::uniform_real_distribution<float> scaleDist(minScale, maxScale);

        // Scalar ZP: fixed non-zero value — u4: 8 (mid-range), i4: 7 (maximum).
        // Per-channel ZP: one independently drawn value per output channel.
        const size_t zpCount = ov::shape_size(zpShape);
        std::vector<int8_t> zpValues(zpCount);
        if (p.perChannelZp) {
            for (auto& v : zpValues) {
                v = static_cast<int8_t>(zpDist(rng));
            }
        } else {
            zpValues[0] = isUnsigned ? int8_t{8} : int8_t{7};
        }

        std::vector<int8_t> wtValues(p.outputChannels * p.inputFeatures);
        for (auto& v : wtValues) {
            v = static_cast<int8_t>(wtDist(rng));
        }

        std::vector<float> scaleValues(p.outputChannels);
        for (auto& v : scaleValues) {
            v = scaleDist(rng);
        }

        const auto weightsConst = std::make_shared<ov::opset1::Constant>(p.weightType, weightsShape, wtValues);
        const auto zpConst = std::make_shared<ov::opset1::Constant>(p.weightType, zpShape, zpValues);
        const auto scaleConst = std::make_shared<ov::opset1::Constant>(ov::element::f16, scaleShape, scaleValues);

        const auto convertWeights = std::make_shared<ov::opset1::Convert>(weightsConst->output(0), ov::element::f16);
        const auto convertZp = std::make_shared<ov::opset1::Convert>(zpConst->output(0), ov::element::f16);

        const auto subtract = std::make_shared<ov::op::v1::Subtract>(convertWeights->output(0), convertZp->output(0));
        const auto multiply = std::make_shared<ov::opset1::Multiply>(subtract->output(0), scaleConst->output(0));

        const std::vector<int64_t> reshapeTarget{static_cast<int64_t>(p.outputChannels),
                                                 static_cast<int64_t>(p.inputFeatures)};
        const auto reshapeConst = std::make_shared<ov::opset1::Constant>(
                ov::element::i64, ov::Shape{reshapeTarget.size()}, reshapeTarget);
        const auto reshape = std::make_shared<ov::op::v1::Reshape>(multiply->output(0), reshapeConst, false);

        const auto matmul = std::make_shared<ov::opset1::MatMul>(input->output(0), reshape->output(0), false, true);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(matmul->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "SubByteWtDq");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SubByteWtDqParams>& obj) {
        const auto& p = obj.param;
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << "_";
        result << "C=" << p.outputChannels << "_";
        result << "K=" << p.inputFeatures << "_";
        result << "WeightType=" << p.weightType << "_";
        result << "ZP=" << (p.perChannelZp ? "perChannel" : "scalar");
        return result.str();
    }
};

//
// Platform test definitions
//

TEST_P(SubByteWeightsDequantize, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(SubByteWeightsDequantize, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(precommit_SubByteWeightsDequantize, SubByteWeightsDequantize,
    ::testing::Values(
        // Scalar ZP
        SubByteWtDqParams{/*outputChannels=*/64, /*inputFeatures=*/128, /*weightType=*/ov::element::u4, /*perChannelZp=*/false},
        SubByteWtDqParams{/*outputChannels=*/64, /*inputFeatures=*/128, /*weightType=*/ov::element::i4, /*perChannelZp=*/false},
        // Per-channel ZP
        SubByteWtDqParams{/*outputChannels=*/64, /*inputFeatures=*/128, /*weightType=*/ov::element::u4, /*perChannelZp=*/true},
        SubByteWtDqParams{/*outputChannels=*/64, /*inputFeatures=*/128, /*weightType=*/ov::element::i4, /*perChannelZp=*/true}
    ),
    SubByteWeightsDequantize::getTestCaseName
);
// clang-format on

}  // namespace SubByteWeightsDequantizeDefinition
