// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset6_decl.hpp"

#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"

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
}  // namespace WeightsDequantizeDefinition
