// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <ov_ops/dynamic_quantize.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset6_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"

using namespace ov::test::utils;
using namespace ov::test;
namespace ov::test::subgraph {

using DQDecompositionParams = std::tuple<ov::Shape,           // input shapes
                                         ov::element::Type>;  // input precision

class FuseDQTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<DQDecompositionParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<DQDecompositionParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 3);
        ASSERT_EQ(expectedTensors.size(), 3);

        const float outputAbsThresholdOutput = 1.0f;
        ov::test::utils::compare(expectedTensors[0], actualTensors[0], outputAbsThresholdOutput);
        const float scaleAbsThresholdOutput = 0.01f;
        ov::test::utils::compare(expectedTensors[1], actualTensors[1], scaleAbsThresholdOutput);
        const float zpAbsThresholdOutput = 1.0f;
        ov::test::utils::compare(expectedTensors[2], actualTensors[2], zpAbsThresholdOutput);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 10, 1, 100);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    std::shared_ptr<ov::Node> find_min_value(const ov::Output<ov::Node>& input) {
        const auto& zero_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        const auto& one_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});

        const auto& input_shape = std::make_shared<ov::op::v3::ShapeOf>(input);
        const auto& input_rank = std::make_shared<ov::op::v3::ShapeOf>(input_shape);
        const auto& input_rank_as_scalar = std::make_shared<ov::op::v0::Squeeze>(input_rank);
        const auto& reduce_axes =
                std::make_shared<ov::op::v4::Range>(zero_node, input_rank_as_scalar, one_node, ov::element::i64);
        const auto& input_min = std::make_shared<ov::op::v1::ReduceMin>(input, reduce_axes);
        const auto& zero_node_u8 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0});
        return std::make_shared<ov::op::v1::Minimum>(zero_node_u8, input_min);
    }

    std::shared_ptr<ov::Node> find_max_value(const ov::Output<ov::Node>& input) {
        const auto& zero_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        const auto& one_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});

        const auto& input_shape = std::make_shared<ov::op::v3::ShapeOf>(input);
        const auto& input_rank = std::make_shared<ov::op::v3::ShapeOf>(input_shape);
        const auto& input_rank_as_scalar = std::make_shared<ov::op::v0::Squeeze>(input_rank);
        const auto& reduce_axes =
                std::make_shared<ov::op::v4::Range>(zero_node, input_rank_as_scalar, one_node, ov::element::i64);
        const auto& input_max = std::make_shared<ov::op::v1::ReduceMax>(input, reduce_axes);
        const auto& zero_node_u8 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0});
        return std::make_shared<ov::op::v1::Maximum>(zero_node_u8, input_max);
    }

    std::shared_ptr<ov::Node> quantize_linear(ov::Output<ov::Node> x, ov::Output<ov::Node> x_span,
                                              ov::Output<ov::Node> quant_range_span,
                                              ov::Output<ov::Node> y_zero_point) {
        const auto& x_scaled = std::make_shared<ov::op::v1::Divide>(
                std::make_shared<ov::op::v1::Multiply>(x, quant_range_span), x_span);
        const auto& x_rounded =
                std::make_shared<ov::op::v5::Round>(x_scaled, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
        const auto& y_zero_point_f32 = std::make_shared<ov::op::v0::Convert>(y_zero_point, ov::element::f32);
        const auto& result_shifted = std::make_shared<ov::op::v1::Add>(x_rounded, y_zero_point_f32);
        const auto& result_clamped = std::make_shared<ov::op::v0::Clamp>(result_shifted, 0, 255);

        return std::make_shared<ov::op::v0::Convert>(result_clamped, ov::element::u8);
    }

    ov::OutputVector dynamic_quantize_linear(const ov::ParameterVector& inputs) {
        const auto& x = inputs[0];
        // quantization range in case of uint8 is [0, 255]
        const auto& quant_range_min = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0});
        const auto& quant_range_max = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {255});
        const auto& quant_range_span = std::make_shared<ov::op::v1::Subtract>(quant_range_max, quant_range_min);

        const auto& x_max = find_max_value(x);
        const auto& x_min = find_min_value(x);
        const auto& x_span = std::make_shared<ov::op::v1::Subtract>(x_max, x_min);
        const auto& y_scale = std::make_shared<ov::op::v1::Divide>(x_span, quant_range_max);
        const auto& x_min_shifted = std::make_shared<ov::op::v1::Subtract>(quant_range_min, x_min);
        const auto& intermediate_zero_point =
                std::make_shared<ov::op::v5::Round>(std::make_shared<ov::op::v1::Divide>(x_min_shifted, y_scale),
                                                    ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
        const auto& y_zero_point = std::make_shared<ov::op::v0::Convert>(
                std::make_shared<ov::op::v0::Clamp>(intermediate_zero_point, 0, 255), ov::element::u8);

        const auto& y = quantize_linear(x, x_span, quant_range_span, y_zero_point);
        return {y, y_scale, y_zero_point};
    }

    std::shared_ptr<ov::Model> init_subgraph(ov::Shape& input_shape, const ov::element::Type input_precision) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(input_precision, input_shape)};

        const auto output = dynamic_quantize_linear(params);
        return std::make_shared<ov::Model>(output, params, "DQDecomposition");
    }

    void SetUp() override {
        ov::Shape input_shape;
        ov::element::Type input_precision;

        std::tie(input_shape, input_precision) = GetParam();
        init_input_shapes(ov::test::static_shapes_to_test_representation({input_shape}));
        function = init_subgraph(input_shape, input_precision);
    }
};

TEST_P(FuseDQTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FuseDQTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(FuseDQTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

namespace {
const std::vector<ov::element::Type> input_precisions = {ov::element::f32};

const std::vector<ov::Shape> input_shapes_basic = {{{1, 304, 560}}, {{2, 2, 6}}};
const std::vector<ov::Shape> input_shapes = {{32}, {{3, 32}}, {{1, 32, 16}}, {{1, 4, 16, 16}}, {{1, 77, 4096}}};

INSTANTIATE_TEST_SUITE_P(precommit_FuseDQ, FuseDQTestCommon,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(input_precisions)),
                         FuseDQTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseDQ, FuseDQTestCommon,
                         ::testing::Combine(::testing::ValuesIn(input_shapes), ::testing::ValuesIn(input_precisions)),
                         FuseDQTestCommon::getTestCaseName);

}  // namespace
}  // namespace ov::test::subgraph
