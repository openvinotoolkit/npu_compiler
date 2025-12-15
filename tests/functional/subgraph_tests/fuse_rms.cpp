// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <ov_ops/rms.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"

using namespace ov::test::utils;
using namespace ov::test;
namespace ov::test::subgraph {

using RMSNormDecompositionParams = std::tuple<ov::Shape,           // input shapes
                                              ov::element::Type>;  // input precision

class FuseRMSTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<RMSNormDecompositionParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RMSNormDecompositionParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 10, 1, 100);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    std::shared_ptr<ov::Model> init_subgraph(std::vector<ov::PartialShape>& input_shapes, const ov::Shape& target_shape,
                                             const ov::element::Type input_precision) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(input_precision, input_shapes[0])};

        // x^2
        auto power_const = ov::op::v0::Constant::create(input_precision, {}, {2.f});
        auto power = std::make_shared<ov::op::v1::Power>(params[0], power_const);

        // ReduceMean(x^2,axes)
        auto mean_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(power, mean_axes, true);

        // ReduceMean(x^2,axes)+eps
        auto eps = ov::op::v0::Constant::create(input_precision, {}, {1e-5f});
        auto add_eps = std::make_shared<ov::op::v1::Add>(mean, eps);

        // Sqrt(ReduceMean(x^2,axes)+eps)
        auto sqrt = std::make_shared<ov::op::v0::Sqrt>(add_eps);

        // 1/Sqrt(ReduceMean(x^2,axes)+eps)
        auto div_const = ov::op::v0::Constant::create(input_precision, {}, {1});
        auto div = std::make_shared<ov::op::v1::Divide>(div_const, sqrt);

        // x * 1/Sqrt(ReduceMean(x^2,axes)+eps)
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(params[0], div);

        // x * 1/Sqrt(ReduceMean(x^2,axes)+eps) * gamma
        auto dim = *target_shape.rbegin();

        auto tensor = ov::test::utils::create_and_fill_tensor(input_precision, ov::Shape{dim});
        auto gamma = std::make_shared<ov::op::v0::Constant>(tensor);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(gamma, mul1);

        auto comp = std::make_shared<ov::op::v0::Convert>(mul2, ov::element::f16);

        return std::make_shared<ov::Model>(ov::NodeVector{comp}, params, "RMSNormDecomposition");
    }
    void SetUp() override {
        ov::Shape input_shapes;
        ov::element::Type input_precision;

        std::tie(input_shapes, input_precision) = GetParam();
        inType = outType = input_precision;
        init_input_shapes(ov::test::static_shapes_to_test_representation({input_shapes}));
        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(input_precision, inputDynamicShapes.front())};

        auto gamma = ov::test::utils::create_and_fill_tensor(ov::element::f32, {input_shapes[input_shapes.size() - 1]},
                                                             10, 0.01f);
        auto rms_const = std::make_shared<ov::op::v0::Constant>(gamma);
        auto rms = std::make_shared<ov::op::internal::RMS>(params[0], rms_const, 1e-5f, ov::element::f16);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(rms)};

        function = std::make_shared<ov::Model>(results, params, "fuse_rms");
    }
};

TEST_P(FuseRMSTestCommon, NPU3720_HW) {
    abs_threshold = 0.11f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FuseRMSTestCommon, NPU4000_HW) {
    abs_threshold = 0.11f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(FuseRMSTestCommon, NPU5010_HW) {
    abs_threshold = 0.11f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
namespace {
const std::vector<ov::element::Type> input_precisions = {ov::element::f32};

const std::vector<ov::Shape> input_shapes_basic = {{{1, 2, 6}}, {{2, 2, 6}}};
const std::vector<ov::Shape> input_shapes = {{32}, {{3, 32}}, {{1, 32, 16}}, {{1, 4, 16, 16}}, {{1, 77, 4096}}};

INSTANTIATE_TEST_SUITE_P(precommit_FuseRMS, FuseRMSTestCommon,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(input_precisions)),
                         FuseRMSTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseRMS, FuseRMSTestCommon,
                         ::testing::Combine(::testing::ValuesIn(input_shapes), ::testing::ValuesIn(input_precisions)),
                         FuseRMSTestCommon::getTestCaseName);

}  // namespace
}  // namespace ov::test::subgraph
