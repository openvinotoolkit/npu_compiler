//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/normalize_l2.hpp"

using namespace ov::test::utils;
using namespace ov::test;
namespace ov::test::subgraph {

// =====================================================================================================================
// NormalizeL2 + Multiply -> RMS fusion test
// Pattern: Input -> NormalizeL2(axes=-1, eps) -> Multiply(gamma) -> Convert(f16)
// =====================================================================================================================
template <class ShapeType>
using IODescription = std::tuple<ShapeType, std::optional<ov::Layout>>;

template <class ShapeType>
using IODescriptions = std::vector<IODescription<ShapeType>>;

using NormalizeL2RMSParams = std::tuple<IODescription<ov::Shape>,  // input shape & optional layout
                                        ov::element::Type,         // input precision
                                        ov::op::EpsMode>;          // eps mode (ADD or MAX)

class FuseNormalizeL2RMSTest : public VpuOv2LayerTest, public testing::WithParamInterface<NormalizeL2RMSParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<NormalizeL2RMSParams> obj) {
        IODescription<ov::Shape> input;
        ov::element::Type inputPrecision;
        ov::op::EpsMode epsMode;
        std::tie(input, inputPrecision, epsMode) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        auto [inputShape, inputLayout] = input;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InputShape=" << inputShape << sep;
        if (inputLayout.has_value()) {
            result << "InputLayout=" << inputLayout->to_string() << sep;
        }
        result << "EpsMode=" << (epsMode == ov::op::EpsMode::ADD ? "ADD" : "MAX") << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 1, 1, 10);
        auto* data = tensorData.data<float>();
        for (size_t i = 0; i < tensorData.get_size(); ++i) {
            data[i] *= 1e-2f;
        }
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    std::shared_ptr<ov::Model> init_subgraph(const ov::PartialShape& input_shape,
                                             const std::optional<ov::Layout>& input_layout,
                                             const ov::Shape& target_shape, const ov::element::Type input_precision,
                                             ov::op::EpsMode epsMode) {
        auto param = std::make_shared<ov::op::v0::Parameter>(input_precision, input_shape);
        if (input_layout.has_value()) {
            param->set_layout(input_layout.value());
        }

        // NormalizeL2(input, axes=[-1], eps, eps_mode)
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        float eps = 1.1920928955078125e-7f;
        auto normalizeL2 = std::make_shared<ov::op::v0::NormalizeL2>(param, axes, eps, epsMode);

        // Multiply(NormalizeL2, gamma)
        ov::Shape gamma_shape = {target_shape.back()};
        auto gamma_tensor = ov::test::utils::create_and_fill_tensor(input_precision, gamma_shape);
        auto gamma = std::make_shared<ov::op::v0::Constant>(gamma_tensor);
        auto mul = std::make_shared<ov::op::v1::Multiply>(normalizeL2, gamma);

        // Convert to f16
        auto comp = std::make_shared<ov::op::v0::Convert>(mul, ov::element::f16);
        auto res = std::make_shared<ov::op::v0::Result>(comp);
        if (input_layout.has_value()) {
            res->set_layout(input_layout.value());
        }
        return std::make_shared<ov::Model>(ov::OutputVector{res}, ov::ParameterVector{param},
                                           "NormalizeL2MultiplyToRMS");
    }

    void SetUp() override {
        IODescription<ov::Shape> input;
        ov::element::Type input_precision;
        ov::op::EpsMode epsMode;

        std::tie(input, input_precision, epsMode) = GetParam();
        const auto& [input_shapes, input_layout] = input;
        inType = outType = input_precision;
        init_input_shapes(ov::test::static_shapes_to_test_representation({input_shapes}));

        function = init_subgraph(inputDynamicShapes.front(), input_layout, input_shapes, input_precision, epsMode);
    }
};

TEST_P(FuseNormalizeL2RMSTest, NPU3720_HW) {
    abs_threshold = 0.11f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FuseNormalizeL2RMSTest, NPU4000_HW) {
    abs_threshold = 0.11f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(FuseNormalizeL2RMSTest, NPU5010_HW) {
    abs_threshold = 0.11f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(FuseNormalizeL2RMSTest, NPU5020_HW) {
    abs_threshold = 0.11f;
    setDefaultHardwareMode();
    // TODO E####-159644
    setBatchCompilerMode("unroll");
    run(Platform::NPU5020);
}

namespace {
const std::vector<ov::element::Type> l2_input_precisions = {ov::element::f32};
const std::vector<ov::op::EpsMode> eps_modes = {ov::op::EpsMode::ADD};

const IODescriptions<ov::Shape> l2_input_shapes_basic = {IODescription<ov::Shape>{{1, 1, 1, 256}, {"NCHW"}}};
const IODescriptions<ov::Shape> l2_input_shapes = {
        {{32}, {"C"}}, {{3, 32}, {"HW"}}, {{1, 32, 16}, {}}, {{1, 4, 16, 16}, {}}, {{1, 77, 4096}, {}}};

INSTANTIATE_TEST_SUITE_P(precommit_FuseNormalizeL2RMS, FuseNormalizeL2RMSTest,
                         ::testing::Combine(::testing::ValuesIn(l2_input_shapes_basic),
                                            ::testing::ValuesIn(l2_input_precisions), ::testing::ValuesIn(eps_modes)),
                         FuseNormalizeL2RMSTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseNormalizeL2RMS, FuseNormalizeL2RMSTest,
                         ::testing::Combine(::testing::ValuesIn(l2_input_shapes),
                                            ::testing::ValuesIn(l2_input_precisions), ::testing::ValuesIn(eps_modes)),
                         FuseNormalizeL2RMSTest::getTestCaseName);

}  // namespace
}  // namespace ov::test::subgraph
