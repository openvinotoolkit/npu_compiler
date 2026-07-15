//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/print_test_case_name.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "pretty_test_arguments.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/convolution.hpp"
#include "openvino/opsets/opset11.hpp"

using namespace ov::test;

namespace ov::test::subgraph {

namespace {

PRETTY_PARAM(Height, BoundedDim);
PRETTY_PARAM(Width, BoundedDim);

using InterpolateSubgraphDynInputParams = std::tuple<Height, Width, ov::element::Type, std::vector<float>>;

// Exercises the real-world subgraph:
//   input (NCHW, f=16) -> Conv1(1×1) -> Interpolate(runtime scales) -> Transpose(NHWC)
class InterpolateSubgraphDynInputLayerTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<InterpolateSubgraphDynInputParams> {
public:
    // Interpolate scales: scale spatial axes (H=dim2, W=dim3) by the given factor in NCHW
    std::vector<float> scalesValues{2.0f, 2.0f};
    const std::vector<int64_t> axesValues{2, 3};

    void SetUp() override {
        Height height;
        Width width;
        std::vector<float> scales;
        std::tie(height, width, modelType, scales) = GetParam();
        scalesValues = scales;
        const auto hDim = height.value();
        const auto wDim = width.value();

        auto inputShape = generateTestShape({BoundedDim(1), BoundedDim(16), hDim, wDim},
                                            hostCompileSmallShapesLimitationCallback);
        const ov::Shape scalesShape{scalesValues.size()};
        const auto scalesInputShape = generateTestShape(scalesShape);
        init_input_shapes({inputShape, scalesInputShape});

        const auto input = std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes.at(0));
        input->set_friendly_name("input");
        const auto scalesParam = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, scalesShape);
        scalesParam->set_friendly_name("scales");

        auto result = buildPattern(input, scalesParam);

        function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(result)},
                                               ov::ParameterVector{input, scalesParam},
                                               "InterpolateSubgraphDynInputLayerTest");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        for (size_t i = 0; i < targetInputStaticShapes.size(); i++) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor inputTensor;
            if (funcInput.get_node()->get_friendly_name() == "scales") {
                inputTensor = ov::Tensor{funcInput.get_element_type(), targetInputStaticShapes[i]};
                auto* data = inputTensor.data<float>();
                for (size_t j = 0; j < scalesValues.size(); j++) {
                    data[j] = scalesValues[j];
                }
            } else {
                inputTensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                      targetInputStaticShapes[i]);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), inputTensor});
        }
    }

private:
    ov::element::Type modelType = ov::element::f16;

    // Build the subgraph:
    //   input [1,16,H,W] -> Conv1(1×1) -> Interpolate(runtime scales, axes{2,3}) -> Transpose(NHWC)
    std::shared_ptr<ov::Node> buildPattern(const std::shared_ptr<ov::op::v0::Parameter>& input,
                                           const std::shared_ptr<ov::op::v0::Parameter>& scales) {
        using ov::op::util::InterpolateBase;

        constexpr int64_t kChannels = 16;
        auto makeWeights = [&]() {
            std::vector<float> w(kChannels * kChannels, 0.f);
            for (int64_t c = 0; c < kChannels; ++c) {
                w[c * kChannels + c] = 1.f;
            }
            return ov::op::v0::Constant::create(
                    modelType, ov::Shape{static_cast<size_t>(kChannels), static_cast<size_t>(kChannels), 1, 1}, w);
        };

        // Conv1: [1,16,H,W] -> [1,16,H,W]
        auto conv1 = std::make_shared<ov::op::v1::Convolution>(input, makeWeights(), ov::Strides{1, 1},
                                                               ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0},
                                                               ov::Strides{1, 1});

        // Interpolate [1,16,H,W] × runtime scales -> [1,16,sH,sW]
        InterpolateBase::InterpolateAttrs interpolateAttrs{
                ov::op::v11::Interpolate::InterpolateMode::LINEAR,
                InterpolateBase::ShapeCalcMode::SCALES,
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                ov::op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL,
                ov::op::v11::Interpolate::NearestMode::FLOOR,
                false,
                -0.75f};

        auto axesConst = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axesValues.size()}, axesValues);
        auto interp = std::make_shared<ov::op::v11::Interpolate>(conv1, scales, axesConst, interpolateAttrs);

        // Transpose NCHW -> NHWC: [1,16,2H,2W] -> [1,2H,2W,16]
        auto transposeOrder =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 3, 1});
        return std::make_shared<ov::op::v1::Transpose>(interp, transposeOrder);
    }
};

TEST_P(InterpolateSubgraphDynInputLayerTest, NPU4000_HC_TestKindSubgraph) {
    abs_threshold = 0.0f;
    enableTurbo();
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(InterpolateSubgraphDynInputLayerTest, NPU5010_HC_TestKindSubgraph) {
    abs_threshold = 0.0f;
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU5010);
}

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<std::vector<float>> scalesList = {
        {1.0f, 1.0f},
        {2.0f, 2.0f},
        {2.0f, 1.0f},
};

// [Tracking number E#196823]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_InterpolateSubgraphDynInputLayerTest, InterpolateSubgraphDynInputLayerTest,
                         ::testing::Combine(::testing::Values(Height{256_Dyn}), ::testing::Values(Width{512_Dyn}),
                                            ::testing::ValuesIn(modelTypes), ::testing::ValuesIn(scalesList)),
                         PrintTestCaseName());

}  // namespace
}  // namespace ov::test::subgraph
