//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/dyn_color_conversion_pattern.hpp"
#include "common/print_test_case_name.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "pretty_test_arguments.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/opsets/opset11.hpp"

using namespace ov::test;
using ov::op::util::InterpolateBase;

namespace ov::test::subgraph {

namespace {

PRETTY_PARAM(Height, BoundedDim);
PRETTY_PARAM(Width, BoundedDim);

using DynamicResizeConvParams = std::tuple<Height, Width, ov::element::Type, std::vector<float>, bool>;

class DynamicResizeConvSubgraphTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<DynamicResizeConvParams> {
public:
    void SetUp() override {
        Height height;
        Width width;
        std::vector<float> scales;
        std::tie(height, width, modelType, scales, useConvTail) = GetParam();
        scalesValues = scales;

        const auto hDim = height.value();
        const auto wDim = width.value();

        const ov::Shape scalesShape{scalesValues.size()};
        const auto scalesInputShape = generateTestShape(scalesShape);

        // Y plane is full resolution: [1,H,W,1]
        auto yShape =
                generateTestShape({BoundedDim(1), hDim, wDim, BoundedDim(1)}, hostCompileSmallShapesLimitationCallback);

        // UV plane is half resolution: [1,H/2,W/2,2]. Its concrete runtime shapes are derived directly from
        // the Y plane's runtime shapes (not regenerated independently) so that Y and UV stay exactly 2x
        // related for every test iteration, as required by the CSC subgraph.
        ov::PartialShape uvPartialShape{
                ov::Dimension(1), hDim.dim == -1 ? ov::Dimension(1, hDim.bound / 2) : ov::Dimension(hDim.dim / 2),
                wDim.dim == -1 ? ov::Dimension(1, wDim.bound / 2) : ov::Dimension(wDim.dim / 2), ov::Dimension(2)};
        std::vector<ov::Shape> uvStaticShapes;
        uvStaticShapes.reserve(yShape.second.size());
        for (const auto& s : yShape.second) {
            uvStaticShapes.push_back(ov::Shape{s[0], s[1] / 2, s[2] / 2, 2});
        }
        ov::test::InputShape uvShape{uvPartialShape, uvStaticShapes};

        init_input_shapes({yShape, uvShape, scalesInputShape});

        const auto yInput = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, inputDynamicShapes.at(0));
        yInput->set_friendly_name("Y");

        const auto uvInput = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, inputDynamicShapes.at(1));
        uvInput->set_friendly_name("UV");

        const auto scalesParam = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, scalesShape);
        scalesParam->set_friendly_name("scales");

        const auto result = buildPattern(yInput, uvInput, scalesParam);
        function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(result)},
                                               ov::ParameterVector{yInput, uvInput, scalesParam},
                                               "DynamicResizeConvSubgraphTest");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        for (size_t i = 0; i < targetInputStaticShapes.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor inputTensor;
            if (funcInput.get_node()->get_friendly_name() == "scales") {
                inputTensor = ov::Tensor{funcInput.get_element_type(), targetInputStaticShapes[i]};
                auto* data = inputTensor.data<float>();
                for (size_t j = 0; j < scalesValues.size(); ++j) {
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
    std::vector<float> scalesValues{2.0f, 2.0f};
    bool useConvTail = true;
    static std::shared_ptr<ov::op::v0::Constant> makeAxes() {
        return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{2, 3});
    }

    // Builds a 1x1 convolution that maps each output channel to a single input channel (cycling through
    // inChannels when outChannels > inChannels). Reduces to an identity mapping when inChannels == outChannels.
    std::shared_ptr<ov::Node> makeChannelMappingConv(const ov::Output<ov::Node>& input, int64_t inChannels,
                                                     int64_t outChannels) const {
        std::vector<float> weightsValues(static_cast<size_t>(outChannels * inChannels), 0.f);
        for (int64_t outCh = 0; outCh < outChannels; ++outCh) {
            const int64_t inCh = outCh % inChannels;
            weightsValues[static_cast<size_t>(outCh * inChannels + inCh)] = 1.f;
        }

        auto weights = ov::op::v0::Constant::create(
                modelType, ov::Shape{static_cast<size_t>(outChannels), static_cast<size_t>(inChannels), 1, 1},
                weightsValues);
        const ov::Strides strides{1, 1};
        const ov::CoordinateDiff padsBegin{0, 0};
        const ov::CoordinateDiff padsEnd{0, 0};
        const ov::Strides dilations{1, 1};
        return std::make_shared<ov::op::v1::Convolution>(input, weights, strides, padsBegin, padsEnd, dilations);
    }

    std::shared_ptr<ov::Node> buildResize(const ov::Output<ov::Node>& nchwInput,
                                          const ov::Output<ov::Node>& scales) const {
        const auto axes = makeAxes();

        InterpolateBase::InterpolateAttrs attrs{ov::op::v11::Interpolate::InterpolateMode::LINEAR,
                                                InterpolateBase::ShapeCalcMode::SCALES,
                                                {0, 0, 0, 0},
                                                {0, 0, 0, 0},
                                                ov::op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL,
                                                ov::op::v11::Interpolate::NearestMode::FLOOR,
                                                false,
                                                -0.75f};

        return std::make_shared<ov::op::v11::Interpolate>(nchwInput, scales, axes, attrs);
    }

    std::shared_ptr<ov::Node> buildConvTail(const ov::Output<ov::Node>& input) const {
        constexpr int64_t rgbChannels = 3;
        constexpr int64_t channels = 32;

        const auto conv1 = makeChannelMappingConv(input, rgbChannels, channels);
        const auto conv2 = makeChannelMappingConv(conv1->output(0), channels, channels);
        return conv2;
    }

    // Subgraph: CSC(Y,UV) -> [Resize(parameter scales)] -> [Convolution(3->32) -> Convolution(32->32)] ->
    // [NHWC] -> [Convert(fp32)]
    std::shared_ptr<ov::Node> buildPattern(const std::shared_ptr<ov::op::v0::Parameter>& yInput,
                                           const std::shared_ptr<ov::op::v0::Parameter>& uvInput,
                                           const std::shared_ptr<ov::op::v0::Parameter>& scales) {
        const auto csc = buildDynYuvToRgbPattern(yInput, uvInput);  // NHWC f16 [1,H,W,3]

        const auto nhwc2nchwOrder =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        ov::Output<ov::Node> stage = std::make_shared<ov::op::v1::Transpose>(csc, nhwc2nchwOrder);

        stage = buildResize(stage, scales)->output(0);
        if (useConvTail) {
            stage = buildConvTail(stage)->output(0);
        }

        stage = std::make_shared<ov::op::v0::Convert>(stage, ov::element::f32)->output(0);
        return stage.get_node_shared_ptr();
    }

    // No-CSC subgraph: Image(u8) -> Convert(modelType) -> [Resize(parameter scales)] ->
    // [Convolution(3->32) -> Convolution(32->32)] -> [NHWC] -> [Convert(fp32)]
    std::shared_ptr<ov::Node> buildPatternNoCsc(const std::shared_ptr<ov::op::v0::Parameter>& imageInput,
                                                const std::shared_ptr<ov::op::v0::Parameter>& scales) {
        // Interpolate requires f16 data for scales-as-parameter legality (ConvertPrecisionToFPPass never
        // auto-promotes u8/ui8), so convert before resize rather than after.
        ov::Output<ov::Node> stage = std::make_shared<ov::op::v0::Convert>(imageInput, modelType);

        stage = buildResize(stage, scales)->output(0);
        if (useConvTail) {
            stage = buildConvTail(stage)->output(0);
        }

        stage = std::make_shared<ov::op::v0::Convert>(stage, ov::element::f32)->output(0);
        return stage.get_node_shared_ptr();
    }
};

class DynamicResizeConvSubgraphTest_TileAgnostic : public DynamicResizeConvSubgraphTest {
    void configure_model() override {
        configuration["NPU_TILES"] = "1";
    }
};

TEST_P(DynamicResizeConvSubgraphTest_TileAgnostic, NPU4000_HC_TestKindSubgraph) {
    abs_threshold = 0.08f;
    enableTurbo();
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(DynamicResizeConvSubgraphTest_TileAgnostic, NPU5010_HC_TestKindSubgraph) {
    abs_threshold = 0.08f;
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU5010);
}

TEST_P(DynamicResizeConvSubgraphTest_TileAgnostic, NPU6010_HC_TestKindSubgraph) {
    abs_threshold = 0.08f;
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU6010);
}

class DynamicResizeConvSubgraphTest_Interpreter : public DynamicResizeConvSubgraphTest {};

TEST_P(DynamicResizeConvSubgraphTest_Interpreter, NPU4000_HC_TestKindSubgraph) {
    abs_threshold = 0.08f;
    enableTurbo();
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(DynamicResizeConvSubgraphTest_Interpreter, NPU5010_HC_TestKindSubgraph) {
    abs_threshold = 0.08f;
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU5010);
}

TEST_P(DynamicResizeConvSubgraphTest_Interpreter, NPU6010_HC_TestKindSubgraph) {
    abs_threshold = 0.08f;
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU6010);
}

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<std::vector<float>> scalesList = {
        {2.0f, 2.0f},
        {1.5f, 2.0f},
};

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_DynamicResizeConvSubgraphTest_TileAgnostic,
                         DynamicResizeConvSubgraphTest_TileAgnostic,
                         ::testing::Combine(::testing::Values(Height{400_Dyn}), ::testing::Values(Width{800_Dyn}),
                                            ::testing::ValuesIn(modelTypes), ::testing::ValuesIn(scalesList),
                                            ::testing::Bool()),
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_DynamicResizeConvSubgraphTest_Interpreter, DynamicResizeConvSubgraphTest_Interpreter,
                         ::testing::Combine(::testing::Values(Height{400_Dyn}), ::testing::Values(Width{800_Dyn}),
                                            ::testing::ValuesIn(modelTypes), ::testing::ValuesIn(scalesList),
                                            ::testing::Bool()),
                         PrintTestCaseName());

}  // namespace
}  // namespace ov::test::subgraph
