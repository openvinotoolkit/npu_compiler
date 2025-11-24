//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"

using namespace ov::test::utils;
using namespace ov::test;

namespace ov::test::subgraph {

struct ColorConversionParams {
    int64_t height;
    int64_t width;

    ov::Shape getYInputShape() const {
        return {1, static_cast<size_t>(height), static_cast<size_t>(width), 1};
    }

    ov::Shape getUVInputShape() const {
        return {1, static_cast<size_t>(height / 2), static_cast<size_t>(width / 2), 2};
    }

    ov::Shape getOutputShape() const {
        return {1, 3, static_cast<size_t>(height), static_cast<size_t>(width)};
    }
};

class FuseColorConversionTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<ColorConversionParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ColorConversionParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;

        const auto& params = obj.param;
        auto yShape = params.getYInputShape();
        auto uvShape = params.getUVInputShape();
        auto outputShape = params.getOutputShape();

        result << "Y:" << yShape << sep;
        result << "UV:" << uvShape << sep;
        result << "Out:" << outputShape;

        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();

        ov::test::utils::InputGenerateData yGenData;
        // The shapes are populated only with value 128
        // Tracking ticket: E#186480
        yGenData.start_from = 128;
        yGenData.range = 1;
        yGenData.resolution = 1;
        ov::Tensor yTensorData = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(),
                                                                         targetInputStaticShapes[0], yGenData);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), yTensorData});

        ov::test::utils::InputGenerateData uvGenData;
        uvGenData.start_from = 128;
        uvGenData.range = 1;
        uvGenData.resolution = 1;
        ov::Tensor uvTensorData = ov::test::utils::create_and_fill_tensor(funcInputs[1].get_element_type(),
                                                                          targetInputStaticShapes[1], uvGenData);
        VpuOv2LayerTest::inputs.insert({funcInputs[1].get_node_shared_ptr(), uvTensorData});
    }

    void SetUp() override {
        inType = outType = ov::element::f32;
        const auto testParams = GetParam();
        const auto yInputShape = testParams.getYInputShape();
        const auto uvInputShape = testParams.getUVInputShape();
        const auto outputShape = testParams.getOutputShape();

        init_input_shapes(ov::test::static_shapes_to_test_representation({yInputShape, uvInputShape}));

        const auto yInput = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));
        const auto uvInput = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(1));

        // Build the YUV to RGB conversion pattern
        auto result = buildYuvToRgbPattern(yInput, uvInput, yInputShape, uvInputShape, outputShape);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(result)};
        function =
                std::make_shared<ov::Model>(results, ov::ParameterVector{yInput, uvInput}, "FuseColorConversionTest");
    }

private:
    std::shared_ptr<ov::Node> buildYuvToRgbPattern(const ov::Output<ov::Node>& yInput,
                                                   const ov::Output<ov::Node>& uvInput, const ov::Shape& yInputShape,
                                                   const ov::Shape& uvInputShape, const ov::Shape& outputShape) {
        ov::Shape yReshapeShape = {yInputShape[0], 1, yInputShape[1], yInputShape[2]};
        auto yReshapeConst =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{yReshapeShape.size()}, yReshapeShape);
        auto yReshape = std::make_shared<ov::op::v1::Reshape>(yInput, yReshapeConst, false);

        std::vector<int64_t> transposeOrder = {0, 3, 1, 2};  // NHWC -> NCHW
        auto transposeConst =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{transposeOrder.size()}, transposeOrder);
        auto uvTranspose = std::make_shared<ov::op::v1::Transpose>(uvInput, transposeConst);

        ov::op::v4::Interpolate::InterpolateAttrs attrs;
        attrs.mode = ov::op::v4::Interpolate::InterpolateMode::NEAREST;
        attrs.shape_calculation_mode = ov::op::v4::Interpolate::ShapeCalcMode::SIZES;
        attrs.coordinate_transformation_mode = ov::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC;
        attrs.nearest_mode = ov::op::v4::Interpolate::NearestMode::FLOOR;
        attrs.antialias = false;
        attrs.pads_begin = {0, 0, 0, 0};
        attrs.pads_end = {0, 0, 0, 0};
        attrs.cube_coeff = -0.75;

        auto targetSize = ov::op::v0::Constant::create(
                ov::element::i64, ov::Shape{2},
                std::vector<int64_t>{static_cast<int64_t>(outputShape[2]), static_cast<int64_t>(outputShape[3])});
        auto scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, std::vector<float>{2.0f, 2.0f});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{2, 3});

        auto interpolate = std::make_shared<ov::op::v4::Interpolate>(uvTranspose, targetSize, scales, axes, attrs);

        // Concat Y and UV channels
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{yReshape, interpolate}, 1);

        // OpenVINO standard
        // R = 1.164*(Y-16) + 1.596*(V-128)
        // G = 1.164*(Y-16) - 0.813*(V-128) - 0.391*(U-128)
        // B = 1.164*(Y-16) + 2.018*(U-128)

        // Weights are normalized: coefficient/255 to work with [0,1] range instead of [0,255]
        // R channel: 1.164/255≈0.00457, 2.018/255≈0.00792, 0
        // G channel: 1.164/255≈0.00457, -0.391/255≈-0.00152, -0.813/255≈-0.00318
        // B channel: 1.164/255≈0.00457, 0, 1.596/255≈0.00627

        std::vector<float> convWeights = {
                0.00456994f, 0.00792123f,  0.00000000f,   // R channel: Y, U, V
                0.00456994f, -0.00152331f, -0.00317720f,  // G channel: Y, U, V
                0.00456994f, 0.00000000f,  0.00626735f    // B channel: Y, U, V
        };

        auto weightsConst = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{3, 3, 1, 1}, convWeights);

        auto convolution = std::make_shared<ov::op::v1::Convolution>(concat, weightsConst, ov::Strides{1, 1},
                                                                     ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0},
                                                                     ov::Strides{1, 1});

        std::vector<float> biasValues = {-1.08561909f, 0.53161991f, -0.87418211f};
        auto biasConst = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 1, 1}, biasValues);

        auto add = std::make_shared<ov::op::v1::Add>(convolution, biasConst);

        auto clamp = std::make_shared<ov::op::v0::Clamp>(add, 0.0, 255.0);

        return clamp;
    }
};

TEST_P(FuseColorConversionTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FuseColorConversionTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

namespace {

static const std::vector<ColorConversionParams> precommit_testValues = {
        // Small resolution
        ColorConversionParams{32, 32},
        // Medium resolution
        ColorConversionParams{64, 48}};

static const std::vector<ColorConversionParams> smoke_testValues = {  // VGA resolution
        ColorConversionParams{480, 640},
        // HD resolution
        ColorConversionParams{720, 1280},
        // 4K resolution (scaled down for testing)
        ColorConversionParams{1080, 1920},
        // Square resolution
        ColorConversionParams{512, 512}};

INSTANTIATE_TEST_SUITE_P(precommit_FuseColorConversion, FuseColorConversionTestCommon,
                         ::testing::ValuesIn(precommit_testValues), FuseColorConversionTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseColorConversion, FuseColorConversionTestCommon,
                         ::testing::ValuesIn(smoke_testValues), FuseColorConversionTestCommon::getTestCaseName);

}  // namespace
}  // namespace ov::test::subgraph
