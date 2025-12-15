//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/prior_box.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/prior_box.hpp"
#include "openvino/op/shape_of.hpp"

using namespace ov::test::utils;

namespace ov {

namespace test {

class PriorBoxLayerTestCommon : public PriorBoxLayerTest, virtual public VpuOv2LayerTest {
    // Cloned 'SetUp' from OpenVino, but with constant foldings enabled.
    void SetUp() override {
        priorBoxSpecificParams specParams;
        ov::element::Type modelType;
        std::vector<InputShape> inputShapes;
        std::vector<float> min_size, max_size, aspect_ratio, density, fixed_ratio, fixed_size, variance;
        float step, offset, scale_all_sizes, min_max_aspect_ratios_order;
        bool clip, flip;
        std::tie(specParams, modelType, inputShapes, std::ignore) = GetParam();

        std::tie(min_size, max_size, aspect_ratio, density, fixed_ratio, fixed_size, clip, flip, step, offset, variance,
                 scale_all_sizes, min_max_aspect_ratios_order) = specParams;

        VpuOv2LayerTest::init_input_shapes(inputShapes);

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(modelType, VpuOv2LayerTest::inputDynamicShapes[0]),
                std::make_shared<ov::op::v0::Parameter>(modelType, VpuOv2LayerTest::inputDynamicShapes[1])};

        ov::op::v8::PriorBox::Attributes attributes;
        attributes.min_size = min_size;
        attributes.max_size = max_size;
        attributes.aspect_ratio = aspect_ratio;
        attributes.density = density;
        attributes.fixed_ratio = fixed_ratio;
        attributes.fixed_size = fixed_size;
        attributes.variance = variance;
        attributes.step = step;
        attributes.offset = offset;
        attributes.clip = clip;
        attributes.flip = flip;
        attributes.scale_all_sizes = scale_all_sizes;

        attributes.min_max_aspect_ratios_order = min_max_aspect_ratios_order;

        auto shape_of_1 = std::make_shared<ov::op::v3::ShapeOf>(params[0]);
        auto shape_of_2 = std::make_shared<ov::op::v3::ShapeOf>(params[1]);
        auto priorBox = std::make_shared<ov::op::v8::PriorBox>(shape_of_1, shape_of_2, attributes);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(priorBox)};
        VpuOv2LayerTest::function = std::make_shared<ov::Model>(results, params, "PriorBoxFunction");
    }
    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }
};

TEST_P(PriorBoxLayerTestCommon, NPU3720_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(PriorBoxLayerTestCommon, NPU4000_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}
TEST_P(PriorBoxLayerTestCommon, NPU5010_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const priorBoxSpecificParams param1 = {
        //(openvino eg)
        std::vector<float>{16.0},                // min_size
        std::vector<float>{38.46},               // max_size
        std::vector<float>{2.0},                 // aspect_ratio
        std::vector<float>{},                    // [density]
        std::vector<float>{},                    // [fixed_ratio]
        std::vector<float>{},                    // [fixed_size]
        false,                                   // clip
        true,                                    // flip
        16.0,                                    // step
        0.5,                                     // offset
        std::vector<float>{0.1, 0.1, 0.2, 0.2},  // variance
        false,                                   // [scale_all_sizes]
        false                                    // min_max_aspect_ratios_order ?
};

const priorBoxSpecificParams param2 = {
        std::vector<float>{2.0},  // min_size
        std::vector<float>{5.0},  // max_size
        std::vector<float>{1.5},  // aspect_ratio
        std::vector<float>{},     // [density]
        std::vector<float>{},     // [fixed_ratio]
        std::vector<float>{},     // [fixed_size]
        false,                    // clip
        false,                    // flip
        1.0,                      // step
        0.0,                      // offset
        std::vector<float>{},     // variance
        false,                    // [scale_all_sizes]
        false                     // min_max_aspect_ratios_order
};

const priorBoxSpecificParams param3 = {
        std::vector<float>{256.0},  // min_size
        std::vector<float>{315.0},  // max_size
        std::vector<float>{2.0},    // aspect_ratio
        std::vector<float>{},       // [density]
        std::vector<float>{},       // [fixed_ratio]
        std::vector<float>{},       // [fixed_size]
        true,                       // clip
        true,                       // flip
        1.0,                        // step
        0.0,                        // offset
        std::vector<float>{},       // variance
        true,                       // [scale_all_sizes]
        false                       // min_max_aspect_ratios_order
};

const priorBoxSpecificParams param4 = {
        //(openvino eg)
        std::vector<float>{8.0},                 // min_size
        std::vector<float>{19.23},               // max_size
        std::vector<float>{1.0},                 // aspect_ratio
        std::vector<float>{},                    // [density]
        std::vector<float>{},                    // [fixed_ratio]
        std::vector<float>{},                    // [fixed_size]
        false,                                   // clip
        true,                                    // flip
        8.0,                                     // step
        0.5,                                     // offset
        std::vector<float>{0.1, 0.1, 0.2, 0.2},  // variance
        false,                                   // [scale_all_sizes]
        false                                    // min_max_aspect_ratios_order ?
};

const std::vector<ov::Shape> inputShape1 = {{24, 42}, {348, 672}};  // inputShape, imageShape
const std::vector<ov::Shape> inputShape2 = {{2, 2}, {10, 10}};
const std::vector<ov::Shape> inputShape3 = {{1, 1}, {300, 300}};
const std::vector<ov::Shape> inputShape4 = {{1, 1}, {5, 5}};

const auto paramsConfig1 = testing::Combine(testing::Values(param1), testing::Values(ov::element::f16),
                                            testing::Values(static_shapes_to_test_representation(inputShape1)),
                                            testing::Values(test_utils::TARGET_DEVICE));

const auto paramsConfig2 = testing::Combine(testing::Values(param2), testing::Values(ov::element::f16),
                                            testing::Values(static_shapes_to_test_representation(inputShape2)),
                                            testing::Values(test_utils::TARGET_DEVICE));

const auto paramsConfig3 = testing::Combine(testing::Values(param3), testing::Values(ov::element::f16),
                                            testing::Values(static_shapes_to_test_representation(inputShape3)),
                                            testing::Values(test_utils::TARGET_DEVICE));

const auto paramsPrecommit = testing::Combine(testing::Values(param4), testing::Values(ov::element::f16),
                                              testing::Values(static_shapes_to_test_representation(inputShape4)),
                                              testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_PriorBox_1, PriorBoxLayerTestCommon, paramsConfig1,
                         PriorBoxLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PriorBox_2, PriorBoxLayerTestCommon, paramsConfig2,
                         PriorBoxLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PriorBox_3, PriorBoxLayerTestCommon, paramsConfig3,
                         PriorBoxLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_PriorBox, PriorBoxLayerTestCommon, paramsPrecommit,
                         PriorBoxLayerTestCommon::getTestCaseName);

}  // namespace
