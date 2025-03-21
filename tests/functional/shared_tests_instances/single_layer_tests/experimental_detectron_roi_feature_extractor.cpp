// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include <vector>

#include "single_op_tests/experimental_detectron_roifeatureextractor.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
namespace ov {

namespace test {
class ExperimentalDetectronROIFeatureExtractorLayerTestCommon :
        public ExperimentalDetectronROIFeatureExtractorLayerTest,
        virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t ind = 0; ind < funcInputs.size(); ind++) {
            ov::Tensor tensorData = create_and_fill_tensor(funcInputs[ind].get_element_type(),
                                                           targetInputStaticShapes[ind], 200, -100, 2, 1);
            inputs.insert({funcInputs[ind].get_node_shared_ptr(), tensorData});
        }
    }

    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

TEST_P(ExperimentalDetectronROIFeatureExtractorLayerTestCommon, NPU3720_SW) {
    const auto type = std::get<5>(GetParam());

    // adjusted for differences when rounding to fp16
    if (type == ov::element::f16) {
        abs_threshold = 0.05f;
    }
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ExperimentalDetectronROIFeatureExtractorLayerTestCommon, NPU4000_SW) {
    const auto type = std::get<5>(GetParam());

    // adjusted for differences when rounding to fp16
    if (type == ov::element::f16) {
        abs_threshold = 0.05f;
    }
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test

}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<int64_t> outputSize = {3, 7};
const std::vector<int64_t> samplingRatio = {0, 2};
const std::vector<bool> aligned = {true, false};

const std::vector<std::vector<int64_t>> pyramidScales = {
        {4, 8, 16}};  // 3 scales, 3 feature inputs, each scale is divided with the original input image which is the
                      // value of scale*(feature_image_height) and scale*(feature_image_width)

const std::vector<std::vector<InputShape>> inputShapesConfig0 = {
        static_shapes_to_test_representation({{100, 4}, {1, 64, 192, 320}, {1, 64, 96, 160}}),
        static_shapes_to_test_representation({{100, 4}, {1, 64, 192, 320}, {1, 64, 96, 160}, {1, 64, 48, 80}}),
};

const std::vector<std::vector<InputShape>> inputShapesConfig1 = {
        static_shapes_to_test_representation({{10, 4}, {1, 64, 96, 160}}),
        static_shapes_to_test_representation({{100, 4}, {1, 64, 48, 80}}),
        static_shapes_to_test_representation({{100, 4}, {1, 2, 10, 10}}),

};

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalROIExtractor, ExperimentalDetectronROIFeatureExtractorLayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(inputShapesConfig0), ::testing::ValuesIn(outputSize),
                                            ::testing::ValuesIn(samplingRatio), ::testing::ValuesIn(pyramidScales),
                                            ::testing::ValuesIn(aligned),
                                            ::testing::Values(ov::element::f32, ov::element::f16),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU)),
                         ExperimentalDetectronROIFeatureExtractorLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(precommit_ExperimentalROIExtractor, ExperimentalDetectronROIFeatureExtractorLayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(inputShapesConfig1), ::testing::ValuesIn(outputSize),
                                            ::testing::ValuesIn(samplingRatio), ::testing::ValuesIn(pyramidScales),
                                            ::testing::ValuesIn(aligned),
                                            ::testing::Values(ov::element::f32, ov::element::f16),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU)),
                         ExperimentalDetectronROIFeatureExtractorLayerTest::getTestCaseName);

}  // namespace
