//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/convert_color_i420.hpp"
#include "single_op_tests/convert_color_nv12.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class ConvertColorYUVLayerTestCommon : virtual public VpuOv2LayerTest {
protected:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        const auto& funcInputs = function->inputs();
        inputs.clear();
        for (size_t i = 0; i < inputShapes.size(); i++) {
            const auto& inputStaticShape = inputShapes[i];
            auto inputTensor = ov::Tensor{ov::element::f16, inputStaticShape};
            auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f16>::value_type>();
            const auto totalSize = ov::shape_size(inputStaticShape);

            // Generate YUV data in range [0, 255] for realistic image values
            for (size_t j = 0; j < totalSize; j++) {
                inputData[j] = static_cast<ov::float16>(rand() % 256);
            }
            inputs[funcInputs[i].get_node_shared_ptr()] = inputTensor;
        }
    }
};

class ConvertColorNV12LayerTestCommon : public ConvertColorNV12LayerTest, public ConvertColorYUVLayerTestCommon {};
class ConvertColorI420LayerTestCommon : public ConvertColorI420LayerTest, public ConvertColorYUVLayerTestCommon {};

class ConvertColorNV12M2ILayerTest : public ConvertColorNV12LayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] =
                "enable-m2i=true workload-management-enable=false";
    }
};
class ConvertColorI420M2ILayerTest : public ConvertColorI420LayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] =
                "enable-m2i=true workload-management-enable=false";
    }
};

// NPU3720
TEST_P(ConvertColorNV12LayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConvertColorI420LayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

// NPU4000
TEST_P(ConvertColorNV12LayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(ConvertColorI420LayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

// NPU4000 M2I
TEST_P(ConvertColorNV12M2ILayerTest, NPU4000_HW) {
    abs_threshold = 1.0f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(ConvertColorI420M2ILayerTest, NPU4000_HW) {
    abs_threshold = 1.0f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

enum ConvertColorType { I420, NV12 };
auto generate_input_static_shapes = [](const std::vector<ov::Shape>& original_shapes, ConvertColorType opType,
                                       bool single_plane) {
    std::vector<std::vector<ov::Shape>> result_shapes;
    for (const auto& original_shape : original_shapes) {
        std::vector<ov::Shape> one_result_shapes;
        if (single_plane) {
            auto shape = original_shape;
            shape[1] = shape[1] * 3 / 2;
            one_result_shapes.push_back(shape);
        } else {
            auto shape = original_shape;
            one_result_shapes.push_back(shape);
            if (opType == I420) {
                auto uvShape = ov::Shape{shape[0], shape[1] / 2, shape[2] / 2, 1};
                one_result_shapes.push_back(uvShape);
                one_result_shapes.push_back(uvShape);
            } else {
                auto uvShape = ov::Shape{shape[0], shape[1] / 2, shape[2] / 2, 2};
                one_result_shapes.push_back(uvShape);
            }
        }
        result_shapes.push_back(one_result_shapes);
    }
    return result_shapes;
};

// N,H,W,C
std::vector<ov::Shape> inShapes = {{1, 368, 432, 1}, {1, 4, 8, 1}, {1, 662, 982, 1}, {3, 128, 128, 1}};
std::vector<ov::Shape> inShapeM2I = {{1, 240, 320, 1}, {1, 64, 64, 1}};

ov::element::Type dTypes[] = {
        ov::element::f16,
};

auto inputShapeTrueI420 = generate_input_static_shapes(inShapes, I420, true);
auto inputShapeFalseI420 = generate_input_static_shapes(inShapes, I420, false);
auto inputShapeTrueNV12 = generate_input_static_shapes(inShapes, NV12, true);
auto inputShapeFalseNV12 = generate_input_static_shapes(inShapes, NV12, false);
// I420
const auto params_trueI420 =
        testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(inputShapeTrueI420)),
                         testing::ValuesIn(dTypes),     // elem Type
                         testing::Values(true, false),  // conv_to_RGB
                         testing::Values(true),         // is_single_plane
                         testing::Values(test_utils::TARGET_DEVICE));

const auto params_falseI420 =
        testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(inputShapeFalseI420)),
                         testing::ValuesIn(dTypes),     // elem Type
                         testing::Values(true, false),  // conv_to_RGB
                         testing::Values(false),        // is_single_plane
                         testing::Values(test_utils::TARGET_DEVICE));
// NV12
const auto params_trueNV12 =
        testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(inputShapeTrueNV12)),
                         testing::ValuesIn(dTypes),     // elem Type
                         testing::Values(true, false),  // conv_to_RGB
                         testing::Values(true),         // is_single_plane
                         testing::Values(test_utils::TARGET_DEVICE));

const auto params_falseNV12 =
        testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(inputShapeFalseNV12)),
                         testing::ValuesIn(dTypes),     // elem Type
                         testing::Values(true, false),  // conv_to_RGB
                         testing::Values(false),        // is_single_plane
                         testing::Values(test_utils::TARGET_DEVICE));
// Case for 4000 M2I
auto inputShapeM2ITrueI420 = generate_input_static_shapes(inShapeM2I, I420, true);
auto inputShapeM2ITrueNV12 = generate_input_static_shapes(inShapeM2I, NV12, true);

// I420
const auto paramsM2II420 =
        testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(inputShapeM2ITrueI420)),  // QVGA
                         testing::Values(ov::element::u8),                                                // elem Type
                         testing::Values(true, false),                                                    // conv_to_RGB
                         testing::Values(true),  // is_single_plane
                         testing::Values(test_utils::TARGET_DEVICE));
// NV12
const auto paramsM2INV12 =
        testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(inputShapeM2ITrueNV12)),  // QVGA
                         testing::Values(ov::element::u8),                                                // elem Type
                         testing::Values(true, false),                                                    // conv_to_RGB
                         testing::Values(true),  // is_single_plane
                         testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorNV12_true, ConvertColorNV12LayerTestCommon, params_trueNV12,
                         ConvertColorNV12LayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorNV12_false, ConvertColorNV12LayerTestCommon, params_falseNV12,
                         ConvertColorNV12LayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorI420_true, ConvertColorI420LayerTestCommon, params_trueI420,
                         ConvertColorI420LayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorI420_false, ConvertColorI420LayerTestCommon, params_falseI420,
                         ConvertColorI420LayerTestCommon::getTestCaseName);

// M2I
// [Tracking number: E#107046]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_ConvertColorNV12_M2I, ConvertColorNV12M2ILayerTest, paramsM2INV12,
                         ConvertColorNV12M2ILayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_ConvertColorI420_M2I, ConvertColorI420M2ILayerTest, paramsM2II420,
                         ConvertColorI420M2ILayerTest::getTestCaseName);

}  // namespace
