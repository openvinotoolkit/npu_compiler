//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/dimension.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"
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

// Suppression for gtest framework internal test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ConvertColorNV12LayerTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ConvertColorI420LayerTest);

class ConvertColorNV12LayerTestCommon : public ConvertColorNV12LayerTest, public ConvertColorYUVLayerTestCommon {};
class ConvertColorI420LayerTestCommon : public ConvertColorI420LayerTest, public ConvertColorYUVLayerTestCommon {};

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

// NPU5010
TEST_P(ConvertColorNV12LayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(ConvertColorI420LayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
// NPU5020
TEST_P(ConvertColorNV12LayerTestCommon, NPU5020_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

TEST_P(ConvertColorI420LayerTestCommon, NPU5020_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

enum ConvertColorType { I420, NV12 };
auto generateInputPartialShape = [](const std::vector<std::vector<ov::Dimension>>& originalShapes,
                                    ConvertColorType opType, bool singlePlane) {
    std::vector<std::vector<ov::PartialShape>> allInputShapes;
    for (const auto& originalShape : originalShapes) {
        std::vector<ov::PartialShape> inputShapes;
        if (singlePlane) {
            auto shape = originalShape;
            shape[1] = shape[1] * 3 / 2;
            inputShapes.emplace_back(shape);
        } else {
            auto shape = originalShape;
            inputShapes.emplace_back(shape);
            if (opType == I420) {
                auto uvShape = ov::PartialShape{shape[0], shape[1] / 2, shape[2] / 2, 1};
                inputShapes.push_back(uvShape);
                inputShapes.push_back(uvShape);
            } else {
                auto uvShape = ov::PartialShape{shape[0], shape[1] / 2, shape[2] / 2, 2};
                inputShapes.push_back(uvShape);
            }
        }
        allInputShapes.push_back(inputShapes);
    }
    return allInputShapes;
};

// N,H,W,C
std::vector<std::vector<ov::Dimension>> inShapes = {{1, 368, 432, 1}, {1, 4, 8, 1}, {1, 662, 982, 1}, {3, 128, 128, 1}};

ov::element::Type dTypes[] = {
        ov::element::f16,
};

auto inputShapeTrueI420 = generateInputPartialShape(inShapes, I420, true);
auto inputShapeFalseI420 = generateInputPartialShape(inShapes, I420, false);
auto inputShapeTrueNV12 = generateInputPartialShape(inShapes, NV12, true);
auto inputShapeFalseNV12 = generateInputPartialShape(inShapes, NV12, false);

inline std::vector<std::vector<InputShape>> partial_shapes_to_test_representation(
        const std::vector<std::vector<ov::PartialShape>>& shapes) {
    std::vector<std::vector<InputShape>> result;
    for (const auto& partialShapes : shapes) {
        std::vector<InputShape> inputShapes;
        for (const auto& partialShape : partialShapes) {
            // Test runtime shape equal to upper bounds
            // E#183027 Incorrect result if the runtime shape is multiple of the step size.
            // Use only full size
            ov::Shape staticShape;
            for (const auto& dim : partialShape) {
                staticShape.push_back(dim.get_interval().get_max_val());
            }

            inputShapes.push_back({{partialShape}, {staticShape}});
        }
        result.push_back(inputShapes);
    }
    return result;
}

//
// Static shapes
//

// I420
const auto paramsTrueI420 =
        testing::Combine(testing::ValuesIn(partial_shapes_to_test_representation(inputShapeTrueI420)),
                         testing::ValuesIn(dTypes),     // elem Type
                         testing::Values(true, false),  // conv_to_RGB
                         testing::Values(true),         // is_single_plane
                         testing::Values(test_utils::TARGET_DEVICE));

const auto paramsFalseI420 =
        testing::Combine(testing::ValuesIn(partial_shapes_to_test_representation(inputShapeFalseI420)),
                         testing::ValuesIn(dTypes),     // elem Type
                         testing::Values(true, false),  // conv_to_RGB
                         testing::Values(false),        // is_single_plane
                         testing::Values(test_utils::TARGET_DEVICE));
// NV12
const auto paramsTrueNV12 =
        testing::Combine(testing::ValuesIn(partial_shapes_to_test_representation(inputShapeTrueNV12)),
                         testing::ValuesIn(dTypes),     // elem Type
                         testing::Values(true, false),  // conv_to_RGB
                         testing::Values(true),         // is_single_plane
                         testing::Values(test_utils::TARGET_DEVICE));

const auto paramsFalseNV12 =
        testing::Combine(testing::ValuesIn(partial_shapes_to_test_representation(inputShapeFalseNV12)),
                         testing::ValuesIn(dTypes),     // elem Type
                         testing::Values(true, false),  // conv_to_RGB
                         testing::Values(false),        // is_single_plane
                         testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorNV12_true, ConvertColorNV12LayerTestCommon, paramsTrueNV12,
                         ConvertColorNV12LayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorNV12_false, ConvertColorNV12LayerTestCommon, paramsFalseNV12,
                         ConvertColorNV12LayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorI420_true, ConvertColorI420LayerTestCommon, paramsTrueI420,
                         ConvertColorI420LayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorI420_false, ConvertColorI420LayerTestCommon, paramsFalseI420,
                         ConvertColorI420LayerTestCommon::getTestCaseName);

//
// Static shapes with scf.for
//

class ConvertColorNV12LayerTestCommon_Scf : public ConvertColorNV12LayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "scf-tiling=true";
        // E-190336 for MC support
        configuration["NPU_TILES"] = "1";
    }
};

class ConvertColorI420LayerTestCommon_Scf : public ConvertColorI420LayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "scf-tiling=true";
        // E-190336 for MC support
        configuration["NPU_TILES"] = "1";
    }
};

// NPU5010
TEST_P(ConvertColorNV12LayerTestCommon_Scf, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(ConvertColorI420LayerTestCommon_Scf, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

// E#208158
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_ConvertColorNV12_single_input, ConvertColorNV12LayerTestCommon_Scf,
                         paramsTrueNV12, ConvertColorNV12LayerTestCommon_Scf::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorNV12_multi_input, ConvertColorNV12LayerTestCommon_Scf, paramsFalseNV12,
                         ConvertColorNV12LayerTestCommon_Scf::getTestCaseName);

// E#208158
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_ConvertColorI420_single_input, ConvertColorI420LayerTestCommon_Scf,
                         paramsTrueI420, ConvertColorI420LayerTestCommon_Scf::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorI420_multi_input, ConvertColorI420LayerTestCommon_Scf, paramsFalseI420,
                         ConvertColorI420LayerTestCommon_Scf::getTestCaseName);

//
// Dynamic shapes
//

class ConvertColorNV12LayerTestCommon_HostCompile : public ConvertColorNV12LayerTestCommon {};
class ConvertColorI420LayerTestCommon_HostCompile : public ConvertColorI420LayerTestCommon {};

// NPU4000
TEST_P(ConvertColorNV12LayerTestCommon_HostCompile, NPU4000_HC) {
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(ConvertColorI420LayerTestCommon_HostCompile, NPU4000_HC) {
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU4000);
}

// NPU5010
TEST_P(ConvertColorNV12LayerTestCommon_HostCompile, NPU5010_HC) {
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU5010);
}

TEST_P(ConvertColorI420LayerTestCommon_HostCompile, NPU5010_HC) {
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU5010);
}

std::vector<std::vector<ov::Dimension>> inShapesDynamic = {
        {1, ov::Dimension(1, 1440), ov::Dimension(1, 2560), 1},
};

auto inputShapeTrueI420Dynamic = generateInputPartialShape(inShapesDynamic, I420, true);
auto inputShapeFalseI420Dynamic = generateInputPartialShape(inShapesDynamic, I420, false);
auto inputShapeTrueNV12Dynamic = generateInputPartialShape(inShapesDynamic, NV12, true);
auto inputShapeFalseNV12Dynamic = generateInputPartialShape(inShapesDynamic, NV12, false);

// I420
const auto paramsTrueI420Dynamic =
        testing::Combine(testing::ValuesIn(partial_shapes_to_test_representation(inputShapeTrueI420Dynamic)),
                         testing::ValuesIn(dTypes),     // elem Type
                         testing::Values(true, false),  // conv_to_RGB
                         testing::Values(true),         // is_single_plane
                         testing::Values(test_utils::TARGET_DEVICE));

const auto paramsFalseI420Dynamic =
        testing::Combine(testing::ValuesIn(partial_shapes_to_test_representation(inputShapeFalseI420Dynamic)),
                         testing::ValuesIn(dTypes),     // elem Type
                         testing::Values(true, false),  // conv_to_RGB
                         testing::Values(false),        // is_single_plane
                         testing::Values(test_utils::TARGET_DEVICE));
// NV12
const auto paramsTrueNV12Dynamic =
        testing::Combine(testing::ValuesIn(partial_shapes_to_test_representation(inputShapeTrueNV12Dynamic)),
                         testing::ValuesIn(dTypes),     // elem Type
                         testing::Values(true, false),  // conv_to_RGB
                         testing::Values(true),         // is_single_plane
                         testing::Values(test_utils::TARGET_DEVICE));

const auto paramsFalseNV12Dynamic =
        testing::Combine(testing::ValuesIn(partial_shapes_to_test_representation(inputShapeFalseNV12Dynamic)),
                         testing::ValuesIn(dTypes),     // elem Type
                         testing::Values(true, false),  // conv_to_RGB
                         testing::Values(false),        // is_single_plane
                         testing::Values(test_utils::TARGET_DEVICE));

// E#208158
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_ConvertColorNV12_single_input, ConvertColorNV12LayerTestCommon_HostCompile,
                         paramsTrueNV12Dynamic, ConvertColorNV12LayerTestCommon_HostCompile::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorNV12_multi_input, ConvertColorNV12LayerTestCommon_HostCompile,
                         paramsFalseNV12Dynamic, ConvertColorNV12LayerTestCommon_HostCompile::getTestCaseName);

// E#208158
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_ConvertColorI420_single_input, ConvertColorI420LayerTestCommon_HostCompile,
                         paramsTrueI420Dynamic, ConvertColorI420LayerTestCommon_HostCompile::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorI420_multi_input, ConvertColorI420LayerTestCommon_HostCompile,
                         paramsFalseI420Dynamic, ConvertColorI420LayerTestCommon_HostCompile::getTestCaseName);

}  // namespace
