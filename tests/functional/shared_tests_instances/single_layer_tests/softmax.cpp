//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/softmax.hpp"
#include "pretty_test_arguments.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test {

class SoftMaxLayerTestCommon : public subgraph::SoftMaxLayerTest, virtual public VpuOv2LayerTest {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

struct SkipDynamicShapes {
    SkipDynamicShapes(SoftMaxLayerTestCommon::ParamType params): params(std::move(params)) {
    }

    inline void operator()(std::stringstream& skip) const {
        const auto inputShapes = std::get<3>(params);
        const auto partialShape = inputShapes.first;
        if (partialShape.is_dynamic()) {
            skip << "Dynamic shapes are not supported";
        }
    }

    SoftMaxLayerTestCommon::ParamType params;
};

// SW pipeline tests are needed to test different axis
// HW pipeline adds reorder to put axis dimension last
TEST_P(SoftMaxLayerTestCommon, NPU3720_SW) {
    setSkipCompilationCallback(SkipDynamicShapes(GetParam()));

    abs_threshold = 0.01;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(SoftMaxLayerTestCommon, NPU3720_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(SoftMaxLayerTestCommon, NPU4000_SW) {
    setSkipCompilationCallback(SkipDynamicShapes(GetParam()));

    abs_threshold = 1e-3;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(SoftMaxLayerTestCommon, NPU4000_HW) {
    abs_threshold = 1e-3;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(SoftMaxLayerTestCommon, NPU5010_SW) {
    setSkipCompilationCallback(SkipDynamicShapes(GetParam()));

    abs_threshold = 1e-3;
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}

TEST_P(SoftMaxLayerTestCommon, NPU5010_HW) {
    setSkipCompilationCallback(SkipDynamicShapes(GetParam()));

    abs_threshold = 1e-3;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
}  // namespace ov::test

using ov::test::SoftMaxLayerTestCommon;

namespace {

//
// Input 2D
//

const std::vector<ov::Shape> inShapes2D = {
        {1, 100}, {100, 1}, {10, 10}, {32, 76}, {72, 2},
};

const std::vector<size_t> axis2D = {0, 1};

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax2D, SoftMaxLayerTestCommon,
        testing::Combine(testing::Values(ov::element::f16),                                              // Model type
                         testing::Values(ov::element::f16),                                              // In type
                         testing::Values(ov::element::f16),                                              // Out type
                         testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes2D)),  // Shape
                         testing::ValuesIn(axis2D),                                                      // Axis
                         testing::Values(test_utils::TARGET_DEVICE), testing::Values(ov::test::Config{})),
        SoftMaxLayerTestCommon::getTestCaseName);

//
// Input 3D
//

const std::vector<ov::Shape> inShapes3D = {{1, 4300, 2}, {8, 182, 182}};

const std::vector<size_t> axis3D = {2};

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax3D, SoftMaxLayerTestCommon,
        testing::Combine(testing::Values(ov::element::f16),                                              // Model type
                         testing::Values(ov::element::f16),                                              // In type
                         testing::Values(ov::element::f16),                                              // Out type
                         testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes3D)),  // Shape
                         testing::ValuesIn(axis3D),                                                      // Axis
                         testing::Values(test_utils::TARGET_DEVICE), testing::Values(ov::test::Config{})),
        SoftMaxLayerTestCommon::getTestCaseName);

//
// Input 4D
//

const std::vector<ov::Shape> inShapes4D = {{1, 2, 108, 60}, {1, 12, 2, 148}, {1, 4, 1, 1}, {1, 100, 1, 1},
                                           {300, 21, 1, 1}, {1, 2, 48, 2},   {1, 3, 83, 4}};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax4D, SoftMaxLayerTestCommon,
        testing::Combine(testing::Values(ov::element::f16),                                              // Model type
                         testing::Values(ov::element::f16),                                              // In type
                         testing::Values(ov::element::f16),                                              // Out type
                         testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes4D)),  // Shape
                         testing::ValuesIn(axis4D),                                                      // Axis
                         testing::Values(test_utils::TARGET_DEVICE), testing::Values(ov::test::Config{})),
        SoftMaxLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_precommit_SoftMax4D, SoftMaxLayerTestCommon,
        testing::Combine(testing::Values(ov::element::f16),  // Model type
                         testing::Values(ov::element::f16),  // In type
                         testing::Values(ov::element::f16),  // Out type
                         testing::ValuesIn(ov::test::static_shapes_to_test_representation({{1, 2, 72, 10}})),  // Shape
                         testing::ValuesIn(axis4D),                                                            // Axis
                         testing::Values(test_utils::TARGET_DEVICE), testing::Values(ov::test::Config{})),
        SoftMaxLayerTestCommon::getTestCaseName);

//
// Input 5D
//

const std::vector<ov::Shape> inShapes5D = {{8, 1, 1, 512, 64}};
const std::vector<size_t> axis5D = {0, 1, 2, 3, 4};

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax5D, SoftMaxLayerTestCommon,
        testing::Combine(testing::Values(ov::element::f16),                                              // Model type
                         testing::Values(ov::element::f16),                                              // In type
                         testing::Values(ov::element::f16),                                              // Out type
                         testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes5D)),  // Shape
                         testing::ValuesIn(axis5D),                                                      // Axis
                         testing::Values(test_utils::TARGET_DEVICE), testing::Values(ov::test::Config{})),
        SoftMaxLayerTestCommon::getTestCaseName);

//
// Test FP32 functionality
//

INSTANTIATE_TEST_SUITE_P(
        smoke_precommit_SoftMaxFP32, SoftMaxLayerTestCommon,
        testing::Combine(testing::Values(ov::element::f32),  // Model type
                         testing::Values(ov::element::f32),  // In type
                         testing::Values(ov::element::f32),  // Out type
                         testing::ValuesIn(ov::test::static_shapes_to_test_representation({{1, 2, 72, 10}})),  // Shape
                         testing::Values(2),                                                                   // Axis
                         testing::Values(test_utils::TARGET_DEVICE), testing::Values(ov::test::Config{})),
        SoftMaxLayerTestCommon::getTestCaseName);

//
// Test tiling functionality
//

INSTANTIATE_TEST_SUITE_P(smoke_TilingSoftMax, SoftMaxLayerTestCommon,
                         testing::Combine(testing::Values(ov::element::f16),  // Model type
                                          testing::Values(ov::element::f16),  // In type
                                          testing::Values(ov::element::f16),  // Out type
                                          testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                  {{1, 20, 64, 512}})),  // Shape
                                          testing::Values(2),            // Axis
                                          testing::Values(test_utils::TARGET_DEVICE),
                                          testing::Values(ov::test::Config{})),
                         SoftMaxLayerTestCommon::getTestCaseName);

//
// Dynamic shape use cases
//

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SoftMax_DynUseCase0, SoftMaxLayerTestCommon,
                         testing::Combine(testing::Values(ov::element::f16),                   // Model type
                                          testing::Values(ov::element::f16),                   // In type
                                          testing::Values(ov::element::f16),                   // Out type
                                          testing::Values(generateTestShape(32_Dyn, 1, 548)),  // Shape
                                          testing::Values(2),                                  // Axis
                                          testing::Values(test_utils::TARGET_DEVICE),
                                          testing::Values(ov::test::Config{})),
                         SoftMaxLayerTestCommon::getTestCaseName);

//
// Dynamic shapes smoke
//

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax_3D_DynSmoke_Axis2, SoftMaxLayerTestCommon,
                         testing::Combine(testing::Values(ov::element::f16),                       // Model type
                                          testing::Values(ov::element::f16),                       // In type
                                          testing::Values(ov::element::f16),                       // Out type
                                          testing::Values(generateTestShape(32_Dyn, 32_Dyn, 64)),  // Shape
                                          testing::Values(2),                                      // Axis
                                          testing::Values(test_utils::TARGET_DEVICE),
                                          testing::Values(ov::test::Config{})),
                         SoftMaxLayerTestCommon::getTestCaseName);

// Hangs: E#145670
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_SoftMax_3D_DynSmoke_Axis1, SoftMaxLayerTestCommon,
                         testing::Combine(testing::Values(ov::element::f16),                       // Model type
                                          testing::Values(ov::element::f16),                       // In type
                                          testing::Values(ov::element::f16),                       // Out type
                                          testing::Values(generateTestShape(32_Dyn, 32, 64_Dyn)),  // Shape
                                          testing::Values(1),                                      // Axis
                                          testing::Values(test_utils::TARGET_DEVICE),
                                          testing::Values(ov::test::Config{})),
                         SoftMaxLayerTestCommon::getTestCaseName);

// Hangs: E#145670
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_SoftMax_3D_DynSmoke_Axis0, SoftMaxLayerTestCommon,
                         testing::Combine(testing::Values(ov::element::f16),                       // Model type
                                          testing::Values(ov::element::f16),                       // In type
                                          testing::Values(ov::element::f16),                       // Out type
                                          testing::Values(generateTestShape(32, 32_Dyn, 64_Dyn)),  // Shape
                                          testing::Values(0),                                      // Axis
                                          testing::Values(test_utils::TARGET_DEVICE),
                                          testing::Values(ov::test::Config{})),
                         SoftMaxLayerTestCommon::getTestCaseName);

}  // namespace
