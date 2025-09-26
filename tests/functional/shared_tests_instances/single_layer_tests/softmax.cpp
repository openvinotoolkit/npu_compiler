//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/softmax.hpp"
#include "pretty_test_arguments.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test {

class SoftMaxLayerTestCommon : public subgraph::SoftMaxLayerTest, virtual public VpuOv2LayerTest {};

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
}  // namespace ov::test

using ov::test::SoftMaxLayerTestCommon;

namespace {

const std::vector<ov::test::ElementType> modelTypes = {
        ov::element::f16,
};

const std::vector<ov::test::ElementType> inputTypes = {
        ov::element::f16,
};

const std::vector<ov::test::ElementType> outputTypes = {
        ov::element::f16,
};

//
// Input 2D
//

const std::vector<ov::Shape> inShapes2D = {
        {1, 100}, {100, 1}, {10, 10}, {32, 76}, {72, 2},
};

const std::vector<size_t> axis2D = {0, 1};

const auto params2D = testing::Combine(
        testing::ValuesIn(modelTypes), testing::ValuesIn(inputTypes), testing::ValuesIn(outputTypes),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes2D)), testing::ValuesIn(axis2D),
        testing::Values(test_utils::TARGET_DEVICE), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax2D, SoftMaxLayerTestCommon, params2D, SoftMaxLayerTestCommon::getTestCaseName);

//
// Input 3D
//

const std::vector<ov::Shape> inShapes3D = {{1, 4300, 2}, {8, 182, 182}};

const std::vector<size_t> axis3D = {2};

const auto params3D = testing::Combine(
        testing::ValuesIn(modelTypes), testing::ValuesIn(inputTypes), testing::ValuesIn(outputTypes),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes3D)), testing::ValuesIn(axis3D),
        testing::Values(test_utils::TARGET_DEVICE), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax3D, SoftMaxLayerTestCommon, params3D, SoftMaxLayerTestCommon::getTestCaseName);

//
// Input 4D
//

const std::vector<ov::Shape> inShapes4D = {{1, 2, 108, 60}, {1, 12, 2, 148}, {1, 4, 1, 1}, {1, 100, 1, 1},
                                           {300, 21, 1, 1}, {1, 2, 48, 2},   {1, 3, 83, 4}};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

const auto params4D = testing::Combine(
        testing::ValuesIn(modelTypes), testing::ValuesIn(inputTypes), testing::ValuesIn(outputTypes),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes4D)), testing::ValuesIn(axis4D),
        testing::Values(test_utils::TARGET_DEVICE), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax4D, SoftMaxLayerTestCommon, params4D, SoftMaxLayerTestCommon::getTestCaseName);

//
// Input 5D
//

const std::vector<ov::Shape> inShapes5D = {{8, 1, 1, 512, 64}};
const std::vector<size_t> axis5D = {0, 1, 2, 3, 4};

const auto params5D = testing::Combine(
        testing::ValuesIn(modelTypes), testing::ValuesIn(inputTypes), testing::ValuesIn(outputTypes),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes5D)), testing::ValuesIn(axis5D),
        testing::Values(test_utils::TARGET_DEVICE), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax5D, SoftMaxLayerTestCommon, params5D, SoftMaxLayerTestCommon::getTestCaseName);

const auto precommit_params4D = testing::Combine(
        testing::ValuesIn(modelTypes), testing::ValuesIn(inputTypes), testing::ValuesIn(outputTypes),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation({{1, 2, 72, 10}})), testing::ValuesIn(axis4D),
        testing::Values(test_utils::TARGET_DEVICE), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SoftMax4D, SoftMaxLayerTestCommon, precommit_params4D,
                         SoftMaxLayerTestCommon::getTestCaseName);

//
// Test tiling functionality
//

const std::vector<ov::Shape> inShapes = {{1, 20, 64, 512}};
const std::vector<size_t> axis = {2};

const auto paramsTilingCases = testing::Combine(
        testing::ValuesIn(modelTypes), testing::ValuesIn(inputTypes), testing::ValuesIn(outputTypes),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)), testing::ValuesIn(axis),
        testing::Values(test_utils::TARGET_DEVICE), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_TilingSoftMax, SoftMaxLayerTestCommon, paramsTilingCases,
                         SoftMaxLayerTestCommon::getTestCaseName);

//
// Dynamic shape use cases
//

const std::vector<ov::test::InputShape> inShapesDynUseCase0 = {
        generateTestShape(32_Dyn, 1, 548),
};

const std::vector<size_t> axisDynUseCase0 = {2};

const auto paramsDynUseCase0 =
        testing::Combine(testing::ValuesIn(modelTypes), testing::ValuesIn(inputTypes), testing::ValuesIn(outputTypes),
                         testing::ValuesIn(inShapesDynUseCase0), testing::ValuesIn(axisDynUseCase0),
                         testing::Values(test_utils::TARGET_DEVICE), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SoftMax_DynUseCase0, SoftMaxLayerTestCommon, paramsDynUseCase0,
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
