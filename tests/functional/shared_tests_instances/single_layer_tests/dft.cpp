//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/dft.hpp"
#include <algorithm>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <vector>
#include "common_test_utils/node_builders/dft.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
using ov::test::utils::DFTOpType;

namespace ov {
namespace test {

class DftLayerTestCommon : public DFTLayerTest, virtual public VpuOv2LayerTest {
    // C-125993
    // Reduce resolution of ov::float16 data generation to prevent NaN values
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        ov::test::utils::InputGenerateData inGenData;
        inGenData.range = 8;
        inGenData.resolution = 32;
        const auto& funcInputs = function->inputs();
        auto funcInput = funcInputs.begin();
        ov::Tensor dataTensor = ov::test::utils::create_and_fill_tensor_act_dft(
                funcInput->get_element_type(), targetInputStaticShapes[0], inGenData.range, inGenData.start_from,
                inGenData.resolution, inGenData.seed);
        inputs.insert({funcInput->get_node_shared_ptr(), dataTensor});
    }
    void SetUp() override {
        ov::element::Type modelType = std::get<1>(GetParam());
        const auto axes = std::get<2>(GetParam());
        if (modelType == ov::element::f16) {
            rel_threshold = 0.15f * axes.size();
        }
        DFTLayerTest::SetUp();
    }
};
TEST_P(DftLayerTestCommon, NPU3720) {
    abs_threshold = 0.2;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DftLayerTestCommon, NPU4000) {
    abs_threshold = 0.2;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

namespace {
using namespace ov::test;

const std::vector<DFTOpType> opTypes = {
        DFTOpType::FORWARD,
        DFTOpType::INVERSE,
};

const std::vector<ov::element::Type> inputType = {
        // disable FP32  tests as default compiler pipelines pass createConvertPrecisionToFP16Pass will convert anyway
        // to fp16 the operation, so test precision will be precision for fp16
        // ov::element::f32,
        ov::element::f16,
};

const auto combine = [](const std::vector<std::vector<ov::Shape>>& inputShapes,
                        const std::vector<std::vector<int64_t>>& axes,
                        const std::vector<std::vector<int64_t>>& signalSizes) {
    return testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(inputShapes)),
                            testing::ValuesIn(inputType), testing::ValuesIn(axes), testing::ValuesIn(signalSizes),
                            testing::ValuesIn(opTypes), testing::Values(DEVICE_NPU));
};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DFT_2d, DftLayerTestCommon,
                         combine(std::vector<std::vector<ov::Shape>>{{{10, 2}}},  // input shapes
                                 {{0}},                                           // axes
                                 {{}, {3}}),                                      // signal sizes
                         DFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DFT_3d, DftLayerTestCommon,
                         combine(std::vector<std::vector<ov::Shape>>{{{10, 4, 2}}},  // input shapes
                                 {{0, 1}},                                           // axes
                                 {{}, {3, 10}}),                                     // signal sizes
                         DFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DFT_4d, DftLayerTestCommon,
                         combine(std::vector<std::vector<ov::Shape>>{{{10, 4, 8, 2}}},  // input shapes
                                 {{0, 1, 2}, {1, 2, 0}},                                // axes
                                 {{}, {3, 10, 8}}),                                     // signal sizes
                         DFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DFT_4d_negative_reversed_axes, DftLayerTestCommon,
                         combine(std::vector<std::vector<ov::Shape>>{{{10, 4, 8, 2}}},  // input shapes
                                 {{-1, -2, -3}},                                        // axes
                                 {{}, {8, 10, 3}}),                                     // signal sizes
                         DFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DFT_4d_single_axis, DftLayerTestCommon,
                         combine(std::vector<std::vector<ov::Shape>>{{{10, 4, 8, 2}}},  // input shapes
                                 {{0}, {1}, {2}},                                       // axes
                                 {{}, {1}, {5}, {20}}),                                 // signal sizes
                         DFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DFT_5d, DftLayerTestCommon,
                         combine(std::vector<std::vector<ov::Shape>>{{{10, 4, 8, 2, 2}}},  // input shapes
                                 {{0, 1, 2, 3}},                                           // axes
                                 {{}, {3, 10, 8, 6}}),                                     // signal sizes
                         DFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DFT_5d_tile, DftLayerTestCommon,
                         combine(std::vector<std::vector<ov::Shape>>{{{1, 80, 64, 64, 2}}},  // input shapes
                                 {{2, 3}},                                                   // axes
                                 {{}}),                                                      // signal sizes
                         DFTLayerTest::getTestCaseName);

}  // namespace
