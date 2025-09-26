//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/grn.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class GRNLayerTestCommon : public GrnLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(GRNLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GRNLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {
        ov::element::f32,  // Testing FP32/FP16 netPrecision functionality only for small scope of
        ov::element::f16   // tests: GRNLayerTest, SplitLayerTest, CTCGreedyDecoderLayerTest
};

const std::vector<float> biases = {
        0.33f,
        1.1f,
};

// OV cases
const std::vector<std::vector<ov::Shape>> inShapes = {{{1, 8, 24, 64}}, {{3, 16, 1, 24}}, {{2, 16, 15, 20}}};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRN, GRNLayerTestCommon,
                         testing::Combine(testing::ValuesIn(modelTypes),
                                          testing::ValuesIn(static_shapes_to_test_representation(inShapes)),
                                          testing::ValuesIn(biases), testing::Values(test_utils::TARGET_DEVICE)),
                         GRNLayerTestCommon::getTestCaseName);

}  // namespace
