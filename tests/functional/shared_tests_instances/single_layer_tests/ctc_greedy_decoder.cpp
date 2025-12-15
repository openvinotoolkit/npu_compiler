//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/ctc_greedy_decoder.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class CTCGreedyDecoderLayerTestCommon : public CTCGreedyDecoderLayerTest, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 8, 0, 32);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }
};

TEST_P(CTCGreedyDecoderLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(CTCGreedyDecoderLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(CTCGreedyDecoderLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,  // Testing FP32/FP16 netPrecision functionality only for small scope of
        ov::element::f16   // tests: GRNLayerTest, SplitLayerTest, CTCGreedyDecoderLayerTest
};

const std::vector<bool> mergeRepeated = {true, false};

// Only batch = 1 is supported
const std::vector<std::vector<ov::Shape>> inputShapes_MLIR = {
        {{88, 1, 71}}, {{10, 1, 16}}, {{50, 3, 3}}, {{50, 3, 128}}, {{1, 1, 16}}};

const auto params_MLIR =
        testing::Combine(testing::ValuesIn(netPrecisions),                                           // Model type
                         testing::ValuesIn(static_shapes_to_test_representation(inputShapes_MLIR)),  // Input shapes
                         testing::ValuesIn(mergeRepeated),                                           // Merge repeated
                         testing::Values(test_utils::TARGET_DEVICE));                                // Device name

INSTANTIATE_TEST_SUITE_P(smoke_CTCGreedyDecoder, CTCGreedyDecoderLayerTestCommon, params_MLIR,
                         CTCGreedyDecoderLayerTest::getTestCaseName);

}  // namespace
