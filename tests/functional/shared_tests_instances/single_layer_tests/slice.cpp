//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/slice.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class SliceLayerTestCommon : public Slice8LayerTest, virtual public VpuOv2LayerTest {};

TEST_P(SliceLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(SliceLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(SliceLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using ov::test::SliceLayerTestCommon;

namespace {

std::vector<ov::test::Slice8SpecificParams> staticParams = {

        ov::test::Slice8SpecificParams{{{{}, {{16}}}}, {4}, {12}, {1}, {0}},
        ov::test::Slice8SpecificParams{{{{}, {{20, 10}}}}, {0, 0}, {10, 20}, {1, 1}, {1, 0}},
        ov::test::Slice8SpecificParams{{{{}, {{1, 12, 100}}}}, {0, 9, 0}, {1, 11, 1}, {1, 1, 1}, {0, 1, -1}},
        ov::test::Slice8SpecificParams{{{{}, {{2, 30, 50}}}}, {0, 0, 4}, {-5, -1, -1}, {1, 2, 1}, {2, 0, 1}},
        ov::test::Slice8SpecificParams{{{{}, {{16}}}}, {0}, {8}, {2}, {0}}};

const std::vector<ov::element::Type> modelTypes = {ov::element::f16, ov::element::f32, ov::element::i32,
                                                   ov::element::u32, ov::element::u8,  ov::element::i8};

const auto sliceParams = testing::Combine(testing::ValuesIn(staticParams),  // params
                                          testing::ValuesIn(modelTypes),    // Model type
                                          testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Slice, SliceLayerTestCommon, sliceParams, SliceLayerTestCommon::getTestCaseName);

}  // namespace
