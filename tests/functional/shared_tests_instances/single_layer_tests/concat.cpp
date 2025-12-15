//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/concat.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class ConcatLayerTestCommon : public ConcatLayerTest, virtual public VpuOv2LayerTest {};
class ConcatSubByteCopyLayerTest : public ConcatLayerTestCommon {};

TEST_P(ConcatLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConcatLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(ConcatSubByteCopyLayerTest, NPU4000_HW) {
    // Tracking number [C#163362]
    setSkipInferenceCallback([](std::stringstream& skip) {
        skip << "OpenVINO error after inference, 'Tensor data with element type u4, is not representable as pointer "
                "to i8'";
    });
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(ConcatLayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

std::vector<int> axes = {0, 1, 2, 3};

std::vector<ov::element::Type> netPrecisions = {ov::element::f16, ov::element::u8};

const auto concatParams = ::testing::Combine(
        ::testing::ValuesIn(axes),
        ::testing::Values(ov::test::static_shapes_to_test_representation({{1, 16, 10, 10}, {1, 16, 10, 10}})),
        ::testing::Values(ov::element::u8), ::testing::Values(test_utils::TARGET_DEVICE));

const auto concatCopyParams = ::testing::Combine(
        ::testing::Values(3),
        ::testing::Values(ov::test::static_shapes_to_test_representation({{1, 2, 3, 3}, {1, 2, 3, 3}})),
        ::testing::ValuesIn({ov::element::u4, ov::element::i4}), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Concat, ConcatLayerTestCommon, concatParams,
                         ConcatLayerTestCommon::getTestCaseName);
// Sub-byte case with non-byte aligned dimension and testing builtin SW.Kernel Copy
// Tracking number [E#158865] - Conversion of non byte-aligned bits to bytes
// Tracking number [E#160558] - Direct access to Network I/O params
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_precommit_ConcatCopySubByte, ConcatSubByteCopyLayerTest, concatCopyParams,
                         ConcatLayerTestCommon::getTestCaseName);

}  // namespace
