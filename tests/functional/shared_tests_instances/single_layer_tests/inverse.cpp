//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/inverse.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class InverseLayerTestCommon : public InverseLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(InverseLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(InverseLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
const std::vector<std::vector<ov::test::InputShape>> input_shapes = {
        {{{10, 10}, {{10, 10}}}}, {{{1, 10, 2, 2}, {{1, 10, 2, 2}}}}, {{{5, 2, 4, 3, 3}, {{5, 2, 4, 3, 3}}}}};

const auto shapes = testing::ValuesIn(input_shapes);
const auto modelTypes = testing::Values(ov::element::f32, ov::element::f16);
const auto adjoint = testing::Values(false, true);
const auto seed = testing::Values(0, 1);

const auto params = ::testing::Combine(shapes, modelTypes, adjoint, seed, testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_InverseStatic, InverseLayerTestCommon, params, InverseLayerTestCommon::getTestCaseName);

}  // namespace
