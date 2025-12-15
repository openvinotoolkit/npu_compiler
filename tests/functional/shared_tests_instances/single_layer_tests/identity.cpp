//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <common_test_utils/ov_tensor_utils.hpp>
#include <common_test_utils/test_constants.hpp>

#include <openvino/op/identity.hpp>
#include <shared_test_classes/single_op/identity.hpp>

#include "vpu_ov2_layer_test.hpp"

using ov::test::IdentityLayerTest;
using namespace ov::test::utils;

namespace ov {
namespace test {

class IdentityLayerTestCommon : public IdentityLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(IdentityLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(IdentityLayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace test
}  // namespace ov

using ov::test::IdentityLayerTestCommon;

namespace {

const std::vector<ov::element::Type> modeltypes = {
        ov::element::f32,
        ov::element::f16,
};

std::vector<std::vector<ov::Shape>> inputShapeStatic4D = {{{2, 2, 2, 2}}, {{1, 10, 2, 3}}, {{2, 3, 4, 5}}};

INSTANTIATE_TEST_SUITE_P(
        smoke_Identity4D, IdentityLayerTestCommon,
        ::testing::Combine(::testing::ValuesIn(modeltypes),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapeStatic4D)),
                           ::testing::Values(test_utils::TARGET_DEVICE)),
        IdentityLayerTestCommon::getTestCaseName);

}  // namespace
