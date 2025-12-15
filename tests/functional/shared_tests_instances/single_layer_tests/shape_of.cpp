//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/shape_of.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class ShapeOfLayerTestCommon : public ShapeOfLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(ShapeOfLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ShapeOfLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(ShapeOfLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using ov::test::ShapeOfLayerTestCommon;

namespace {
const std::vector<ov::element::Type> modelTypes = {ov::element::f16, ov::element::u8};

const std::vector<std::vector<ov::Shape>> inShapes = {
        std::vector<ov::Shape>{{10}},
        std::vector<ov::Shape>{{10, 11}},
        std::vector<ov::Shape>{{10, 11, 12}},
        std::vector<ov::Shape>{{10, 11, 12, 13}},
        std::vector<ov::Shape>{{10, 11, 12, 13, 14}},
        std::vector<ov::Shape>{{2, 3, 244, 244}},
        std::vector<ov::Shape>{{2, 4, 8, 16, 32}},
};

const std::vector<std::vector<ov::Shape>> inShapes_precommit = {
        std::vector<ov::Shape>{{3, 3, 5}},
        std::vector<ov::Shape>{{5, 7, 6, 3}},
};

const auto paramsConfig1 =
        testing::Combine(::testing::ValuesIn(modelTypes), ::testing::Values(ov::element::i32),
                         ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                         ::testing::Values(test_utils::TARGET_DEVICE));
const auto paramsPrecommit =
        testing::Combine(::testing::Values(ov::element::f16), ::testing::Values(ov::element::i32),
                         ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes_precommit)),
                         ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf, ShapeOfLayerTestCommon, paramsConfig1, ShapeOfLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ShapeOf, ShapeOfLayerTestCommon, paramsPrecommit,
                         ShapeOfLayerTestCommon::getTestCaseName);

}  // namespace
