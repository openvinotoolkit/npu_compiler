//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/batch_norm.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
namespace ov {
namespace test {

class BatchNormLayerTestCommon : public BatchNormLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(BatchNormLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(BatchNormLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<std::vector<ov::Shape>> inShapes_precommit = {
        {{1, 5, 20, 20}},
};
const std::vector<std::vector<ov::Shape>> inShapes = {
        {{6, 7}},
        {{3, 3, 5}},
        {{5, 7, 6, 3}},
        {{1, 3, 256, 256}},
};

const auto paramsConfig =
        testing::Combine(testing::Values(0.001),                                             // epsilon
                         testing::Values(ov::element::f16),                                  // Model type
                         testing::ValuesIn(static_shapes_to_test_representation(inShapes)),  // Input shape
                         testing::Values(test_utils::TARGET_DEVICE));                        // Target device name

const auto paramsPrecommit =
        testing::Combine(testing::Values(0.001),                                                       // epsilon
                         testing::Values(ov::element::f16),                                            // Model type
                         testing::ValuesIn(static_shapes_to_test_representation(inShapes_precommit)),  // Input shape
                         testing::Values(test_utils::TARGET_DEVICE));  // Target device name

INSTANTIATE_TEST_SUITE_P(smoke_precommit_BatchNorm, BatchNormLayerTestCommon, paramsPrecommit,
                         BatchNormLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BatchNorm, BatchNormLayerTestCommon, paramsConfig,
                         BatchNormLayerTestCommon::getTestCaseName);

}  // namespace
