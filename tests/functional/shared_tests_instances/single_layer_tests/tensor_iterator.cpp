//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/tensor_iterator.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov;
using namespace element;

namespace ov::test {

class TensorIteratorLayerTestCommon : public TensorIteratorTest, virtual public VpuOv2LayerTest {};

class TensorIteratorLayerTest_NPU3720 : public TensorIteratorLayerTestCommon {};

TEST_P(TensorIteratorLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<ov::op::RecurrentSequenceDirection> directions = {ov::op::RecurrentSequenceDirection::FORWARD,
                                                                    ov::op::RecurrentSequenceDirection::REVERSE};

const std::vector<ov::test::utils::TensorIteratorBody> tiBodyTypes = {ov::test::utils::TensorIteratorBody::GRU};

const std::vector<ov::test::utils::TensorIteratorBody> tiBodyTypes_RNN = {ov::test::utils::TensorIteratorBody::RNN};

const auto tensorIteratorPrecommitParams_RNN =
        ::testing::Combine(testing::Values(false),                                        // should_decompose
                           testing::Values(2),                                            // seq_lengths
                           testing::Values(4),                                            // batch
                           testing::Values(4),                                            // hidden_size
                           testing::Values(1),                                            // sequence_axis
                           testing::ValuesIn({0.0f}),                                     // clip
                           testing::ValuesIn(tiBodyTypes_RNN),                            // ti_body
                           testing::Values(ov::op::RecurrentSequenceDirection::FORWARD),  // direction
                           testing::ValuesIn(modelTypes),                                 // netPrecision
                           testing::Values(test_utils::TARGET_DEVICE));                   // targetDevice

const auto tensorIteratorPrecommitParamsAll_1 =
        ::testing::Combine(testing::Values(false),                       // should_decompose
                           testing::Values(2),                           // seq_lengths
                           testing::Values(4),                           // batch
                           testing::Values(4),                           // hidden_size
                           testing::Values(1),                           // sequence_axis
                           testing::ValuesIn({0.0f}),                    // clip
                           testing::ValuesIn(tiBodyTypes),               // ti_body
                           testing::ValuesIn(directions),                // direction
                           testing::ValuesIn(modelTypes),                // netPrecision
                           testing::Values(test_utils::TARGET_DEVICE));  // targetDevice

const auto tensorIteratorPrecommitParamsAll_2 =
        ::testing::Combine(testing::Values(false),                       // should_decompose
                           testing::Values(4),                           // seq_lengths
                           testing::Values(4),                           // batch
                           testing::Values(4),                           // hidden_size
                           testing::Values(1),                           // sequence_axis
                           testing::ValuesIn({0.0f}),                    // clip
                           testing::ValuesIn(tiBodyTypes),               // ti_body
                           testing::ValuesIn(directions),                // direction
                           testing::ValuesIn(modelTypes),                // netPrecision
                           testing::Values(test_utils::TARGET_DEVICE));  // targetDevice

// RNN Op is not supported. Tracked By Issue: [E#-117139]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_precommit_TensorIterator, TensorIteratorLayerTest_NPU3720,
                         tensorIteratorPrecommitParams_RNN, TensorIteratorLayerTestCommon ::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_TensorIterator_1, TensorIteratorLayerTest_NPU3720,
                         tensorIteratorPrecommitParamsAll_1, TensorIteratorLayerTestCommon ::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_TensorIterator_2, TensorIteratorLayerTest_NPU3720,
                         tensorIteratorPrecommitParamsAll_2, TensorIteratorLayerTestCommon ::getTestCaseName);

}  // namespace ov::test
