// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/einsum.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class EinsumLayerTestCommon : public EinsumLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(EinsumLayerTestCommon, NPU3720) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(EinsumLayerTestCommon, NPU4000) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::test::EinsumEquationWithInput> equationsWithInput = {
        {"ij->ji", ov::test::static_shapes_to_test_representation({{1, 2}})},
        {"ij->i", ov::test::static_shapes_to_test_representation({{2, 3}})},
        {"ab,cd->abcd", ov::test::static_shapes_to_test_representation({{1, 2}, {3, 4}})},
        {"ab,bc->ac", ov::test::static_shapes_to_test_representation({{2, 3}, {3, 2}})},
};

const std::vector<ov::test::EinsumEquationWithInput> equationsWithInput_PBI = {
        {"ibnd,hnd->ibh", ov::test::static_shapes_to_test_representation({{128, 1, 12, 64}, {768, 12, 64}})},
        {"ijbn->bnij", ov::test::static_shapes_to_test_representation({{128, 128, 1, 1}})}};
const std::vector<ov::element::Type> model_types = {
        ov::element::f32, ov::element::i32
        // ov::element::f16 Unsupported precision [C-138797]
};
const auto params = ::testing::Combine(::testing::ValuesIn(model_types), ::testing::ValuesIn(equationsWithInput_PBI),
                                       ::testing::Values(DEVICE_NPU));

const auto params_precommit = ::testing::Combine(
        ::testing::ValuesIn(model_types), ::testing::ValuesIn(equationsWithInput), ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Einsum, EinsumLayerTestCommon, params_precommit,
                         EinsumLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Einsum, EinsumLayerTestCommon, params, EinsumLayerTest::getTestCaseName);

}  // namespace
