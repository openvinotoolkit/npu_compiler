//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pretty_test_arguments.hpp"
#include "vpu_ov2_layer_test.hpp"

#include <common/print_test_case_name.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/op/exp.hpp>
#include <openvino/op/parameter.hpp>

namespace ov::test {

PRETTY_PARAM(Input, ov::test::InputShape);

using DynamicExpTestParams = std::tuple<Input>;

class DynamicExpLayerTest : public testing::WithParamInterface<DynamicExpTestParams>, public VpuOv2LayerTest {
protected:
    void SetUp() override {
        const auto& [inputShape] = GetParam();
        init_input_shapes({inputShape.value()});
        auto shape = inputDynamicShapes.front();
        const auto dataParam = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape);
        const auto exp = std::make_shared<ov::op::v0::Exp>(dataParam);
        function = std::make_shared<ov::Model>(exp, ov::ParameterVector{dataParam}, "DynamicExp");
    }
};

TEST_P(DynamicExpLayerTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicExpLayerTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicExpLayerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(DynamicExpLayerTest, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<Input> inputShapes = {generateTestShape(14_Dyn, 1), generateTestShape(128_Dyn)};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicExp, DynamicExpLayerTest, ::testing::Combine(::testing::ValuesIn(inputShapes)),
                         PrintTestCaseName());

}  // namespace ov::test
