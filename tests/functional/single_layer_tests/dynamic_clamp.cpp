//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pretty_test_arguments.hpp"
#include "vpu_ov2_layer_test.hpp"

#include <common/print_test_case_name.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/parameter.hpp>

namespace ov::test {

PRETTY_PARAM(Input, ov::test::InputShape);

using DynamicClampTestParams = std::tuple<Input>;

class DynamicClampLayerTest : public testing::WithParamInterface<DynamicClampTestParams>, public VpuOv2LayerTest {
protected:
    void SetUp() override {
        const auto& [inputShape] = GetParam();
        init_input_shapes({inputShape.value()});
        auto shape = inputDynamicShapes.front();
        const auto dataParam = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape);
        const auto clamp = std::make_shared<ov::op::v0::Clamp>(dataParam, -1.0, 1.0);
        function = std::make_shared<ov::Model>(clamp, ov::ParameterVector{dataParam}, "DynamicClamp");
    }
};

TEST_P(DynamicClampLayerTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicClampLayerTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicClampLayerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(DynamicClampLayerTest, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<Input> inputShapes = {generateTestShape(14_Dyn, 1), generateTestShape(128_Dyn)};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicClamp, DynamicClampLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes)), PrintTestCaseName());

}  // namespace ov::test
