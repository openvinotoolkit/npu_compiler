//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pretty_test_arguments.hpp"
#include "vpu_ov2_layer_test.hpp"

#include <common/print_test_case_name.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/opsets/opset1_decl.hpp>

namespace ov::test {

PRETTY_PARAM(Input, ov::test::InputShape);
PRETTY_PARAM(NewShape, std::vector<int64_t>);

using ReshapeLayerTestParams = std::tuple<Input, NewShape>;

class DynamicReshapeLayerTest : public testing::WithParamInterface<ReshapeLayerTestParams>, public VpuOv2LayerTest {
public:
protected:
    void SetUp() override {
        const auto& [inputShape, newShape] = GetParam();
        init_input_shapes({inputShape.value()});
        auto shape = inputDynamicShapes.front();
        const auto dataParam = std::make_shared<ov::opset1::Parameter>(ov::element::i32, shape);
        const auto newShapeConstant = std::make_shared<ov::opset1::Constant>(
                ov::element::i64, ov::Shape{newShape.value().size()}, newShape.value());
        const auto reshape = std::make_shared<ov::opset1::Reshape>(dataParam, newShapeConstant, false);
        function = std::make_shared<ov::Model>(reshape, ov::ParameterVector{dataParam}, "Reshape");
    }
};

TEST_P(DynamicReshapeLayerTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicReshapeLayerTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicReshapeLayerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
TEST_P(DynamicReshapeLayerTest, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<Input> inputShapes = {generateTestShape(128_Dyn, 1)};

const std::vector<NewShape> newShapes = {{NewShape{-1}}};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicReshape, DynamicReshapeLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(newShapes)),
                         PrintTestCaseName());

}  // namespace ov::test
