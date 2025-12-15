// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_ov2_layer_test.hpp"
#include "vpux/utils/core/error.hpp"

#include <common/print_test_case_name.hpp>
#include <openvino/core/type/element_type.hpp>
#include <pretty_test_arguments.hpp>
#include "vpu_ov2_layer_test.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/opsets/opset8.hpp>
#include <random>
#include <vector>
#include "openvino/opsets/opset1.hpp"

namespace ov::test {

PRETTY_PARAM(InputType, ov::element::Type);

using DynamicGatherTestParams = std::tuple<std::vector<int32_t>, InputType>;

//
// DynamicGatherLayerTest
//

class DynamicGatherLayerTest : public testing::WithParamInterface<DynamicGatherTestParams>, public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        const int32_t startFrom = 0;
        const int32_t range = 3;

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                        targetInputStaticShapes[i], range, startFrom);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    std::vector<int64_t> generateConst(const ov::Shape& shape) {
        size_t totalElements = 1;
        for (size_t dim : shape) {
            totalElements *= dim;
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int64_t> dis(0, 100);

        std::vector<int64_t> randomNumbers(totalElements);
        for (size_t i = 0; i < totalElements; ++i) {
            randomNumbers[i] = dis(gen);
        }

        return randomNumbers;
    }

protected:
    void SetUp() override {
        const auto& [InputShape, type] = this->GetParam();
        ov::Shape inputConstShape(InputShape.begin(), InputShape.end());

        std::vector<int64_t> randomInput = generateConst(inputConstShape);
        auto inputConst = ov::op::v0::Constant::create(type.value(), inputConstShape, randomInput);

        auto axisParam = ov::op::v0::Constant::create(ov::element::i32, Shape{}, std::vector<int32_t>{0});
        const auto inferenceIndicesShapes = std::vector<ov::Shape>{{1, 32}};
        const auto indicesShape = ov::test::InputShape{{1, ov::Dimension(1, 64)}, inferenceIndicesShapes};
        init_input_shapes({indicesShape});
        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::i64, inputDynamicShapes.at(0));
        auto gather = std::make_shared<ov::op::v8::Gather>(inputConst, param->output(0), axisParam);
        function = std::make_shared<ov::Model>(gather, ov::ParameterVector{param}, "DynamicGather");
    }
};

TEST_P(DynamicGatherLayerTest, NPU3720_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicGatherLayerTest, NPU4000_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicGatherLayerTest, NPU5010_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<InputType> inputPrecision = {ov::element::f16};

const std::vector<std::vector<int32_t>> inShapes = {{2, 128}, {8, 64}};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicGather, DynamicGatherLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(inputPrecision)),
                         PrintTestCaseName());
}  // namespace ov::test
