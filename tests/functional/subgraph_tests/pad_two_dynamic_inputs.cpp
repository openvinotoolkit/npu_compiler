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

using Input1AndInput2Shape = std::tuple<std::vector<int32_t>, std::vector<int32_t>>;
using DynamicGatherwithAddTestParams = std::tuple<Input1AndInput2Shape, InputType>;

//
// DynamicGatherwithAddNPUTest
//

//     *----------------*     *----------------*
//     |  DynamicGather |     |  DynamicGather |
//     *----------------*     *----------------*
//             |                      |
//              \                    /
//               \                  /
//                *----------------*
//                |      Add       |
//                *----------------*

class DynamicGatherwithAddNPUTest :
        public testing::WithParamInterface<DynamicGatherwithAddTestParams>,
        public VpuOv2LayerTest {
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
        const auto& [Inputs, type] = this->GetParam();
        const auto& [Input1ConstSize, Input2ConstSize] = Inputs;

        ov::Shape input1ConstShape(Input1ConstSize.begin(), Input1ConstSize.end());
        std::vector<int64_t> randomInput1 = generateConst(input1ConstShape);
        auto inputConst1 = ov::op::v0::Constant::create(type.value(), input1ConstShape, randomInput1);

        ov::Shape input2ConstShape(Input2ConstSize.begin(), Input2ConstSize.end());
        std::vector<int64_t> randomInput2 = generateConst(input2ConstShape);
        auto inputConst2 = ov::op::v0::Constant::create(type.value(), input2ConstShape, randomInput2);

        auto axisParam = ov::op::v0::Constant::create(ov::element::i32, Shape{}, std::vector<int32_t>{0});
        const auto inferenceIndicesShapes = std::vector<ov::Shape>{{1, 32}};
        const auto indicesShape = ov::test::InputShape{{1, ov::Dimension(1, 64)}, inferenceIndicesShapes};
        init_input_shapes({indicesShape});
        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::i64, inputDynamicShapes.at(0));
        auto gather1 = std::make_shared<ov::op::v8::Gather>(inputConst1, param->output(0), axisParam);
        auto gather2 = std::make_shared<ov::op::v8::Gather>(inputConst2, param->output(0), axisParam);
        auto addResult = std::make_shared<ov::op::v1::Add>(gather1, gather2);
        function = std::make_shared<ov::Model>(addResult, ov::ParameterVector{param}, "DynamicGatherwithAdd");
    }
};

TEST_P(DynamicGatherwithAddNPUTest, NPU3720_HW_TestKindSubgraph) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicGatherwithAddNPUTest, NPU4000_HW_TestKindSubgraph) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicGatherwithAddNPUTest, NPU5010_HW_TestKindSubgraph) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<InputType> inputPrecision = {ov::element::f16};

const std::vector<Input1AndInput2Shape> inShapes = {
        {{8, 128}, {2, 128}},
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicGatherwithAdd, DynamicGatherwithAddNPUTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(inputPrecision)),
                         PrintTestCaseName());
}  // namespace ov::test
