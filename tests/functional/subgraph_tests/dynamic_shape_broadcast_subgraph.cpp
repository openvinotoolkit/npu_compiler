//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_ov2_layer_test.hpp"

#include <common/print_test_case_name.hpp>
#include <pretty_test_arguments.hpp>

#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/opsets/opset1_decl.hpp>
#include <openvino/opsets/opset3_decl.hpp>

#include "openvino/op/broadcast.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"

using namespace ov::test;
using namespace ov::test::utils;

namespace {

// Subgraph with ShapeOf

using BroadcastSubgraphShapeOfParams =
        std::tuple<std::vector<ov::test::InputShape>, ov::element::Type, ov::op::BroadcastType>;

class BroadcastWithShapeOfNPUTest :
        public testing::WithParamInterface<BroadcastSubgraphShapeOfParams>,
        public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BroadcastSubgraphShapeOfParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "IS=";

        ov::element::Type inputType;
        std::vector<ov::test::InputShape> shapes;
        ov::op::BroadcastType mode;

        std::tie(shapes, inputType, mode) = obj.param;

        for (auto shape : shapes) {
            result << vec2str(shape.second) << sep;
        }

        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        int32_t i = 0;
        for (const auto& funcInput : funcInputs) {
            if (funcInput.get_element_type() == ov::element::boolean) {
                auto tensor = ov::Tensor{funcInput.get_element_type(), targetInputStaticShapes[i]};
                auto inputData = tensor.data<ov::element_type_traits<ov::element::boolean>::value_type>();
                const auto totalSize = std::accumulate(targetInputStaticShapes[i].begin(),
                                                       targetInputStaticShapes[i].end(), 1, std::multiplies<size_t>());
                for (size_t i = 0; i < totalSize; i += 2) {
                    inputData[i] = false;
                }
                for (size_t i = 1; i < totalSize; i += 2) {
                    inputData[i] = true;
                }
                inputs.insert({funcInput.get_node_shared_ptr(), tensor});
                i++;
            } else {
                ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                            targetInputStaticShapes[i], 100, 0);
                inputs.insert({funcInput.get_node_shared_ptr(), tensor});
                i++;
            }
        }
    }

protected:
    void SetUp() override {
        const auto& [shapes, typeForInput, mode] = this->GetParam();

        const auto& inputShapeForBroadcast = shapes[0];
        const auto& inputShapeForParamBeforeShapeOf = shapes[1];

        init_input_shapes({inputShapeForBroadcast, inputShapeForParamBeforeShapeOf});
        ov::ParameterVector inputParams;

        // broadcast input
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(typeForInput, inputDynamicShapes[0]));
        // parameter input - to be followed by ShapeOf
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(typeForInput, inputDynamicShapes[1]));

        inputParams[1]->set_friendly_name("input_0");
        auto shapeOf = std::make_shared<ov::opset3::ShapeOf>(inputParams[1]);

        inputParams[0]->set_friendly_name("input_1");
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(inputParams[0], shapeOf, mode);

        auto results = ov::ResultVector();
        for (size_t i = 0; i < broadcast->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::opset3::Result>(broadcast->output(i)));
        }

        function = std::make_shared<ov::Model>(results, inputParams, "DynamicBroadcastShapeSubgraph");
    }
};

TEST_P(BroadcastWithShapeOfNPUTest, NPU3720_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(BroadcastWithShapeOfNPUTest, NPU4000_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(BroadcastWithShapeOfNPUTest, NPU5010_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<ov::element::Type> inputType = {ov::element::i64, ov::element::i32, ov::element::f16,
                                                  ov::element::f32};

const std::vector<ov::op::BroadcastType> broadcastModes = {ov::op::BroadcastType::NUMPY,
                                                           ov::op::BroadcastType::BIDIRECTIONAL};
//     *---------------*
//     | Dynamic shape |
//     *---------------*
//             |
//             |
//     *----------------*     *--------------*
//     |     ShapeOf    |     | Static shape |
//     | <target_shape> |     |   <data>     |
//     *----------------*     *--------------*
//             |                     |
//             |                     |
//              \                    /
//               \                  /
//                *----------------*
//                |    Broadcast   |
//                *----------------*
const std::vector<std::vector<ov::test::InputShape>> inShapesShapeOfBroadcastDataStaticUnknownTargetShape = {
        {generateTestShape(1), {{1, 4, ov::Dimension(1, 5)}, {{1, 4, 4}}}},

        {generateTestShape(1, 1), {{1, 1, ov::Dimension(1, 5)}, {{1, 1, 3}}}},

        {generateTestShape(4, 1, 1), {{1, 4, ov::Dimension(1, 5), ov::Dimension(1, 5)}, {{1, 4, 2, 2}}}},

        /* E#151546 this case fails sporadically. Upd: Could not reproduce the failure */
        {generateTestShape(1, 1, 1), {{ov::Dimension(1, 10), 1, ov::Dimension(1, 5)}, {{8, 1, 3}}}},

        {generateTestShape(1, 16, 1, 1), {{1, 16, 1, ov::Dimension(1, 5)}, {{1, 16, 1, 3}}}}};

//     *---------------*
//     | Dynamic shape |
//     *---------------*
//             |
//             |
//     *----------------*     *---------------*
//     |     ShapeOf    |     | Dynamic shape |
//     | <target_shape> |     |    <data>     |
//     *----------------*     *---------------*
//              |                   |
//              |                   |
//               \                 /
//                \               /
//                *----------------*
//                |    Broadcast   |
//                *----------------*
const std::vector<std::vector<ov::test::InputShape>> inShapesShapeOfBroadcastDataDynamicUnknownTargetShape = {
        {{{ov::Dimension(1, 5), ov::Dimension(1, 10), ov::Dimension(1, 15)}, {{1, 1, 1}}},
         {{ov::Dimension(1, 2), ov::Dimension(1, 15), ov::Dimension(1, 25)}, {{1, 4, 3}}}},

        {{{ov::Dimension(1, 5), ov::Dimension(1, 15)}, {{2, 1}}},
         {{ov::Dimension(1, 15), ov::Dimension(1, 25)}, {{2, 8}}}},

        {{{ov::Dimension(1, 5)}, {{1}}}, {{ov::Dimension(1, 2), ov::Dimension(1, 5)}, {{1, 4}}}},

        {{{ov::Dimension(1, 5), ov::Dimension(1, 10), ov::Dimension(1, 15)}, {{1, 1, 1}}},
         {{1, 4, ov::Dimension(1, 15), ov::Dimension(1, 25)}, {{1, 4, 3, 20}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_BroadcastWithShapeOfDataStaticUnknownTargetShape, BroadcastWithShapeOfNPUTest,
                         ::testing::Combine(::testing::ValuesIn(inShapesShapeOfBroadcastDataStaticUnknownTargetShape),
                                            ::testing::ValuesIn(inputType), ::testing::ValuesIn(broadcastModes)),
                         BroadcastWithShapeOfNPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BroadcastWithShapeOfDataDynamicUnknownTargetShape, BroadcastWithShapeOfNPUTest,
                         ::testing::Combine(::testing::ValuesIn(inShapesShapeOfBroadcastDataDynamicUnknownTargetShape),
                                            ::testing::ValuesIn(inputType), ::testing::ValuesIn(broadcastModes)),
                         BroadcastWithShapeOfNPUTest::getTestCaseName);

// Subgraph with Select

using BroadcastSubgraphSelectParams = std::tuple<std::vector<ov::test::InputShape>, ov::element::Type>;

class BroadcastWithSelectNPUTest :
        public testing::WithParamInterface<BroadcastSubgraphSelectParams>,
        public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BroadcastSubgraphSelectParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "IS=";

        ov::element::Type inputType;
        std::vector<ov::test::InputShape> shapes;

        std::tie(shapes, inputType) = obj.param;

        for (auto shape : shapes) {
            result << vec2str(shape.second) << sep;
        }

        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        const int32_t startFrom = 0;
        const int32_t range = 2;

        for (const auto& funcInput : funcInputs) {
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                        targetInputStaticShapes[0], range, startFrom);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        const auto& [shapes, typeForSelect] = this->GetParam();

        const auto& inputShapeForBroadcast = shapes[0];

        init_input_shapes({inputShapeForBroadcast});
        ov::ParameterVector inputParams;

        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(typeForSelect, shape));
        }

        const ov::Shape constShape({3});
        auto shapeOfResult = ov::op::v0::Constant::create(typeForSelect, constShape, {1, 1, 4});
        const auto constTensor = ov::op::v0::Constant::create(typeForSelect, constShape, {1, 4, 4});

        auto equal = std::make_shared<ov::opset1::Equal>(shapeOfResult, constTensor);
        auto select = std::make_shared<ov::opset1::Select>(equal, constTensor, shapeOfResult);

        inputParams[0]->set_friendly_name("input_1");
        auto broadcast =
                std::make_shared<ov::op::v3::Broadcast>(inputParams[0], select, ov::op::BroadcastType::BIDIRECTIONAL);

        auto results = ov::ResultVector();
        for (size_t i = 0; i < broadcast->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::opset3::Result>(broadcast->output(i)));
        }

        function = std::make_shared<ov::Model>(results, inputParams, "DynamicBroadcastShapeSubgraph");
    }
};

TEST_P(BroadcastWithSelectNPUTest, NPU3720_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(BroadcastWithSelectNPUTest, NPU4000_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(BroadcastWithSelectNPUTest, NPU5010_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

// *----------------*
// |     ShapeOf    |
// | <target_shape> |
// *----------------*
//         |
// *----------------*
// |     Select     |    *------------------------*
// | <target_shape> |    |         <data>         |
// *----------------*    *------------------------*
//         |                     |
//         |                     |
//         \                    /
//          \                  /
//           *----------------*
//           |   Broadcast    |
//           *----------------*
const std::vector<std::vector<ov::test::InputShape>> inShapesSelectBroadcast = {
        {{{ov::Dimension(1, 5), ov::Dimension(1, 10)}, {{1, 1}}}},

        {{{ov::Dimension(1, 5)}, {{4}}}},
};

const std::vector<ov::element::Type> inputTypeSelect = {ov::element::i64, ov::element::i32};

INSTANTIATE_TEST_SUITE_P(smoke_BroadcastWithSelect, BroadcastWithSelectNPUTest,
                         ::testing::Combine(::testing::ValuesIn(inShapesSelectBroadcast),
                                            ::testing::ValuesIn(inputTypeSelect)),
                         BroadcastWithSelectNPUTest::getTestCaseName);

}  // namespace
